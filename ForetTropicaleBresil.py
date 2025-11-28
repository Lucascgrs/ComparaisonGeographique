# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import argparse
import shutil
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import shapely.geometry
from PIL import Image
from tqdm import tqdm
import sentinelhub
import warnings

warnings.filterwarnings("ignore")

try:
    import mycredentials
except ImportError:
    print("⚠️ mycredentials.py manquant.")

# ============================================================================
# CONSTANTES
# ============================================================================
BRAZIL_STATES = [
    # --- Priorité 1 : Amazonie (Le front de déforestation) ---
    "Pará, Brazil", 
    "Mato Grosso, Brazil", 
    "Rondônia, Brazil", 
    "Amazonas, Brazil", 
    "Acre, Brazil", 
    "Maranhão, Brazil", 
    "Roraima, Brazil", 
    "Tocantins, Brazil", 
    "Amapá, Brazil",

    # --- Priorité 2 : Le Cerrado & Centre (Agriculture intensive) ---
    "Goiás, Brazil", 
    "Mato Grosso do Sul, Brazil", 
    "Piauí, Brazil", 
    "Bahia, Brazil", 
    "Minas Gerais, Brazil", 
    "Distrito Federal, Brazil",

    # --- Priorité 3 : Sud & Côte (Zones déjà très urbanisées/agricoles) ---
    "São Paulo, Brazil", 
    "Paraná, Brazil", 
    "Rio Grande do Sul, Brazil", 
    "Santa Catarina, Brazil", 
    "Ceará, Brazil", 
    "Rio de Janeiro, Brazil", 
    "Pernambuco, Brazil", 
    "Espírito Santo, Brazil", 
    "Paraíba, Brazil", 
    "Rio Grande do Norte, Brazil", 
    "Alagoas, Brazil", 
    "Sergipe, Brazil"
]

# ============================================================================
# MODULE GÉOGRAPHIE (OPTIMISÉ IBGE)
# ============================================================================
class GeoManager:
    def __init__(self):
        self.municipalities = None
        self.current_aoi = None
        self.current_bbox = None
        self.current_name = None
        
        # Codes des états pour l'API IBGE
        self.ibge_codes = {
            "Rondônia": 11, "Acre": 12, "Amazonas": 13, "Roraima": 14,
            "Pará": 15, "Amapá": 16, "Tocantins": 17, "Maranhão": 21,
            "Piauí": 22, "Ceará": 23, "Rio Grande do Norte": 24, "Paraíba": 25,
            "Pernambuco": 26, "Alagoas": 27, "Sergipe": 28, "Bahia": 29,
            "Minas Gerais": 31, "Espírito Santo": 32, "Rio de Janeiro": 33,
            "São Paulo": 35, "Paraná": 41, "Santa Catarina": 42,
            "Rio Grande do Sul": 43, "Mato Grosso do Sul": 50,
            "Mato Grosso": 51, "Goiás": 52, "Distrito Federal": 53
        }

    def get_municipalities_list(self, state_full_name, limit=None, min_area_km2=2000):
        """Récupération IBGE : Stratégie Hybride (Carte + Noms)"""
        state_name = state_full_name.split(",")[0].strip()
        tqdm.write(f"🌍 Récupération IBGE pour : {state_name}...")

        code = self.ibge_codes.get(state_name)
        if not code: return []

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}

            # ÉTAPE 1 : Télécharger la CARTE (Géométries + Codes)
            url_geo = f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{code}?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=municipio"
            r_geo = requests.get(url_geo, headers=headers)
            r_geo.raise_for_status()

            from io import BytesIO
            gdf = gpd.read_file(BytesIO(r_geo.content))

            # ÉTAPE 2 : Télécharger les NOMS (Métadonnées)
            # On récupère la liste de tous les municipes de l'état avec leurs noms et codes
            url_names = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{code}/municipios"
            r_names = requests.get(url_names, headers=headers)
            r_names.raise_for_status()
            names_data = r_names.json()  # C'est une liste de dicts [{'id': 1500107, 'nome': 'Abaetetuba'...}]

            # Création d'un dictionnaire de mapping : Code -> Nom
            # Attention : codarea dans le GeoJSON est souvent un string, id dans le JSON est un int
            # On convertit tout en string pour être sûr
            code_to_name = {str(item['id']): item['nome'] for item in names_data}

            # ÉTAPE 3 : FUSION (Jointure)
            # La colonne code s'appelle souvent 'codarea' ou 'CD_MUN'
            geo_col = 'codarea' if 'codarea' in gdf.columns else gdf.columns[0]  # On prend la 1ere si on trouve pas

            # On crée la colonne 'name' en mappant
            # On prend les 7 premiers chiffres du code (parfois le code geo a un suffixe)
            gdf['clean_code'] = gdf[geo_col].astype(str).str.slice(0, 7)
            gdf['name_only'] = gdf['clean_code'].map(code_to_name)

            # Nettoyage : On vire ceux qu'on a pas trouvés (rare)
            gdf = gdf.dropna(subset=['name_only'])
            gdf['name'] = gdf['name_only'] + ", Brazil"

            # ÉTAPE 4 : SURFACE & FILTRE
            gdf_area = gdf.to_crs("EPSG:5880")
            gdf['area_km2'] = gdf_area.geometry.area / 10 ** 6

            gdf = gdf[gdf['area_km2'] >= min_area_km2]
            gdf = gdf.sort_values(by='area_km2', ascending=False)

            tqdm.write(f"✅ {len(gdf)} villes chargées (Fusion Carte+Noms réussie).")

            if limit and limit > 0:
                gdf = gdf.head(limit)

            self.municipalities = gdf
            return gdf['name'].tolist()

        except Exception as e:
            tqdm.write(f"❌ Erreur Stratégie Hybride : {e}")
            import traceback
            traceback.print_exc()
            return []

    def set_current_municipality(self, name):
        # Si on a chargé via IBGE, c'est immédiat
        if self.municipalities is not None and name in self.municipalities['name'].values:
            self.current_aoi = self.municipalities[self.municipalities['name'] == name].iloc[[0]]
            # Conversion CRS si nécessaire (IBGE est souvent en SIRGAS 2000 / EPSG:4674)
            if self.current_aoi.crs != "EPSG:4326":
                self.current_aoi = self.current_aoi.to_crs("EPSG:4326")
        else:
            # Fallback OSM si jamais
            try:
                gdf = ox.geocode_to_gdf(name)
                self.current_aoi = gdf.iloc[[0]]
            except: return False
            
        self.current_name = name
        self.current_bbox = self.current_aoi.total_bounds
        return True

    def save_current_map(self, output_dir):
        filename = os.path.join(output_dir, "map_location.html")
        try:
            m = self.current_aoi.explore(color='red', tiles='OpenStreetMap', style_kwds={'fillOpacity': 0.1})
            m.save(filename)
        except: pass

# ============================================================================
# SENTINEL HUB (AUTHENTIFIÉ AVEC CREDENTIALS)
# ============================================================================
class SentinelHubProcessor:
    def __init__(self, bbox, resolution=500):
        # ICI : On configure SentinelHub avec tes clés explicites
        self.config = sentinelhub.SHConfig()
        self.config.sh_client_id = mycredentials.client_id
        self.config.sh_client_secret = mycredentials.client_secret
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        
        self.aoi_bbox_sh = sentinelhub.BBox(bbox=list(bbox), crs=sentinelhub.CRS.WGS84)
        try:
            self.aoi_size = sentinelhub.bbox_to_dimensions(self.aoi_bbox_sh, resolution=resolution)
            max_px = 2500
            if max(self.aoi_size) > max_px:
                scale = max_px / max(self.aoi_size)
                self.aoi_size = (int(self.aoi_size[0] * scale), int(self.aoi_size[1] * scale))
        except: self.aoi_size = (100, 100)

    def get_image(self, evalscript, start_date, end_date, brightness=1.0, filename=None):
        request = sentinelhub.SentinelHubRequest(
            evalscript=evalscript,
            input_data=[sentinelhub.SentinelHubRequest.input_data(
                data_collection=sentinelhub.DataCollection.SENTINEL2_L2A.define_from(name="s2", service_url="https://sh.dataspace.copernicus.eu"),
                time_interval=(start_date, end_date),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
            )],
            responses=[sentinelhub.SentinelHubRequest.output_response("default", sentinelhub.MimeType.TIFF)],
            bbox=self.aoi_bbox_sh, size=self.aoi_size, config=self.config
        )
        try:
            data = request.get_data()
            if not data: return None
            img_array = data[0]
            if brightness != 1.0: img_array = np.uint8((img_array * brightness).clip(0, 255))
            if filename:
                to_save = img_array if img_array.shape[-1] != 1 else img_array.squeeze()
                Image.fromarray(to_save).save(filename)
            return img_array
        except: return None

EVALSCRIPTS = {
    "TRUE_COLOR": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B02", "B03", "B04"] }], output: { bands: 3 } }; }
        function evaluatePixel(sample) { return [sample.B04, sample.B03, sample.B02]; }
    """,
    "NDVI": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B04", "B08"] }], output: { bands: 4 } }; }
        const whiteGreen = [[1.0, 0xFFFFFF], [0.5, 0x000000], [0.0, 0x00FF00]];
        let viz = new ColorGradientVisualizer(whiteGreen, -1.0, 1.0);
        function evaluatePixel(samples) {
            let val = ((samples.B08 + samples.B04)==0) ? 0 : ((samples.B08 - samples.B04) / (samples.B08 + samples.B04));
            let col = viz.process(val); col.push(255); return col;
        }
    """,
    "BURNED_INDEX_RAW": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B08", "B12"] }], output: { bands: 1, sampleType: SampleType.FLOAT32 } }; }
        function evaluatePixel(samples) {
            return [((samples.B08 + samples.B12)==0) ? 0 : ((samples.B08 - samples.B12) / (samples.B08 + samples.B12))];
        }
    """
}

# ============================================================================
# LOGIQUE PRINCIPALE
# ============================================================================

def analyze_municipality_year(geo, name, year_start, year_end, args, is_deep_scan=False):
    """Analyse une paire d'années spécifique pour une ville."""
    try:
        if not geo.set_current_municipality(name): return 0

        # Dossier spécifique : Nom_Ville / 2019-2020
        clean_name = name.replace(" ", "_").replace(",", "")
        period_name = f"{year_start}-{year_end}"
        output_dir = None
        
        if is_deep_scan:
            output_dir = os.path.join("results", clean_name, period_name)
            os.makedirs(output_dir, exist_ok=True)
            # On sauve la carte à la racine de la ville (une seule fois suffit)
            geo.save_current_map(os.path.join("results", clean_name))
            tqdm.write(f"   📸 Deep Scan: {name} ({period_name})")

        res = args.resolution if is_deep_scan else args.scan_resolution
        sh_proc = SentinelHubProcessor(geo.current_bbox, resolution=res)
        
        # Période sèche (Juillet-Septembre)
        interval_before = (f"{year_start}-07-01", f"{year_start}-09-30")
        interval_after = (f"{year_end}-07-01", f"{year_end}-09-30")

        # 1. Calcul Score
        raw_before = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_before)
        raw_after = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_after)

        score = 0
        if raw_before is not None and raw_after is not None:
            diff = raw_before - raw_after
            score = (np.sum(diff > 0.25) / diff.size) * 100
        else:
            return 0

        if not is_deep_scan: return score

        # 2. Génération Images (Seulement Deep Scan)
        # Sauvegarde RAW pour analyse scientifique
        Image.fromarray(raw_before.squeeze()).save(os.path.join(output_dir, "burn_raw_before.tif"))
        
        # Images Visuelles
        img_after = sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_after, brightness=3.5, filename=os.path.join(output_dir, "true_color_after.png"))
        sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_before, brightness=3.5, filename=os.path.join(output_dir, "true_color_before.png"))
        sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_before, filename=os.path.join(output_dir, "ndvi_before.png"))
        sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_after, filename=os.path.join(output_dir, "ndvi_after.png"))

        # Composite
        if img_after is not None:
            composite = np.array(img_after)
            diff = raw_before - raw_after
            if diff.shape[:2] == composite.shape[:2]:
                composite[diff > 0.25] = [255, 128, 0]
                composite[diff > 0.60] = [255, 0, 0]
                Image.fromarray(composite).save(os.path.join(output_dir, "analysis_composite.png"))

        return score

    except Exception as e:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Time Machine Déforestation")
    parser.add_argument("--max_states", type=int, default=1)
    parser.add_argument("--limit_per_state", type=int, default=3)
    parser.add_argument("--top_n", type=int, default=80, help="Nombre de cas critiques à générer")
    parser.add_argument("--start_year", type=int, default=2017)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--scan_resolution", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=210)
    parser.add_argument("--min_area", type=int, default=5000, help="Surface minimale en km² pour analyser une municipalité")
    args = parser.parse_args()

    # Génération des intervalles (ex: 2018-2019, 2019-2020...)
    years_intervals = [(y, y+1) for y in range(args.start_year, args.end_year)]
    columns = ["Municipality", "State"] + [f"{y1}-{y2}" for y1, y2 in years_intervals] + ["Max_Score", "Worst_Year"]
    
    results_data = [] # Liste pour le DataFrame final
    critical_cases = [] # Liste des (Ville, Année, Score) pour le deep scan

    states = BRAZIL_STATES[:args.max_states]
    geo = GeoManager()

    print(f"🚀 TIME MACHINE INITIALISÉE ({args.start_year} -> {args.end_year})")
    print(f"📊 États: {len(states)} | Intervalles: {len(years_intervals)}")

    for state in states:
        cities = geo.get_municipalities_list(state, limit=args.limit_per_state, min_area_km2=args.min_area)
        if not cities: continue

        # Barre de progression des villes
        pbar = tqdm(cities, desc=f"Scan {state.split(',')[0]}")
        for city in pbar:
            row = {"Municipality": city, "State": state}
            max_city_score = 0
            worst_interval = None

            # Analyse Temporelle
            for y1, y2 in years_intervals:
                score = analyze_municipality_year(geo, city, y1, y2, args, is_deep_scan=False)
                col_name = f"{y1}-{y2}"
                row[col_name] = round(score, 2)

                # Tracking du pire moment
                if score > max_city_score:
                    max_city_score = score
                    worst_interval = (y1, y2)
            
            row["Max_Score"] = round(max_city_score, 2)
            row["Worst_Year"] = f"{worst_interval[0]}-{worst_interval[1]}" if worst_interval else "N/A"
            results_data.append(row)

            # Si le score est significatif, on l'ajoute aux candidats pour le deep scan
            if max_city_score > 0.2:
                critical_cases.append({
                    "name": city,
                    "year_start": worst_interval[0],
                    "year_end": worst_interval[1],
                    "score": max_city_score
                })
                pbar.set_postfix({"Pire": f"{max_city_score:.1f}% ({worst_interval[0]})"})

    # --- EXPORT EXCEL ---
    print("\n💾 Génération du fichier Excel...")
    df = pd.DataFrame(results_data, columns=columns)
    df = df.sort_values(by="Max_Score", ascending=False) # Les pires en premier
    df.to_excel("deforestation_report.xlsx", index=False)
    print("✅ Rapport sauvegardé : deforestation_report.xlsx")

    # --- DEEP SCAN SUR LES PIRES CAS ---
    print("\n" + "="*60)
    print(f"🔬 ANALYSE DÉTAILLÉE DES {args.top_n} PIRES CAS")
    print("="*60)
    
    # On trie les cas critiques par score
    critical_cases.sort(key=lambda x: x["score"], reverse=True)
    
    # On prend le top N
    for case in critical_cases[:args.top_n]:
        print(f"👉 {case['name']} : Pic en {case['year_start']}-{case['year_end']} (Score: {case['score']:.2f})")
        analyze_municipality_year(
            geo, 
            case['name'], 
            case['year_start'], 
            case['year_end'], 
            args, 
            is_deep_scan=True
        )

    print("\n✅ Mission Terminée.")

if __name__ == "__main__":
    main()