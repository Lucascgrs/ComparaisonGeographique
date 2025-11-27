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

# Ignorer les warnings géométriques mineurs
warnings.filterwarnings("ignore")

# On essaie d'importer les credentials, sinon on crée un mock
try:
    import mycredentials
except ImportError:
    print("⚠️ mycredentials.py manquant. Certaines fonctions pourraient échouer.")
    class mycredentials:
        username = ""
        password = ""

# ============================================================================
# CONSTANTES : LISTE DES ÉTATS DU BRÉSIL (Ordre Prioritaire Déforestation)
# ============================================================================
# On commence par l'Amazonie Légale ("Legal Amazon") où se concentre la déforestation
BRAZIL_STATES = [
    "Pará, Brazil",
    "Mato Grosso, Brazil",
    "Rondônia, Brazil",
    "Amazonas, Brazil",
    "Acre, Brazil",
    "Maranhão, Brazil",
    "Roraima, Brazil",
    "Tocantins, Brazil",
    "Amapá, Brazil",
    # Autres états (Cerrado / Mata Atlântica)
    "Goiás, Brazil",
    "Bahia, Brazil",
    "Minas Gerais, Brazil",
    "Mato Grosso do Sul, Brazil",
    "Piauí, Brazil",
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
    "Sergipe, Brazil",
    "Distrito Federal, Brazil"
]

# ============================================================================
# MODULE 1: GÉOGRAPHIE & GESTION DE VILLES
# ============================================================================
class GeoManager:
    def __init__(self):
        self.municipalities = None
        self.current_aoi = None
        self.current_bbox = None
        self.current_name = None

    def get_municipalities_list(self, state_name, limit=None):
        """Récupère la liste des municipalités pour un état donné."""
        tqdm.write(f"🌍 Récupération des municipalités pour : {state_name}...")
        try:
            # On récupère les frontières administratives niveau 8 (villes)
            gdf = ox.features_from_place(
                state_name,
                tags={'admin_level': '8', 'boundary': 'administrative'}
            )
            
            if 'name' not in gdf.columns and 'display_name' in gdf.columns:
                gdf['name'] = gdf['display_name']
            
            # Nettoyage
            gdf = gdf[gdf['name'].notna()]
            
            # Tri aléatoire pour ne pas toujours faire les mêmes si on met une limite
            # ou tri par taille (optionnel), ici on prend les premiers retournés
            
            if limit and limit > 0:
                gdf = gdf.head(limit)
            
            self.municipalities = gdf
            tqdm.write(f"✅ {len(gdf)} municipalités trouvées dans {state_name}.")
            return gdf['name'].tolist()
            
        except Exception as e:
            tqdm.write(f"❌ Erreur récupération liste {state_name} : {e}")
            return []

    def set_current_municipality(self, name):
        """Définit la municipalité active pour l'analyse."""
        # Si la municipalité est dans le buffer chargé (le cas normal dans la boucle)
        if self.municipalities is not None and name in self.municipalities['name'].values:
            self.current_aoi = self.municipalities[self.municipalities['name'] == name].iloc[[0]]
        else:
            # Fallback : géocodage direct (pour le deep scan final si besoin)
            try:
                gdf = ox.geocode_to_gdf(f"{name}, Brazil")
                self.current_aoi = gdf.iloc[[0]]
            except:
                return False

        self.current_name = name
        self.current_bbox = self.current_aoi.total_bounds # (minx, miny, maxx, maxy)
        # Note: SentinelHub gère le bbox en liste, pas besoin de polygone shapely ici
        return True

    def save_current_map(self, output_dir):
        """Sauvegarde la carte HTML."""
        filename = os.path.join(output_dir, "map_location.html")
        try:
            m = self.current_aoi.explore(color='red', tiles='OpenStreetMap', style_kwds={'fillOpacity': 0.1})
            m.save(filename)
        except Exception as e:
            print(f"⚠️ Erreur carte HTML : {e}")


# ============================================================================
# MODULE 2: SENTINEL HUB (ANALYSE)
# ============================================================================
class SentinelHubProcessor:
    def __init__(self, bbox, resolution=500):
        self.config = sentinelhub.SHConfig("cdse")
        self.aoi_bbox_sh = sentinelhub.BBox(bbox=list(bbox), crs=sentinelhub.CRS.WGS84)
        
        # Calcul dynamique taille image
        try:
            self.aoi_size = sentinelhub.bbox_to_dimensions(self.aoi_bbox_sh, resolution=resolution)
            # Sécurité taille max (SentinelHub bloque souvent > 2500px en free tier/standard)
            max_px = 2500
            if max(self.aoi_size) > max_px:
                scale = max_px / max(self.aoi_size)
                self.aoi_size = (int(self.aoi_size[0] * scale), int(self.aoi_size[1] * scale))
        except:
            self.aoi_size = (100, 100)

    def get_image(self, evalscript, start_date, end_date, brightness=1.0, filename=None):
        request = sentinelhub.SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                sentinelhub.SentinelHubRequest.input_data(
                    data_collection=sentinelhub.DataCollection.SENTINEL2_L2A.define_from(
                        name="s2", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(start_date, end_date),
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
                )
            ],
            responses=[sentinelhub.SentinelHubRequest.output_response("default", sentinelhub.MimeType.TIFF)],
            bbox=self.aoi_bbox_sh,
            size=self.aoi_size,
            config=self.config,
        )
        
        try:
            data = request.get_data()
            if not data: return None
            img_array = data[0]
            
            if brightness != 1.0:
                img_array = np.uint8((img_array * brightness).clip(0, 255))
                
            if filename:
                to_save = img_array if img_array.shape[-1] != 1 else img_array.squeeze()
                Image.fromarray(to_save).save(filename)
                
            return img_array
        except Exception as e:
            # tqdm.write(f"⚠️ Erreur SH : {e}")
            return None

# ============================================================================
# EVALSCRIPTS
# ============================================================================
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
# CORE LOGIC
# ============================================================================

def analyze_municipality(geo, name, args, is_deep_scan=False):
    """Retourne un score de déforestation (0-100). Génère des fichiers si is_deep_scan=True."""
    try:
        if not geo.set_current_municipality(name):
            return 0

        # Dossier de sortie (seulement pour le deep scan)
        output_dir = None
        if is_deep_scan:
            clean_name = name.replace(" ", "_").replace(",", "")
            output_dir = os.path.join("results", clean_name)
            os.makedirs(output_dir, exist_ok=True)
            geo.save_current_map(output_dir)
            tqdm.write(f"📂 Génération rapport pour {name}...")

        # Résolution adaptative
        res = args.resolution if is_deep_scan else args.scan_resolution
        sh_proc = SentinelHubProcessor(geo.current_bbox, resolution=res)
        
        interval_before = (f"{args.year_before}-07-01", f"{args.year_before}-09-30")
        interval_after = (f"{args.year_after}-07-01", f"{args.year_after}-09-30")

        # 1. Calcul Score (Burn Index)
        raw_before = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_before)
        raw_after = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_after)

        score = 0
        if raw_before is not None and raw_after is not None:
            difference = raw_before - raw_after
            # On compte les pixels > seuil (0.2)
            affected_pixels = np.sum(difference > 0.25)
            total_pixels = difference.size
            score = (affected_pixels / total_pixels) * 100
        else:
            return 0

        if not is_deep_scan:
            return score

        # 2. Génération Deep Scan
        Image.fromarray(raw_before if raw_before.shape[-1] != 1 else raw_before.squeeze()).save(os.path.join(output_dir, "burn_raw_before.tif"))
        
        img_after = sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_after, brightness=3.5, filename=os.path.join(output_dir, "true_color_after.png"))
        sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_before, brightness=3.5, filename=os.path.join(output_dir, "true_color_before.png"))
        
        sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_before, filename=os.path.join(output_dir, "ndvi_before.png"))
        sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_after, filename=os.path.join(output_dir, "ndvi_after.png"))

        # Composite
        if img_after is not None and raw_before is not None and raw_after is not None:
            composite = np.array(img_after)
            diff = raw_before - raw_after
            if diff.shape[:2] == composite.shape[:2]:
                composite[diff > 0.25] = [255, 128, 0]
                composite[diff > 0.60] = [255, 0, 0]
                Image.fromarray(composite).save(os.path.join(output_dir, "analysis_composite.png"))

        return score

    except Exception as e:
        # tqdm.write(f"Erreur {name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Scanner National de Déforestation - Brésil")
    parser.add_argument("--max_states", type=int, default=1, help="Nombre d'états à scanner (dans l'ordre prioritaire)")
    parser.add_argument("--limit_per_state", type=int, default=5, help="Nombre de villes à scanner par état")
    parser.add_argument("--top_n", type=int, default=2, help="Top N final des pires villes à analyser en détail")
    
    parser.add_argument("--year_before", type=str, default="2018")
    parser.add_argument("--year_after", type=str, default="2021")
    parser.add_argument("--scan_resolution", type=int, default=1000, help="Résolution du scan (m)")
    parser.add_argument("--resolution", type=int, default=200, help="Résolution du deep scan (m)")
    
    args = parser.parse_args()

    # Sélection des états
    states_to_scan = BRAZIL_STATES[:args.max_states]
    
    print(f"🚀 DÉMARRAGE DU SCAN GLOBAL ({args.year_before} -> {args.year_after})")
    print(f"📍 États ciblés ({len(states_to_scan)}): {', '.join([s.split(',')[0] for s in states_to_scan])}")
    print(f"⚡ Résolution scan: {args.scan_resolution}m | Limite: {args.limit_per_state} villes/état")
    print("="*60)

    geo = GeoManager()
    global_scores = {}

    # --- BOUCLE SUR LES ÉTATS ---
    for state_name in states_to_scan:
        muni_list = geo.get_municipalities_list(state_name, limit=args.limit_per_state)
        
        if not muni_list:
            continue
            
        print(f"   🔍 Analyse de {len(muni_list)} villes dans {state_name}...")
        
        # Barre de progression par état
        for name in tqdm(muni_list, desc=f"Scan {state_name.split(',')[0]}", leave=False):
            score = analyze_municipality(geo, name, args, is_deep_scan=False)
            
            # On garde le score s'il est pertinent
            if score > 0:
                global_scores[name] = score
                # Affichage dynamique des "gros" cas
                if score > 2.0:
                    tqdm.write(f"      ⚠️  Alert: {name} -> Score: {score:.2f}")

    # --- RÉSULTATS GLOBAUX ---
    sorted_scores = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print(f"🏆 CLASSEMENT NATIONAL (TOP 20) - Score de changement")
    print("="*60)
    for i, (name, score) in enumerate(sorted_scores[:20]):
        print(f"{i+1}. {name:<30} : {score:.2f}")

    # --- DEEP SCAN FINAL ---
    print("\n" + "="*60)
    print(f"🔬 GÉNÉRATION DES RAPPORTS DÉTAILLÉS (TOP {args.top_n})")
    print("="*60)
    
    for name, score in sorted_scores[:args.top_n]:
        analyze_municipality(geo, name, args, is_deep_scan=True)
    
    print(f"\n✅ Analyse terminée. Voir le dossier 'results/'.")

if __name__ == "__main__":
    main()