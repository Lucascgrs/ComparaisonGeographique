# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import argparse
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import shapely.geometry
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image
import cv2
from tqdm import tqdm
import sentinelhub


def display_info(title, data):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(data)


import mycredentials

# ============================================================================
# MODULE 1: GÉOGRAPHIE & ZONE D'INTÉRÊT
# ============================================================================
class GeoManager:
    def __init__(self, place_name="Rio de Janeiro, Brazil"):
        self.place_name = place_name
        self.municipalities = None
        self.aoi = None
        self.bbox = None
        self.bbox_polygon = None
        
    def fetch_data(self):
        display_info("GÉOGRAPHIE", f"Récupération des données pour : {self.place_name}")
        try:
            # OPTIMISATION : On utilise geocode_to_gdf qui est beaucoup plus léger
            # Il ne télécharge que la frontière (polygone), pas les rues ni les bâtiments
            self.municipalities = ox.geocode_to_gdf(self.place_name)
            
            # On renomme la colonne 'display_name' en 'name' pour garder la compatibilité avec le reste du script
            if 'display_name' in self.municipalities.columns:
                self.municipalities['name'] = self.municipalities['display_name']
                
            print(f"✅ Frontières récupérées rapidement pour {self.place_name}")
            
        except Exception as e:
            print(f"⚠️ Erreur méthode rapide, tentative méthode complète... ({e})")
            # Fallback sur l'ancienne méthode si jamais ça échoue
            try:
                self.municipalities = ox.features_from_place(
                    self.place_name,
                    tags={'admin_level': '8', 'boundary': 'administrative'}
                )
            except Exception as e2:
                print(f"❌ Échec total récupération géo : {e2}")
                sys.exit(1)

    def save_map(self, filename="municipalities_map.html"):
        if self.municipalities is not None:
            print(f"💾 Sauvegarde de la carte dans {filename}...")
            m = self.municipalities.explore(column='name', cmap='Set3')
            m.save(filename)

    def select_municipality(self, search_str):
        if self.municipalities is None:
            self.fetch_data()
            
        mask = self.municipalities['name'].str.contains(search_str, case=False, na=False)
        subset = self.municipalities[mask]
        
        if subset.empty:
            raise ValueError(f"Aucune municipalité trouvée avec le nom : {search_str}")
            
        self.aoi = subset.iloc[[0]] # Garder en GeoDataFrame
        muni_name = self.aoi.iloc[0].get('name', 'N/A')
        print(f"Municipalité sélectionnée : {muni_name}")
        
        # Calcul de la BBOX
        self.bbox = self.aoi.total_bounds # (minx, miny, maxx, maxy)
        self.bbox_polygon = shapely.geometry.box(*self.bbox)
        print(f"BBOX : {self.bbox}")
        return self.aoi

# ============================================================================
# MODULE 2: COPERNICUS ODATA API (RECHERCHE & TÉLÉCHARGEMENT DIRECT)
# ============================================================================
class CopernicusDirect:
    def __init__(self, bbox_polygon, start_date, end_date, cloud_cover=100.0):
        self.bbox_polygon = bbox_polygon
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover
        self.products = pd.DataFrame()
        self.session = requests.Session()

    def search_products(self):
        display_info("RECHERCHE SATELLITE", f"Recherche images Sentinel-2 (Nuages < {self.cloud_cover}%)")
        
        url = (
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            "&$filter=Collection/Name eq 'SENTINEL-2'"
            " and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{self.bbox_polygon}')"
            f" and ContentDate/Start gt {self.start_date}T00:00:00.000Z"
            f" and ContentDate/Start lt {self.end_date}T00:00:00.000Z"
            f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {self.cloud_cover})"
            "&$expand=Assets"
            "&$expand=Attributes"
            "&$orderby=ContentDate/Start"
            "&$top=50"
        )
        
        try:
            response = requests.get(url).json()
            if "value" in response:
                self.products = pd.DataFrame.from_dict(response['value'])
                print(f"{len(self.products)} images trouvées.")
            else:
                print(f"Erreur ou aucun résultat : {response}")
        except Exception as e:
            print(f"Erreur de requête : {e}")

    def authenticate(self):
        data = {
            "client_id": "cdse-public",
            "username": mycredentials.username,
            "password": mycredentials.password,
            "grant_type": "password",
        }
        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", data=data)
            r.raise_for_status()
            token = r.json()["access_token"]
            self.session.headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            print(f"Échec authentification (vérifiez mycredentials.py) : {e}")

    def download_first_product_bands(self, output_dir="."):
        if self.products.empty:
            print("Aucun produit à télécharger.")
            return

        product = self.products.iloc[0]
        pid = product['Id']
        pname = product['Name']
        print(f"⬇️ Téléchargement pour le produit : {pname}")

        # 1. Télécharger Manifeste
        manifest_path = os.path.join(output_dir, "Manifest.xml")
        self._download_file(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({pid})/Nodes({pname})/Nodes(MTD_MSIL2A.xml)/$value", manifest_path)

        # 2. Parser Manifeste pour trouver les bandes
        bands_to_download = self._parse_manifest(manifest_path)
        
        # 3. Télécharger les bandes
        downloaded_files = []
        for band_path in bands_to_download:
            # Construction URL complexe (simplifiée ici selon logique notebook)
            parts = band_path.split("/") # Structure typique Sentinel
            node_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({pid})/Nodes({pname})"
            for p in parts:
                node_url += f"/Nodes({p})"
            node_url += "/$value"
            
            filename = parts[-1]
            filepath = os.path.join(output_dir, filename)
            self._download_file(node_url, filepath)
            downloaded_files.append(filepath)
            
        return downloaded_files

    def _download_file(self, url, save_path):
        # Gestion des redirections manuelle comme dans le notebook
        response = self.session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = self.session.get(url, allow_redirects=False)
        
        response = self.session.get(url, verify=False, allow_redirects=True, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        
        print(f"   💾 {os.path.basename(save_path)}")
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

    def _parse_manifest(self, manifest_path):
        # Logique extraite du notebook pour filtrer 20m et bandes spécifiques
        desired_channels = ["B02", "B03", "B04", "B8A"]
        gsd = "20m"
        band_locations = []
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        ns = {'ns': 'https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd'} # Namespace peut varier
        
        # Recherche simple via itération pour éviter les soucis de namespace complexes
        for elem in root.iter():
            if elem.text and gsd in elem.text and any(ch in elem.text for ch in desired_channels):
                if elem.text.endswith('.jp2'): # Sécurité
                    band_locations.append(f"{elem.text}")
                else:
                    band_locations.append(f"{elem.text}.jp2")
        return list(set(band_locations)) # Uniques

# ============================================================================
# MODULE 3: SENTINEL HUB (PROCESSUS COMPARATIF)
# ============================================================================
class SentinelHubProcessor:
    def __init__(self, bbox, resolution=200):
        self.bbox_list = bbox
        self.resolution = resolution
        # Configuration SentinelHub
        self.config = sentinelhub.SHConfig("cdse") # Profil par défaut 'cdse'
        self.aoi_bbox_sh = sentinelhub.BBox(bbox=list(bbox), crs=sentinelhub.CRS.WGS84)
        self.aoi_size = sentinelhub.bbox_to_dimensions(self.aoi_bbox_sh, resolution=resolution)
        print(f"📐 Taille image cible : {self.aoi_size} pixels")

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
        
        data = request.get_data()
        if not data:
            print("⚠️ Aucune donnée reçue.")
            return None

        img_array = data[0]
        
        # Ajustement luminosité
        if brightness != 1.0:
            img_array = np.uint8((img_array * brightness).clip(0, 255))
            
        # Sauvegarde
        if filename:
            Image.fromarray(img_array if img_array.shape[-1] != 1 else img_array.squeeze()).save(filename)
            print(f"💾 Image sauvegardée : {filename}")
            
        return img_array

# ============================================================================
# SCRIPTS D'ÉVALUATION (EVALSCRIPTS)
# ============================================================================
EVALSCRIPTS = {
    "TRUE_COLOR": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B02", "B03", "B04"] }], output: { bands: 3 } }; }
        function evaluatePixel(sample) { return [sample.B04, sample.B03, sample.B02]; }
    """,
    "FALSE_COLOR": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B03", "B04", "B08"] }], output: { bands: 3 } }; }
        function evaluatePixel(sample) { return [sample.B08, sample.B04, sample.B03]; }
    """,
    "NDVI": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B04", "B08", "dataMask"] }], output: { bands: 4 } }; }
        const whiteGreen = [[1.0, 0xFFFFFF], [0.5, 0x000000], [0.0, 0x00FF00]];
        let viz = new ColorGradientVisualizer(whiteGreen, -1.0, 1.0);
        function evaluatePixel(samples) {
            let val = ((samples.B08 + samples.B04)==0) ? 0 : ((samples.B08 - samples.B04) / (samples.B08 + samples.B04));
            let col = viz.process(val);
            col.push(samples.dataMask);
            return col;
        }
    """,
    "BURNED_INDEX_RAW": """
        //VERSION=3
        function setup() { return { input: [{ bands: ["B08", "B12", "dataMask"] }], output: { bands: 1, sampleType: SampleType.FLOAT32 } }; }
        function evaluatePixel(samples) {
            let val = ((samples.B08 + samples.B12)==0) ? 0 : ((samples.B08 - samples.B12) / (samples.B08 + samples.B12));
            return [val];
        }
    """
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Outil de surveillance forestière Sentinel-2")
    parser.add_argument("--city", type=str, default="Lábrea, Brazil", help="Nom de la municipalité (ex: 'Novo Progresso, Brazil')")
    parser.add_argument("--cloud", type=float, default=10.0, help="Pourcentage maximum de nuages")
    parser.add_argument("--start", type=str, default="2019-07-01", help="Date début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2019-09-30", help="Date fin (YYYY-MM-DD)")
    parser.add_argument("--compare_year_before", type=str, default="2018", help="Année 'Avant' pour comparaison")
    parser.add_argument("--compare_year_after", type=str, default="2021", help="Année 'Après' pour comparaison")
    
    args = parser.parse_args()

    # --- CORRECTION ICI ---
    # On initialise GeoManager avec la ville demandée, et pas Rio par défaut
    geo = GeoManager(place_name=args.city)
    
    # Pour la recherche, on prend juste le nom de la ville avant la virgule si présent
    # Ex: "Novo Progresso, Brazil" -> recherche "Novo Progresso" dans le résultat
    search_name = args.city.split(",")[0]
    geo.select_municipality(search_name)
    # ----------------------

    geo.save_map(f"map_{search_name.replace(' ', '_')}.html")

    # 2. Recherche API (Mode Découverte)
    copernicus = CopernicusDirect(geo.bbox_polygon, args.start, args.end, args.cloud)
    copernicus.search_products()
    
    # 3. Analyse SentinelHub
    if 'sentinelhub' in sys.modules:
        try:
            sh_proc = SentinelHubProcessor(geo.bbox, resolution=210) # Resolution ajustée
            
            interval_before = (f"{args.compare_year_before}-07-01", f"{args.compare_year_before}-09-30")
            interval_after = (f"{args.compare_year_after}-07-01", f"{args.compare_year_after}-09-30")
            
            print(f"\n🔄 Génération des images comparatives ({args.compare_year_before} vs {args.compare_year_after})...")

            # True Color
            img_after = sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_after, brightness=3.5, filename="true_color_after.png")
            sh_proc.get_image(EVALSCRIPTS["TRUE_COLOR"], *interval_before, brightness=3.5, filename="true_color_before.png")

            # NDVI
            sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_before, filename="ndvi_before.png")
            sh_proc.get_image(EVALSCRIPTS["NDVI"], *interval_after, filename="ndvi_after.png")

            # Burn Index
            raw_before = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_before, filename="burned_raw_before.tif")
            raw_after = sh_proc.get_image(EVALSCRIPTS["BURNED_INDEX_RAW"], *interval_after, filename="burned_raw_after.tif")
            
            if raw_before is not None and raw_after is not None:
                print("\n🔥 Calcul de la différence des surfaces brûlées...")
                difference = raw_before - raw_after
                
                if img_after is not None:
                    composite = np.array(img_after)
                    # Masque rouge pour les zones brûlées
                    # On redimensionne le masque pour être sûr qu'il colle à l'image (sécurité)
                    if difference.shape[:2] == composite.shape[:2]:
                        composite[difference > 0.27] = [255, 128, 0]
                        composite[difference > 0.66] = [255, 0, 0]
                    
                    Image.fromarray(composite).save('burn_composite_analysis.png')
                    print("✅ Image composite sauvegardée : burn_composite_analysis.png")

        except Exception as e:
            print(f"❌ Erreur Sentinel Hub : {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Module SentinelHub manquant, analyse ignorée.")

if __name__ == "__main__":
    main()