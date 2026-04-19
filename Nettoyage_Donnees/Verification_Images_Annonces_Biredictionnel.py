import pandas as pd
from pathlib import Path
import re


fichier_donnees = Path(r"C:\Users\messo\Desktop\Manel\PrédictionImmobilier\Fichiers_Rapports\fichier_sans_colonnes_hors_model.csv")
dossier_images = Path(r"C:\Users\messo\Desktop\Manel\PrédictionImmobilier\London")

extensions_valides = {".jpg", ".jpeg", ".png", ".webp"}


# FONCTIONS

def lire_fichier_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str, low_memory=False)
    elif suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl", dtype=str)
    elif suffix == ".xls":
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Format non pris en charge : {path.suffix}")

def nettoyer_id(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = re.sub(r"\.0$", "", s)
    return s

def extraire_id_depuis_listing_url(url):
    if pd.isna(url):
        return None
    s = str(url).strip()
    match = re.search(r"/rooms/(\d+)", s)
    if match:
        return match.group(1)
    return None


# 1. LECTURE DU FICHIER DE DONNÉES

df = lire_fichier_table(fichier_donnees)
df.columns = df.columns.str.strip()

# 2. CONSTRUIRE L'ID DE COMPARAISON

if "listing_url" in df.columns:
    df["_id_compare"] = df["listing_url"].apply(extraire_id_depuis_listing_url)

    if "id" in df.columns:
        masque_vides = df["_id_compare"].isna()
        df.loc[masque_vides, "_id_compare"] = df.loc[masque_vides, "id"].apply(nettoyer_id)

elif "id" in df.columns:
    df["_id_compare"] = df["id"].apply(nettoyer_id)

else:
    raise ValueError("Le fichier doit contenir au moins 'listing_url' ou 'id'.")

df = df[df["_id_compare"].notna()].copy()


# 3. LECTURE DES IMAGES DU DOSSIER

ids_images = set()

for img in dossier_images.iterdir():
    if img.is_file() and img.suffix.lower() in extensions_valides:
        ids_images.add(img.stem.strip())

ids_donnees = set(df["_id_compare"].astype(str))


# 4. COMPARAISON BIDIRECTIONNELLE

ids_images_a_supprimer = ids_images - ids_donnees
ids_annonces_a_supprimer = ids_donnees - ids_images
ids_communs = ids_donnees & ids_images

# 5. RÉSULTATS

print(f"Nombre total d'annonces dans le fichier : {len(ids_donnees)}")
print(f"Nombre total d'images dans le dossier : {len(ids_images)}")
print(f"Nombre d'ids communs : {len(ids_communs)}")
print(f"Nombre d'images à supprimer : {len(ids_images_a_supprimer)}")
print(f"Nombre d'annonces à supprimer : {len(ids_annonces_a_supprimer)}")


