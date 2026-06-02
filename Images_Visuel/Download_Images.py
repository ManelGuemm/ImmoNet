import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse
import mimetypes
import re

# Configuration
FICHIER_CSV = "Donnees/listings_price_valide.csv"
COLONNE_ID = "id"
COLONNE_URL = "picture_url"

DOSSIER_IMAGES = Path("Images_Visuel") / "Images_London"
FICHIER_EXCEL = Path("Fichiers_Rapports") / "Rapport_telechargement_images_london.xlsx"

TIMEOUT = 20

DOSSIER_IMAGES.mkdir(parents=True, exist_ok=True)
FICHIER_EXCEL.parent.mkdir(parents=True, exist_ok=True)

def nettoyer_nom(valeur):
    """Nettoie un texte pour l'utiliser dans un nom de fichier."""
    valeur = str(valeur).strip()
    valeur = re.sub(r"[^\w\-]", "_", valeur)
    return valeur

def trouver_extension(url, content_type):
    """
    Essaie de trouver une extension correcte à partir :
    1. du Content-Type
    2. de l'URL
    3. sinon .jpg par défaut
    """
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]:
            return ".jpg" if ext == ".jpe" else ext

    path = urlparse(url).path
    ext_url = Path(path).suffix.lower()
    if ext_url in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]:
        return ext_url

    return ".jpg"

def generer_nom_fichier_unique(id_annonce, extension, dossier):
    """Évite d'écraser un fichier si un nom existe déjà."""
    base = nettoyer_nom(id_annonce)
    chemin = dossier / f"{base}{extension}"

    compteur = 1
    while chemin.exists():
        chemin = dossier / f"{base}_{compteur}{extension}"
        compteur += 1

    return chemin

def telecharger_image(id_annonce, url):
    """
    Télécharge une image si possible.
    Retourne un dictionnaire de suivi.
    """
    resultat = {
        "id": id_annonce,
        "picture_url": url,
        "statut": None,
        "nom_fichier": None,
        "chemin_fichier": None,
        "http_status": None,
        "content_type": None,
        "message": None
    }

    if pd.isna(url) or str(url).strip() == "":
        resultat["statut"] = "error"
        resultat["message"] = "URL vide"
        return resultat

    url = str(url).strip()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, stream=True, timeout=TIMEOUT, headers=headers, allow_redirects=True)
        resultat["http_status"] = response.status_code
        resultat["content_type"] = response.headers.get("Content-Type", "")

        if response.status_code != 200:
            resultat["statut"] = "error"
            resultat["message"] = f"HTTP {response.status_code}"
            return resultat

        if "image" not in resultat["content_type"].lower():
            resultat["statut"] = "error"
            resultat["message"] = f"Contenu non image : {resultat['content_type']}"
            return resultat

        extension = trouver_extension(url, resultat["content_type"])
        chemin_image = generer_nom_fichier_unique(id_annonce, extension, DOSSIER_IMAGES)

        with open(chemin_image, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        resultat["statut"] = "ok"
        resultat["nom_fichier"] = chemin_image.name
        resultat["chemin_fichier"] = str(chemin_image)

        return resultat

    except requests.exceptions.RequestException as e:
        resultat["statut"] = "error"
        resultat["message"] = str(e)
        return resultat

    except Exception as e:
        resultat["statut"] = "error"
        resultat["message"] = f"Erreur inattendue : {e}"
        return resultat


print("Chargement du fichier CSV...")
df = pd.read_csv(FICHIER_CSV, sep=None, engine="python", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

if COLONNE_ID not in df.columns:
    raise ValueError(f"La colonne '{COLONNE_ID}' est introuvable dans le fichier.")

if COLONNE_URL not in df.columns:
    raise ValueError(f"La colonne '{COLONNE_URL}' est introuvable dans le fichier.")


df[COLONNE_ID] = df[COLONNE_ID].astype(str).str.strip()


resultats = []

total = len(df)
print(f"Nombre total d'annonces à traiter : {total}")

for i, row in enumerate(df[[COLONNE_ID, COLONNE_URL]].itertuples(index=False), start=1):
    id_annonce, url = row

    print(f"\r{i}/{total} - téléchargement de l'annonce {id_annonce}", end="", flush=True)

    resultat = telecharger_image(id_annonce, url)
    resultats.append(resultat)

print("\nTéléchargement terminé.")


df_resultats = pd.DataFrame(resultats)
df_erreurs = df_resultats[df_resultats["statut"] == "error"].copy()

with pd.ExcelWriter(FICHIER_EXCEL, engine="openpyxl") as writer:
    df_resultats.to_excel(writer, sheet_name="suivi_images", index=False)
    df_erreurs.to_excel(writer, sheet_name="images_erreur", index=False)

nb_ok = (df_resultats["statut"] == "ok").sum()
nb_error = (df_resultats["statut"] == "error").sum()

print("\n===== RÉSUMÉ =====")
print(f"Images téléchargées avec succès : {nb_ok}")
print(f"Images en erreur : {nb_error}")
print(f"Dossier images : {DOSSIER_IMAGES}")
print(f"Fichier Excel : {FICHIER_EXCEL}")

