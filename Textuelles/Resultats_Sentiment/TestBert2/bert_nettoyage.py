# ============================================================
# NETTOYAGE DES AVIS AIRBNB AVANT ANALYSE BERT
# Objectifs :
# 1. Charger le fichier reviews_filtrees_annonces_finales.csv
# 2. Nettoyer les commentaires
# 3. Supprimer les avis vides
# 4. Sauvegarder un fichier propre pour BERT


from pathlib import Path
import re
import html
import pandas as pd


# ============================================================
# ÉTAPE 0 — Chemins du projet
# ============================================================

# Dossier où se trouve ce script Python
BASE_DIR = Path(__file__).resolve().parent

# Dossier principal du projet ImmoNet
PROJECT_DIR = BASE_DIR.parent

# Dossier contenant les fichiers textuels
dossier_sortie = PROJECT_DIR / "Textuelles" / "Resultats_Sentiment"
dossier_sortie.mkdir(parents=True, exist_ok=True)

# Fichier d'entrée : avis déjà filtrés avec les annonces finales
fichier_reviews_filtre = dossier_sortie / "reviews_filtrees_annonces_finales.csv"

# Fichier de sortie : avis nettoyés avant BERT
fichier_reviews_clean = dossier_sortie / "reviews_filtrees_clean.csv"


print("Dossier du projet :", PROJECT_DIR)
print("Fichier reviews filtré :", fichier_reviews_filtre)
print("Existe reviews filtré :", fichier_reviews_filtre.exists())
print("Fichier de sortie nettoyé :", fichier_reviews_clean)


if not fichier_reviews_filtre.exists():
    raise FileNotFoundError(
        f"Le fichier reviews filtré est introuvable : {fichier_reviews_filtre}"
    )


# ============================================================
# PARAMÈTRES
# ============================================================

CHUNKSIZE_CLEAN = 100_000


# ============================================================
# ÉTAPE 1 — Fonction de nettoyage des commentaires
# ============================================================

def nettoyer_commentaire(texte):


    if pd.isna(texte):
        return ""

    texte = str(texte)

    # Convertir les caractères HTML, par exemple &amp; devient &
    texte = html.unescape(texte)

    # Remplacer les retours à la ligne HTML
    texte = re.sub(r"<br\s*/?>", " ", texte, flags=re.IGNORECASE)

    # Supprimer les autres balises HTML
    texte = re.sub(r"<[^>]+>", " ", texte)

    # Supprimer les espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()


# ============================================================
# ÉTAPE 2 — Nettoyage du fichier reviews

print("\n===== NETTOYAGE DES COMMENTAIRES =====")

nb_avant_nettoyage = 0
nb_avis_vides = 0
nb_avis_gardes = 0
premier_morceau = True

for chunk in pd.read_csv(
    fichier_reviews_filtre,
    dtype=str,
    chunksize=CHUNKSIZE_CLEAN,
    encoding="utf-8-sig",
    on_bad_lines="skip"
):
    # Nettoyer les noms de colonnes
    chunk.columns = chunk.columns.str.strip()

    # Vérification des colonnes nécessaires
    if "listing_id" not in chunk.columns:
        raise ValueError("La colonne listing_id est absente du fichier reviews filtré.")

    if "comments" not in chunk.columns:
        raise ValueError("La colonne comments est absente du fichier reviews filtré.")

    nb_avant_nettoyage += len(chunk)

    # Nettoyage de l'identifiant de l'annonce
    chunk["listing_id"] = (
        chunk["listing_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # Nettoyage du texte des commentaires
    chunk["comments_clean"] = chunk["comments"].apply(nettoyer_commentaire)

    # Suppression des commentaires vides
    chunk_clean = chunk[
        (chunk["comments_clean"].notna())
        & (chunk["comments_clean"].str.strip() != "")
        & (chunk["comments_clean"].str.lower() != "nan")
    ].copy()

    nb_avis_vides += len(chunk) - len(chunk_clean)
    nb_avis_gardes += len(chunk_clean)

    # Sauvegarde progressive du fichier nettoyé
    chunk_clean.to_csv(
        fichier_reviews_clean,
        mode="w" if premier_morceau else "a",
        header=premier_morceau,
        index=False,
        encoding="utf-8-sig"
    )

    premier_morceau = False


# ============================================================
# ÉTAPE 3 — Résumé

print("\n===== RÉSULTAT DU NETTOYAGE =====")
print("Nombre d'avis avant nettoyage :", nb_avant_nettoyage)
print("Nombre d'avis vides supprimés :", nb_avis_vides)
print("Nombre d'avis textuels gardés :", nb_avis_gardes)
print("Fichier nettoyé créé :", fichier_reviews_clean)

print("\nNettoyage terminé avec succès.")