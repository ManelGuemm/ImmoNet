# Supprimer les annonces dont les images sont en erreur et sauvegarder les résultats

import pandas as pd
from pathlib import Path

# ========= 1. Dossier de base =========
base_dir = Path("Fichiers_Rapports")
base_dir1 = Path("Nettoyage_Donnees")

fichier_images = base_dir / "Rapport_telechargement_images_london.xlsx"
fichier_donnees = base_dir / "fichier_sans_colonnes_hors_model.csv"
fichier_sortie = base_dir / "donnees_airbnb_nettoyees.xlsx"
fichier_supprimees = base_dir / "annonces_supprimees_images_error.xlsx"

# ========= 2. Lecture des fichiers =========
df_images = pd.read_excel(fichier_images, engine="openpyxl")
df_data = pd.read_csv(fichier_donnees, low_memory=False)

# ========= 3. Vérification des colonnes =========
if "picture_url" not in df_images.columns:
    raise ValueError("La colonne 'picture_url' est absente du fichier images.")

if "picture_url" not in df_data.columns:
    raise ValueError("La colonne 'picture_url' est absente du fichier principal.")

if "statut" not in df_images.columns:
    raise ValueError("La colonne 'statut' est absente du fichier images.")

# ========= 4. Nettoyage léger de picture_url =========
df_images["picture_url"] = df_images["picture_url"].astype(str).str.strip()
df_data["picture_url"] = df_data["picture_url"].astype(str).str.strip()

# ========= 5. Récupérer les picture_url en erreur =========
urls_error = df_images.loc[
    df_images["statut"].astype(str).str.lower().str.strip() == "error",
    "picture_url"
].dropna().unique()

# ========= 6. Lignes supprimées =========
df_supprimees = df_data[df_data["picture_url"].isin(urls_error)].copy()

# ========= 7. Suppression =========
df_data_clean = df_data[~df_data["picture_url"].isin(urls_error)].copy()

# ========= 8. Sauvegarde =========
df_data_clean.to_excel(fichier_sortie, index=False)
df_supprimees.to_excel(fichier_supprimees, index=False)

# ========= 9. Résumé =========
print(f"Nombre total d'annonces au départ : {len(df_data)}")
print(f"Nombre de picture_url en erreur : {len(urls_error)}")
print(f"Nombre d'annonces supprimées : {len(df_supprimees)}")
print(f"Nombre d'annonces restantes : {len(df_data_clean)}")
print(f"\nFichier nettoyé enregistré sous : {fichier_sortie}")
print(f"Fichier des annonces supprimées enregistré sous : {fichier_supprimees}")