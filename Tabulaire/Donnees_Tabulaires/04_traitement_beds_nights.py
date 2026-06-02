# ============================================================
# 04_traitement_beds_nights.py
# Objectif :
# - Lire la base après traitement des catégories
# - Traiter uniquement :
#   - beds
#   - minimum_nights
#   - maximum_nights


import pandas as pd
import numpy as np
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent

def find_data_dir():
    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR / "Donnees_Tabulaires",
        SCRIPT_DIR.parent / "Donnees_Tabulaires",
        SCRIPT_DIR.parent.parent / "Tabulaire" / "Donnees_Tabulaires",
    ]

    for path in candidates:
        if path.exists() and path.is_dir() and path.name.lower() == "donnees_tabulaires":
            return path

    raise FileNotFoundError("Impossible de trouver le dossier Donnees_Tabulaires.")


DATA_DIR = find_data_dir()

print("Dossier des données utilisé :")
print(DATA_DIR)


# 2. Lecture du fichier avec catégories déjà traitées

input_file = DATA_DIR / "airbnb_tabulaire_clean_avec_categories.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Le fichier {input_file.name} est introuvable. "
        "Lance d'abord le script 03_traitement_categories.py."
    )

df = pd.read_csv(
    input_file,
    dtype={"id_clean": str},
    low_memory=False
)

print("\nFichier utilisé :")
print(input_file)

print("\nDimensions initiales :")
print(df.shape)

# 3. Conversion numérique 

for col in ["beds", "minimum_nights", "maximum_nights"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"{col} convertie en numérique.")
    else:
        print(f"{col} absente.")


# 4. Traitement de beds
# Objectif :
# - créer beds_clean
# - mettre beds = 0 en NaN
# - garder l'information avec beds_was_zero
#
# Pourquoi ?
# - un logement avec 0 lit est suspect
# - CatBoost sait gérer les NaN
# - l'indicateur beds_was_zero permet au modèle de savoir que
#   l'information était présente mais incohérente


if "beds" in df.columns:
    df["beds_clean"] = df["beds"].copy()

    df["beds_was_zero"] = (df["beds_clean"] == 0).astype(int)

    n_zero_beds = (df["beds_clean"] == 0).sum()
    n_negative_beds = (df["beds_clean"] < 0).sum()

    df.loc[df["beds_clean"] == 0, "beds_clean"] = np.nan
    df.loc[df["beds_clean"] < 0, "beds_clean"] = np.nan

    print("\nTraitement de beds :")
    print("beds = 0 mis en NaN :", n_zero_beds)
    print("beds < 0 mis en NaN :", n_negative_beds)
    print("NaN beds_clean :", df["beds_clean"].isna().sum())
    print(df["beds_clean"].describe())

    print("\nContrôle des grandes valeurs de beds_clean, sans modification :")
    print(df["beds_clean"].value_counts(dropna=False).sort_index().tail(15))

else:
    print("\nColonne beds absente.")



# 5. Traitement de minimum_nights
# Objectif :
# - créer minimum_nights_clean
# - traiter uniquement les valeurs impossibles ou très extrêmes
#
# Décision :
# - minimum_nights < 1 devient NaN
# - minimum_nights > 365 devient NaN
#
# Pourquoi 365 ?
# - une durée minimale supérieure à un an est très difficile
#   à justifier pour une annonce Airbnb classique
# - les valeurs longues comme 30, 90 ou 180 peuvent être réelles
#   donc on ne les supprime pas
# On crée aussi des indicateurs de séjour long.

if "minimum_nights" in df.columns:
    df["minimum_nights_clean"] = df["minimum_nights"].copy()

    df["minimum_nights_is_long"] = (df["minimum_nights_clean"] > 30).astype(int)
    df["minimum_nights_is_very_long"] = (df["minimum_nights_clean"] > 90).astype(int)
    df["minimum_nights_is_extreme"] = (df["minimum_nights_clean"] > 365).astype(int)

    n_invalid_min = (df["minimum_nights_clean"] < 1).sum()
    n_extreme_min = (df["minimum_nights_clean"] > 365).sum()

    df.loc[df["minimum_nights_clean"] < 1, "minimum_nights_clean"] = np.nan
    df.loc[df["minimum_nights_clean"] > 365, "minimum_nights_clean"] = np.nan

    print("\nTraitement de minimum_nights :")
    print("minimum_nights < 1 mis en NaN :", n_invalid_min)
    print("minimum_nights > 365 mis en NaN :", n_extreme_min)
    print("Nombre minimum_nights > 30 :", df["minimum_nights_is_long"].sum())
    print("Nombre minimum_nights > 90 :", df["minimum_nights_is_very_long"].sum())
    print("NaN minimum_nights_clean :", df["minimum_nights_clean"].isna().sum())
    print(df["minimum_nights_clean"].describe())

else:
    print("\nColonne minimum_nights absente.")


# 6. Traitement de maximum_nights
# Objectif :
# - créer maximum_nights_clean
# - traiter uniquement les valeurs impossibles ou extrêmes
#
# Décision :
# - maximum_nights < 1 devient NaN
# - maximum_nights > 1125 devient NaN
#
# Pourquoi 1125 ?
# - 1125 est une valeur fréquente dans les données Airbnb
# - les valeurs au-dessus sont très rares dans ta base
# - ta valeur maximale 524855552 est clairement incohérente

if "maximum_nights" in df.columns:
    df["maximum_nights_clean"] = df["maximum_nights"].copy()

    df["maximum_nights_is_extreme"] = (df["maximum_nights_clean"] > 1125).astype(int)

    n_invalid_max = (df["maximum_nights_clean"] < 1).sum()
    n_extreme_max = (df["maximum_nights_clean"] > 1125).sum()

    df.loc[df["maximum_nights_clean"] < 1, "maximum_nights_clean"] = np.nan
    df.loc[df["maximum_nights_clean"] > 1125, "maximum_nights_clean"] = np.nan

    print("\nTraitement de maximum_nights :")
    print("maximum_nights < 1 mis en NaN :", n_invalid_max)
    print("maximum_nights > 1125 mis en NaN :", n_extreme_max)
    print("NaN maximum_nights_clean :", df["maximum_nights_clean"].isna().sum())
    print(df["maximum_nights_clean"].describe())

else:
    print("\nColonne maximum_nights absente.")


# 7. Cohérence entre minimum_nights et maximum_nights
# Objectif :
# - vérifier que maximum_nights_clean >= minimum_nights_clean
#
# Si maximum_nights_clean < minimum_nights_clean :
# - on crée nights_range_is_incoherent = 1
# - on met maximum_nights_clean en NaN



if "minimum_nights_clean" in df.columns and "maximum_nights_clean" in df.columns:
    condition_incoherent = (
        df["minimum_nights_clean"].notna()
        & df["maximum_nights_clean"].notna()
        & (df["maximum_nights_clean"] < df["minimum_nights_clean"])
    )

    df["nights_range_is_incoherent"] = condition_incoherent.astype(int)

    n_incoherent = condition_incoherent.sum()

    df.loc[condition_incoherent, "maximum_nights_clean"] = np.nan

    print("\nContrôle cohérence minimum_nights / maximum_nights :")
    print("Cas maximum_nights_clean < minimum_nights_clean :", n_incoherent)


# 8. Suppression des colonnes originales remplacées
# Objectif :
# - éviter deux versions de la même information
# On garde :
# - beds_clean
# - minimum_nights_clean
# - maximum_nights_clean
# - les indicateurs créés

cols_to_drop = [
    "beds",
    "minimum_nights",
    "maximum_nights"
]

existing_cols_to_drop = [
    col for col in cols_to_drop
    if col in df.columns
]

df = df.drop(columns=existing_cols_to_drop)

print("\nColonnes originales supprimées :")
print(existing_cols_to_drop)



# 9. Rapport final des valeurs manquantes

missing_report = pd.DataFrame({
    "colonne": df.columns,
    "type": df.dtypes.astype(str).values,
    "valeurs_manquantes": df.isna().sum().values,
    "pourcentage_manquant": (df.isna().sum().values / len(df) * 100).round(2)
})

missing_report = missing_report.sort_values(
    by="pourcentage_manquant",
    ascending=False
)

print("\nTop 30 des valeurs manquantes après traitement beds/nights :")
print(missing_report.head(30))


# 10. Sauvegarde de la base finale

df["id_clean"] = df["id_clean"].astype(str)

output_csv = DATA_DIR / "airbnb_tabulaire_final_catboost.csv"
output_excel = DATA_DIR / "airbnb_tabulaire_final_catboost.xlsx"

df.to_csv(output_csv, index=False, encoding="utf-8-sig")
df.to_excel(output_excel, index=False)

print("\nFichiers sauvegardés :")
print(output_csv)
print(output_excel)

# 11. Sauvegarde du rapport

report_dir = DATA_DIR / "rapports_final_catboost"
report_dir.mkdir(exist_ok=True)

missing_report.to_csv(
    report_dir / "rapport_valeurs_manquantes_final_catboost.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRapport sauvegardé dans :")
print(report_dir)

print("\nTraitement beds / minimum_nights / maximum_nights terminé avec succès.")