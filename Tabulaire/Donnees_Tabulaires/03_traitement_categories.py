# ============================================================
# 03_traitement_categories.py
# Objectif :
# - Lire le fichier nettoyé précédent
# - Traiter uniquement les variables catégorielles
# - Remplacer les valeurs manquantes par "missing"
# - Garder les catégories sous forme texte pour CatBoost
# - Regrouper les modalités rares de property_type
# - Supprimer les colonnes catégorielles originales
#
# Fichier d'entrée :
# - airbnb_tabulaire_clean_sans_categories.csv
#
# Fichiers de sortie :
# - airbnb_tabulaire_clean_avec_categories.csv
# - airbnb_tabulaire_clean_avec_categories.xlsx
# - rapport_categories.csv
#
# Important :
# - pas de one-hot encoding
# - les variables restent en texte pour CatBoost
# ============================================================

import pandas as pd
import numpy as np
import re
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent

def find_data_dir():
    """
    Recherche automatique du dossier Donnees_Tabulaires.
  
    """

    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR / "Donnees_Tabulaires",
        SCRIPT_DIR.parent / "Donnees_Tabulaires",
        SCRIPT_DIR.parent.parent / "Tabulaire" / "Donnees_Tabulaires",
    ]

    for path in candidates:
        if path.exists() and path.is_dir() and path.name.lower() == "donnees_tabulaires":
            return path

    raise FileNotFoundError(
        "Impossible de trouver le dossier Donnees_Tabulaires."
    )


DATA_DIR = find_data_dir()

print("Dossier des données utilisé :")
print(DATA_DIR)



input_file = DATA_DIR / "airbnb_tabulaire_clean_sans_categories.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Le fichier {input_file.name} est introuvable.\n"
        "Lance d'abord le script 02_nettoyage_tabulaire.py."
    )

print("\nFichier utilisé :")
print(input_file)

# id_clean est forcé en texte pour éviter les problèmes de jointure plus tard.
df = pd.read_csv(
    input_file,
    dtype={
        "id_clean": str,
        "host_response_time": str,
        "room_type": str,
        "neighbourhood_cleansed": str,
        "property_type": str
    },
    low_memory=False
)

print("\nDimensions initiales :")
print(df.shape)

print("\nColonnes initiales :")
print(df.columns.tolist())


# 3. Fonction de nettoyage des catégories
# Objectif :
# - transformer les NaN en "missing"
# - mettre le texte en minuscules
# - enlever les espaces inutiles
# - éviter les valeurs textuelles vides comme "nan", "none", "null"


def clean_category_value(value):
    """
    Nettoie une valeur catégorielle.

    Règles :
    - NaN devient "missing"
    - chaîne vide devient "missing"
    - "nan", "none", "null" deviennent "missing"
    - texte mis en minuscules
    - espaces multiples réduits
    """

    if pd.isna(value):
        return "missing"

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    if value in ["", "nan", "none", "null"]:
        return "missing"

    return value



# 4. Traitement de host_response_time
# Objectif :
# - remplacer les valeurs manquantes par "missing"
# - garder la variable en texte pour CatBoost



if "host_response_time" in df.columns:
    df["host_response_time_clean"] = df["host_response_time"].apply(clean_category_value)

    print("\nDistribution host_response_time_clean :")
    print(df["host_response_time_clean"].value_counts(dropna=False))

else:
    print("\nColonne host_response_time absente.")


# 5. Traitement de room_type
# Objectif :
# - garder les 4 modalités principales
# - nettoyer seulement le texte
# - ne pas regrouper les modalités rares ici


if "room_type" in df.columns:
    df["room_type_clean"] = df["room_type"].apply(clean_category_value)

    print("\nDistribution room_type_clean :")
    print(df["room_type_clean"].value_counts(dropna=False))

else:
    print("\nColonne room_type absente.")


# 6. Traitement de neighbourhood_cleansed
# Objectif :
# - garder les quartiers
# - nettoyer seulement le texte
# - ne pas regrouper les quartiers


if "neighbourhood_cleansed" in df.columns:
    df["neighbourhood_cleansed_clean"] = df["neighbourhood_cleansed"].apply(clean_category_value)

    print("\nDistribution neighbourhood_cleansed_clean :")
    print(df["neighbourhood_cleansed_clean"].value_counts(dropna=False).head(15))
    print("Nombre de modalités :", df["neighbourhood_cleansed_clean"].nunique())

else:
    print("\nColonne neighbourhood_cleansed absente.")



# 7. Traitement de property_type
# Objectif :
# - nettoyer le texte
# - regrouper les modalités rares dans "other_rare"
# une modalité apparaissant moins de 50 fois est considérée rare


if "property_type" in df.columns:
    df["property_type_clean"] = df["property_type"].apply(clean_category_value)

    rare_threshold = 50

    property_counts_before = df["property_type_clean"].value_counts(dropna=False)

    rare_property_types = property_counts_before[
        property_counts_before < rare_threshold
    ].index

    df["property_type_clean"] = df["property_type_clean"].where(
        ~df["property_type_clean"].isin(rare_property_types),
        "other_rare"
    )

    property_counts_after = df["property_type_clean"].value_counts(dropna=False)

    print("\nDistribution property_type_clean après regroupement :")
    print(property_counts_after.head(30))

    print("\nNombre de modalités property_type avant regroupement :")
    print(property_counts_before.shape[0])

    print("Nombre de modalités rares regroupées :")
    print(len(rare_property_types))

    print("Nombre de modalités property_type après regroupement :")
    print(df["property_type_clean"].nunique())

else:
    print("\nColonne property_type absente.")

# 8. Suppression des colonnes catégorielles originales
# - éviter d'avoir deux versions de la même variable
# - garder uniquement les colonnes clean
# Colonnes supprimées :
# - host_response_time
# - room_type
# - neighbourhood_cleansed
# - property_type
# Colonnes conservées :
# - host_response_time_clean
# - room_type_clean
# - neighbourhood_cleansed_clean
# - property_type_clean

categorical_original_cols = [
    "host_response_time",
    "room_type",
    "neighbourhood_cleansed",
    "property_type"
]

existing_categorical_original_cols = [
    col for col in categorical_original_cols
    if col in df.columns
]

df = df.drop(columns=existing_categorical_original_cols)

print("\nColonnes catégorielles originales supprimées :")
print(existing_categorical_original_cols)

# 9. Contrôle des variables catégorielles finales

cat_features = [
    "host_response_time_clean",
    "room_type_clean",
    "neighbourhood_cleansed_clean",
    "property_type_clean"
]

existing_cat_features = [
    col for col in cat_features
    if col in df.columns
]

print("\nVariables catégorielles finales pour CatBoost :")
for col in existing_cat_features:
    print(f"\n{col}")
    print("Type :", df[col].dtype)
    print("Nombre de modalités :", df[col].nunique(dropna=False))
    print(df[col].value_counts(dropna=False).head(15))


# 10. Rapport des valeurs manquantes après traitement catégories

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

print("\nTop 30 des valeurs manquantes après traitement des catégories :")
print(missing_report.head(30))

# 11. Contrôle des colonnes à exclure du futur modèle
# Important :
# - id_clean sert aux jointures futures, pas au modèle
# - price sert à l'interprétation en euros, pas aux variables explicatives
# - log_price sera la cible d'entraînement probable

cols_not_for_model = [
    "id_clean",
    "price",
    "log_price"
]

feature_cols_provisoires = [
    col for col in df.columns
    if col not in cols_not_for_model
]

print("\nNombre de variables explicatives provisoires après catégories :")
print(len(feature_cols_provisoires))

print("\nVariables explicatives provisoires :")
for col in feature_cols_provisoires:
    print("-", col)


# 12. Sauvegarde du fichier final avec catégories traitées
output_csv = DATA_DIR / "airbnb_tabulaire_clean_avec_categories.csv"
output_excel = DATA_DIR / "airbnb_tabulaire_clean_avec_categories.xlsx"

df.to_csv(output_csv, index=False, encoding="utf-8-sig")
df.to_excel(output_excel, index=False)

print("\nFichiers sauvegardés :")
print(output_csv)
print(output_excel)


# 13. Sauvegarde des rapports

report_dir = DATA_DIR / "rapports_categories"
report_dir.mkdir(exist_ok=True)

missing_report.to_csv(
    report_dir / "rapport_valeurs_manquantes_apres_categories.csv",
    index=False,
    encoding="utf-8-sig"
)

category_report_rows = []

for col in existing_cat_features:
    counts = df[col].value_counts(dropna=False)
    for modality, count in counts.items():
        category_report_rows.append({
            "colonne": col,
            "modalite": modality,
            "effectif": count,
            "pourcentage": round(count / len(df) * 100, 4)
        })

category_report = pd.DataFrame(category_report_rows)

category_report.to_csv(
    report_dir / "rapport_modalites_categories.csv",
    index=False,
    encoding="utf-8-sig"
)

# Sauvegarde de la liste cat_features pour le futur script CatBoost
cat_features_file = report_dir / "cat_features.txt"

with open(cat_features_file, "w", encoding="utf-8") as f:
    for col in existing_cat_features:
        f.write(col + "\n")

print("\nRapports sauvegardés dans :")
print(report_dir)

print("\nListe cat_features sauvegardée dans :")
print(cat_features_file)



# 14. Fin du script
print("\nTraitement des variables catégorielles terminé avec succès.")