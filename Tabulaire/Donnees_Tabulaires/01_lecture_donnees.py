# ============================================================
# 01_lecture_donnees.py
# Objectif :
# - Lire le fichier Airbnb tabulaire
# - Afficher la structure générale
# - Vérifier les colonnes, types, valeurs manquantes, doublons



import pandas as pd
import numpy as np
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent

def find_data_dir():
    """
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
        "Impossible de trouver le dossier Donnees_Tabulaires. "
        "Vérifie l'emplacement du script."
    )


DATA_DIR = find_data_dir()

print("Dossier des données utilisé :")
print(DATA_DIR)



excel_files = [
    f for f in DATA_DIR.glob("*.xlsx")
    if not f.name.startswith("~$")
]

print("\nFichiers Excel trouvés :")
for f in excel_files:
    print("-", f.name)

if len(excel_files) == 0:
    raise FileNotFoundError("Aucun fichier Excel .xlsx trouvé dans Donnees_Tabulaires.")

preferred_names = [
    "Donnees_Airbnb_Finales_Tabulaire.xlsx",
    "Donnees_Airbnb_Tabulaire.xlsx"
]

input_file = None

for name in preferred_names:
    candidate = DATA_DIR / name
    if candidate.exists():
        input_file = candidate
        break

if input_file is None:
    input_file = excel_files[0]

print("\nFichier utilisé :")
print(input_file)


# Lecture du fichier

df = pd.read_excel(input_file)

print("\nLecture terminée.")
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])


# Affichage des colonnes

print("\nListe des colonnes :")
for col in df.columns:
    print("-", col)

# 5. Types détectés 


print("\nTypes détectés :")
print(df.dtypes)

# 6. Valeurs manquantes par colonne

missing_report = pd.DataFrame({
    "colonne": df.columns,
    "type_detecte": df.dtypes.astype(str).values,
    "valeurs_manquantes": df.isna().sum().values,
    "pourcentage_manquant": (df.isna().sum().values / len(df) * 100).round(2)
})

missing_report = missing_report.sort_values(
    by="pourcentage_manquant",
    ascending=False
)

print("\nValeurs manquantes :")
print(missing_report)

# 7. Doublons

print("\nDoublons exacts dans tout le fichier :")
print(df.duplicated().sum())

if "id" in df.columns:
    print("\nDoublons sur id :")
    print(df["id"].duplicated().sum())

if "listing_url" in df.columns:
    print("\nDoublons sur listing_url :")
    print(df["listing_url"].duplicated().sum())


# 8. Analyse rapide de la cible price


if "price" in df.columns:
    price_numeric = pd.to_numeric(df["price"], errors="coerce")

    print("\nAnalyse rapide de price :")
    print(price_numeric.describe())

    print("\nNombre de price manquants après conversion numérique :")
    print(price_numeric.isna().sum())

    print("\nNombre de price <= 0 :")
    print((price_numeric <= 0).sum())

    print("\nQuantiles de price :")
    print(price_numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))
else:
    print("\nLa colonne price est absente.")


# 9. Sauvegarde des rapports de lecture

REPORT_DIR = DATA_DIR / "rapports_lecture"
REPORT_DIR.mkdir(exist_ok=True)

missing_report.to_csv(
    REPORT_DIR / "rapport_valeurs_manquantes.csv",
    index=False,
    encoding="utf-8-sig"
)

dtypes_report = pd.DataFrame({
    "colonne": df.columns,
    "type_detecte": df.dtypes.astype(str).values
})

dtypes_report.to_csv(
    REPORT_DIR / "rapport_types_colonnes.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRapports sauvegardés dans :")
print(REPORT_DIR)

print("\nFin de la lecture des données.")