# ============================================================
# 05_correction_rates_sur_fichier_final.py
# Objectif :
# - Lire le fichier final déjà traité :
#   airbnb_tabulaire_final_catboost.xlsx ou .csv
# - Lire le fichier original :
#   Donnees/listings.csv
# - Récupérer les vraies valeurs de :
#   host_response_rate
#   host_acceptance_rate
# - Convertir les taux :
#   "90%"  -> 0.90
#   "100%" -> 1.00
#   "N/A"  -> NaN
#
# - Faire la correspondance par id_clean extrait depuis listing_url


import pandas as pd
import numpy as np
import re
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent

DATA_TAB_DIR = SCRIPT_DIR

if DATA_TAB_DIR.name.lower() != "donnees_tabulaires":
    candidates = [
        SCRIPT_DIR / "Donnees_Tabulaires",
        SCRIPT_DIR.parent / "Donnees_Tabulaires",
        SCRIPT_DIR.parent.parent / "Tabulaire" / "Donnees_Tabulaires",
    ]

    for path in candidates:
        if path.exists() and path.is_dir() and path.name.lower() == "donnees_tabulaires":
            DATA_TAB_DIR = path
            break

if DATA_TAB_DIR.name.lower() != "donnees_tabulaires":
    raise FileNotFoundError("Impossible de trouver le dossier Donnees_Tabulaires.")


# Dossier Donnees contenant listings.csv
DATA_RAW_DIR = DATA_TAB_DIR.parent.parent / "Donnees"

if not DATA_RAW_DIR.exists():
    raise FileNotFoundError(
        f"Impossible de trouver le dossier Donnees : {DATA_RAW_DIR}"
    )

print("Dossier Donnees_Tabulaires :")
print(DATA_TAB_DIR)

print("\nDossier Donnees original :")
print(DATA_RAW_DIR)

# 2. Définir les fichiers à utiliser

# Fichier final déjà obtenu après traitement beds/nights
final_xlsx = DATA_TAB_DIR / "airbnb_tabulaire_final_catboost.xlsx"
final_csv = DATA_TAB_DIR / "airbnb_tabulaire_final_catboost.csv"

# Fichier original contenant les vrais pourcentages
listings_file = DATA_RAW_DIR / "listings.csv"

if not listings_file.exists():
    raise FileNotFoundError(
        f"Fichier listings.csv introuvable : {listings_file}"
    )

# On lit de préférence le CSV final s'il existe, sinon on lit le XLSX.
if final_csv.exists():
    final_file = final_csv
elif final_xlsx.exists():
    final_file = final_xlsx
else:
    raise FileNotFoundError(
        "Impossible de trouver airbnb_tabulaire_final_catboost.csv ou .xlsx"
    )

print("\nFichier final utilisé :")
print(final_file)

print("\nFichier listings utilisé :")
print(listings_file)

# 3. Lecture du fichier final

if final_file.suffix.lower() == ".csv":
    df_final = pd.read_csv(
        final_file,
        dtype={"id_clean": str},
        low_memory=False
    )

elif final_file.suffix.lower() == ".xlsx":
    df_final = pd.read_excel(
        final_file,
        dtype={"id_clean": str},
        engine="openpyxl"
    )

else:
    raise ValueError(f"Format non reconnu pour le fichier final : {final_file.suffix}")

print("\nFichier final lu :")
print(df_final.shape)

# 4. Lecture de listings.csv

df_listings = pd.read_csv(
    listings_file,
    dtype=str,
    low_memory=False
)

print("\nFichier listings lu :")
print(df_listings.shape)


# 5. Nettoyage des noms de colonnes du fichier listings

def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w_]", "", col)
    col = re.sub(r"_+", "_", col)
    return col


df_listings.columns = [clean_column_name(c) for c in df_listings.columns]

print("\nColonnes principales dans listings :")
print(df_listings.columns[:30].tolist())

# 6. Vérification des colonnes nécessaires

required_cols_listings = [
    "listing_url",
    "host_response_rate",
    "host_acceptance_rate"
]

for col in required_cols_listings:
    if col not in df_listings.columns:
        raise ValueError(f"Colonne absente dans listings.csv : {col}")

if "id_clean" not in df_final.columns:
    raise ValueError("La colonne id_clean est absente du fichier final.")


# 7. Création de id_clean depuis listing_url dans listings

df_listings["id_clean"] = (
    df_listings["listing_url"]
    .astype(str)
    .str.extract(r"/rooms/(\d+)")[0]
)

print("\nContrôle id_clean dans listings :")
print("id_clean manquants :", df_listings["id_clean"].isna().sum())
print("id_clean uniques :", df_listings["id_clean"].nunique())
print("doublons id_clean :", df_listings["id_clean"].duplicated().sum())



# 8. Sécurisation du id_clean dans le fichier final


df_final["id_clean"] = df_final["id_clean"].astype(str).str.strip()

# Si jamais un id est lu avec ".0", on nettoie.
df_final["id_clean"] = df_final["id_clean"].str.replace(r"\.0$", "", regex=True)

print("\nContrôle id_clean dans le fichier final :")
print("id_clean manquants :", df_final["id_clean"].isna().sum())
print("id_clean uniques :", df_final["id_clean"].nunique())
print("doublons id_clean :", df_final["id_clean"].duplicated().sum())


# 9. Fonction de conversion des taux
# Exemples :
# "90%"  -> 0.90
# "85%"  -> 0.85
# "100%" -> 1.00
# "N/A"  -> NaN
def clean_rate_percent(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["", "nan", "none", "null", "n/a", "na", "not available"]:
        return np.nan

    value = value.replace("%", "")
    value = value.replace(",", ".")
    value = re.sub(r"[^\d.]", "", value)

    if value == "":
        return np.nan

    try:
        value = float(value)
    except ValueError:
        return np.nan

    # Si on a 90, 85, 100, on convertit en 0.90, 0.85, 1.00.
    if value > 1:
        value = value / 100

    # Sécurité : un taux doit être entre 0 et 1.
    if value < 0 or value > 1:
        return np.nan

    return value

# 10. Conversion des taux dans listings

df_listings["host_response_rate_correct"] = (
    df_listings["host_response_rate"]
    .apply(clean_rate_percent)
)

df_listings["host_acceptance_rate_correct"] = (
    df_listings["host_acceptance_rate"]
    .apply(clean_rate_percent)
)

print("\nContrôle des taux corrigés dans listings :")

print("\nhost_response_rate_correct :")
print(df_listings["host_response_rate_correct"].describe())
print("NaN :", df_listings["host_response_rate_correct"].isna().sum())

print("\nhost_acceptance_rate_correct :")
print(df_listings["host_acceptance_rate_correct"].describe())
print("NaN :", df_listings["host_acceptance_rate_correct"].isna().sum())


# 11. Création de la table de correspondance

rates_map = df_listings[
    [
        "id_clean",
        "host_response_rate_correct",
        "host_acceptance_rate_correct"
    ]
].copy()

rates_map = rates_map.dropna(subset=["id_clean"])

# Si doublons éventuels, on garde la première valeur.
rates_map = (
    rates_map
    .groupby("id_clean", as_index=False)
    .agg({
        "host_response_rate_correct": "first",
        "host_acceptance_rate_correct": "first"
    })
)

print("\nTable de correspondance créée :")
print(rates_map.shape)
print("id_clean uniques :", rates_map["id_clean"].nunique())


# 12. Fusion avec le fichier final

n_before = len(df_final)

df_corrected = df_final.merge(
    rates_map,
    on="id_clean",
    how="left"
)

n_after = len(df_corrected)

print("\nContrôle fusion :")
print("Lignes avant fusion :", n_before)
print("Lignes après fusion :", n_after)

if n_before != n_after:
    raise ValueError("Erreur : le nombre de lignes a changé après la fusion.")


# 13. Analyse de correspondance

ids_final = set(df_final["id_clean"].astype(str))
ids_listings = set(rates_map["id_clean"].astype(str))

missing_ids = ids_final - ids_listings

print("\nAnalyse correspondance id_clean :")
print("Nombre de id_clean du fichier final absents de listings :", len(missing_ids))

if len(missing_ids) > 0:
    print("Exemples id_clean absents :")
    print(list(missing_ids)[:10])


# 14. Remplacement des anciennes colonnes de taux
# On remplace les deux colonnes du fichier final par les vraies
# valeurs récupérées depuis listings.

df_corrected["host_response_rate"] = df_corrected["host_response_rate_correct"]
df_corrected["host_acceptance_rate"] = df_corrected["host_acceptance_rate_correct"]

# On recrée les indicateurs de présence.
df_corrected["has_host_response_rate"] = (
    df_corrected["host_response_rate"]
    .notna()
    .astype(int)
)

df_corrected["has_host_acceptance_rate"] = (
    df_corrected["host_acceptance_rate"]
    .notna()
    .astype(int)
)

# Suppression des colonnes temporaires
df_corrected = df_corrected.drop(
    columns=[
        "host_response_rate_correct",
        "host_acceptance_rate_correct"
    ]
)

# 15. Contrôle après correction

print("\nContrôle final après correction des taux :")

print("\nhost_response_rate :")
print(df_corrected["host_response_rate"].describe())
print("NaN :", df_corrected["host_response_rate"].isna().sum())
print("Valeurs uniques exemple :")
print(np.sort(df_corrected["host_response_rate"].dropna().unique())[:20])
print("...")
print(np.sort(df_corrected["host_response_rate"].dropna().unique())[-20:])

print("\nhost_acceptance_rate :")
print(df_corrected["host_acceptance_rate"].describe())
print("NaN :", df_corrected["host_acceptance_rate"].isna().sum())
print("Valeurs uniques exemple :")
print(np.sort(df_corrected["host_acceptance_rate"].dropna().unique())[:20])
print("...")
print(np.sort(df_corrected["host_acceptance_rate"].dropna().unique())[-20:])

print("\nhas_host_response_rate :")
print(df_corrected["has_host_response_rate"].value_counts(dropna=False))

print("\nhas_host_acceptance_rate :")
print(df_corrected["has_host_acceptance_rate"].value_counts(dropna=False))


# 16. Sauvegarde du fichier final corrigé


output_csv = DATA_TAB_DIR / "airbnb_tabulaire_final_catboost_rates_corriges.csv"
output_excel = DATA_TAB_DIR / "airbnb_tabulaire_final_catboost_rates_corriges.xlsx"

df_corrected.to_csv(output_csv, index=False, encoding="utf-8-sig")
df_corrected.to_excel(output_excel, index=False)

print("\nFichiers finaux corrigés sauvegardés :")
print(output_csv)
print(output_excel)

# 17. Rapport de correction

report_dir = DATA_TAB_DIR / "rapports_correction_rates_final"
report_dir.mkdir(exist_ok=True)

rapport_rates = pd.DataFrame({
    "colonne": [
        "host_response_rate",
        "host_acceptance_rate",
        "has_host_response_rate",
        "has_host_acceptance_rate"
    ],
    "valeurs_manquantes": [
        df_corrected["host_response_rate"].isna().sum(),
        df_corrected["host_acceptance_rate"].isna().sum(),
        df_corrected["has_host_response_rate"].isna().sum(),
        df_corrected["has_host_acceptance_rate"].isna().sum()
    ],
    "pourcentage_manquant": [
        round(df_corrected["host_response_rate"].isna().mean() * 100, 2),
        round(df_corrected["host_acceptance_rate"].isna().mean() * 100, 2),
        round(df_corrected["has_host_response_rate"].isna().mean() * 100, 2),
        round(df_corrected["has_host_acceptance_rate"].isna().mean() * 100, 2)
    ],
    "min": [
        df_corrected["host_response_rate"].min(),
        df_corrected["host_acceptance_rate"].min(),
        df_corrected["has_host_response_rate"].min(),
        df_corrected["has_host_acceptance_rate"].min()
    ],
    "max": [
        df_corrected["host_response_rate"].max(),
        df_corrected["host_acceptance_rate"].max(),
        df_corrected["has_host_response_rate"].max(),
        df_corrected["has_host_acceptance_rate"].max()
    ]
})

rapport_rates.to_csv(
    report_dir / "rapport_correction_rates_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRapport sauvegardé dans :")
print(report_dir)

print("\nCorrection des taux sur le fichier final terminée avec succès.")