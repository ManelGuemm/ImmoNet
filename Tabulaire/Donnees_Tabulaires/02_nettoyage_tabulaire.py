
# Objectif :
# - Nettoyer la base Airbnb tabulaire pour CatBoost
# - Créer id_clean depuis listing_url
# - Nettoyer price et créer log_price
# - Transformer host_about en has_host_about
# - Nettoyer les booléens
# - Nettoyer certains numériques 
# - Nettoyer les reviews
# - Transformer amenities en variables binaires



import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path



# 1. Gestion des chemins

SCRIPT_DIR = Path(__file__).resolve().parent

def find_data_dir():
    """
    Recherche automatique du dossier Donnees_Tabulaires.
    Le script peut être placé dans :
    - Tabulaire/ModeleTabulaire__Cat_Boost/
    - ou directement dans Tabulaire/Donnees_Tabulaires/
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



# 2. Recherche du fichier Excel source

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
    excel_files = [
        f for f in DATA_DIR.glob("*.xlsx")
        if not f.name.startswith("~$")
    ]

    if len(excel_files) == 0:
        raise FileNotFoundError("Aucun fichier Excel .xlsx trouvé.")

    input_file = excel_files[0]

print("\nFichier utilisé :")
print(input_file)



# 3. Lecture des données

df = pd.read_excel(input_file)

print("\nDimensions initiales :")
print(df.shape)



# 4. Nettoyage des noms de colonnes
# Objectif :
# - mettre les noms en minuscules
# - remplacer les espaces par des underscores
# - supprimer les caractères problématiques
# Exemple :"Review Scores Rating" devient "review_scores_rating"


def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w_]", "", col)
    col = re.sub(r"_+", "_", col)
    return col

df.columns = [clean_column_name(c) for c in df.columns]

print("\nColonnes après nettoyage des noms :")
print(df.columns.tolist())


# 5. Création de id_clean depuis listing_url



if "listing_url" not in df.columns:
    raise ValueError("La colonne listing_url est absente. Impossible de créer id_clean.")

df["id_clean"] = df["listing_url"].astype(str).str.extract(r"/rooms/(\d+)")[0]

print("\nContrôle id_clean :")
print("id_clean manquants :", df["id_clean"].isna().sum())
print("id_clean uniques :", df["id_clean"].nunique())
print("doublons id_clean :", df["id_clean"].duplicated().sum())



# 6. Nettoyage de la cible price
# Objectif :
# - convertir price en numérique
# - enlever les symboles monétaires si jamais ils existent
# - supprimer les prix impossibles : NaN, 0, négatifs
# Gestion des valeurs nulles : les price non convertibles deviennent NaN, les lignes avec price NaN ou <= 0 sont supprimées


if "price" not in df.columns:
    raise ValueError("La colonne price est absente.")

def clean_price(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace("€", "")
    value = value.replace("$", "")
    value = value.replace("£", "")
    value = value.replace(",", "")
    value = value.replace(" ", "")

    value = re.sub(r"[^\d.]", "", value)

    if value == "":
        return np.nan

    return float(value)

df["price"] = df["price"].apply(clean_price)

n_before_price = len(df)

df = df[df["price"].notna()]
df = df[df["price"] > 0]

n_after_price = len(df)

print("\nNettoyage de price :")
print("Lignes supprimées :", n_before_price - n_after_price)
print(df["price"].describe())



# 7. Création de log_price
# Objectif :
# - réduire l'effet des prix très élevés pour aprés entraîner le modèle sur log_price


df["log_price"] = np.log1p(df["price"])

print("\nContrôle log_price :")
print(df[["price", "log_price"]].describe())


# 8. Transformation de host_about

# Objectif :
# - NaN ou texte vide => has_host_about = 0
# - texte présent => has_host_about = 1

if "host_about" in df.columns:
    host_about_clean = (
        df["host_about"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["has_host_about"] = (
        ~host_about_clean.isin(["", "nan", "none", "null"])
    ).astype(int)

    print("\nDistribution has_host_about :")
    print(df["has_host_about"].value_counts(dropna=False))
else:
    print("\nColonne host_about absente. Aucun indicateur créé.")



# 9. Conversion des variables booléennes

# Variables concernées :
# - host_is_superhost
# - host_has_profile_pic
# - host_identity_verified
# - instant_bookable

# Objectif :
# - transformer t/f, True/False, yes/no en 1/0
# - les valeurs manquantes restent NaN, on ne les remplace pas maintenant
# - CatBoost pourra gérer les NaN numériques


def convert_bool_to_01(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in [0, 1]:
            return int(value)

    value = str(value).strip().lower()

    mapping = {
        "t": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "1": 1,
        "f": 0,
        "false": 0,
        "no": 0,
        "n": 0,
        "0": 0
    }

    return mapping.get(value, np.nan)

bool_cols = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable"
]

print("\nConversion des variables booléennes :")

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].apply(convert_bool_to_01)
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False))
    else:
        print(f"{col} absente.")



# 10. Nettoyage des taux hôte
# Variables concernées :
# - host_response_rate
# - host_acceptance_rate
#
# Objectif :
# - convertir les valeurs en numérique entre 0 et 1
# - accepter les formats 1, 0, 95%, "95"
#
# Gestion des valeurs nulles :
# - les NaN restent NaN
# - on crée une variable has_col pour indiquer si l'information existe

def clean_rate(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
    else:
        value = str(value).strip().lower()
        value = value.replace("%", "")
        value = value.replace(",", ".")
        value = re.sub(r"[^\d.]", "", value)

        if value == "":
            return np.nan

        value = float(value)

    if value > 1:
        value = value / 100

    if value < 0 or value > 1:
        return np.nan

    return value

rate_cols = [
    "host_response_rate",
    "host_acceptance_rate"
]

print("\nNettoyage des taux hôte :")

for col in rate_cols:
    if col in df.columns:
        df[f"has_{col}"] = df[col].notna().astype(int)
        df[col] = df[col].apply(clean_rate)

        print(f"\n{col}")
        print(df[col].describe())
        print(f"has_{col}")
        print(df[f"has_{col}"].value_counts(dropna=False))
    else:
        print(f"{col} absente.")



# 11. Conversion des variables numériques 
# Objectif :
# - convertir en numérique 
# - on NE TOUCHE PAS à beds
# - on NE TOUCHE PAS à minimum_nights
# - on NE TOUCHE PAS à maximum_nights
# - les valeurs non convertibles deviennent NaN
# - on ne fait pas d'imputation agressive


numeric_cols_to_convert = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

print("\nConversion des variables numériques validées :")

for col in numeric_cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"{col} convertie en numérique.")
    else:
        print(f"{col} absente.")



# 12. Nettoyage de bathrooms
# Objectif :
# - créer une version bathrooms_clean
# - remplacer uniquement les valeurs négatives, car elles sont impossibles
# - ne PAS modifier les grandes valeurs sans décision manuelle
# - les NaN restent NaN
# - CatBoost pourra les gérer


if "bathrooms" in df.columns:
    df["bathrooms_clean"] = df["bathrooms"].copy()

    # Valeurs impossibles : un nombre de salles de bain négatif
    df.loc[df["bathrooms_clean"] < 0, "bathrooms_clean"] = np.nan

    print("\nContrôle bathrooms_clean :")
    print(df["bathrooms_clean"].describe())
    print("NaN bathrooms_clean :", df["bathrooms_clean"].isna().sum())

    print("\nContrôle des grandes valeurs de bathrooms_clean, sans modification :")
    print(df["bathrooms_clean"].value_counts(dropna=False).sort_index().tail(15))

    print("\nNombre de logements avec bathrooms_clean > 10 :")
    print((df["bathrooms_clean"] > 10).sum())

else:
    print("\nColonne bathrooms absente.")



# 12 bis. Transformation de bathrooms_text en is_bathroom_shared
   
# Objectif :
# - ne pas garder le texte brut bathrooms_text dans le modèle
# - extraire uniquement une information utile : salle de bain partagée ou non
# - bathrooms_text manquant => is_bathroom_shared = NaN
# - CatBoost pourra gérer cette valeur manquante


if "bathrooms_text" in df.columns:
    bathrooms_text_clean = (
        df["bathrooms_text"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    df["is_bathroom_shared"] = np.where(
        bathrooms_text_clean.isna(),
        np.nan,
        bathrooms_text_clean.str.contains("shared", na=False).astype(int)
    )

    print("\nDistribution de is_bathroom_shared :")
    print(df["is_bathroom_shared"].value_counts(dropna=False))

    print("\nExemples bathrooms_text -> is_bathroom_shared :")
    print(
        df[["bathrooms_text", "is_bathroom_shared"]]
        .drop_duplicates()
        .head(20)
    )

else:
    print("\nColonne bathrooms_text absente. Variable is_bathroom_shared non créée.")


# 13. Nettoyage de bedrooms
# Objectif :
# - créer une version bedrooms_clean
# - conserver bedrooms = 0, car cela peut correspondre à un studio
# - remplacer uniquement les valeurs négatives, car elles sont impossibles
# - les NaN restent NaN
# - CatBoost pourra les gérer


if "bedrooms" in df.columns:
    df["bedrooms_clean"] = df["bedrooms"].copy()

    # Valeurs impossibles : un nombre de chambres négatif
    df.loc[df["bedrooms_clean"] < 0, "bedrooms_clean"] = np.nan

    print("\nContrôle bedrooms_clean :")
    print(df["bedrooms_clean"].describe())
    print("NaN bedrooms_clean :", df["bedrooms_clean"].isna().sum())

    print("\nContrôle des grandes valeurs de bedrooms_clean, sans modification :")
    print(df["bedrooms_clean"].value_counts(dropna=False).sort_index().tail(15))

    print("\nNombre de logements avec bedrooms_clean > 10 :")
    print((df["bedrooms_clean"] > 10).sum())

else:
    print("\nColonne bedrooms absente.")

# 14. Reviews : création de has_reviews
# Objectif :
# - garder les reviews pour le modèle tabulaire actuel
# - créer has_reviews pour distinguer les annonces avec/sans avis
# - si number_of_reviews est manquant, on considère 0 avis
# - has_reviews = 1 si number_of_reviews > 0
# - has_reviews = 0 sinon
# ============================================================

if "number_of_reviews" in df.columns:
    df["has_reviews"] = (df["number_of_reviews"].fillna(0) > 0).astype(int)

    print("\nDistribution has_reviews :")
    print(df["has_reviews"].value_counts(dropna=False))
else:
    print("\nColonne number_of_reviews absente.")


# ============================================================
# 15. Nettoyage des scores de reviews
# ============================================================
# Objectif :
# - créer une version clean pour chaque score
# - les scores doivent être compris entre 1 et 5
#
# Gestion des valeurs nulles :
# - les annonces sans reviews gardent NaN
# - les valeurs < 1 deviennent NaN
# - les valeurs > 5 deviennent NaN

review_score_cols = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

print("\nNettoyage des scores de reviews :")

for col in review_score_cols:
    if col in df.columns:
        clean_col = f"{col}_clean"

        df[clean_col] = df[col].copy()

        df.loc[df[clean_col] < 1, clean_col] = np.nan
        df.loc[df[clean_col] > 5, clean_col] = np.nan

        print(f"\n{clean_col}")
        print(df[clean_col].describe())
        print("NaN :", df[clean_col].isna().sum())
    else:
        print(f"{col} absente.")


# 16. Parsing de amenities
# Objectif :
# - transformer la liste texte des équipements en vraie liste Python
# - créer amenities_count
# - NaN ou liste vide => []
# - amenities_count = 0

def parse_amenities(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip().lower() for x in value]

    value = str(value).strip()

    if value == "" or value == "[]":
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except Exception:
        pass

    value = value.replace("[", "").replace("]", "")
    value = value.replace('"', "").replace("'", "")

    return [
        x.strip().lower()
        for x in value.split(",")
        if x.strip() != ""
    ]

if "amenities" in df.columns:
    df["amenities_list"] = df["amenities"].apply(parse_amenities)
    df["amenities_count"] = df["amenities_list"].apply(len)

    print("\nContrôle amenities_count :")
    print(df["amenities_count"].describe())
else:
    print("\nColonne amenities absente.")


# ============================================================
# 17. Création des variables binaires issues de amenities
# ============================================================
# Objectif :
# - créer des variables 0/1 simples et interprétables
# - éviter les faux positifs
#
# Exemples de faux positifs évités :
# - washer ne doit pas détecter dishwasher
# - dryer ne doit pas détecter hair dryer
# - air conditioning ne doit pas utiliser le mot trop vague "ac"
# - pool ne doit pas détecter pool table ou whirlpool
#
# Gestion des valeurs nulles :
# - si amenities_list est vide, toutes les variables has_* valent 0
# ============================================================

def normalize_amenity_item(item):
    """
    Nettoie un équipement individuel.
    On travaille item par item, pas sur toute la liste concaténée,
    pour éviter les détections trop larges.
    """
    item = str(item).lower().strip()
    item = item.replace("–", "-").replace("—", "-").replace("’", "'")
    item = re.sub(r"\s+", " ", item)
    return item


amenity_rules = {
    "has_wifi": {
        "include": ["wifi", "internet"],
        "exclude": []
    },
    "has_kitchen": {
        "include": ["kitchen", "kitchenette"],
        "exclude": []
    },
    "has_parking": {
        "include": ["parking", "garage", "driveway", "street parking", "valet parking"],
        "exclude": []
    },
    "has_air_conditioning": {
        "include": ["air conditioning"],
        "exclude": []
    },
    "has_heating": {
        "include": ["heating"],
        "exclude": []
    },
    "has_pool": {
        "include": ["pool"],
        "exclude": ["pool table", "whirlpool", "pool view"]
    },
    "has_washer": {
        "include": ["washer", "washing machine"],
        "exclude": ["dishwasher"]
    },
    "has_dryer": {
        "include": ["dryer"],
        "exclude": ["hair dryer"]
    },
    "has_tv": {
        "include": ["tv", "television", "hdtv"],
        "exclude": []
    },
    "has_workspace": {
        "include": ["dedicated workspace", "workspace", "desk"],
        "exclude": []
    },
    "has_balcony": {
        "include": ["balcony", "patio", "terrace"],
        "exclude": []
    },
    "has_gym": {
        "include": ["gym", "exercise equipment"],
        "exclude": []
    },
    "has_elevator": {
        "include": ["elevator", "lift"],
        "exclude": []
    },
    "has_smoke_alarm": {
        "include": ["smoke alarm"],
        "exclude": []
    },
    "has_self_checkin": {
        "include": ["self check-in", "self check in", "lockbox", "keypad", "smart lock"],
        "exclude": []
    },
    "has_dishwasher": {
        "include": ["dishwasher"],
        "exclude": []
    },
    "has_microwave": {
        "include": ["microwave"],
        "exclude": []
    },
    "has_refrigerator": {
        "include": ["refrigerator", "fridge"],
        "exclude": []
    },
    "has_oven": {
        "include": ["oven"],
        "exclude": []
    },
    "has_bathtub": {
        "include": ["bathtub", "bath tub"],
        "exclude": []
    },
    "has_private_entrance": {
        "include": ["private entrance"],
        "exclude": []
    },
    "has_luggage_dropoff": {
        "include": ["luggage dropoff", "luggage drop-off"],
        "exclude": []
    },
    "has_long_term_stays": {
        "include": ["long term stays allowed", "long-term stays allowed"],
        "exclude": []
    },
    "has_hot_tub": {
        "include": ["hot tub", "jacuzzi"],
        "exclude": []
    },
    "has_pets_allowed": {
        "include": ["pets allowed", "pet friendly"],
        "exclude": []
    },
    "has_baby_equipment": {
        "include": ["crib", "high chair", "baby", "children"],
        "exclude": []
    },
    "has_security_camera": {
        "include": ["security camera", "security cameras", "cameras on property"],
        "exclude": []
    }
}


def has_amenity_rule(amenities_list, rule):
    """
    Retourne 1 si au moins un équipement correspond à la règle.
    Retourne 0 sinon.
    """
    include_keywords = rule.get("include", [])
    exclude_keywords = rule.get("exclude", [])

    for raw_item in amenities_list:
        item = normalize_amenity_item(raw_item)

        if any(excluded in item for excluded in exclude_keywords):
            continue

        if any(included in item for included in include_keywords):
            return 1

    return 0


if "amenities_list" in df.columns:
    for new_col, rule in amenity_rules.items():
        df[new_col] = df["amenities_list"].apply(
            lambda x: has_amenity_rule(x, rule)
        )

    print("\nVariables amenities créées :")
    amenity_cols = list(amenity_rules.keys())
    print(df[amenity_cols].mean().sort_values(ascending=False))
else:
    print("\namenities_list absente. Variables amenities non créées.")



# 18. Suppression des colonnes 
# Objectif :
# - supprimer les identifiants techniques
# - supprimer les textes bruts
# - supprimer amenities brut après transformation
# Important :
# - id_clean est conservé dans le fichier final pour les jointures futures


cols_to_drop = [
    "id",
    "listing_url",
    "picture_url",
    "host_about",
    "bathrooms_text",
    "amenities",
    "amenities_list"
]

existing_cols_to_drop = [
    col for col in cols_to_drop
    if col in df.columns
]

df_clean = df.drop(columns=existing_cols_to_drop)

print("\nColonnes supprimées :")
print(existing_cols_to_drop)

print("\nDimensions après premières suppressions :")
print(df_clean.shape)



# 19. Suppression des colonnes originales remplacées par clean
# Objectif :
# - éviter d'avoir deux versions de la même information
# On supprime :
# - bathrooms car on garde bathrooms_clean
# - bedrooms car on garde bedrooms_clean
# - review_scores_* car on garde review_scores_*_clean


cols_replaced_by_clean = [
    "bathrooms",
    "bedrooms",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

existing_replaced_cols = [
    col for col in cols_replaced_by_clean
    if col in df_clean.columns
]

df_clean = df_clean.drop(columns=existing_replaced_cols)

print("\nColonnes originales remplacées supprimées :")
print(existing_replaced_cols)

print("\nDimensions finales provisoires :")
print(df_clean.shape)



# 20. Contrôle : variables qu'on n'a volontairement pas touchées


not_touched_cols = [
    "beds",
    "minimum_nights",
    "maximum_nights"
]

print("\nContrôle des variables laissées pour décision plus tard :")

for col in not_touched_cols:
    if col in df_clean.columns:
        print(f"\n{col}")
        print("Type :", df_clean[col].dtype)
        print(df_clean[col].describe(include="all"))
    else:
        print(f"{col} absente.")


# 21. Contrôle : variables catégorielles non traitées

# Objectif :
# - vérifier qu'elles existent toujours
# - ne pas les transformer maintenant
#Elles seront décidées plus tard :
# - room_type
# - property_type
# - neighbourhood_cleansed
# - host_response_time


categorical_cols_not_processed = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_response_time"
]

print("\nContrôle des variables catégorielles non traitées :")

for col in categorical_cols_not_processed:
    if col in df_clean.columns:
        print(f"\n{col}")
        print("Type :", df_clean[col].dtype)
        print("Nombre de modalités :", df_clean[col].nunique(dropna=False))
        print(df_clean[col].value_counts(dropna=False).head(10))
    else:
        print(f"{col} absente.")



# 22. Rapport final des valeurs manquantes


missing_report = pd.DataFrame({
    "colonne": df_clean.columns,
    "type": df_clean.dtypes.astype(str).values,
    "valeurs_manquantes": df_clean.isna().sum().values,
    "pourcentage_manquant": (df_clean.isna().sum().values / len(df_clean) * 100).round(2)
})

missing_report = missing_report.sort_values(
    by="pourcentage_manquant",
    ascending=False
)

print("\nTop 30 des valeurs manquantes après nettoyage :")
print(missing_report.head(30))



# 23. Contrôle des colonnes disponibles pour le futur modèle
# - price est la cible en euros, gardée pour interprétation
# - log_price est la cible d'entraînement probable
# - id_clean est gardé pour jointure, mais pas pour le modèle

cols_not_for_model = [
    "id_clean",
    "price",
    "log_price"
]

feature_cols_provisoires = [
    col for col in df_clean.columns
    if col not in cols_not_for_model
]

print("\nNombre de variables explicatives provisoires :")
print(len(feature_cols_provisoires))

print("\nVariables explicatives provisoires :")
for col in feature_cols_provisoires:
    print("-", col)



# 24. Sauvegarde du fichier nettoyé

output_csv = DATA_DIR / "airbnb_tabulaire_clean_sans_categories.csv"
output_excel = DATA_DIR / "airbnb_tabulaire_clean_sans_categories.xlsx"

df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")

# Sauvegarde Excel aussi, pratique pour vérification visuelle dans VS Code
df_clean.to_excel(output_excel, index=False)

print("\nFichiers nettoyés sauvegardés :")
print(output_csv)
print(output_excel)

# 25. Sauvegarde du rapport de valeurs manquantes


report_dir = DATA_DIR / "rapports_nettoyage"
report_dir.mkdir(exist_ok=True)

missing_report.to_csv(
    report_dir / "rapport_valeurs_manquantes_apres_nettoyage.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRapport de nettoyage sauvegardé dans :")
print(report_dir)

print("\nNettoyage terminé avec succès.")