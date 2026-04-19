import pandas as pd


''' Fichier qui convertit les % en nombres et les booléens en 0/1 dans le fichier 
    listings_dates_converties.csv '''


# Fichier d'entrée
fichier_csv = "listings_dates_converties.csv"

# Fichier de sortie
fichier_sortie = "listings_types_corriges.csv"

# Charger le fichier
df = pd.read_csv(fichier_csv, sep=None, engine="python", encoding="utf-8-sig")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()


# 1. CONVERSION DES POURCENTAGES

colonnes_pourcentages = [
    "host_response_rate",
    "host_acceptance_rate"
]

for col in colonnes_pourcentages:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .replace("nan", pd.NA)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")


# 2. CONVERSION DES BOOLÉENS EN 0/1

colonnes_booleennes = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable"
]

mapping_bool = {
    "t": 1,
    "f": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0
}

for col in colonnes_booleennes:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping_bool)
        )

# Vérification dans le terminal
print(df[colonnes_pourcentages + [c for c in colonnes_booleennes if c in df.columns]].dtypes)

# Sauvegarde
df.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

print(f"Nouveau fichier enregistré : {fichier_sortie}")