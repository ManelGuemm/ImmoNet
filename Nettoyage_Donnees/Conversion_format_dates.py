import pandas as pd

' Fichier qui convertit les colonnes de dates du fichier listings_avec_annonces_supprimees '
' en format date et sauvegarde le résultat dans listings_dates_converties.csv '


# Fichier d'entrée
fichier_csv = "listings_avec_annonces_supprimees.csv"

# Fichier de sortie
fichier_sortie = "listings_dates_converties.csv"

# Charger le fichier
df = pd.read_csv(fichier_csv, sep=None, engine="python", encoding="utf-8-sig")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Colonnes de dates à convertir
colonnes_dates = [
    "last_scraped",
    "host_since",
    "calendar_last_scraped",
    "first_review",
    "last_review"
]

# Conversion en format date
for col in colonnes_dates:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

# Vérification dans le terminal
print(df[colonnes_dates].dtypes)

# Sauvegarde
df.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

print(f"Nouveau fichier enregistré : {fichier_sortie}")