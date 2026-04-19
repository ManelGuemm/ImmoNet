import pandas as pd

''' Fichier qui supprime les colonnes vides et constantes du fichier listings_price_valide.csv
    et sauvegarde le résultat dans listings_sans_vides_constantes.csv '''


# Charger le fichier
df = pd.read_csv("listings_price_valide.csv", sep=None, engine="python", encoding="utf-8-sig")

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Colonnes à supprimer
colonnes_a_supprimer = [
    "neighbourhood_group_cleansed",
    "calendar_updated",
    "license",
    "scrape_id",
    "source",
    "neighbourhood",
    "has_availability"
]

# Supprimer les colonnes
df = df.drop(columns=colonnes_a_supprimer)

# Sauvegarder le nouveau fichier
df.to_csv("listings_sans_vides_constantes.csv", index=False, encoding="utf-8-sig")

print("Colonnes supprimées avec succès.")
print("Nouveau fichier enregistré : listings_sans_vides_constantes.csv")
print(f"Nombre de colonnes restantes : {df.shape[1]}")
print(f"Nombre de lignes : {df.shape[0]}")