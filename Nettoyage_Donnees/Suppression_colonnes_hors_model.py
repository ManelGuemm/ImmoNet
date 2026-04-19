import pandas as pd

''' Fichier qui supprime les colonnes qui ne sont pas utilisées dans aucun modèle de prédiction '''



# Fichier d'entrée
fichier_entree = "listings_types_corriges.csv"

# Fichier de sortie
fichier_sortie = "fichier_sans_colonnes.csv"

# Charger le fichier Excel
df = pd.read_csv(fichier_entree)

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Colonnes à supprimer
colonnes_a_supprimer = [
    "last_scraped",
    "neighborhood_overview",
    "host_id",
    "host_url",
    "host_name",
    "host_since",
    "host_location",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "calendar_last_scraped",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "availability_eoy",
    "number_of_reviews_ly",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "first_review",
    "last_review",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "reviews_per_month"
]

# Supprimer seulement les colonnes qui existent dans le fichier
df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns])

# Sauvegarder dans un nouveau fichier Excel
df.to_csv(fichier_sortie, index=False)

print("Colonnes supprimées avec succès.")
print(f"Nouveau fichier enregistré : {fichier_sortie}")
print(f"Nombre de colonnes restantes : {df.shape[1]}")
print(f"Nombre de lignes : {df.shape[0]}")