import pandas as pd


''' Fichier qui vérifie les annonces avec un prix supérieur à 2000 euros dans le fichier 
    listings_sans_vides_constantes.csv
    et qui sauvegarde ces annonces dans un fichier Excel pour les supprimer ensuite du 
    fichier final listings_avec_annonces_supprimees.csv ''' 


# Fichier d'entrée
fichier_csv = "listings_sans_vides_constantes.csv"

# Fichier de sortie
fichier_excel = "annonces_price_sup_2000.xlsx"

# Charger le CSV
df = pd.read_csv(fichier_csv, sep=None, engine="python", encoding="utf-8-sig")

# Nettoyer seulement les noms de colonnes
df.columns = df.columns.str.strip()

# Sélection des annonces avec price > 2000
df_sup_2000 = df[df["price"] > 2000][["id", "listing_url", "price"]]

# Compter le nombre d'annonces
nb_annonces = len(df_sup_2000)

# Afficher dans le terminal
print(f"Nombre d'annonces avec un prix supérieur à 2000 : {nb_annonces}")

# Enregistrer dans un fichier Excel
df_sup_2000.to_excel(fichier_excel, index=False)

print(f"Fichier Excel enregistré : {fichier_excel}")