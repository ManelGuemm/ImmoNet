# Fichier qui supprime les annonces fausses du fichier listings_sans_vides_constantes.csv
# On se base sur le fichier annonces_price_sup_2000.xlsx qui contient les annonces avec 
# un prix supérieur à 2000 euros

import pandas as pd

# Fichiers
fichier_brut = "listings_sans_vides_constantes.csv"
fichier_excel = "annonces_price_sup_2000.xlsx"
fichier_sortie = "listings_avec_annonces_supprimees.csv"

# Lire les fichiers
df_brut = pd.read_csv(fichier_brut, sep=None, engine="python", encoding="utf-8-sig")
df_excel = pd.read_excel(fichier_excel)

# Nettoyer les noms de colonnes
df_brut.columns = df_brut.columns.str.strip()
df_excel.columns = df_excel.columns.str.strip()

# Mettre les URL au même format
df_brut["listing_url"] = df_brut["listing_url"].astype(str).str.strip()
df_excel["listing_url"] = df_excel["listing_url"].astype(str).str.strip()

# Supprimer selon listing_url
urls_a_supprimer = set(df_excel["listing_url"])
df_final = df_brut[~df_brut["listing_url"].isin(urls_a_supprimer)]

# Résultats
nb_avant = len(df_brut)
nb_apres = len(df_final)
nb_supprimees = nb_avant - nb_apres

print(f"Nombre de lignes avant suppression : {nb_avant}")
print(f"Nombre de lignes supprimées : {nb_supprimees}")
print(f"Nombre de lignes après suppression : {nb_apres}")

# Sauvegarde
df_final.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

print(f"Nouveau fichier enregistré : {fichier_sortie}")