import pandas as pd

fichier_csv = "Donnees/listings_price_valide.csv"
# fichier_csv = "Donnees/listings.csv"

df = pd.read_csv(fichier_csv, sep=None, engine="python", encoding="utf-8-sig")

df.columns = df.columns.str.strip()

if "id" not in df.columns:
    raise ValueError("La colonne 'id' est introuvable dans le fichier.")

df["id"] = df["id"].astype(str).str.strip()

# 1. Doublons exacts
nb_doublons_exacts = df.duplicated().sum()

print("\n===== DOUBLONS EXACTS =====")
print(f"Nombre de doublons exacts : {nb_doublons_exacts}")

if nb_doublons_exacts > 0:
    print("\nLignes concernées par les doublons exacts :")
    print(df[df.duplicated(keep=False)])
else:
    print("Aucun doublon exact trouvé.")

# 2. Même id répété
ids_dupliques = df[df["id"].duplicated(keep=False)].sort_values("id")
nb_lignes_id_dupliques = len(ids_dupliques)
nb_id_distincts_dupliques = ids_dupliques["id"].nunique()

print("\n===== ID DUPLIQUÉS =====")
print(f"Nombre de lignes ayant un id répété : {nb_lignes_id_dupliques}")
print(f"Nombre d'id distincts répétés : {nb_id_distincts_dupliques}")

if nb_lignes_id_dupliques > 0:
    print("\nRésumé des id répétés :")
    print(df["id"].value_counts()[df["id"].value_counts() > 1])
else:
    print("Tous les id sont uniques.")