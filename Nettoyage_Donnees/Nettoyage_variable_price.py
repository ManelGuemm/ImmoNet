import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "listings_sans_vides_constantes.csv"   # Remplacer par ton vrai fichier : .csv ou .xlsx

if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH)
elif FILE_PATH.endswith(".xlsx"):
    df = pd.read_excel(FILE_PATH)
else:
    raise ValueError("Format non supporté. Utilise un fichier .csv ou .xlsx")

# 3. Vérifier la présence de la colonne price
if "price" not in df.columns:
    raise ValueError("La colonne 'price' est introuvable dans le fichier.")

# 4. S'assurer que price est numérique
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# 5. Vérifications de base

nb_missing = df["price"].isna().sum()
nb_zero = (df["price"] == 0).sum()
nb_negative = (df["price"] < 0).sum()

print("===== Vérification de la colonne price =====")
print(f"Nombre total de lignes : {len(df)}")
print(f"Nombre de valeurs manquantes dans price : {nb_missing}")
print(f"Nombre de prix égaux à 0 : {nb_zero}")
print(f"Nombre de prix négatifs : {nb_negative}")

# 6. Statistiques descriptives
print("\n===== Statistiques descriptives de price =====")
print(df["price"].describe())

# 7. Détection des valeurs extrêmes avec l'IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

prix_extremes = df[(df["price"] < borne_basse) | (df["price"] > borne_haute)]

print("\n===== Détection des valeurs extrêmes =====")
print(f"Q1 : {Q1}")
print(f"Q3 : {Q3}")
print(f"IQR : {IQR}")
print(f"Borne basse : {borne_basse}")
print(f"Borne haute : {borne_haute}")
print(f"Nombre de prix extrêmes détectés : {len(prix_extremes)}")

print("\nExemples de prix extrêmes :")
print(prix_extremes[["price"]].sort_values(by="price", ascending=False).head(20))

# 8. Visualisation de la distribution
plt.figure(figsize=(8, 5))
plt.hist(df["price"].dropna(), bins=50, edgecolor="black")
plt.title("Distribution de la variable price")
plt.xlabel("Prix")
plt.ylabel("Nombre d'annonces")
plt.tight_layout()
plt.savefig("histogramme_price.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(df["price"].dropna(), vert=False)
plt.title("Boxplot de la variable price")
plt.xlabel("Prix")
plt.tight_layout()
plt.savefig("boxplot_price.png", dpi=300)
plt.close()

print("Les graphiques ont été enregistrés : histogramme_price.png et boxplot_price.png")
# 9. Export des résultats dans un fichier Excel

output_file = "verification_price_resultats.xlsx"

# Tableau récapitulatif
resume_df = pd.DataFrame({
    "indicateur": [
        "nombre_total_lignes",
        "nombre_valeurs_manquantes_price",
        "nombre_price_egaux_0",
        "nombre_price_negatifs",
        "Q1",
        "Q3",
        "IQR",
        "borne_basse_IQR",
        "borne_haute_IQR",
        "nombre_prix_extremes"
    ],
    "valeur": [
        len(df),
        nb_missing,
        nb_zero,
        nb_negative,
        Q1,
        Q3,
        IQR,
        borne_basse,
        borne_haute,
        len(prix_extremes)
    ]
})

# Statistiques descriptives
stats_df = df["price"].describe().reset_index()
stats_df.columns = ["statistique", "valeur"]

# Prix extrêmes
prix_extremes_df = prix_extremes[["price"]].sort_values(by="price", ascending=False).copy()

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    resume_df.to_excel(writer, sheet_name="resume_verification", index=False)
    stats_df.to_excel(writer, sheet_name="statistiques_price", index=False)
    prix_extremes_df.to_excel(writer, sheet_name="prix_extremes", index=False)

print(f"\nLes résultats ont été enregistrés dans : {output_file}")