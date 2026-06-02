from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

fichier_tabulaire = (
    PROJECT_DIR
    / "Tabulaire"
    / "Donnees_Tabulaires"
    / "Donnees_Airbnb_Finales_Tabulaire.xlsx"
)

fichier_reviews = PROJECT_DIR / "reviews.csv"

dossier_sortie = PROJECT_DIR / "Textuelles" / "Resultats_Sentiment"
dossier_sortie.mkdir(parents=True, exist_ok=True)

fichier_reviews_filtre = dossier_sortie / "reviews_filtrees_annonces_finales.csv"


print("Dossier du script :", BASE_DIR)
print("Dossier du projet :", PROJECT_DIR)
print("Chemin tabulaire :", fichier_tabulaire)
print("Existe tabulaire :", fichier_tabulaire.exists())
print("Chemin reviews :", fichier_reviews)
print("Existe reviews :", fichier_reviews.exists())
print("Dossier de sortie :", dossier_sortie)


if not fichier_tabulaire.exists():
    raise FileNotFoundError(f"Fichier tabulaire introuvable : {fichier_tabulaire}")

if not fichier_reviews.exists():
    raise FileNotFoundError(f"Fichier reviews introuvable : {fichier_reviews}")


df_tab = pd.read_excel(fichier_tabulaire, dtype=str)
df_tab.columns = df_tab.columns.str.strip()


if "id_clean" in df_tab.columns:
    print("\nLa colonne id_clean existe déjà. Elle sera utilisée.")

    df_tab["id_clean"] = (
        df_tab["id_clean"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

else:
    print("\nLa colonne id_clean n'existe pas. Création depuis listing_url.")

    if "listing_url" not in df_tab.columns:
        raise ValueError(
            "Impossible de créer id_clean : la colonne listing_url est absente."
        )

    df_tab["id_clean"] = (
        df_tab["listing_url"]
        .astype("string")
        .str.extract(r"/rooms/(\d+)")[0]
    )


print("\n===== Vérification du fichier tabulaire =====")
print("Nombre de lignes :", len(df_tab))
print("Nombre d'identifiants uniques :", df_tab["id_clean"].nunique())
print("Nombre d'identifiants manquants :", df_tab["id_clean"].isna().sum())
print("Nombre de doublons :", df_tab["id_clean"].duplicated().sum())


if df_tab["id_clean"].isna().sum() > 0:
    raise ValueError(
        "Certains identifiants id_clean sont manquants. "
        "Il faut vérifier listing_url ou la colonne id_clean."
    )

if df_tab["id_clean"].duplicated().sum() > 0:
    raise ValueError(
        "Des doublons existent dans id_clean. "
        "Il faut les vérifier avant de continuer."
    )


ids_annonces_finales = set(df_tab["id_clean"].dropna().astype(str))

print("\n===== Filtrage du fichier reviews =====")
print("Nombre d'annonces finales utilisées pour le filtrage :", len(ids_annonces_finales))


nb_total_reviews = 0
nb_reviews_gardees = 0
nb_reviews_supprimees = 0

chunksize = 100_000
premier_morceau = True


for chunk in pd.read_csv(
    fichier_reviews,
    dtype=str,
    chunksize=chunksize,
    encoding="utf-8",
    on_bad_lines="skip"
):
    chunk.columns = chunk.columns.str.strip()

    if "listing_id" not in chunk.columns:
        raise ValueError("La colonne 'listing_id' est absente du fichier reviews.")

    if "comments" not in chunk.columns:
        raise ValueError("La colonne 'comments' est absente du fichier reviews.")

    nb_total_reviews += len(chunk)

    chunk["listing_id"] = (
        chunk["listing_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    chunk_filtre = chunk[
        chunk["listing_id"].isin(ids_annonces_finales)
    ].copy()

    nb_reviews_gardees += len(chunk_filtre)
    nb_reviews_supprimees += len(chunk) - len(chunk_filtre)

    chunk_filtre.to_csv(
        fichier_reviews_filtre,
        mode="w" if premier_morceau else "a",
        header=premier_morceau,
        index=False,
        encoding="utf-8-sig"
    )

    premier_morceau = False


print("\n===== Résultat du filtrage =====")
print("Nombre total d'avis dans reviews :", nb_total_reviews)
print("Nombre d'avis gardés :", nb_reviews_gardees)
print("Nombre d'avis supprimés :", nb_reviews_supprimees)
print("Fichier filtré créé :", fichier_reviews_filtre)

print("\nFiltrage terminé avec succès.")