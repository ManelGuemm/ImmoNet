from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

fichier_reviews_clean = BASE_DIR / "reviews_filtrees_clean.csv"
fichier_reviews_bert = BASE_DIR / "reviews_avec_sentiment_bert_COMPLET.csv"
fichier_sentiment_annonce = BASE_DIR / "sentiment_bert_par_annonce_COMPLET.csv"


print("\n============================================================")
print("BLOC 1 — VALIDATION TECHNIQUE DU TRAITEMENT BERT")
print("============================================================")


print("\n============================================================")
print("1. Vérification de l'existence des fichiers")
print("============================================================")

print("Fichier reviews nettoyé existe :", fichier_reviews_clean.exists())
print("Fichier avis avec sentiment BERT existe :", fichier_reviews_bert.exists())
print("Fichier sentiment agrégé par annonce existe :", fichier_sentiment_annonce.exists())

if not fichier_reviews_clean.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_reviews_clean}")

if not fichier_reviews_bert.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_reviews_bert}")

if not fichier_sentiment_annonce.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_sentiment_annonce}")


print("\n============================================================")
print("2. Comptage des lignes")
print("============================================================")

def compter_lignes_csv(chemin):
    with open(chemin, "r", encoding="utf-8-sig", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


nb_reviews_clean = compter_lignes_csv(fichier_reviews_clean)
nb_reviews_bert = compter_lignes_csv(fichier_reviews_bert)

print("Nombre d'avis dans reviews_filtrees_clean.csv :", nb_reviews_clean)
print("Nombre d'avis dans reviews_avec_sentiment_bert_COMPLET.csv :", nb_reviews_bert)
print("Différence :", nb_reviews_clean - nb_reviews_bert)

if nb_reviews_clean == nb_reviews_bert:
    print("OK : tous les avis nettoyés semblent avoir été traités par BERT.")
else:
    print("ATTENTION : le nombre de lignes est différent.")
    print("Cela peut être normal si certains commentaires vides ont été retirés avant BERT.")
    print("Il faut vérifier la différence.")


print("\n============================================================")
print("3. Chargement du fichier BERT")
print("============================================================")

df = pd.read_csv(
    fichier_reviews_bert,
    dtype=str,
    encoding="utf-8-sig",
    on_bad_lines="skip"
)

print("Dimensions du fichier avis BERT :", df.shape)


print("\n============================================================")
print("4. Vérification des colonnes attendues")
print("============================================================")

colonnes_attendues = [
    "listing_id",
    "comments_clean",
    "bert_label_raw",
    "bert_confidence",
    "bert_stars",
    "sentiment_score",
    "sentiment_label"
]

for col in colonnes_attendues:
    print(f"{col} :", col in df.columns)

colonnes_absentes = [col for col in colonnes_attendues if col not in df.columns]

if len(colonnes_absentes) > 0:
    raise ValueError(f"Colonnes absentes : {colonnes_absentes}")
else:
    print("OK : toutes les colonnes attendues sont présentes.")


print("\n============================================================")
print("5. Valeurs manquantes dans les sorties BERT")
print("============================================================")

colonnes_sorties_bert = [
    "bert_label_raw",
    "bert_confidence",
    "bert_stars",
    "sentiment_score",
    "sentiment_label"
]

missing_bert = df[colonnes_sorties_bert].isna().sum()

print(missing_bert)

if missing_bert.sum() == 0:
    print("OK : aucune valeur manquante dans les sorties BERT.")
else:
    print("ATTENTION : il existe des valeurs manquantes dans les sorties BERT.")


print("\n============================================================")
print("6. Conversion des scores numériques")
print("============================================================")

df["bert_confidence"] = pd.to_numeric(df["bert_confidence"], errors="coerce")
df["bert_stars"] = pd.to_numeric(df["bert_stars"], errors="coerce")
df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

print("Valeurs non numériques après conversion :")
print("bert_confidence :", df["bert_confidence"].isna().sum())
print("bert_stars :", df["bert_stars"].isna().sum())
print("sentiment_score :", df["sentiment_score"].isna().sum())


print("\n============================================================")
print("7. Vérification des valeurs possibles")
print("============================================================")

print("\nLabels BERT bruts :")
print(df["bert_label_raw"].value_counts(dropna=False))

print("\nValeurs de bert_stars :")
print(df["bert_stars"].value_counts(dropna=False).sort_index())

print("\nValeurs de sentiment_label :")
print(df["sentiment_label"].value_counts(dropna=False))

print("\nValeurs de sentiment_score :")
print(df["sentiment_score"].value_counts(dropna=False).sort_index())

valeurs_stars_valides = set([1, 2, 3, 4, 5])
valeurs_stars_observees = set(df["bert_stars"].dropna().unique())

valeurs_labels_valides = set(["positif", "neutre", "negatif"])
valeurs_labels_observees = set(df["sentiment_label"].dropna().unique())

if valeurs_stars_observees.issubset(valeurs_stars_valides):
    print("OK : bert_stars contient uniquement des valeurs entre 1 et 5.")
else:
    print("ATTENTION : bert_stars contient des valeurs inattendues.")
    print("Valeurs observées :", valeurs_stars_observees)

if valeurs_labels_observees.issubset(valeurs_labels_valides):
    print("OK : sentiment_label contient uniquement positif, neutre ou negatif.")
else:
    print("ATTENTION : sentiment_label contient des valeurs inattendues.")
    print("Valeurs observées :", valeurs_labels_observees)


print("\n============================================================")
print("8. Distribution des sentiments")
print("============================================================")

distribution_labels = df["sentiment_label"].value_counts(dropna=False)
distribution_pourcentages = (
    df["sentiment_label"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print("\nDistribution en nombre :")
print(distribution_labels)

print("\nDistribution en pourcentage :")
print(distribution_pourcentages)

pct_positif = distribution_pourcentages.get("positif", 0)
pct_neutre = distribution_pourcentages.get("neutre", 0)
pct_negatif = distribution_pourcentages.get("negatif", 0)

print("\nRésumé :")
print("Pourcentage positif :", pct_positif, "%")
print("Pourcentage neutre :", pct_neutre, "%")
print("Pourcentage négatif :", pct_negatif, "%")

if pct_positif > 70:
    print("Distribution cohérente avec le contexte Airbnb : les avis sont majoritairement positifs.")
else:
    print("Distribution à vérifier : la proportion d'avis positifs semble faible pour Airbnb.")

if pct_negatif == 0:
    print("ATTENTION : aucun avis négatif détecté. Il faut vérifier les prédictions.")
else:
    print("OK : des avis négatifs sont bien détectés.")


print("\n============================================================")
print("9. Statistiques des scores BERT")
print("============================================================")

print(df[["bert_confidence", "bert_stars", "sentiment_score"]].describe())

print("\nNombre d'avis avec confiance < 0.50 :", (df["bert_confidence"] < 0.50).sum())
print("Nombre d'avis avec confiance < 0.70 :", (df["bert_confidence"] < 0.70).sum())
print("Nombre d'avis avec confiance >= 0.90 :", (df["bert_confidence"] >= 0.90).sum())


print("\n============================================================")
print("10. Chargement du fichier agrégé par annonce")
print("============================================================")

df_annonce = pd.read_csv(
    fichier_sentiment_annonce,
    dtype=str,
    encoding="utf-8-sig",
    on_bad_lines="skip"
)

print("Dimensions du fichier agrégé :", df_annonce.shape)

colonnes_agregees_attendues = [
    "id_clean",
    "nb_avis_textuels_bert",
    "bert_stars_moyen",
    "bert_confiance_moyenne",
    "sentiment_bert_moyen",
    "sentiment_bert_min",
    "sentiment_bert_max",
    "sentiment_bert_ecart_type",
    "nb_avis_bert_positifs",
    "nb_avis_bert_neutres",
    "nb_avis_bert_negatifs",
    "ratio_avis_bert_positifs",
    "ratio_avis_bert_neutres",
    "ratio_avis_bert_negatifs"
]

print("\nVérification des colonnes agrégées :")
for col in colonnes_agregees_attendues:
    print(f"{col} :", col in df_annonce.columns)

colonnes_agregees_absentes = [
    col for col in colonnes_agregees_attendues
    if col not in df_annonce.columns
]

if len(colonnes_agregees_absentes) > 0:
    raise ValueError(f"Colonnes agrégées absentes : {colonnes_agregees_absentes}")
else:
    print("OK : toutes les colonnes agrégées attendues sont présentes.")


print("\n============================================================")
print("11. Valeurs manquantes dans le fichier agrégé")
print("============================================================")

print(df_annonce[colonnes_agregees_attendues].isna().sum())


print("\n============================================================")
print("12. Statistiques du fichier agrégé")
print("============================================================")

colonnes_agregees_numeriques = [
    "nb_avis_textuels_bert",
    "bert_stars_moyen",
    "bert_confiance_moyenne",
    "sentiment_bert_moyen",
    "sentiment_bert_min",
    "sentiment_bert_max",
    "sentiment_bert_ecart_type",
    "nb_avis_bert_positifs",
    "nb_avis_bert_neutres",
    "nb_avis_bert_negatifs",
    "ratio_avis_bert_positifs",
    "ratio_avis_bert_neutres",
    "ratio_avis_bert_negatifs"
]

for col in colonnes_agregees_numeriques:
    df_annonce[col] = pd.to_numeric(df_annonce[col], errors="coerce")

print(df_annonce[colonnes_agregees_numeriques].describe())


print("\n============================================================")
print("13. Contrôle des ratios")
print("============================================================")

df_annonce["somme_ratios"] = (
    df_annonce["ratio_avis_bert_positifs"]
    + df_annonce["ratio_avis_bert_neutres"]
    + df_annonce["ratio_avis_bert_negatifs"]
)

print(df_annonce["somme_ratios"].describe())

nb_ratios_incorrects = (
    (df_annonce["somme_ratios"] < 0.99)
    | (df_annonce["somme_ratios"] > 1.01)
).sum()

print("Nombre d'annonces avec somme des ratios différente de 1 :", nb_ratios_incorrects)

if nb_ratios_incorrects == 0:
    print("OK : les ratios positif/neutre/négatif sont cohérents.")
else:
    print("ATTENTION : certains ratios ne somment pas à 1.")


print("\n============================================================")
print("VALIDATION TECHNIQUE TERMINÉE")
print("============================================================")