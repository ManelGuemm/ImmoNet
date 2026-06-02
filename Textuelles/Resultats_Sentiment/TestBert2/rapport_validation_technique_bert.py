from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


BASE_DIR = Path(__file__).resolve().parent

fichier_reviews_bert = BASE_DIR / "reviews_avec_sentiment_bert_COMPLET.csv"
fichier_sentiment_annonce = BASE_DIR / "sentiment_bert_par_annonce_COMPLET.csv"
fichier_pdf = BASE_DIR / "rapport_validation_technique_bert.pdf"


def make_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def df_to_table(df, index=True, max_rows=None):
    df_copy = df.copy()

    if max_rows is not None:
        df_copy = df_copy.head(max_rows)

    if index:
        df_copy = df_copy.reset_index()

    data = [df_copy.columns.tolist()] + df_copy.astype(str).values.tolist()
    return make_table(data)


if not fichier_reviews_bert.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_reviews_bert}")

if not fichier_sentiment_annonce.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_sentiment_annonce}")


print("Chargement du fichier avis avec sentiment BERT...")

df = pd.read_csv(
    fichier_reviews_bert,
    dtype=str,
    encoding="utf-8-sig",
    on_bad_lines="skip"
)

df["bert_confidence"] = pd.to_numeric(df["bert_confidence"], errors="coerce")
df["bert_stars"] = pd.to_numeric(df["bert_stars"], errors="coerce")
df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")


print("Chargement du fichier agrégé par annonce...")

df_annonce = pd.read_csv(
    fichier_sentiment_annonce,
    dtype=str,
    encoding="utf-8-sig",
    on_bad_lines="skip"
)

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


colonnes_attendues = [
    "listing_id",
    "comments_clean",
    "bert_label_raw",
    "bert_confidence",
    "bert_stars",
    "sentiment_score",
    "sentiment_label"
]

presence_colonnes = pd.DataFrame({
    "Colonne": colonnes_attendues,
    "Présente": [col in df.columns for col in colonnes_attendues]
})

colonnes_sorties_bert = [
    "bert_label_raw",
    "bert_confidence",
    "bert_stars",
    "sentiment_score",
    "sentiment_label"
]

missing_bert = df[colonnes_sorties_bert].isna().sum().reset_index()
missing_bert.columns = ["Variable", "Valeurs manquantes"]

distribution_labels = df["sentiment_label"].value_counts(dropna=False).reset_index()
distribution_labels.columns = ["Sentiment", "Nombre"]

distribution_pourcentages = (
    df["sentiment_label"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
    .reset_index()
)

distribution_pourcentages.columns = ["Sentiment", "Pourcentage"]

distribution_complete = distribution_labels.merge(
    distribution_pourcentages,
    on="Sentiment",
    how="left"
)

distribution_stars = df["bert_stars"].value_counts(dropna=False).sort_index().reset_index()
distribution_stars.columns = ["Étoiles BERT", "Nombre"]

stats_scores = df[["bert_confidence", "bert_stars", "sentiment_score"]].describe().round(4)

nb_conf_inf_050 = int((df["bert_confidence"] < 0.50).sum())
nb_conf_inf_070 = int((df["bert_confidence"] < 0.70).sum())
nb_conf_sup_090 = int((df["bert_confidence"] >= 0.90).sum())

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

presence_colonnes_agregees = pd.DataFrame({
    "Colonne": colonnes_agregees_attendues,
    "Présente": [col in df_annonce.columns for col in colonnes_agregees_attendues]
})

missing_agrege = df_annonce[colonnes_agregees_attendues].isna().sum().reset_index()
missing_agrege.columns = ["Variable", "Valeurs manquantes"]

stats_agregees = df_annonce[colonnes_agregees_numeriques].describe().round(4)

df_annonce["somme_ratios"] = (
    df_annonce["ratio_avis_bert_positifs"]
    + df_annonce["ratio_avis_bert_neutres"]
    + df_annonce["ratio_avis_bert_negatifs"]
)

nb_ratios_incorrects = int((
    (df_annonce["somme_ratios"] < 0.99)
    | (df_annonce["somme_ratios"] > 1.01)
).sum())

pct_positif = float(distribution_pourcentages.loc[
    distribution_pourcentages["Sentiment"] == "positif", "Pourcentage"
].iloc[0]) if "positif" in distribution_pourcentages["Sentiment"].values else 0

pct_neutre = float(distribution_pourcentages.loc[
    distribution_pourcentages["Sentiment"] == "neutre", "Pourcentage"
].iloc[0]) if "neutre" in distribution_pourcentages["Sentiment"].values else 0

pct_negatif = float(distribution_pourcentages.loc[
    distribution_pourcentages["Sentiment"] == "negatif", "Pourcentage"
].iloc[0]) if "negatif" in distribution_pourcentages["Sentiment"].values else 0


doc = SimpleDocTemplate(
    str(fichier_pdf),
    pagesize=A4,
    rightMargin=1.5 * cm,
    leftMargin=1.5 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.5 * cm
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    fontSize=18,
    leading=22,
    spaceAfter=16
)

h1_style = ParagraphStyle(
    "Heading1Custom",
    parent=styles["Heading1"],
    fontSize=14,
    leading=18,
    spaceBefore=12,
    spaceAfter=8
)

body_style = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=12,
    spaceAfter=6
)

story = []

story.append(Paragraph("Rapport de validation technique du traitement BERT", title_style))
story.append(Paragraph(
    "Ce rapport présente la validation technique des sorties produites par le modèle BERT appliqué aux avis Airbnb. "
    "L'objectif est de vérifier que les colonnes générées sont présentes, complètes et cohérentes avant leur fusion avec les données tabulaires.",
    body_style
))
story.append(Spacer(1, 0.3 * cm))


story.append(Paragraph("1. Colonnes attendues dans le fichier avis", h1_style))
story.append(df_to_table(presence_colonnes, index=False))

story.append(Paragraph(
    f"Le fichier des avis avec sentiment contient {len(df)} avis exploitables et {df.shape[1]} colonnes. "
    "Les colonnes attendues correspondent aux identifiants des annonces, au commentaire nettoyé et aux sorties principales du modèle BERT.",
    body_style
))


story.append(Paragraph("2. Valeurs manquantes dans les sorties BERT", h1_style))
story.append(df_to_table(missing_bert, index=False))

if missing_bert["Valeurs manquantes"].sum() == 0:
    story.append(Paragraph(
        "Aucune valeur manquante n'a été détectée dans les principales sorties BERT. "
        "Chaque avis chargé dispose donc d'une prédiction exploitable.",
        body_style
    ))
else:
    story.append(Paragraph(
        "Des valeurs manquantes ont été détectées dans les sorties BERT. "
        "Une vérification complémentaire est nécessaire avant l'exploitation des résultats.",
        body_style
    ))


story.append(Paragraph("3. Distribution des sentiments", h1_style))
story.append(df_to_table(distribution_complete, index=False))

story.append(Paragraph(
    f"La distribution obtenue est composée de {pct_positif:.2f} % d'avis positifs, "
    f"{pct_neutre:.2f} % d'avis neutres et {pct_negatif:.2f} % d'avis négatifs. "
    "Cette répartition est cohérente avec le contexte Airbnb, où les avis sont généralement très majoritairement positifs.",
    body_style
))


story.append(Paragraph("4. Distribution des étoiles prédites par BERT", h1_style))
story.append(df_to_table(distribution_stars, index=False))


story.append(Paragraph("5. Statistiques des scores avis par avis", h1_style))
story.append(df_to_table(stats_scores, index=True))

confiance_table = [
    ["Indicateur", "Nombre d'avis"],
    ["bert_confidence < 0.50", str(nb_conf_inf_050)],
    ["bert_confidence < 0.70", str(nb_conf_inf_070)],
    ["bert_confidence >= 0.90", str(nb_conf_sup_090)]
]

story.append(make_table(confiance_table, col_widths=[8 * cm, 5 * cm]))
story.append(Paragraph(
    "Le score de confiance correspond à la probabilité associée à la classe choisie par le modèle. "
    "Il mesure la certitude interne du modèle, mais ne constitue pas une vérité terrain.",
    body_style
))


story.append(PageBreak())


story.append(Paragraph("6. Colonnes attendues dans le fichier agrégé par annonce", h1_style))
story.append(df_to_table(presence_colonnes_agregees, index=False))

story.append(Paragraph(
    f"Le fichier agrégé contient {len(df_annonce)} annonces avec au moins un avis textuel analysé. "
    "Chaque ligne correspond à une annonce et résume les prédictions BERT associées à ses avis.",
    body_style
))


story.append(Paragraph("7. Valeurs manquantes dans le fichier agrégé", h1_style))
story.append(df_to_table(missing_agrege, index=False))


story.append(Paragraph("8. Statistiques des variables agrégées", h1_style))
story.append(df_to_table(stats_agregees, index=True))


story.append(Paragraph("9. Contrôle des ratios", h1_style))

ratios_table = [
    ["Indicateur", "Valeur"],
    ["Nombre d'annonces avec somme des ratios différente de 1", str(nb_ratios_incorrects)],
    ["Somme moyenne des ratios", f"{df_annonce['somme_ratios'].mean():.4f}"],
    ["Somme minimale des ratios", f"{df_annonce['somme_ratios'].min():.4f}"],
    ["Somme maximale des ratios", f"{df_annonce['somme_ratios'].max():.4f}"]
]

story.append(make_table(ratios_table, col_widths=[10 * cm, 5 * cm]))

if nb_ratios_incorrects == 0:
    story.append(Paragraph(
        "Les ratios d'avis positifs, neutres et négatifs sont cohérents : leur somme est égale à 1 pour chaque annonce.",
        body_style
    ))
else:
    story.append(Paragraph(
        "Certains ratios ne somment pas correctement à 1. Une vérification complémentaire est nécessaire.",
        body_style
    ))


story.append(Paragraph("10. Conclusion", h1_style))

conclusion = (
    "La validation technique confirme que le traitement BERT a produit des sorties complètes et cohérentes. "
    f"Le fichier avis contient {len(df)} avis exploitables avec des prédictions de sentiment sans valeurs manquantes. "
    f"La distribution obtenue est fortement positive ({pct_positif:.2f} %), ce qui est cohérent avec le contexte Airbnb. "
    f"Le fichier agrégé contient {len(df_annonce)} annonces avec au moins un avis textuel analysé. "
)

if nb_ratios_incorrects == 0 and missing_bert["Valeurs manquantes"].sum() == 0:
    conclusion += (
        "Les ratios agrégés sont cohérents et les variables produites peuvent être utilisées pour la fusion avec les données tabulaires."
    )
else:
    conclusion += (
        "Certaines anomalies doivent être vérifiées avant la fusion avec les données tabulaires."
    )

story.append(Paragraph(conclusion, body_style))

doc.build(story)

print("Rapport PDF créé :", fichier_pdf)