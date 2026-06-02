from pathlib import Path
from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent

fichier_reviews_clean = BASE_DIR / "reviews_filtrees_clean.csv"
fichier_reviews_sentiment = BASE_DIR / "reviews_avec_sentiment_bert_COMPLET.csv"
fichier_sentiment_annonce = BASE_DIR / "sentiment_bert_par_annonce_COMPLET.csv"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
CHUNKSIZE = 10000
BATCH_SIZE = 64 if torch.cuda.is_available() else 8
DEVICE = 0 if torch.cuda.is_available() else -1

print("Fichier reviews :", fichier_reviews_clean)
print("Existe :", fichier_reviews_clean.exists())
print("GPU disponible :", torch.cuda.is_available())
print("GPU utilisé :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun")
print("Device :", DEVICE)
print("Batch size :", BATCH_SIZE)

if not fichier_reviews_clean.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_reviews_clean}")

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE
)

def compter_lignes_csv(chemin):
    if not chemin.exists():
        return 0

    with open(chemin, "r", encoding="utf-8-sig", errors="ignore") as f:
        nb_lignes = sum(1 for _ in f)

    return max(nb_lignes - 1, 0)

def extraire_nombre_etoiles(label):
    return int(str(label).split()[0])

def score_depuis_etoiles(nb_etoiles):
    return (nb_etoiles - 3) / 2

def classer_sentiment_depuis_etoiles(nb_etoiles):
    if nb_etoiles <= 2:
        return "negatif"
    elif nb_etoiles == 3:
        return "neutre"
    else:
        return "positif"

nb_deja_traites = compter_lignes_csv(fichier_reviews_sentiment)

print("Nombre d'avis déjà traités :", nb_deja_traites)

premier_morceau = not fichier_reviews_sentiment.exists()
nb_total_lus = 0
nb_total_traites_session = 0

for chunk in tqdm(
    pd.read_csv(
        fichier_reviews_clean,
        dtype=str,
        chunksize=CHUNKSIZE,
        encoding="utf-8-sig",
        on_bad_lines="skip"
    ),
    desc="Analyse BERT complète"
):
    chunk.columns = chunk.columns.str.strip()

    if "listing_id" not in chunk.columns:
        raise ValueError("La colonne listing_id est absente du fichier.")

    if "comments_clean" not in chunk.columns:
        if "comments" in chunk.columns:
            chunk["comments_clean"] = chunk["comments"]
        else:
            raise ValueError("La colonne comments_clean est absente du fichier.")

    debut_chunk = nb_total_lus
    fin_chunk = nb_total_lus + len(chunk)
    nb_total_lus = fin_chunk

    if fin_chunk <= nb_deja_traites:
        continue

    if debut_chunk < nb_deja_traites < fin_chunk:
        position_depart = nb_deja_traites - debut_chunk
        chunk = chunk.iloc[position_depart:].copy()

    chunk["listing_id"] = (
        chunk["listing_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    chunk["comments_clean"] = (
        chunk["comments_clean"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    chunk = chunk[chunk["comments_clean"] != ""].copy()

    textes = chunk["comments_clean"].tolist()

    predictions = []

    for i in range(0, len(textes), BATCH_SIZE):
        batch_textes = textes[i:i + BATCH_SIZE]

        preds = sentiment_pipeline(
            batch_textes,
            truncation=True,
            max_length=512
        )

        predictions.extend(preds)

    labels_raw = [p["label"] for p in predictions]
    confidences = [p["score"] for p in predictions]
    etoiles = [extraire_nombre_etoiles(label) for label in labels_raw]

    chunk["bert_label_raw"] = labels_raw
    chunk["bert_confidence"] = confidences
    chunk["bert_stars"] = etoiles
    chunk["sentiment_score"] = chunk["bert_stars"].apply(score_depuis_etoiles)
    chunk["sentiment_label"] = chunk["bert_stars"].apply(classer_sentiment_depuis_etoiles)

    chunk.to_csv(
        fichier_reviews_sentiment,
        mode="w" if premier_morceau else "a",
        header=premier_morceau,
        index=False,
        encoding="utf-8-sig"
    )

    premier_morceau = False
    nb_total_traites_session += len(chunk)

    print("Avis traités pendant cette session :", nb_total_traites_session)
    print("Total approximatif traité :", nb_deja_traites + nb_total_traites_session)

print("Analyse BERT terminée.")
print("Fichier avis avec sentiment :", fichier_reviews_sentiment)

stats = defaultdict(lambda: {
    "nb": 0,
    "sum_stars": 0.0,
    "sum_conf": 0.0,
    "sum_score": 0.0,
    "sum_score_square": 0.0,
    "min_score": None,
    "max_score": None,
    "nb_pos": 0,
    "nb_neu": 0,
    "nb_neg": 0
})

for chunk in tqdm(
    pd.read_csv(
        fichier_reviews_sentiment,
        dtype=str,
        chunksize=100000,
        encoding="utf-8-sig",
        usecols=[
            "listing_id",
            "bert_confidence",
            "bert_stars",
            "sentiment_score",
            "sentiment_label"
        ],
        on_bad_lines="skip"
    ),
    desc="Agrégation par annonce"
):
    chunk["listing_id"] = (
        chunk["listing_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    chunk["bert_confidence"] = pd.to_numeric(chunk["bert_confidence"], errors="coerce")
    chunk["bert_stars"] = pd.to_numeric(chunk["bert_stars"], errors="coerce")
    chunk["sentiment_score"] = pd.to_numeric(chunk["sentiment_score"], errors="coerce")

    for row in chunk.itertuples(index=False):
        listing_id = str(row.listing_id)
        confidence = row.bert_confidence
        stars = row.bert_stars
        score = row.sentiment_score
        label = row.sentiment_label

        if pd.isna(listing_id) or pd.isna(score):
            continue

        s = stats[listing_id]

        s["nb"] += 1
        s["sum_stars"] += 0 if pd.isna(stars) else stars
        s["sum_conf"] += 0 if pd.isna(confidence) else confidence
        s["sum_score"] += score
        s["sum_score_square"] += score ** 2

        if s["min_score"] is None or score < s["min_score"]:
            s["min_score"] = score

        if s["max_score"] is None or score > s["max_score"]:
            s["max_score"] = score

        if label == "positif":
            s["nb_pos"] += 1
        elif label == "neutre":
            s["nb_neu"] += 1
        elif label == "negatif":
            s["nb_neg"] += 1

lignes = []

for listing_id, s in stats.items():
    nb = s["nb"]

    if nb == 0:
        continue

    mean_score = s["sum_score"] / nb
    variance = (s["sum_score_square"] / nb) - (mean_score ** 2)

    if variance < 0:
        variance = 0

    lignes.append({
        "id_clean": listing_id,
        "nb_avis_textuels_bert": nb,
        "bert_stars_moyen": s["sum_stars"] / nb,
        "bert_confiance_moyenne": s["sum_conf"] / nb,
        "sentiment_bert_moyen": mean_score,
        "sentiment_bert_min": s["min_score"],
        "sentiment_bert_max": s["max_score"],
        "sentiment_bert_ecart_type": variance ** 0.5,
        "nb_avis_bert_positifs": s["nb_pos"],
        "nb_avis_bert_neutres": s["nb_neu"],
        "nb_avis_bert_negatifs": s["nb_neg"],
        "ratio_avis_bert_positifs": s["nb_pos"] / nb,
        "ratio_avis_bert_neutres": s["nb_neu"] / nb,
        "ratio_avis_bert_negatifs": s["nb_neg"] / nb
    })

df_sentiment_annonce = pd.DataFrame(lignes)

df_sentiment_annonce.to_csv(
    fichier_sentiment_annonce,
    index=False,
    encoding="utf-8-sig"
)

print("Agrégation terminée.")
print("Nombre d'annonces avec avis analysés :", len(df_sentiment_annonce))
print("Fichier agrégé :", fichier_sentiment_annonce)