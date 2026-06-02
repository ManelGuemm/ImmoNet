import os
import re
import gc
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/workspace/airbnb_text_e5_lora_stacking")

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs_text_e5_lora"
MODEL_DIR = PROJECT_DIR / "saved_lora_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "Donnees_Airbnb_Finales_Textuelle.xlsx"

MODEL_NAME = "intfloat/multilingual-e5-large"

URL_COL = "listing_url"
ID_COL = "listing_id_clean"
TARGET_COL = "price"

RANDOM_STATE = 42

# Split global
TEST_SIZE = 0.20

# KFold sur train uniquement
N_FOLDS = 5

# Training
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRADIENT_CLIP = 1.0

# PCA
PCA_COMPONENTS = 100

# GPU
USE_AMP = True
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(RANDOM_STATE)

# 4. FONCTIONS UTILES

def clean_text(x):
    if pd.isna(x):
        return ""

    x = str(x)

    # Retirer balises HTML
    x = re.sub(r"<[^>]+>", " ", x)

    # Retirer URLs
    x = re.sub(r"http\S+|www\.\S+", " ", x)

    # Normaliser espaces
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()

    return x


def parse_price(x):
    if pd.isna(x):
        return np.nan

    x = str(x)
    x = x.replace(",", "")
    x = re.sub(r"[^0-9.]", "", x)

    if x == "":
        return np.nan

    try:
        return float(x)
    except Exception:
        return np.nan


def extract_listing_id_from_url(url):
    if pd.isna(url):
        return np.nan

    url = str(url)
    match = re.search(r"/rooms/(\d+)", url)

    if match:
        return match.group(1)

    return np.nan


def build_listing_text(name, description):
    name = clean_text(name)
    description = clean_text(description)

    # Pour E5, on ajoute un préfixe.
    # Ici on traite chaque annonce comme un texte descriptif à encoder.
    text = f"query: Title: {name}. Description: {description}"

    return text.strip()


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


def make_regression_bins(y, n_bins=10):
    """
    Crée des bins pour stratifier un problème de régression.
    On utilise des quantiles de log_price.
    """
    y_series = pd.Series(y)

    try:
        bins = pd.qcut(y_series, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(y_series, bins=n_bins, labels=False)

    return bins.astype(int).values


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# 5. AFFICHAGE DEVICE
# ============================================================

print("=" * 80)
print("DEVICE :", DEVICE)

if DEVICE == "cuda":
    print("GPU :", torch.cuda.get_device_name(0))
    print(
        "VRAM totale :",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("=" * 80)


# ============================================================
# 6. CHARGEMENT DU FICHIER
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFichier introuvable : {INPUT_FILE}\n\n"
        f"Place ton fichier ici :\n"
        f"{DATA_DIR}\n\n"
        f"Nom attendu :\n"
        f"Donnees_Airbnb_Finales_Textuelle.xlsx\n"
    )

print(f"Lecture du fichier : {INPUT_FILE}")

if INPUT_FILE.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(INPUT_FILE)
elif INPUT_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(INPUT_FILE)
else:
    raise ValueError("Format non supporté. Utilise .xlsx, .xls ou .csv")

print("Shape initiale :", df.shape)
print("Colonnes trouvées :", list(df.columns))


# ============================================================
# 7. CONTRÔLE DES COLONNES
# ============================================================

required_cols = [URL_COL, "name", "description", TARGET_COL]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(
        f"Colonnes manquantes : {missing_cols}\n"
        f"Colonnes disponibles : {list(df.columns)}"
    )

cols_to_keep = []

# id est gardé seulement pour audit pas pour la fusion
if "id" in df.columns:
    cols_to_keep.append("id")

cols_to_keep += [URL_COL, "name", "description", TARGET_COL]

df = df[cols_to_keep].copy()


# ============================================================
# 8. CRÉATION DE listing_id_clean
# ============================================================

df[ID_COL] = df[URL_COL].apply(extract_listing_id_from_url)

n_missing_key = df[ID_COL].isna().sum()
print(f"Nombre de listing_id_clean manquants : {n_missing_key}")

if n_missing_key > 0:
    df = df.dropna(subset=[ID_COL]).copy()

df[ID_COL] = df[ID_COL].astype(str)

n_rows_before_dup = len(df)
n_unique_key = df[ID_COL].nunique()
n_duplicates = n_rows_before_dup - n_unique_key

print("Nombre de lignes avant suppression doublons :", n_rows_before_dup)
print("Nombre de listing_id_clean uniques          :", n_unique_key)
print("Nombre de doublons sur listing_id_clean     :", n_duplicates)

if n_duplicates > 0:
    df = df.drop_duplicates(subset=[ID_COL], keep="first").copy()

print("Nombre de lignes après clé propre :", len(df))
print("listing_id_clean est unique :", df[ID_COL].is_unique)


# ============================================================
# 9. NETTOYAGE TEXTE + CIBLE
# ============================================================

df["name"] = df["name"].fillna("").apply(clean_text)
df["description"] = df["description"].fillna("").apply(clean_text)

df["price_clean"] = df[TARGET_COL].apply(parse_price)

before_price = len(df)
df = df.dropna(subset=["price_clean"]).copy()
df = df[df["price_clean"] > 0].copy()
after_price = len(df)

print(f"Lignes supprimées à cause du prix invalide : {before_price - after_price}")

df["log_price"] = np.log1p(df["price_clean"])

df["listing_text"] = [
    build_listing_text(n, d)
    for n, d in zip(df["name"].values, df["description"].values)
]

df["name_n_words"] = df["name"].apply(lambda x: len(str(x).split()))
df["description_n_words"] = df["description"].apply(lambda x: len(str(x).split()))
df["description_is_missing"] = (df["description"].str.len() == 0).astype(int)

df = df.reset_index(drop=True)

print("\nShape après nettoyage :", df.shape)
print(df[[ID_COL, URL_COL, "name", "description", "price_clean", "log_price"]].head())


# 10. SPLIT GLOBAL TRAIN / TEST
# On stratifie approximativement avec des bins de log_price.
# Ça évite d'avoir un train/test avec une distribution de prix trop différente.
split_bins = make_regression_bins(df["log_price"].values, n_bins=10)

train_idx_global, test_idx_global = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=split_bins
)

df["split"] = "train"
df.loc[test_idx_global, "split"] = "test"

train_df = df.iloc[train_idx_global].copy().reset_index(drop=True)
test_df = df.iloc[test_idx_global].copy().reset_index(drop=True)

print("\nSplit global :")
print("Train :", train_df.shape)
print("Test  :", test_df.shape)

# Sauvegarde du split pour réutiliser exactement le même split avec le tabulaire
split_df = df[[ID_COL, "split"]].copy()
split_path = OUTPUT_DIR / "split_listing_ids.csv"
split_df.to_csv(split_path, index=False)

print(f"Split sauvegardé : {split_path}")

# 11. DATASET PYTORCH

class AirbnbTextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = list(texts)
        self.targets = np.array(targets, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.targets[idx], dtype=torch.float32)
        }

        return item

# 12. MODÈLE E5 + LoRA + TÊTE DE RÉGRESSION

class E5LoraRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        base_model = AutoModel.from_pretrained(model_name)
        hidden_size = base_model.config.hidden_size

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        self.encoder = get_peft_model(base_model, lora_config)

        self.regressor = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        emb = mean_pooling(outputs.last_hidden_state, attention_mask)
        pred = self.regressor(emb).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(pred, labels)

        return {
            "loss": loss,
            "pred": pred,
            "embedding": emb
        }


def get_trainable_state_dict(model):
    trainable_state = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()

    return trainable_state

# 13. FONCTIONS TRAIN / PREDICT

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, desc):
    model.train()
    total_loss = 0.0

    progress = tqdm(train_loader, desc=desc)

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        if USE_AMP and DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()

        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def predict_with_embeddings(model, data_loader, desc):
    model.eval()

    preds = []
    embeddings = []
    losses = []

    progress = tqdm(data_loader, desc=desc)

    for batch in progress:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        if USE_AMP and DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        preds.append(out["pred"].detach().cpu().numpy())
        embeddings.append(out["embedding"].detach().cpu().numpy())

        if out["loss"] is not None:
            losses.append(out["loss"].item())

    preds = np.concatenate(preds, axis=0)
    embeddings = np.concatenate(embeddings, axis=0)

    return preds, embeddings, np.mean(losses) if losses else None


def create_loader(texts, targets, tokenizer, shuffle):
    dataset = AirbnbTextDataset(
        texts=texts,
        targets=targets,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return loader


def build_optimizer_scheduler(model, train_loader, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


# 14. TOKENIZER

print("\nChargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 15. KFold SUR TRAIN UNIQUEMENT : OOF TRAIN

train_texts = train_df["listing_text"].values
train_y = train_df["log_price"].values.astype(np.float32)

test_texts = test_df["listing_text"].values
test_y = test_df["log_price"].values.astype(np.float32)

n_train = len(train_df)
n_test = len(test_df)

EMBED_DIM = 1024

oof_pred_train = np.zeros(n_train, dtype=np.float32)
oof_embeddings_train = np.zeros((n_train, EMBED_DIM), dtype=np.float32)

fold_metrics = []

train_bins = make_regression_bins(train_y, n_bins=10)

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (fold_train_idx, fold_valid_idx) in enumerate(
    skf.split(train_df, train_bins),
    start=1
):
    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_FOLDS} SUR TRAIN GLOBAL")
    print("Fold train size :", len(fold_train_idx))
    print("Fold valid size :", len(fold_valid_idx))
    print("=" * 80)

    fold_train_loader = create_loader(
        texts=train_texts[fold_train_idx],
        targets=train_y[fold_train_idx],
        tokenizer=tokenizer,
        shuffle=True
    )

    fold_valid_loader = create_loader(
        texts=train_texts[fold_valid_idx],
        targets=train_y[fold_valid_idx],
        tokenizer=tokenizer,
        shuffle=False
    )

    model = E5LoraRegressor(MODEL_NAME)
    model.to(DEVICE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Paramètres entraînables : {trainable_params:,}")
    print(f"Paramètres totaux       : {total_params:,}")
    print(f"% entraînable           : {100 * trainable_params / total_params:.4f}%")

    optimizer, scheduler = build_optimizer_scheduler(
        model=model,
        train_loader=fold_train_loader,
        epochs=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    best_rmse = float("inf")
    best_state_path = MODEL_DIR / f"fold_{fold}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=fold_train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            desc=f"Fold {fold} | Epoch {epoch} | train"
        )

        valid_preds_tmp, _, valid_loss = predict_with_embeddings(
            model=model,
            data_loader=fold_valid_loader,
            desc=f"Fold {fold} | Epoch {epoch} | valid"
        )

        valid_rmse = rmse(train_y[fold_valid_idx], valid_preds_tmp)
        valid_mae = mean_absolute_error(train_y[fold_valid_idx], valid_preds_tmp)

        print(
            f"Fold {fold} | Epoch {epoch} | "
            f"Train loss: {train_loss:.5f} | "
            f"Valid loss: {valid_loss:.5f} | "
            f"Valid RMSE: {valid_rmse:.5f} | "
            f"Valid MAE: {valid_mae:.5f}"
        )

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            torch.save(get_trainable_state_dict(model), best_state_path)
            print(f"Best fold model sauvegardé : {best_state_path}")

    model.load_state_dict(torch.load(best_state_path, map_location=DEVICE), strict=False)

    valid_preds, valid_embeddings, _ = predict_with_embeddings(
        model=model,
        data_loader=fold_valid_loader,
        desc=f"Fold {fold} | best valid predict"
    )

    oof_pred_train[fold_valid_idx] = valid_preds.astype(np.float32)
    oof_embeddings_train[fold_valid_idx] = valid_embeddings.astype(np.float32)

    fold_rmse = rmse(train_y[fold_valid_idx], valid_preds)
    fold_mae = mean_absolute_error(train_y[fold_valid_idx], valid_preds)
    fold_r2 = r2_score(train_y[fold_valid_idx], valid_preds)

    fold_metrics.append({
        "fold": fold,
        "rmse_log": fold_rmse,
        "mae_log": fold_mae,
        "r2_log": fold_r2,
        "n_train": len(fold_train_idx),
        "n_valid": len(fold_valid_idx)
    })

    print(f"Fold {fold} terminé | RMSE log: {fold_rmse:.5f} | MAE log: {fold_mae:.5f}")

    del model
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 16. MÉTRIQUES OOF TRAIN

oof_rmse = rmse(train_y, oof_pred_train)
oof_mae = mean_absolute_error(train_y, oof_pred_train)
oof_r2 = r2_score(train_y, oof_pred_train)

print("\n" + "=" * 80)
print("MÉTRIQUES OOF SUR TRAIN GLOBAL")
print(f"OOF RMSE log_price : {oof_rmse:.5f}")
print(f"OOF MAE  log_price : {oof_mae:.5f}")
print(f"OOF R2   log_price : {oof_r2:.5f}")
print("=" * 80)

metrics_df = pd.DataFrame(fold_metrics)

metrics_df.loc[len(metrics_df)] = {
    "fold": "OOF_TRAIN_GLOBAL",
    "rmse_log": oof_rmse,
    "mae_log": oof_mae,
    "r2_log": oof_r2,
    "n_train": len(train_df),
    "n_valid": len(train_df)
}

metrics_path = OUTPUT_DIR / "oof_train_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"Métriques OOF sauvegardées : {metrics_path}")


# ============================================================
# 17. ENTRAÎNER MODÈLE TEXTE FINAL SUR TOUT LE TRAIN

print("\n" + "=" * 80)
print("ENTRAÎNEMENT DU MODÈLE TEXTE FINAL SUR TOUT LE TRAIN GLOBAL")
print("=" * 80)

full_train_loader = create_loader(
    texts=train_texts,
    targets=train_y,
    tokenizer=tokenizer,
    shuffle=True
)

test_loader = create_loader(
    texts=test_texts,
    targets=test_y,
    tokenizer=tokenizer,
    shuffle=False
)

final_model = E5LoraRegressor(MODEL_NAME)
final_model.to(DEVICE)

optimizer, scheduler = build_optimizer_scheduler(
    model=final_model,
    train_loader=full_train_loader,
    epochs=EPOCHS
)

scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(
        model=final_model,
        train_loader=full_train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        desc=f"Final model | Epoch {epoch} | train"
    )

    print(f"Final model | Epoch {epoch} | Train loss: {train_loss:.5f}")

final_model_path = MODEL_DIR / "final_text_model_train_global.pt"
torch.save(get_trainable_state_dict(final_model), final_model_path)
print(f"Modèle texte final sauvegardé : {final_model_path}")


# 18. PRÉDICTION TEST AVEC MODÈLE TEXTE FINAL

test_pred, test_embeddings, test_loss = predict_with_embeddings(
    model=final_model,
    data_loader=test_loader,
    desc="Final model | test predict"
)

test_rmse = rmse(test_y, test_pred)
test_mae = mean_absolute_error(test_y, test_pred)
test_r2 = r2_score(test_y, test_pred)

print("\n" + "=" * 80)
print("PERFORMANCE TEXTE-ONLY SUR TEST GLOBAL")
print(f"Test RMSE log_price : {test_rmse:.5f}")
print(f"Test MAE  log_price : {test_mae:.5f}")
print(f"Test R2   log_price : {test_r2:.5f}")
print("=" * 80)

test_metrics_path = OUTPUT_DIR / "text_only_test_metrics.csv"

pd.DataFrame([{
    "rmse_log": test_rmse,
    "mae_log": test_mae,
    "r2_log": test_r2,
    "n_test": len(test_df)
}]).to_csv(test_metrics_path, index=False)

print(f"Métriques test sauvegardées : {test_metrics_path}")

del final_model
gc.collect()

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# 19. PCA FIT SUR TRAIN UNIQUEMENT, TRANSFORM TRAIN ET TEST

print("\nApplication StandardScaler + PCA...")

scaler_emb = StandardScaler()
train_emb_scaled = scaler_emb.fit_transform(oof_embeddings_train)
test_emb_scaled = scaler_emb.transform(test_embeddings)

pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)

train_emb_pca = pca.fit_transform(train_emb_scaled)
test_emb_pca = pca.transform(test_emb_scaled)

explained_variance = float(pca.explained_variance_ratio_.sum())

print("Variance expliquée cumulée PCA sur train :", round(explained_variance, 4))

joblib.dump(scaler_emb, OUTPUT_DIR / "text_embedding_scaler_train_only.joblib")
joblib.dump(pca, OUTPUT_DIR / "text_embedding_pca_train_only.joblib")

print("Scaler et PCA sauvegardés.")

# 20. CONSTRUCTION DES FEATURES TRAIN / TEST

def build_features_file(base_df, text_pred, emb_pca, split_name):
    features_df = pd.DataFrame()

    features_df[ID_COL] = base_df[ID_COL].values
    features_df["split"] = split_name

    # Prédiction texte-only
    # Pour train : text_pred_oof
    # Pour test  : prédiction du modèle texte final entraîné sur train seulement
    if split_name == "train":
        features_df["text_pred_oof"] = text_pred.astype(np.float32)
    else:
        features_df["text_pred_test"] = text_pred.astype(np.float32)

    features_df["name_n_words"] = base_df["name_n_words"].values
    features_df["description_n_words"] = base_df["description_n_words"].values
    features_df["description_is_missing"] = base_df["description_is_missing"].values

    for i in range(PCA_COMPONENTS):
        features_df[f"txt_e5_{i:03d}"] = emb_pca[:, i].astype(np.float32)

    assert features_df[ID_COL].is_unique, f"Erreur : {ID_COL} non unique dans {split_name}"
    assert features_df[ID_COL].isna().sum() == 0, f"Erreur : {ID_COL} manquant dans {split_name}"

    return features_df


train_features_df = build_features_file(
    base_df=train_df,
    text_pred=oof_pred_train,
    emb_pca=train_emb_pca,
    split_name="train"
)

test_features_df = build_features_file(
    base_df=test_df,
    text_pred=test_pred,
    emb_pca=test_emb_pca,
    split_name="test"
)

# Pour faciliter la fusion plus tard, on crée aussi une colonne au même nom.
# Dans train, text_pred_final = text_pred_oof.
# Dans test, text_pred_final = text_pred_test.
train_features_df["text_pred_final"] = train_features_df["text_pred_oof"]
test_features_df["text_pred_final"] = test_features_df["text_pred_test"]


# 21. AUDIT TRAIN / TEST

def build_audit_file(base_df, text_pred, split_name):
    audit_cols = [ID_COL, URL_COL, "name", "description", "price_clean", "log_price"]

    if "id" in base_df.columns:
        audit_cols = ["id"] + audit_cols

    audit_df = base_df[audit_cols].copy()
    audit_df["split"] = split_name
    audit_df["text_pred_log"] = text_pred.astype(np.float32)
    audit_df["text_pred_price"] = np.expm1(text_pred)

    return audit_df


train_audit_df = build_audit_file(
    base_df=train_df,
    text_pred=oof_pred_train,
    split_name="train"
)

test_audit_df = build_audit_file(
    base_df=test_df,
    text_pred=test_pred,
    split_name="test"
)


# ============================================================
# 22. SAUVEGARDES

train_parquet_path = OUTPUT_DIR / "airbnb_text_features_train_e5_lora.parquet"
test_parquet_path = OUTPUT_DIR / "airbnb_text_features_test_e5_lora.parquet"

train_csv_path = OUTPUT_DIR / "airbnb_text_features_train_e5_lora.csv"
test_csv_path = OUTPUT_DIR / "airbnb_text_features_test_e5_lora.csv"

train_xlsx_path = OUTPUT_DIR / "airbnb_text_features_train_e5_lora.xlsx"
test_xlsx_path = OUTPUT_DIR / "airbnb_text_features_test_e5_lora.xlsx"

train_audit_path = OUTPUT_DIR / "airbnb_text_audit_train_e5_lora.xlsx"
test_audit_path = OUTPUT_DIR / "airbnb_text_audit_test_e5_lora.xlsx"

# Formats principaux
train_features_df.to_parquet(train_parquet_path, index=False)
test_features_df.to_parquet(test_parquet_path, index=False)

train_features_df.to_csv(train_csv_path, index=False)
test_features_df.to_csv(test_csv_path, index=False)

train_features_df.to_excel(train_xlsx_path, index=False)
test_features_df.to_excel(test_xlsx_path, index=False)

train_audit_df.to_excel(train_audit_path, index=False)
test_audit_df.to_excel(test_audit_path, index=False)

all_features_df = pd.concat([train_features_df, test_features_df], axis=0, ignore_index=True)
all_features_path = OUTPUT_DIR / "airbnb_text_features_all_e5_lora.parquet"
all_features_df.to_parquet(all_features_path, index=False)

print("\n" + "=" * 80)
print("FICHIERS SAUVEGARDÉS")
print(f"Split IDs             : {split_path}")
print(f"Train features parquet: {train_parquet_path}")
print(f"Test features parquet : {test_parquet_path}")
print(f"Train features csv    : {train_csv_path}")
print(f"Test features csv     : {test_csv_path}")
print(f"Train features xlsx   : {train_xlsx_path}")
print(f"Test features xlsx    : {test_xlsx_path}")
print(f"All features parquet  : {all_features_path}")
print(f"Train audit xlsx      : {train_audit_path}")
print(f"Test audit xlsx       : {test_audit_path}")
print(f"OOF train metrics     : {metrics_path}")
print(f"Text test metrics     : {test_metrics_path}")
print("=" * 80)

print("\nAperçu train features :")
print(train_features_df.head())

print("\nAperçu test features :")
print(test_features_df.head())

print("\nNombre train :", len(train_features_df))
print("Nombre test  :", len(test_features_df))

print("\nColonnes finales train :")
print(train_features_df.columns.tolist())

print("\nTerminé.")