# -*- coding: utf-8 -*-
"""
Entraînement direct ConvNeXt image seule -> log_price
Objectif : créer une branche visuelle indépendante pour une future late fusion.
Le modèle ne fait PAS : EfficientNet embeddings -> CatBoost.
Il apprend directement : image -> ConvNeXt -> log_price.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# Lecture fichiers

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=str)

    return pd.read_csv(path, dtype=str, low_memory=False)



def find_id_column(df: pd.DataFrame) -> str:
    candidates = ["listing_id_clean", "id_clean", "listing_id", "id"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Aucune colonne ID trouvée. Colonnes acceptées : "
        "listing_id_clean, id_clean, listing_id, id."
    )



def normalize_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)



def clean_numeric_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9\.\-]", "", regex=True),
        errors="coerce",
    )

# Images : alignement ID -> chemin


def build_image_index(image_dir: str | Path) -> Dict[str, Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable : {image_dir}")

    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]

    image_index: Dict[str, Path] = {}
    duplicate_ids = []
    for p in image_paths:
        img_id = str(p.stem).strip()
        if img_id in image_index:
            duplicate_ids.append(img_id)
            continue
        image_index[img_id] = p

    if len(image_index) == 0:
        raise RuntimeError(f"Aucune image exploitable trouvée dans : {image_dir}")

    return image_index



def align_dataset_with_images(
    df: pd.DataFrame,
    image_dir: str | Path,
    output_dir: Path,
    id_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index(image_dir)

    df = df.copy()
    df[id_col] = normalize_id(df[id_col])

    df["image_path"] = df[id_col].map(lambda x: str(image_index.get(str(x), "")))
    missing_mask = df["image_path"].eq("")

    missing_df = df.loc[missing_mask, [id_col]].copy()
    missing_df = missing_df.rename(columns={id_col: "listing_id_clean"})
    missing_df.to_csv(output_dir / "images_absentes.csv", index=False, encoding="utf-8-sig")

    aligned = df.loc[~missing_mask].copy()

    image_ids = set(image_index.keys())
    data_ids = set(df[id_col].astype(str))
    extra_ids = sorted(image_ids - data_ids)
    extra_df = pd.DataFrame({"listing_id_clean": extra_ids})
    extra_df.to_csv(output_dir / "images_en_trop_dans_dossier.csv", index=False, encoding="utf-8-sig")

    return aligned, missing_df, extra_df

# Split

def price_segment_series(price: pd.Series) -> pd.Series:
    price_num = pd.to_numeric(price, errors="coerce")
    return pd.cut(
        price_num,
        bins=[0, 100, 200, 400, 800, np.inf],
        labels=["<100", "100-200", "200-400", "400-800", ">800"],
        include_lowest=True,
    ).astype(str)



def summarize_split(df: pd.DataFrame, split_col: str, price_col: str) -> pd.DataFrame:
    rows = []
    for split_value, g in df.groupby(split_col, dropna=False):
        price = pd.to_numeric(g[price_col], errors="coerce")
        rows.append({
            "split": split_value,
            "n": len(g),
            "price_mean": price.mean(),
            "price_median": price.median(),
            "price_min": price.min(),
            "price_max": price.max(),
            "log_price_mean": pd.to_numeric(g["log_price"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)



def load_or_create_split(
    df: pd.DataFrame,
    id_col: str,
    price_col: str,
    split_file: Optional[str],
    split_col: str,
    test_size: float,
    random_state: int,
    output_dir: Path,
) -> pd.DataFrame:
    df = df.copy()
    df["split"] = None

    if split_file is not None and Path(split_file).exists():
        split_df = read_table(split_file)
        split_id_col = find_id_column(split_df)
        split_df[split_id_col] = normalize_id(split_df[split_id_col])

        available_split_cols = [c for c in [split_col, "split", "set", "dataset", "subset"] if c in split_df.columns]

        if len(available_split_cols) > 0:
            real_split_col = available_split_cols[0]
            tmp = split_df[[split_id_col, real_split_col]].copy()
            tmp = tmp.rename(columns={split_id_col: id_col, real_split_col: "split_source"})
            tmp[id_col] = normalize_id(tmp[id_col])
            tmp["split_source"] = tmp["split_source"].astype(str).str.lower().str.strip()

            df = df.merge(tmp, on=id_col, how="inner")

            train_values = {"train", "train_dev", "dev", "development", "apprentissage", "oof"}
            test_values = {"test", "test_final", "final", "holdout"}

            df["split"] = np.where(
                df["split_source"].isin(test_values),
                "test_final",
                np.where(df["split_source"].isin(train_values), "train_dev", None),
            )

            if df["split"].isna().any():
                unknown = sorted(df.loc[df["split"].isna(), "split_source"].dropna().unique().tolist())
                raise ValueError(
                    "Certaines valeurs du split ne sont pas reconnues : "
                    f"{unknown}. Utilise par exemple train_dev / test_final."
                )

            df = df.drop(columns=["split_source"])
            print(f"Split chargé depuis {split_file}.")

        else:
            # Le fichier contient seulement une liste d'IDs : on filtre puis on crée le split.
            expected_ids = set(split_df[split_id_col].astype(str))
            before = len(df)
            df = df[df[id_col].astype(str).isin(expected_ids)].copy()
            print(
                f"Le fichier {split_file} ne contient pas de colonne split. "
                f"Il est utilisé comme filtre d'IDs : {before} -> {len(df)} lignes."
            )

    # Si aucun split explicite n'a été chargé on crée un split.
    if df["split"].isna().all():
        segments = price_segment_series(df[price_col])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, segments))
        df.iloc[train_idx, df.columns.get_loc("split")] = "train_dev"
        df.iloc[test_idx, df.columns.get_loc("split")] = "test_final"
        print("Split train_dev/test_final créé automatiquement par stratification sur les segments de prix.")

    split_report = summarize_split(df, "split", price_col)
    split_report.to_csv(output_dir / "02_split_train_test_report.csv", index=False, encoding="utf-8-sig")

    return df


# Poids par segment de prix

def compute_sample_weights(price: np.ndarray) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    weights = np.ones_like(price, dtype=np.float32)

    # Stratégie manual aggressive C05 proche de tes expériences précédentes.
    weights[price < 100] = 1.00
    weights[(price >= 100) & (price < 200)] = 1.00
    weights[(price >= 200) & (price < 400)] = 1.15
    weights[(price >= 400) & (price < 800)] = 2.00
    weights[price >= 800] = 3.50

    # Normalisation pour garder une échelle de loss stable.
    weights = weights / np.nanmean(weights)
    return weights.astype(np.float32)


# Dataset PyTorch

class AirbnbImagePriceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        image_col: str,
        target_col: str,
        price_col: str,
        transform,
        use_weights: bool,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.id_col = id_col
        self.image_col = image_col
        self.target_col = target_col
        self.price_col = price_col
        self.transform = transform
        self.use_weights = use_weights

        price_values = pd.to_numeric(self.df[price_col], errors="coerce").values
        self.weights = compute_sample_weights(price_values) if use_weights else np.ones(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        listing_id = str(row[self.id_col])
        y = np.float32(row[self.target_col])
        price = np.float32(row[self.price_col])
        weight = np.float32(self.weights[idx])

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            failed = False
            error = ""
        except Exception as e:
            # On garde la ligne avec une image noire pour ne pas casser l'alignement.
            # Les cas sont enregistrés dans images_non_lisibles_pendant_training.csv.
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            image = self.transform(image)
            failed = True
            error = str(e)

        return {
            "image": image,
            "y": torch.tensor(y, dtype=torch.float32),
            "price": torch.tensor(price, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "listing_id": listing_id,
            "image_path": str(image_path),
            "failed": failed,
            "error": error,
        }

# Modèle ConvNeXt


def build_convnext_regressor(model_size: str):
    model_size = model_size.lower()

    if model_size == "tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
    elif model_size == "small":
        weights = ConvNeXt_Small_Weights.DEFAULT
        model = convnext_small(weights=weights)
    elif model_size == "base":
        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights)
    elif model_size == "large":
        weights = ConvNeXt_Large_Weights.DEFAULT
        model = convnext_large(weights=weights)
    else:
        raise ValueError("model_size doit être : tiny, small, base ou large.")

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    transform = weights.transforms()
    return model, transform



def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    # Dans torchvision ConvNeXt : model.features = backbone, model.classifier = tête.
    for p in model.features.parameters():
        p.requires_grad = trainable
    for p in model.classifier.parameters():
        p.requires_grad = True



def make_optimizer(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# Metrics

def inverse_log_price(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float))



def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def metrics_log(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": safe_rmse(y_true, y_pred),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Biais_moyen": float(np.mean(y_pred - y_true)),
        "Sous_estimation_pct": float(np.mean(y_pred < y_true) * 100.0),
        "Surestimation_pct": float(np.mean(y_pred > y_true) * 100.0),
        "Erreur_absolue_P90": float(np.percentile(np.abs(y_pred - y_true), 90)),
        "Erreur_absolue_P95": float(np.percentile(np.abs(y_pred - y_true), 95)),
    }



def metrics_euro(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true = inverse_log_price(y_true_log)
    y_pred = inverse_log_price(y_pred_log)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": safe_rmse(y_true, y_pred),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Biais_moyen": float(np.mean(y_pred - y_true)),
        "Sous_estimation_pct": float(np.mean(y_pred < y_true) * 100.0),
        "Surestimation_pct": float(np.mean(y_pred > y_true) * 100.0),
        "Erreur_absolue_P90": float(np.percentile(np.abs(y_pred - y_true), 90)),
        "Erreur_absolue_P95": float(np.percentile(np.abs(y_pred - y_true), 95)),
    }



def segment_report(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> pd.DataFrame:
    y_true_price = inverse_log_price(y_true_log)
    y_pred_price = inverse_log_price(y_pred_log)

    df = pd.DataFrame({
        "price_true": y_true_price,
        "price_pred": y_pred_price,
    })
    df["error_euro"] = df["price_pred"] - df["price_true"]
    df["abs_error_euro"] = np.abs(df["error_euro"])
    df["segment_price"] = pd.cut(
        df["price_true"],
        bins=[0, 100, 200, 400, 800, np.inf],
        labels=["<100", "100-200", "200-400", "400-800", ">800"],
        include_lowest=True,
    )

    rows = []
    for seg, g in df.groupby("segment_price", observed=True):
        rows.append({
            "segment_price": str(seg),
            "n": int(len(g)),
            "price_mean": float(g["price_true"].mean()),
            "pred_mean": float(g["price_pred"].mean()),
            "MAE": float(g["abs_error_euro"].mean()),
            "RMSE": float(np.sqrt(np.mean(g["error_euro"] ** 2))),
            "Biais_moyen": float(g["error_euro"].mean()),
            "Sous_estimation_pct": float(np.mean(g["price_pred"] < g["price_true"]) * 100.0),
        })
    return pd.DataFrame(rows)


# Entraînement et prédiction

@dataclass
class TrainResult:
    best_epoch: int
    best_val_mae_log: float
    history: pd.DataFrame
    failed_images: List[Dict[str, str]]



def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    output_model_path: Path,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    freeze_backbone_epochs: int,
    patience: int,
    amp: bool,
    desc_prefix: str,
) -> TrainResult:
    criterion = nn.SmoothL1Loss(reduction="none", beta=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_val_mae = math.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []
    failed_images: List[Dict[str, str]] = []

    # Départ : backbone gelé ou non.
    set_backbone_trainable(model, trainable=(freeze_backbone_epochs <= 0))
    optimizer = make_optimizer(model, lr_backbone, lr_head, weight_decay)

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, lr_backbone, lr_head, weight_decay)
            print(f"[{desc_prefix}] Backbone dégelé à l'époque {epoch}.")

        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"{desc_prefix} | epoch {epoch}/{epochs} | train", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True)

            if any(batch["failed"]):
                for listing_id, path, err, failed in zip(batch["listing_id"], batch["image_path"], batch["error"], batch["failed"]):
                    if failed:
                        failed_images.append({"listing_id_clean": listing_id, "image_path": path, "error": err})

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(images).squeeze(1)
                loss_raw = criterion(pred, y)
                loss = (loss_raw * w).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix({"loss": np.mean(train_losses)})

        train_loss = float(np.mean(train_losses)) if len(train_losses) else np.nan

        if val_loader is not None:
            val_pred_df, failed_val = predict_model(model, val_loader, device, amp=amp, desc=f"{desc_prefix} | val")
            failed_images.extend(failed_val)
            val_mae_log = float(mean_absolute_error(val_pred_df["log_price_true"], val_pred_df["log_price_pred"]))
            val_rmse_log = safe_rmse(val_pred_df["log_price_true"].values, val_pred_df["log_price_pred"].values)
        else:
            val_mae_log = train_loss
            val_rmse_log = np.nan

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae_log": val_mae_log,
            "val_rmse_log": val_rmse_log,
            "backbone_trainable": bool(epoch > freeze_backbone_epochs or freeze_backbone_epochs <= 0),
        })

        print(
            f"[{desc_prefix}] epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_MAE_log={val_mae_log:.5f} | val_RMSE_log={val_rmse_log:.5f}"
        )

        if val_mae_log < best_val_mae - 1e-6:
            best_val_mae = val_mae_log
            best_epoch = epoch
            patience_counter = 0
            output_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val_mae_log": best_val_mae,
            }, output_model_path)
        else:
            patience_counter += 1

        if val_loader is not None and patience_counter >= patience:
            print(f"[{desc_prefix}] Early stopping à l'époque {epoch}.")
            break

    history = pd.DataFrame(history_rows)

    # Recharge le meilleur modèle si validation fournie.
    if output_model_path.exists():
        checkpoint = torch.load(output_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    return TrainResult(best_epoch=best_epoch, best_val_mae_log=best_val_mae, history=history, failed_images=failed_images)



def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    output_model_path: Path,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    freeze_backbone_epochs: int,
    amp: bool,
    desc_prefix: str,
) -> pd.DataFrame:
    criterion = nn.SmoothL1Loss(reduction="none", beta=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    set_backbone_trainable(model, trainable=(freeze_backbone_epochs <= 0))
    optimizer = make_optimizer(model, lr_backbone, lr_head, weight_decay)

    history_rows = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, lr_backbone, lr_head, weight_decay)
            print(f"[{desc_prefix}] Backbone dégelé à l'époque {epoch}.")

        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"{desc_prefix} | epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(images).squeeze(1)
                loss_raw = criterion(pred, y)
                loss = (loss_raw * w).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix({"loss": np.mean(losses)})

        train_loss = float(np.mean(losses)) if len(losses) else np.nan
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae_log": np.nan,
            "val_rmse_log": np.nan,
            "backbone_trainable": bool(epoch > freeze_backbone_epochs or freeze_backbone_epochs <= 0),
        })
        print(f"[{desc_prefix}] epoch {epoch:03d} | train_loss={train_loss:.5f}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epochs": epochs}, output_model_path)
    return pd.DataFrame(history_rows)



def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    desc: str,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    model.eval()
    rows = []
    failed_images: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].cpu().numpy()
            price = batch["price"].cpu().numpy()
            ids = batch["listing_id"]
            paths = batch["image_path"]

            if any(batch["failed"]):
                for listing_id, path, err, failed in zip(batch["listing_id"], batch["image_path"], batch["error"], batch["failed"]):
                    if failed:
                        failed_images.append({"listing_id_clean": listing_id, "image_path": path, "error": err})

            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(images).squeeze(1).detach().cpu().numpy()

            for i in range(len(ids)):
                rows.append({
                    "listing_id_clean": ids[i],
                    "image_path": paths[i],
                    "log_price_true": float(y[i]),
                    "log_price_pred": float(pred[i]),
                    "price_true": float(price[i]),
                    "price_pred": float(np.expm1(pred[i])),
                })

    pred_df = pd.DataFrame(rows)
    pred_df["error_log"] = pred_df["log_price_pred"] - pred_df["log_price_true"]
    pred_df["abs_error_log"] = pred_df["error_log"].abs()
    pred_df["error_euro"] = pred_df["price_pred"] - pred_df["price_true"]
    pred_df["abs_error_euro"] = pred_df["error_euro"].abs()
    return pred_df, failed_images

# Figures

def save_figures(output_dir: Path, test_pred: pd.DataFrame, seg_test: pd.DataFrame, history_all: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1 : prix réel vs prix prédit
    plt.figure(figsize=(7, 5))
    plt.scatter(test_pred["price_true"], test_pred["price_pred"], s=6, alpha=0.35)
    max_val = np.nanpercentile(test_pred[["price_true", "price_pred"]].values, 99)
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("ConvNeXt image seule - prix prédit vs prix réel")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_01_pred_vs_true_test.png", dpi=200)
    plt.close()

    # Figure 2 : erreur absolue selon prix réel
    plt.figure(figsize=(7, 5))
    plt.scatter(test_pred["price_true"], test_pred["abs_error_euro"], s=6, alpha=0.35)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Erreur absolue (€)")
    plt.title("ConvNeXt image seule - erreur absolue selon le prix réel")
    plt.xlim(0, np.nanpercentile(test_pred["price_true"], 99))
    plt.ylim(0, np.nanpercentile(test_pred["abs_error_euro"], 99))
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_02_abs_error_by_price_test.png", dpi=200)
    plt.close()

    # Figure 3 : MAE par segment
    plt.figure(figsize=(7, 5))
    plt.bar(seg_test["segment_price"], seg_test["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("ConvNeXt image seule - MAE par segment")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_03_mae_by_segment_test.png", dpi=200)
    plt.close()

    # Figure 4 : biais par segment
    plt.figure(figsize=(7, 5))
    plt.bar(seg_test["segment_price"], seg_test["Biais_moyen"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("ConvNeXt image seule - biais moyen par segment")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_04_bias_by_segment_test.png", dpi=200)
    plt.close()

    # Figure 5 : courbes de loss / val MAE
    if not history_all.empty:
        plt.figure(figsize=(8, 5))
        for run_name, g in history_all.groupby("run"):
            if "final_full_train" in str(run_name):
                continue
            plt.plot(g["epoch"], g["val_mae_log"], alpha=0.8, label=str(run_name))
        plt.xlabel("Époque")
        plt.ylabel("MAE validation sur log_price")
        plt.title("ConvNeXt image seule - évolution validation")
        if history_all["run"].nunique() <= 8:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_05_training_val_mae.png", dpi=200)
        plt.close()


# Rapport texte

def write_text_report(
    output_dir: Path,
    config: dict,
    dataset_report: pd.DataFrame,
    split_report: pd.DataFrame,
    metrics_oof_log: Dict[str, float],
    metrics_oof_euro: Dict[str, float],
    metrics_test_log: Dict[str, float],
    metrics_test_euro: Dict[str, float],
    best_epochs: List[int],
) -> None:
    lines = []
    lines.append("Rapport ConvNeXt image seule - prédiction directe de log_price")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Objectif")
    lines.append("-" * 90)
    lines.append("Ce modèle constitue une branche visuelle indépendante pour une future late fusion.")
    lines.append("Il apprend directement une relation image -> log_price, sans passer par CatBoost.")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 90)
    for k, v in config.items():
        lines.append(f"{k} : {v}")
    lines.append("")
    lines.append("Alignement des données")
    lines.append("-" * 90)
    for _, row in dataset_report.iterrows():
        lines.append(f"{row['indicateur']} : {row['valeur']}")
    lines.append("")
    lines.append("Split")
    lines.append("-" * 90)
    lines.append(split_report.to_string(index=False))
    lines.append("")
    lines.append("Résultats OOF train_dev - log_price")
    lines.append("-" * 90)
    for k, v in metrics_oof_log.items():
        lines.append(f"{k} : {v:.6f}")
    lines.append("")
    lines.append("Résultats OOF train_dev - euros")
    lines.append("-" * 90)
    for k, v in metrics_oof_euro.items():
        lines.append(f"{k} : {v:.6f}")
    lines.append("")
    lines.append("Résultats test_final - log_price")
    lines.append("-" * 90)
    for k, v in metrics_test_log.items():
        lines.append(f"{k} : {v:.6f}")
    lines.append("")
    lines.append("Résultats test_final - euros")
    lines.append("-" * 90)
    for k, v in metrics_test_euro.items():
        lines.append(f"{k} : {v:.6f}")
    lines.append("")
    lines.append("Époques retenues")
    lines.append("-" * 90)
    if len(best_epochs) > 0:
        lines.append(f"Best epochs folds : {best_epochs}")
        lines.append(f"Moyenne best epoch : {np.mean(best_epochs):.2f}")
        lines.append(f"Médiane best epoch : {np.median(best_epochs):.2f}")
    lines.append("")
    lines.append("Fichiers importants pour la late fusion")
    lines.append("-" * 90)
    lines.append("09_predictions_oof_for_late_fusion.csv")
    lines.append("10_predictions_test_for_late_fusion.csv")
    lines.append("")
    lines.append("Remarque méthodologique")
    lines.append("-" * 90)
    lines.append(
        "Les prédictions OOF du train_dev sont produites fold par fold. Chaque annonce du train_dev "
        "est donc prédite par un modèle qui ne l'a pas vue pendant son entraînement. "
        "C'est indispensable pour entraîner ensuite un méta-modèle de late fusion sans fuite de données."
    )

    with open(output_dir / "12_rapport_final.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main(args):
    start = time.time()
    seed_everything(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    if torch.cuda.is_available():
        print(f"GPU détecté : {torch.cuda.get_device_name(0)}")

    # Chargement et préparation cible

    df_raw = read_table(args.target_path)
    id_col = find_id_column(df_raw)
    df_raw[id_col] = normalize_id(df_raw[id_col])

    if id_col != "listing_id_clean":
        df_raw = df_raw.rename(columns={id_col: "listing_id_clean"})
        id_col = "listing_id_clean"

    if args.price_col not in df_raw.columns:
        raise ValueError(f"Colonne prix introuvable : {args.price_col}")

    df_raw[args.price_col] = clean_numeric_price(df_raw[args.price_col])

    if args.target_col in df_raw.columns:
        df_raw[args.target_col] = pd.to_numeric(df_raw[args.target_col], errors="coerce")
    else:
        print(f"Colonne {args.target_col} absente. Création avec np.log1p({args.price_col}).")
        df_raw[args.target_col] = np.log1p(df_raw[args.price_col])

    df_raw = df_raw.dropna(subset=[id_col, args.price_col, args.target_col]).copy()
    df_raw = df_raw[df_raw[args.price_col] > 0].copy()
    df_raw = df_raw.drop_duplicates(subset=[id_col], keep="first").copy()

    if args.max_samples is not None and args.max_samples > 0:
        # Sous-échantillon pour test rapide.
        segments = price_segment_series(df_raw[args.price_col])
        _, sample_idx = train_test_split(
            np.arange(len(df_raw)),
            test_size=min(args.max_samples, len(df_raw)) / len(df_raw),
            random_state=args.random_state,
            stratify=segments,
        )
        df_raw = df_raw.iloc[sample_idx].copy().reset_index(drop=True)
        print(f"Mode test : max_samples={args.max_samples}, lignes retenues={len(df_raw)}")

    # Alignement images
    df_aligned, missing_df, extra_df = align_dataset_with_images(df_raw, args.image_dir, output_dir, id_col)

    # Split
    df = load_or_create_split(
        df=df_aligned,
        id_col=id_col,
        price_col=args.price_col,
        split_file=args.split_file,
        split_col=args.split_col,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=output_dir,
    )

    # Renommage standard pour la suite
    df = df.rename(columns={args.target_col: "log_price", args.price_col: "price"})
    df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["log_price", "price", "image_path", "split"]).copy()

    # Dataset alignment report
    dataset_report = pd.DataFrame([
        {"indicateur": "lignes_fichier_initial", "valeur": len(df_raw)},
        {"indicateur": "lignes_apres_alignement_images", "valeur": len(df_aligned)},
        {"indicateur": "images_absentes", "valeur": len(missing_df)},
        {"indicateur": "images_en_trop_dossier", "valeur": len(extra_df)},
        {"indicateur": "lignes_finales_modelisation", "valeur": len(df)},
        {"indicateur": "prix_moyen", "valeur": float(df["price"].mean())},
        {"indicateur": "prix_median", "valeur": float(df["price"].median())},
        {"indicateur": "prix_min", "valeur": float(df["price"].min())},
        {"indicateur": "prix_max", "valeur": float(df["price"].max())},
    ])
    dataset_report.to_csv(output_dir / "01_dataset_alignment_report.csv", index=False, encoding="utf-8-sig")

    # Mise à jour split report après renommage
    split_report = summarize_split(df, "split", "price")
    split_report.to_csv(output_dir / "02_split_train_test_report.csv", index=False, encoding="utf-8-sig")

    df.to_csv(output_dir / "dataset_aligne_splits.csv", index=False, encoding="utf-8-sig")

    train_dev_df = df[df["split"] == "train_dev"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test_final"].copy().reset_index(drop=True)

    if len(train_dev_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Split invalide : train_dev ou test_final est vide.")

    print(f"Train_dev : {len(train_dev_df)} | Test_final : {len(test_df)}")

    # Modèle et transform
    tmp_model, transform = build_convnext_regressor(args.model_size)
    del tmp_model

    # OOF K-Fold sur train_dev
    y_segments = price_segment_series(train_dev_df["price"])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)

    oof_predictions = []
    fold_metrics_log = []
    fold_metrics_euro = []
    history_all = []
    failed_all: List[Dict[str, str]] = []
    best_epochs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_dev_df, y_segments), start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold}/{args.n_folds}")
        print("=" * 80)

        fold_train_df = train_dev_df.iloc[tr_idx].copy().reset_index(drop=True)
        fold_val_df = train_dev_df.iloc[va_idx].copy().reset_index(drop=True)

        train_dataset = AirbnbImagePriceDataset(
            fold_train_df, id_col, "image_path", "log_price", "price", transform, args.use_weights
        )
        val_dataset = AirbnbImagePriceDataset(
            fold_val_df, id_col, "image_path", "log_price", "price", transform, False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        model, _ = build_convnext_regressor(args.model_size)
        model = model.to(device)

        fold_model_path = output_dir / "models" / f"fold_{fold}_best.pt"
        result = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_model_path=fold_model_path,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            patience=args.patience,
            amp=args.amp,
            desc_prefix=f"fold_{fold}",
        )

        best_epochs.append(int(result.best_epoch))
        failed_all.extend(result.failed_images)

        h = result.history.copy()
        h.insert(0, "run", f"fold_{fold}")
        history_all.append(h)

        val_pred_df, failed_pred = predict_model(model, val_loader, device, args.amp, desc=f"fold_{fold} | predict val")
        failed_all.extend(failed_pred)
        val_pred_df.insert(0, "fold", fold)
        val_pred_df["prediction_type"] = "oof_train_dev"
        oof_predictions.append(val_pred_df)

        mlog = metrics_log(val_pred_df["log_price_true"].values, val_pred_df["log_price_pred"].values)
        meuro = metrics_euro(val_pred_df["log_price_true"].values, val_pred_df["log_price_pred"].values)
        mlog = {"fold": fold, **mlog}
        meuro = {"fold": fold, **meuro}
        fold_metrics_log.append(mlog)
        fold_metrics_euro.append(meuro)

        pd.DataFrame(fold_metrics_log).to_csv(output_dir / "03_fold_metrics_log_price.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_metrics_euro).to_csv(output_dir / "04_fold_metrics_price_euros.csv", index=False, encoding="utf-8-sig")

        # Libération mémoire GPU entre folds
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    oof_df = pd.concat(oof_predictions, ignore_index=True)
    oof_df.to_csv(output_dir / "09_predictions_oof_for_late_fusion.csv", index=False, encoding="utf-8-sig")

    # Metrics OOF globales
    oof_metrics_log = metrics_log(oof_df["log_price_true"].values, oof_df["log_price_pred"].values)
    oof_metrics_euro = metrics_euro(oof_df["log_price_true"].values, oof_df["log_price_pred"].values)

    oof_metrics_df = pd.DataFrame([
        {"dataset": "OOF_train_dev", "scale": "log_price", **oof_metrics_log},
        {"dataset": "OOF_train_dev", "scale": "price_euros", **oof_metrics_euro},
    ])
    oof_metrics_df.to_csv(output_dir / "05_oof_metrics_global.csv", index=False, encoding="utf-8-sig")

    oof_segments = segment_report(oof_df["log_price_true"].values, oof_df["log_price_pred"].values)
    oof_segments.to_csv(output_dir / "06_oof_segments_price.csv", index=False, encoding="utf-8-sig")


    # Modèle final pour prédire test_final
    # Étape 1 : trouver un bon nombre d'époques via validation interne
    # Étape 2 : réentraîner sur tout train_dev pendant ce nombre d'époques
    print("\n" + "=" * 80)
    print("Modèle final test_final")
    print("=" * 80)

    final_segments = price_segment_series(train_dev_df["price"])
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(train_dev_df)),
        test_size=args.final_val_size,
        random_state=args.random_state,
        stratify=final_segments,
    )

    inner_train_df = train_dev_df.iloc[inner_train_idx].copy().reset_index(drop=True)
    inner_val_df = train_dev_df.iloc[inner_val_idx].copy().reset_index(drop=True)

    inner_train_dataset = AirbnbImagePriceDataset(inner_train_df, id_col, "image_path", "log_price", "price", transform, args.use_weights)
    inner_val_dataset = AirbnbImagePriceDataset(inner_val_df, id_col, "image_path", "log_price", "price", transform, False)

    inner_train_loader = DataLoader(
        inner_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    inner_val_loader = DataLoader(
        inner_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    final_probe_model, _ = build_convnext_regressor(args.model_size)
    final_probe_model = final_probe_model.to(device)
    final_probe_path = output_dir / "models" / "final_probe_best.pt"

    final_probe_result = train_one_model(
        model=final_probe_model,
        train_loader=inner_train_loader,
        val_loader=inner_val_loader,
        device=device,
        output_model_path=final_probe_path,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        patience=args.patience,
        amp=args.amp,
        desc_prefix="final_probe",
    )

    h_probe = final_probe_result.history.copy()
    h_probe.insert(0, "run", "final_probe")
    history_all.append(h_probe)
    failed_all.extend(final_probe_result.failed_images)

    final_best_epoch = max(1, int(final_probe_result.best_epoch))
    print(f"Nombre d'époques retenu pour réentraînement final complet : {final_best_epoch}")

    # Réentraîne sur tout train_dev
    full_train_dataset = AirbnbImagePriceDataset(train_dev_df, id_col, "image_path", "log_price", "price", transform, args.use_weights)
    test_dataset = AirbnbImagePriceDataset(test_df, id_col, "image_path", "log_price", "price", transform, False)

    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    final_model, _ = build_convnext_regressor(args.model_size)
    final_model = final_model.to(device)
    final_model_path = output_dir / "models" / "final_model_full_train_dev.pt"

    h_final = train_fixed_epochs(
        model=final_model,
        train_loader=full_train_loader,
        device=device,
        output_model_path=final_model_path,
        epochs=final_best_epoch,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        amp=args.amp,
        desc_prefix="final_full_train_dev",
    )
    h_final.insert(0, "run", "final_full_train_dev")
    history_all.append(h_final)

    test_pred_df, failed_test = predict_model(final_model, test_loader, device, args.amp, desc="final | predict test_final")
    failed_all.extend(failed_test)
    test_pred_df["prediction_type"] = "test_final"
    test_pred_df.to_csv(output_dir / "10_predictions_test_for_late_fusion.csv", index=False, encoding="utf-8-sig")

    test_metrics_log = metrics_log(test_pred_df["log_price_true"].values, test_pred_df["log_price_pred"].values)
    test_metrics_euro = metrics_euro(test_pred_df["log_price_true"].values, test_pred_df["log_price_pred"].values)

    test_metrics_df = pd.DataFrame([
        {"dataset": "test_final", "scale": "log_price", **test_metrics_log},
        {"dataset": "test_final", "scale": "price_euros", **test_metrics_euro},
    ])
    test_metrics_df.to_csv(output_dir / "07_test_metrics_global.csv", index=False, encoding="utf-8-sig")

    test_segments = segment_report(test_pred_df["log_price_true"].values, test_pred_df["log_price_pred"].values)
    test_segments.to_csv(output_dir / "08_test_segments_price.csv", index=False, encoding="utf-8-sig")

    # Historique complet
    history_all_df = pd.concat(history_all, ignore_index=True) if len(history_all) > 0 else pd.DataFrame()
    history_all_df.to_csv(output_dir / "11_training_history.csv", index=False, encoding="utf-8-sig")

    # Images non lisibles rencontrées pendant training/prédiction
    failed_df = pd.DataFrame(failed_all).drop_duplicates() if len(failed_all) > 0 else pd.DataFrame(columns=["listing_id_clean", "image_path", "error"])
    failed_df.to_csv(output_dir / "images_non_lisibles_pendant_training.csv", index=False, encoding="utf-8-sig")

    # Figures
    save_figures(output_dir, test_pred_df, test_segments, history_all_df)

    elapsed_min = (time.time() - start) / 60.0

    config = {
        "target_path": args.target_path,
        "image_dir": args.image_dir,
        "split_file": args.split_file,
        "output_dir": str(output_dir),
        "model": f"ConvNeXt-{args.model_size}",
        "target": "log_price",
        "price_col": "price",
        "n_rows_initial": int(len(df_raw)),
        "n_rows_aligned_images": int(len(df_aligned)),
        "n_rows_final": int(len(df)),
        "n_train_dev": int(len(train_dev_df)),
        "n_test_final": int(len(test_df)),
        "n_folds": int(args.n_folds),
        "epochs_max": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr_backbone": float(args.lr_backbone),
        "lr_head": float(args.lr_head),
        "weight_decay": float(args.weight_decay),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
        "patience": int(args.patience),
        "use_weights": bool(args.use_weights),
        "amp": bool(args.amp),
        "random_state": int(args.random_state),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "best_epochs_folds": best_epochs,
        "final_best_epoch": int(final_best_epoch),
        "elapsed_minutes": float(elapsed_min),
    }

    with open(output_dir / "13_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    rapport_final_json = {
        "oof_log_price": oof_metrics_log,
        "oof_price_euros": oof_metrics_euro,
        "test_log_price": test_metrics_log,
        "test_price_euros": test_metrics_euro,
        "config": config,
    }
    with open(output_dir / "rapport_final.json", "w", encoding="utf-8") as f:
        json.dump(rapport_final_json, f, indent=2, ensure_ascii=False)

    write_text_report(
        output_dir=output_dir,
        config=config,
        dataset_report=dataset_report,
        split_report=split_report,
        metrics_oof_log=oof_metrics_log,
        metrics_oof_euro=oof_metrics_euro,
        metrics_test_log=test_metrics_log,
        metrics_test_euro=test_metrics_euro,
        best_epochs=best_epochs,
    )

    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print(f"Dossier résultats : {output_dir}")
    print(f"Temps total : {elapsed_min:.2f} minutes")
    print("Fichiers late fusion :")
    print(f"- {output_dir / '09_predictions_oof_for_late_fusion.csv'}")
    print(f"- {output_dir / '10_predictions_test_for_late_fusion.csv'}")
    print("Rapports :")
    print(f"- {output_dir / '07_test_metrics_global.csv'}")
    print(f"- {output_dir / '08_test_segments_price.csv'}")
    print(f"- {output_dir / '12_rapport_final.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_path", type=str, required=True, help="Fichier tabulaire contenant id_clean/listing_id_clean, price et log_price.")
    parser.add_argument("--image_dir", type=str, required=True, help="Dossier contenant les images nommées par ID.")
    parser.add_argument("--split_file", type=str, default=None, help="Fichier de split. Peut contenir seulement les IDs ou une colonne split.")
    parser.add_argument("--split_col", type=str, default="split", help="Nom de la colonne split si elle existe.")
    parser.add_argument("--output_dir", type=str, default="Resultats_ImageOnly_ConvNeXtBase_RapportComplet")

    parser.add_argument("--target_col", type=str, default="log_price")
    parser.add_argument("--price_col", type=str, default="price")

    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--final_val_size", type=float, default=0.10)

    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Pour test rapide uniquement. Exemple : 5000.")

    parser.add_argument("--use_weights", action="store_true", help="Active une loss pondérée pour les logements chers.")
    parser.add_argument("--amp", action="store_true", help="Active mixed precision CUDA.")

    args = parser.parse_args()
    main(args)
