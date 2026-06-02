# -*- coding: utf-8 -*-
"""
Fusion tabulaire + embeddings texte pour la prédiction du prix Airbnb.

Objectif :
- exclure text_pred_final,
- garder les embeddings txt_e5_000 à txt_e5_099,
- utiliser le split train/test déjà défini par la branche texte,
- faire une CV 5-fold uniquement sur le train,
- ne jamais utiliser le test final pour le choix du modèle,
- calculer les poids uniquement sur le train de chaque fold,
- choisir le nombre d'arbres final avec la CV,
- entraîner un modèle final sur tout le train,
- évaluer une seule fois sur le test final.

"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")


# 1. CONFIGURATION

@dataclass
class Config:
    random_state: int = 42

    project_dir: str = "/workspace/airbnb_text_e5_lora_stacking"

    tabular_file: str = "airbnb_tabulaire_fusionné_sentiment.csv"

    text_train_file: str = "airbnb_text_features_train_e5_lora.parquet"
    text_test_file: str = "airbnb_text_features_test_e5_lora.parquet"
    split_file: str = "split_listing_ids.csv"

    target_log: str = "log_price"
    target_real: str = "price"

    id_tabular_col: str = "id_clean"
    id_text_col: str = "listing_id_clean"

    # text_pred_final est volontairement exclue dans cette expérience.
    cols_to_exclude: Tuple[str, ...] = (
        "id",
        "id_clean",
        "listing_id_clean",
        "listing_url",
        "picture_url",

        "price",
        "price_clean",
        "log_price",
        "price_txt",
        "price_clean_txt",
        "log_price_txt",

        "split",
        "split_txt",

        "nights_range_is_incoherent",

        # Variables très redondantes avec la présence/quantité d'avis.
        "has_reviews",
        "nb_avis_textuels_bert",
        "bert_stars_moyen",

        "text_pred_final",
        "text_pred_oof",
        "text_pred_test",

        "name",
        "description",
        "neighborhood_overview",
        "host_about",
        "amenities",
    )

    base_cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # K-folds sur le train uniquement
    n_splits: int = 5
    strat_bins: int = 20

    # Pondération manual_aggressive
    price_segment_bins: Tuple[float, ...] = (0, 100, 200, 400, 800, np.inf)
    price_segment_labels: Tuple[str, ...] = (
        "< 100 €",
        "100-200 €",
        "200-400 €",
        "400-800 €",
        "> 800 €",
    )
    manual_price_weights: Tuple[float, ...] = (1.00, 1.00, 1.15, 2.00, 3.50)
    weight_clip_min: float = 0.5
    weight_clip_max: float = 6.0

    # False = la validation du fold reste non pondérée.
    # C'est plus neutre pour comparer les modèles.
    weight_validation_pool: bool = False

    # CatBoost C05_premium_oriented
    iterations: int = 8000
    learning_rate: float = 0.015
    depth: int = 7
    l2_leaf_reg: float = 8.0
    random_strength: float = 1.0
    bagging_temperature: float = 0.6
    rsm: float = 0.95
    border_count: int = 254
    leaf_estimation_iterations: int = 10
    early_stopping_rounds: int = 300

    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"

    task_type: str = "CPU"
    thread_count: int = -1
    used_ram_limit: Optional[str] = "16gb"
    allow_writing_files: bool = False
    verbose_eval: int = 200

    compute_shap_final: bool = True
    shap_max_rows_final: int = 3000
    shap_top_n: int = 30

    min_expected_merge_ratio: float = 0.95


CFG = Config()

# 2. OUTILS

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val):
            return None
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_id_series(s: pd.Series) -> pd.Series:
    def clean_one(x):
        if pd.isna(x):
            return np.nan

        if isinstance(x, (int, np.integer)):
            return str(int(x))

        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return np.nan
            return str(int(x))

        x = str(x).strip()

        if x.endswith(".0"):
            x = x[:-2]

        return x

    return s.apply(clean_one).astype(str)


def make_regression_strat_bins(y: pd.Series, n_splits: int, max_bins: int = 20) -> pd.Series:
    """
    Stratification adaptée à une régression :
    on transforme log_price en classes par quantiles.
    """
    y = pd.Series(y).reset_index(drop=True)
    max_bins = int(min(max_bins, max(2, len(y) // n_splits)))

    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
            vc = pd.Series(bins).value_counts(dropna=False)

            if len(vc) >= 2 and vc.min() >= n_splits:
                return pd.Series(bins, index=y.index).astype(int)

        except Exception:
            continue

    fallback = pd.cut(
        y,
        bins=np.unique(np.quantile(y, [0.0, 0.5, 1.0])),
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )

    return pd.Series(fallback, index=y.index).fillna(0).astype(int)


# 3. CHEMINS

def get_paths(cfg: Config) -> Dict[str, Path]:
    project_dir = Path(cfg.project_dir)
    data_dir = project_dir / "data"
    text_dir = project_dir / "outputs_text_e5_lora"

    output_dir = project_dir / "outputs_fusion_embeddings_only_kfold_manual_aggressive_c05"
    reports_dir = output_dir / "rapports"
    plots_dir = output_dir / "graphiques"
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "modeles"
    merged_dir = output_dir / "merged_data"
    shap_dir = output_dir / "shap"

    for p in [
        output_dir,
        reports_dir,
        plots_dir,
        predictions_dir,
        models_dir,
        merged_dir,
        shap_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "text_dir": text_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "predictions_dir": predictions_dir,
        "models_dir": models_dir,
        "merged_dir": merged_dir,
        "shap_dir": shap_dir,
        "tabular_path": data_dir / cfg.tabular_file,
        "text_train_path": text_dir / cfg.text_train_file,
        "text_test_path": text_dir / cfg.text_test_file,
        "split_path": text_dir / cfg.split_file,
    }

# 4. CHARGEMENT

def load_inputs(paths: Dict[str, Path], cfg: Config):
    tabular_path = paths["tabular_path"]
    text_train_path = paths["text_train_path"]
    text_test_path = paths["text_test_path"]
    split_path = paths["split_path"]

    if not tabular_path.exists():
        raise FileNotFoundError(f"Fichier tabulaire introuvable : {tabular_path}")
    if not text_train_path.exists():
        raise FileNotFoundError(f"Features texte train introuvables : {text_train_path}")
    if not text_test_path.exists():
        raise FileNotFoundError(f"Features texte test introuvables : {text_test_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Fichier split introuvable : {split_path}")

    print("\n================ CHARGEMENT ================")

    if tabular_path.suffix.lower() == ".csv":
        tab = pd.read_csv(tabular_path, dtype={cfg.id_tabular_col: str}, low_memory=False)
    elif tabular_path.suffix.lower() in [".xlsx", ".xls"]:
        tab = pd.read_excel(tabular_path, dtype={cfg.id_tabular_col: str})
    else:
        raise ValueError("Format tabulaire non supporté. Utilise .csv ou .xlsx.")

    text_train = pd.read_parquet(text_train_path)
    text_test = pd.read_parquet(text_test_path)
    split_df = pd.read_csv(split_path, dtype={cfg.id_text_col: str})

    print("Tabulaire   :", tab.shape, tabular_path)
    print("Texte train :", text_train.shape, text_train_path)
    print("Texte test  :", text_test.shape, text_test_path)
    print("Split       :", split_df.shape, split_path)

    return tab, text_train, text_test, split_df

# 5. FUSION TRAIN / TEST

def prepare_and_merge(tab, text_train, text_test, split_df, cfg: Config, paths: Dict[str, Path]):
    required_tab_cols = [cfg.id_tabular_col, cfg.target_log, cfg.target_real]
    missing_tab = [c for c in required_tab_cols if c not in tab.columns]
    if missing_tab:
        raise ValueError(f"Colonnes manquantes dans le tabulaire : {missing_tab}")

    text_all = pd.concat([text_train, text_test], axis=0, ignore_index=True)

    required_text_cols = [
        cfg.id_text_col,
        "split",
        "txt_e5_000",
        "txt_e5_099",
    ]

    missing_text = [c for c in required_text_cols if c not in text_all.columns]
    if missing_text:
        raise ValueError(f"Colonnes manquantes dans les features texte : {missing_text}")

    if "text_pred_final" in text_all.columns:
        print("Info : text_pred_final existe, mais il sera exclu des variables explicatives.")

    tab = tab.copy()
    text_train = text_train.copy()
    text_test = text_test.copy()
    split_df = split_df.copy()

    tab[cfg.id_text_col] = clean_id_series(tab[cfg.id_tabular_col])
    text_train[cfg.id_text_col] = clean_id_series(text_train[cfg.id_text_col])
    text_test[cfg.id_text_col] = clean_id_series(text_test[cfg.id_text_col])
    split_df[cfg.id_text_col] = clean_id_series(split_df[cfg.id_text_col])

    print("\n================ VÉRIFICATION DES CLÉS ================")

    checks = {
        "tab_listing_id_unique": tab[cfg.id_text_col].is_unique,
        "text_train_listing_id_unique": text_train[cfg.id_text_col].is_unique,
        "text_test_listing_id_unique": text_test[cfg.id_text_col].is_unique,
        "split_listing_id_unique": split_df[cfg.id_text_col].is_unique,
    }

    for k, v in checks.items():
        print(k, ":", v)

    if not checks["text_train_listing_id_unique"]:
        raise ValueError("Doublons dans text_train sur listing_id_clean.")
    if not checks["text_test_listing_id_unique"]:
        raise ValueError("Doublons dans text_test sur listing_id_clean.")
    if not checks["split_listing_id_unique"]:
        raise ValueError("Doublons dans split_listing_ids.csv sur listing_id_clean.")

    if not checks["tab_listing_id_unique"]:
        print("Attention : doublons tabulaires détectés. On garde la première occurrence.")
        tab = tab.drop_duplicates(subset=[cfg.id_text_col], keep="first").copy()

    train_ids = set(text_train[cfg.id_text_col])
    test_ids = set(text_test[cfg.id_text_col])
    inter = train_ids & test_ids

    if len(inter) > 0:
        raise ValueError(f"Fuite : {len(inter)} IDs sont présents à la fois dans text_train et text_test.")

    tab_split = tab.merge(
        split_df[[cfg.id_text_col, "split"]],
        on=cfg.id_text_col,
        how="inner",
        validate="one_to_one",
    )

    missing_in_split = len(tab) - len(tab_split)
    merge_ratio = len(tab_split) / max(1, len(tab))

    print("\n================ ALIGNEMENT SPLIT ================")
    print("Tabulaire avant split :", len(tab))
    print("Tabulaire après split :", len(tab_split))
    print("Lignes tabulaires sans split texte :", missing_in_split)
    print("Ratio de conservation :", round(merge_ratio, 4))
    print(tab_split["split"].value_counts())

    if merge_ratio < cfg.min_expected_merge_ratio:
        raise ValueError(
            f"Trop de lignes perdues pendant l'alignement avec split_listing_ids.csv : "
            f"ratio={merge_ratio:.4f}. Vérifie id_clean/listing_id_clean."
        )

    tab_train = tab_split[tab_split["split"] == "train"].copy()
    tab_test = tab_split[tab_split["split"] == "test"].copy()

    merged_train = tab_train.merge(
        text_train,
        on=cfg.id_text_col,
        how="inner",
        suffixes=("", "_txt"),
        validate="one_to_one",
    )

    merged_test = tab_test.merge(
        text_test,
        on=cfg.id_text_col,
        how="inner",
        suffixes=("", "_txt"),
        validate="one_to_one",
    )

    if "split_txt" in merged_train.columns:
        merged_train = merged_train.drop(columns=["split_txt"])
    if "split_txt" in merged_test.columns:
        merged_test = merged_test.drop(columns=["split_txt"])

    print("\n================ FUSION ================")
    print("Tab train    :", tab_train.shape)
    print("Text train   :", text_train.shape)
    print("Merged train :", merged_train.shape)

    print("Tab test     :", tab_test.shape)
    print("Text test    :", text_test.shape)
    print("Merged test  :", merged_test.shape)

    if len(merged_train) != len(tab_train):
        raise ValueError("Perte de lignes dans la fusion train.")
    if len(merged_test) != len(tab_test):
        raise ValueError("Perte de lignes dans la fusion test.")

    merged_train.to_parquet(paths["merged_dir"] / "merged_train_tab_embeddings.parquet", index=False)
    merged_test.to_parquet(paths["merged_dir"] / "merged_test_tab_embeddings.parquet", index=False)

    merge_report = {
        "experiment": "fusion_tabular_embeddings_only_kfold_without_text_pred_final",
        "tabular_rows": int(len(tab)),
        "tabular_rows_after_split_inner": int(len(tab_split)),
        "missing_tabular_in_text_split": int(missing_in_split),
        "merge_ratio": float(merge_ratio),
        "tab_train_rows": int(len(tab_train)),
        "tab_test_rows": int(len(tab_test)),
        "merged_train_rows": int(len(merged_train)),
        "merged_test_rows": int(len(merged_test)),
        "merged_train_cols": int(merged_train.shape[1]),
        "merged_test_cols": int(merged_test.shape[1]),
        "text_pred_final_excluded": True,
        "kfold_done_only_on_train": True,
    }

    with open(paths["reports_dir"] / "00_merge_report.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(merge_report), f, indent=2, ensure_ascii=False)

    return merged_train, merged_test

# 6. PRÉPARATION FEATURES

def prepare_features_for_catboost(merged_train, merged_test, cfg: Config, paths: Dict[str, Path]):
    print("\n================ PRÉPARATION FEATURES ================")

    for df_name, df in [("train", merged_train), ("test", merged_test)]:
        if cfg.target_log not in df.columns:
            raise ValueError(f"{df_name} : cible absente {cfg.target_log}")
        if cfg.target_real not in df.columns:
            raise ValueError(f"{df_name} : prix réel absent {cfg.target_real}")

    y_train_log = pd.to_numeric(merged_train[cfg.target_log], errors="coerce")
    y_train_real = pd.to_numeric(merged_train[cfg.target_real], errors="coerce")

    y_test_log = pd.to_numeric(merged_test[cfg.target_log], errors="coerce")
    y_test_real = pd.to_numeric(merged_test[cfg.target_real], errors="coerce")

    valid_train = y_train_log.notna() & y_train_real.notna()
    valid_test = y_test_log.notna() & y_test_real.notna()

    merged_train = merged_train.loc[valid_train].reset_index(drop=True)
    merged_test = merged_test.loc[valid_test].reset_index(drop=True)

    y_train_log = y_train_log.loc[valid_train].reset_index(drop=True)
    y_train_real = y_train_real.loc[valid_train].reset_index(drop=True)

    y_test_log = y_test_log.loc[valid_test].reset_index(drop=True)
    y_test_real = y_test_real.loc[valid_test].reset_index(drop=True)

    ids_train = merged_train[cfg.id_text_col].astype(str).reset_index(drop=True)
    ids_test = merged_test[cfg.id_text_col].astype(str).reset_index(drop=True)

    cols_to_exclude_present = sorted(
        set([c for c in cfg.cols_to_exclude if c in merged_train.columns])
        | set([c for c in cfg.cols_to_exclude if c in merged_test.columns])
    )

    train_feature_cols = [c for c in merged_train.columns if c not in cols_to_exclude_present]
    test_feature_cols = [c for c in merged_test.columns if c not in cols_to_exclude_present]

    common_cols = [c for c in train_feature_cols if c in test_feature_cols]

    X_train_full = merged_train[common_cols].copy()
    X_test = merged_test[common_cols].copy()

    constant_cols = [c for c in X_train_full.columns if X_train_full[c].nunique(dropna=False) <= 1]

    if constant_cols:
        X_train_full = X_train_full.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    cat_features = []

    for c in cfg.base_cat_features:
        if c in X_train_full.columns:
            cat_features.append(c)

    object_cols = X_train_full.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for c in object_cols:
        if c not in cat_features:
            cat_features.append(c)

    for c in cat_features:
        X_train_full[c] = X_train_full[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)

    for c in X_train_full.columns:
        if c not in cat_features:
            X_train_full[c] = pd.to_numeric(X_train_full[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    feature_names = list(X_train_full.columns)

    forbidden_features = [
        "id",
        "id_clean",
        "listing_id_clean",
        "listing_url",
        "picture_url",
        "price",
        "price_clean",
        "log_price",
        "price_txt",
        "price_clean_txt",
        "log_price_txt",
        "split",
        "split_txt",
        "nights_range_is_incoherent",
        "has_reviews",
        "nb_avis_textuels_bert",
        "bert_stars_moyen",
        "text_pred_final",
        "text_pred_oof",
        "text_pred_test",
    ]

    for forbidden in forbidden_features:
        if forbidden in feature_names:
            raise ValueError(f"FUITE OU COLONNE INTERDITE DANS X : {forbidden}")

    txt_cols = [c for c in feature_names if c.startswith("txt_e5_")]

    if len(txt_cols) != 100:
        raise ValueError(f"Nombre de composantes texte inattendu : {len(txt_cols)} au lieu de 100.")

    if "text_pred_final" in feature_names:
        raise ValueError("text_pred_final ne doit pas être dans les features.")

    if list(X_train_full.columns) != list(X_test.columns):
        raise ValueError("Colonnes train/test non alignées.")

    print("Train features :", X_train_full.shape)
    print("Test features  :", X_test.shape)
    print("Nombre variables :", len(feature_names))
    print("Nombre embeddings txt_e5_* :", len(txt_cols))
    print("Catégorielles :", cat_features)
    print("Colonnes exclues présentes :", cols_to_exclude_present)
    print("Colonnes constantes supprimées :", constant_cols)

    pd.DataFrame({"feature": feature_names}).to_csv(
        paths["reports_dir"] / "01_features_utilisees_fusion_embeddings_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"cat_feature": cat_features}).to_csv(
        paths["reports_dir"] / "02_cat_features_fusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"excluded_column": cols_to_exclude_present}).to_csv(
        paths["reports_dir"] / "03_colonnes_exclues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"constant_column_removed": constant_cols}).to_csv(
        paths["reports_dir"] / "04_colonnes_constantes_supprimees.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "X_train_full": X_train_full,
        "X_test": X_test,
        "y_train_log": y_train_log,
        "y_train_real": y_train_real,
        "y_test_log": y_test_log,
        "y_test_real": y_test_real,
        "ids_train": ids_train,
        "ids_test": ids_test,
        "cat_features": cat_features,
        "feature_names": feature_names,
        "constant_cols": constant_cols,
        "cols_to_exclude_present": cols_to_exclude_present,
    }

# 7. PONDÉRATION

def build_price_segments(y_price, cfg: Config):
    return pd.cut(
        y_price,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    )


def normalize_and_clip_weights(weights, clip_min, clip_max, normalizer=None):
    w = pd.Series(weights, dtype=float).copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w = w.clip(lower=clip_min, upper=clip_max)

    if normalizer is None:
        normalizer = w.mean()

    if normalizer <= 0 or np.isnan(normalizer):
        normalizer = 1.0

    w = w / normalizer

    return w.astype(float), float(normalizer)


def make_manual_aggressive_weights(y_real, cfg: Config, normalizer=None):
    y_real = pd.Series(y_real).reset_index(drop=True).astype(float)

    segments = pd.cut(
        y_real,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    ).astype(str)

    mapping = dict(zip(cfg.price_segment_labels, cfg.manual_price_weights))
    raw_weights = segments.map(mapping).astype(float)

    return normalize_and_clip_weights(
        raw_weights,
        cfg.weight_clip_min,
        cfg.weight_clip_max,
        normalizer=normalizer,
    )


def compute_weight_diagnostics(y_real, weights, cfg: Config, split_name: str, fold_name: str):
    df_w = pd.DataFrame({
        "y_real": pd.Series(y_real).reset_index(drop=True),
        "weight": pd.Series(weights).reset_index(drop=True),
    })

    df_w["segment"] = build_price_segments(df_w["y_real"], cfg).astype(str)

    rows = []

    for seg in cfg.price_segment_labels:
        g = df_w[df_w["segment"] == seg]

        rows.append({
            "split": split_name,
            "fold": fold_name,
            "segment": seg,
            "n": int(len(g)),
            "price_mean": float(g["y_real"].mean()) if len(g) else np.nan,
            "price_median": float(g["y_real"].median()) if len(g) else np.nan,
            "weight_mean": float(g["weight"].mean()) if len(g) else np.nan,
            "weight_min": float(g["weight"].min()) if len(g) else np.nan,
            "weight_max": float(g["weight"].max()) if len(g) else np.nan,
        })

    return pd.DataFrame(rows)

# 8. MÉTRIQUES

def safe_mape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "R2": np.nan,
            "MAPE_pct": np.nan,
            "SMAPE_pct": np.nan,
            "Mean_Error": np.nan,
            "Underestimation_Rate_pct": np.nan,
            "Overestimation_Rate_pct": np.nan,
            "Abs_Error_P75": np.nan,
            "Abs_Error_P90": np.nan,
            "Abs_Error_P95": np.nan,
        }

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    if len(y_true) >= 2 and len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "R2": r2,
        "MAPE_pct": safe_mape_pct(y_true, y_pred),
        "SMAPE_pct": smape_pct(y_true, y_pred),
        "Mean_Error": float(np.mean(residuals)),
        "Underestimation_Rate_pct": float(np.mean(residuals < 0) * 100.0),
        "Overestimation_Rate_pct": float(np.mean(residuals > 0) * 100.0),
        "Abs_Error_P75": float(np.percentile(abs_errors, 75)),
        "Abs_Error_P90": float(np.percentile(abs_errors, 90)),
        "Abs_Error_P95": float(np.percentile(abs_errors, 95)),
    }


def make_predictions_df(ids, split_name, y_true_log, pred_log, y_true_price, fold_name=None):
    pred_price = np.maximum(np.expm1(pred_log), 0)

    pred_df = pd.DataFrame({
        "listing_id_clean": ids.values,
        "split": split_name,
        "fold": fold_name,
        "y_true_log": y_true_log.values,
        "y_pred_log": pred_log,
        "y_true_price": y_true_price.values,
        "y_pred_price": pred_price,
    })

    pred_df["residual_log"] = pred_df["y_pred_log"] - pred_df["y_true_log"]
    pred_df["residual_price"] = pred_df["y_pred_price"] - pred_df["y_true_price"]
    pred_df["abs_error_log"] = np.abs(pred_df["residual_log"])
    pred_df["abs_error_price"] = np.abs(pred_df["residual_price"])

    return pred_df


def compute_segment_metrics(pred_df, cfg: Config, split_name: str):
    tmp = pred_df.copy()
    tmp["price_segment"] = build_price_segments(tmp["y_true_price"], cfg).astype(str)

    rows = []

    for seg in cfg.price_segment_labels:
        g = tmp[tmp["price_segment"] == seg]
        metrics = compute_metrics(g["y_true_price"].values, g["y_pred_price"].values)

        rows.append({
            "split": split_name,
            "segment": seg,
            "price_mean": float(g["y_true_price"].mean()) if len(g) else np.nan,
            "price_median": float(g["y_true_price"].median()) if len(g) else np.nan,
            **metrics,
        })

    return pd.DataFrame(rows)

# 9. CATBOOST PARAMS

def make_catboost_params(
    cfg: Config,
    seed: int,
    iterations_override: Optional[int] = None,
    with_early_stopping: bool = True,
):
    params = {
        "iterations": cfg.iterations if iterations_override is None else int(iterations_override),
        "learning_rate": cfg.learning_rate,
        "depth": cfg.depth,
        "l2_leaf_reg": cfg.l2_leaf_reg,
        "random_strength": cfg.random_strength,
        "bagging_temperature": cfg.bagging_temperature,
        "rsm": cfg.rsm,
        "border_count": cfg.border_count,
        "leaf_estimation_iterations": cfg.leaf_estimation_iterations,
        "loss_function": cfg.loss_function,
        "eval_metric": cfg.eval_metric,
        "random_seed": seed,
        "allow_writing_files": cfg.allow_writing_files,
        "task_type": cfg.task_type,
        "thread_count": cfg.thread_count,
        "verbose": cfg.verbose_eval,
    }

    if with_early_stopping:
        params["early_stopping_rounds"] = cfg.early_stopping_rounds

    if cfg.used_ram_limit:
        params["used_ram_limit"] = cfg.used_ram_limit

    return params

# 10. K-FOLD CV SUR TRAIN UNIQUEMENT

def run_kfold_cv(data: Dict[str, Any], cfg: Config, paths: Dict[str, Path]):
    X = data["X_train_full"]
    y_log = data["y_train_log"]
    y_real = data["y_train_real"]
    ids = data["ids_train"]
    cat_features = data["cat_features"]

    print("\n" + "=" * 100)
    print("K-FOLD CROSS-VALIDATION SUR LE TRAIN UNIQUEMENT")
    print("=" * 100)
    print("Nombre de folds :", cfg.n_splits)
    print("Train complet utilisé pour CV :", X.shape)
    print("Test final non utilisé ici.")

    strat_bins = make_regression_strat_bins(
        y=y_log,
        n_splits=cfg.n_splits,
        max_bins=cfg.strat_bins,
    )

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    oof_predictions = []
    fold_metrics_log_rows = []
    fold_metrics_price_rows = []
    fold_segment_rows = []
    fold_weight_rows = []
    fold_iteration_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, strat_bins), start=1):
        fold_name = f"fold_{fold_id}"
        fold_seed = cfg.random_state + fold_id

        print("\n" + "-" * 100)
        print(f"{fold_name}")
        print("-" * 100)

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)

        y_train_log = y_log.iloc[train_idx].reset_index(drop=True)
        y_val_log = y_log.iloc[val_idx].reset_index(drop=True)

        y_train_real = y_real.iloc[train_idx].reset_index(drop=True)
        y_val_real = y_real.iloc[val_idx].reset_index(drop=True)

        ids_val = ids.iloc[val_idx].reset_index(drop=True)

        # Poids calculés seulement sur le train du fold.
        w_train, normalizer = make_manual_aggressive_weights(
            y_train_real,
            cfg,
            normalizer=None,
        )

        w_val, _ = make_manual_aggressive_weights(
            y_val_real,
            cfg,
            normalizer=normalizer,
        )

        fold_weight_rows.append(
            compute_weight_diagnostics(
                y_real=y_train_real,
                weights=w_train,
                cfg=cfg,
                split_name="fold_train",
                fold_name=fold_name,
            )
        )

        train_pool = Pool(
            data=X_train,
            label=y_train_log,
            cat_features=cat_features,
            weight=w_train.values,
        )

        if cfg.weight_validation_pool:
            val_pool = Pool(
                data=X_val,
                label=y_val_log,
                cat_features=cat_features,
                weight=w_val.values,
            )
        else:
            val_pool = Pool(
                data=X_val,
                label=y_val_log,
                cat_features=cat_features,
            )

        model = CatBoostRegressor(
            **make_catboost_params(
                cfg=cfg,
                seed=fold_seed,
                with_early_stopping=True,
            )
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=cfg.verbose_eval,
        )

        model_path = paths["models_dir"] / f"catboost_cv_{fold_name}.cbm"
        model.save_model(str(model_path))

        best_iteration = model.get_best_iteration()
        tree_count = int(model.tree_count_)

        fold_iteration_rows.append({
            "fold": fold_name,
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "tree_count": tree_count,
            "model_path": str(model_path),
        })

        pred_val_log = model.predict(val_pool)

        pred_df = make_predictions_df(
            ids=ids_val,
            split_name="oof",
            y_true_log=y_val_log,
            pred_log=pred_val_log,
            y_true_price=y_val_real,
            fold_name=fold_name,
        )

        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)

        pred_df.to_csv(
            paths["predictions_dir"] / f"predictions_oof_{fold_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        oof_predictions.append(pred_df)

        metrics_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        metrics_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])

        fold_metrics_log_rows.append({
            "fold": fold_name,
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "tree_count": tree_count,
            **metrics_log,
        })

        fold_metrics_price_rows.append({
            "fold": fold_name,
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "tree_count": tree_count,
            **metrics_price,
        })

        seg_fold = compute_segment_metrics(pred_df, cfg, split_name=fold_name)
        fold_segment_rows.append(seg_fold)

        print(
            f"{fold_name} terminé | "
            f"MAE € = {metrics_price['MAE']:.2f} | "
            f"RMSE € = {metrics_price['RMSE']:.2f} | "
            f"Biais € = {metrics_price['Mean_Error']:.2f} | "
            f"tree_count = {tree_count}"
        )

    oof_df = pd.concat(oof_predictions, ignore_index=True)
    fold_metrics_log_df = pd.DataFrame(fold_metrics_log_rows)
    fold_metrics_price_df = pd.DataFrame(fold_metrics_price_rows)
    fold_segments_df = pd.concat(fold_segment_rows, ignore_index=True)
    fold_weights_df = pd.concat(fold_weight_rows, ignore_index=True)
    fold_iterations_df = pd.DataFrame(fold_iteration_rows)

    oof_df.to_csv(
        paths["predictions_dir"] / "10_oof_predictions_all_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics_log_df.to_csv(
        paths["reports_dir"] / "10_cv_fold_metrics_log_price.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics_price_df.to_csv(
        paths["reports_dir"] / "11_cv_fold_metrics_price_euros.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_segments_df.to_csv(
        paths["reports_dir"] / "12_cv_segments_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_weights_df.to_csv(
        paths["reports_dir"] / "13_cv_weight_diagnostics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_iterations_df.to_csv(
        paths["reports_dir"] / "14_cv_best_iterations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_metrics_log = compute_metrics(oof_df["y_true_log"], oof_df["y_pred_log"])
    oof_metrics_price = compute_metrics(oof_df["y_true_price"], oof_df["y_pred_price"])
    oof_segments = compute_segment_metrics(oof_df, cfg, split_name="oof_global")

    oof_metrics_rows = []

    for metric, value in oof_metrics_log.items():
        oof_metrics_rows.append({
            "split": "oof_global",
            "scale": "log_price",
            "metric": metric,
            "value": value,
        })

    for metric, value in oof_metrics_price.items():
        oof_metrics_rows.append({
            "split": "oof_global",
            "scale": "price_euros",
            "metric": metric,
            "value": value,
        })

    oof_metrics_df = pd.DataFrame(oof_metrics_rows)

    oof_metrics_df.to_csv(
        paths["reports_dir"] / "15_cv_oof_metrics_global.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_segments.to_csv(
        paths["reports_dir"] / "16_cv_oof_segments_global.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tree_counts = fold_iterations_df["tree_count"].dropna().astype(int).values

    if len(tree_counts) == 0:
        final_iterations = cfg.iterations
    else:
        final_iterations = int(np.median(tree_counts))
        final_iterations = max(1, final_iterations)

    cv_summary = {
        "n_splits": cfg.n_splits,
        "oof_metrics_log": oof_metrics_log,
        "oof_metrics_price": oof_metrics_price,
        "tree_counts": tree_counts.tolist(),
        "final_iterations_selected_by_median_tree_count": final_iterations,
        "selection_rule": "median_tree_count_from_kfold_cv_on_train_only",
        "test_final_used_in_cv": False,
        "weight_validation_pool": cfg.weight_validation_pool,
    }

    with open(paths["reports_dir"] / "17_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(cv_summary), f, indent=2, ensure_ascii=False)

    with open(paths["reports_dir"] / "18_cv_resume.txt", "w", encoding="utf-8") as f:
        f.write("Résumé CV - fusion tabulaire + embeddings texte\n")
        f.write("=" * 90 + "\n\n")
        f.write("La validation croisée est faite uniquement sur le train.\n")
        f.write("Le test final n'est jamais utilisé pendant la CV.\n")
        f.write("Les poids sont calculés uniquement sur le train de chaque fold.\n")
        f.write(f"Validation pondérée : {cfg.weight_validation_pool}\n")
        f.write(f"Nombre de folds : {cfg.n_splits}\n\n")

        f.write("Métriques OOF globales - price_euros\n")
        f.write("-" * 90 + "\n")
        for k, v in oof_metrics_price.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\nNombre d'arbres par fold\n")
        f.write("-" * 90 + "\n")
        f.write(fold_iterations_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"Nombre d'arbres retenu pour le modèle final : {final_iterations}\n")

    plot_cv_outputs(oof_df, oof_segments, paths)

    return {
        "oof_predictions": oof_df,
        "fold_metrics_log": fold_metrics_log_df,
        "fold_metrics_price": fold_metrics_price_df,
        "oof_metrics_log": oof_metrics_log,
        "oof_metrics_price": oof_metrics_price,
        "oof_segments": oof_segments,
        "fold_iterations": fold_iterations_df,
        "final_iterations": final_iterations,
    }

# 11. ENTRAÎNEMENT FINAL + TEST FINAL

def train_final_model_and_evaluate(
    data: Dict[str, Any],
    cfg: Config,
    paths: Dict[str, Path],
    final_iterations: int,
    cv_results: Dict[str, Any],
):
    X_train_full = data["X_train_full"]
    X_test = data["X_test"]

    y_train_log = data["y_train_log"]
    y_train_real = data["y_train_real"]

    y_test_log = data["y_test_log"]
    y_test_real = data["y_test_real"]

    ids_train = data["ids_train"]
    ids_test = data["ids_test"]

    cat_features = data["cat_features"]
    feature_names = data["feature_names"]

    print("\n" + "=" * 100)
    print("ENTRAÎNEMENT DU MODÈLE FINAL SUR TOUT LE TRAIN")
    print("=" * 100)
    print("Itérations finales :", final_iterations)
    print("Train complet :", X_train_full.shape)
    print("Test final :", X_test.shape)

    w_full, normalizer = make_manual_aggressive_weights(
        y_train_real,
        cfg,
        normalizer=None,
    )

    weight_diag = compute_weight_diagnostics(
        y_real=y_train_real,
        weights=w_full,
        cfg=cfg,
        split_name="full_train",
        fold_name="final_model",
    )

    weight_diag.to_csv(
        paths["reports_dir"] / "20_final_weight_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_pool = Pool(
        data=X_train_full,
        label=y_train_log,
        cat_features=cat_features,
        weight=w_full.values,
    )

    final_model = CatBoostRegressor(
        **make_catboost_params(
            cfg=cfg,
            seed=cfg.random_state + 3030,
            iterations_override=final_iterations,
            with_early_stopping=False,
        )
    )

    final_model.fit(train_pool, verbose=cfg.verbose_eval)

    final_model_path = paths["models_dir"] / "catboost_final_full_train_embeddings_only_kfold.cbm"
    final_model.save_model(str(final_model_path))

    print("Modèle final sauvegardé :", final_model_path)

    split_items = [
        ("train_full", X_train_full, y_train_log, y_train_real, ids_train),
        ("test_final", X_test, y_test_log, y_test_real, ids_test),
    ]

    preds = {}
    metrics_rows = []
    segment_rows = []

    for split_name, X_part, y_log_part, y_real_part, ids_part in split_items:
        pool = Pool(
            data=X_part,
            label=y_log_part,
            cat_features=cat_features,
        )

        pred_log = final_model.predict(pool)

        pred_df = make_predictions_df(
            ids=ids_part,
            split_name=split_name,
            y_true_log=y_log_part,
            pred_log=pred_log,
            y_true_price=y_real_part,
            fold_name="final_model",
        )

        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)

        pred_df.to_csv(
            paths["predictions_dir"] / f"20_predictions_{split_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        preds[split_name] = pred_df

        m_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        m_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])

        for metric, value in m_log.items():
            metrics_rows.append({
                "model": "final_model_full_train",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "tabular_plus_embeddings_only_kfold_without_text_pred_final",
                "split": split_name,
                "scale": "log_price",
                "metric": metric,
                "value": value,
            })

        for metric, value in m_price.items():
            metrics_rows.append({
                "model": "final_model_full_train",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "tabular_plus_embeddings_only_kfold_without_text_pred_final",
                "split": split_name,
                "scale": "price_euros",
                "metric": metric,
                "value": value,
            })

        seg = compute_segment_metrics(pred_df, cfg, split_name=split_name)
        segment_rows.append(seg)

    metrics_df = pd.DataFrame(metrics_rows)
    segments_df = pd.concat(segment_rows, ignore_index=True)

    metrics_df.to_csv(
        paths["reports_dir"] / "21_final_metrics_train_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segments_df.to_csv(
        paths["reports_dir"] / "22_final_segments_train_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = metrics_df.pivot_table(
        index=["scale", "metric"],
        columns="split",
        values="value",
    ).reset_index()

    pivot.to_csv(
        paths["reports_dir"] / "23_final_metrics_pivot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_feature_importance(final_model, feature_names, paths)
    plot_final_outputs(preds, segments_df, paths)

    if cfg.compute_shap_final:
        save_shap_final(final_model, X_train_full, y_train_log, cat_features, cfg, paths)

    write_final_report(
        model=final_model,
        metrics_df=metrics_df,
        segments_df=segments_df,
        cv_results=cv_results,
        cfg=cfg,
        paths=paths,
        final_model_path=final_model_path,
    )

    config = {
        "candidate": "manual_aggressive__C05_premium_oriented",
        "experiment": "tabular_plus_embeddings_only_kfold_without_text_pred_final",
        "config": asdict(cfg),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "constant_cols": data["constant_cols"],
        "excluded_cols_present": data["cols_to_exclude_present"],
        "final_model": {
            "model_path": str(final_model_path),
            "iterations": int(final_iterations),
            "selection_rule": "median tree_count from K-fold CV on train only",
        },
        "important": (
            "text_pred_final est exclu. "
            "La CV est faite uniquement sur le train. "
            "Le test final est utilisé une seule fois après le choix du nombre d'arbres."
        ),
    }

    with open(paths["models_dir"] / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(config), f, indent=2, ensure_ascii=False)

    return {
        "model": final_model,
        "metrics": metrics_df,
        "segments": segments_df,
        "predictions": preds,
    }

# 12. IMPORTANCE, SHAP, GRAPHIQUES, RAPPORTS

def save_feature_importance(model, feature_names, paths: Dict[str, Path]):
    importance = model.get_feature_importance(type="FeatureImportance")

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        paths["reports_dir"] / "24_final_feature_importance_catboost.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = fi.head(30).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance CatBoost")
    plt.title("Fusion tabulaire + embeddings texte - Top 30 variables")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "24_final_feature_importance_top30.png", dpi=160)
    plt.close()


def save_shap_final(model, X_train, y_train_log, cat_features, cfg: Config, paths: Dict[str, Path]):
    sample_size = min(cfg.shap_max_rows_final, len(X_train))

    X_sample = X_train.sample(sample_size, random_state=cfg.random_state)
    y_sample = y_train_log.loc[X_sample.index]

    shap_pool = Pool(
        data=X_sample,
        label=y_sample,
        cat_features=cat_features,
    )

    shap_values = model.get_feature_importance(
        data=shap_pool,
        type="ShapValues",
    )

    shap_contrib = shap_values[:, :-1]
    mean_abs_shap = np.abs(shap_contrib).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(
        paths["shap_dir"] / "final_mean_abs_shap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = shap_df.head(cfg.shap_top_n).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP|")
    plt.title("Fusion tabulaire + embeddings texte - Top 30 SHAP")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "25_final_shap_top30.png", dpi=160)
    plt.close()


def plot_cv_outputs(oof_df, oof_segments, paths: Dict[str, Path]):
    max_value = np.percentile(
        np.concatenate([oof_df["y_true_price"], oof_df["y_pred_price"]]),
        99,
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(oof_df["y_true_price"], oof_df["y_pred_price"], alpha=0.20, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("OOF CV - Prix prédit vs réel\nFusion tabulaire + embeddings texte")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "10_cv_oof_pred_vs_true.png", dpi=160)
    plt.close()

    residuals_clip = np.clip(
        oof_df["residual_price"],
        np.percentile(oof_df["residual_price"], 1),
        np.percentile(oof_df["residual_price"], 99),
    )

    plt.figure(figsize=(9, 5))
    plt.hist(residuals_clip, bins=70)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Résidu en euros")
    plt.ylabel("Fréquence")
    plt.title("OOF CV - Résidus")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "11_cv_oof_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(oof_df["y_true_price"], oof_df["abs_error_price"], alpha=0.20, s=8)
    plt.xlim(0, np.percentile(oof_df["y_true_price"], 99))
    plt.ylim(0, np.percentile(oof_df["abs_error_price"], 99))
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Erreur absolue (€)")
    plt.title("OOF CV - Erreur absolue selon le prix réel")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "12_cv_oof_abs_error_vs_price.png", dpi=160)
    plt.close()

    seg_plot = oof_segments.copy()
    order = list(CFG.price_segment_labels)
    seg_plot["segment"] = pd.Categorical(seg_plot["segment"], categories=order, ordered=True)
    seg_plot = seg_plot.sort_values("segment")

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("OOF CV - MAE par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "13_cv_oof_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("OOF CV - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "14_cv_oof_bias_by_segment.png", dpi=160)
    plt.close()


def plot_final_outputs(preds, segments_df, paths: Dict[str, Path]):
    test = preds["test_final"]

    max_value = np.percentile(
        np.concatenate([test["y_true_price"], test["y_pred_price"]]),
        99,
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(test["y_true_price"], test["y_pred_price"], alpha=0.25, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("Test final - Prix prédit vs réel\nFusion tabulaire + embeddings texte")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "20_final_test_pred_vs_true.png", dpi=160)
    plt.close()

    residuals_clip = np.clip(
        test["residual_price"],
        np.percentile(test["residual_price"], 1),
        np.percentile(test["residual_price"], 99),
    )

    plt.figure(figsize=(9, 5))
    plt.hist(residuals_clip, bins=70)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Résidu en euros")
    plt.ylabel("Fréquence")
    plt.title("Test final - Résidus")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "21_final_test_residuals.png", dpi=160)
    plt.close()

    seg_test = segments_df[segments_df["split"] == "test_final"].copy()
    order = list(CFG.price_segment_labels)
    seg_test["segment"] = pd.Categorical(seg_test["segment"], categories=order, ordered=True)
    seg_test = seg_test.sort_values("segment")

    plt.figure(figsize=(8, 5))
    plt.bar(seg_test["segment"].astype(str), seg_test["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("Test final - MAE par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "22_final_test_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_test["segment"].astype(str), seg_test["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("Test final - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "23_final_test_bias_by_segment.png", dpi=160)
    plt.close()


def get_metric(metrics_df, split, scale, metric):
    row = metrics_df[
        (metrics_df["split"] == split)
        & (metrics_df["scale"] == scale)
        & (metrics_df["metric"] == metric)
    ]

    if len(row) == 0:
        return np.nan

    return float(row["value"].iloc[0])


def write_final_report(
    model,
    metrics_df,
    segments_df,
    cv_results,
    cfg: Config,
    paths: Dict[str, Path],
    final_model_path: Path,
):
    test_mae = get_metric(metrics_df, "test_final", "price_euros", "MAE")
    test_rmse = get_metric(metrics_df, "test_final", "price_euros", "RMSE")
    test_r2 = get_metric(metrics_df, "test_final", "price_euros", "R2")
    test_bias = get_metric(metrics_df, "test_final", "price_euros", "Mean_Error")

    test_log_mae = get_metric(metrics_df, "test_final", "log_price", "MAE")
    test_log_rmse = get_metric(metrics_df, "test_final", "log_price", "RMSE")
    test_log_r2 = get_metric(metrics_df, "test_final", "log_price", "R2")

    seg_test = segments_df[segments_df["split"] == "test_final"].copy()
    order = list(cfg.price_segment_labels)
    seg_test["segment"] = pd.Categorical(seg_test["segment"], categories=order, ordered=True)
    seg_test = seg_test.sort_values("segment")

    report_path = paths["reports_dir"] / "30_resume_final_fusion_embeddings_only_kfold_catboost.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Résumé final - Fusion tabulaire + embeddings texte avec CatBoost + K-fold CV\n")
        f.write("=" * 100 + "\n\n")

        f.write("Candidat retenu\n")
        f.write("-" * 100 + "\n")
        f.write("manual_aggressive__C05_premium_oriented\n\n")

        f.write("Protocole\n")
        f.write("-" * 100 + "\n")
        f.write("Le split train/test vient de la branche texte via split_listing_ids.csv.\n")
        f.write("La validation croisée 5-fold est faite uniquement sur le train.\n")
        f.write("Le test final n'est jamais utilisé pendant la CV.\n")
        f.write("Les poids manual_aggressive sont calculés seulement sur le train de chaque fold.\n")
        f.write("La validation du fold reste non pondérée, sauf si weight_validation_pool=True.\n")
        f.write("Le nombre d'arbres du modèle final est la médiane des tree_count obtenus en CV.\n")
        f.write("Le modèle final est entraîné sur tout le train, puis évalué une seule fois sur le test final.\n")
        f.write("La variable text_pred_final est volontairement exclue.\n")
        f.write("Les embeddings txt_e5_000 à txt_e5_099 sont conservés.\n\n")

        f.write("Configuration CatBoost\n")
        f.write("-" * 100 + "\n")
        f.write(f"iterations max CV : {cfg.iterations}\n")
        f.write(f"iterations final : {cv_results['final_iterations']}\n")
        f.write(f"learning_rate : {cfg.learning_rate}\n")
        f.write(f"depth : {cfg.depth}\n")
        f.write(f"l2_leaf_reg : {cfg.l2_leaf_reg}\n")
        f.write(f"random_strength : {cfg.random_strength}\n")
        f.write(f"bagging_temperature : {cfg.bagging_temperature}\n")
        f.write(f"rsm : {cfg.rsm}\n")
        f.write(f"early_stopping_rounds : {cfg.early_stopping_rounds}\n")
        f.write(f"task_type : {cfg.task_type}\n")
        f.write(f"modèle final : {final_model_path}\n\n")

        f.write("Pondération manual_aggressive\n")
        f.write("-" * 100 + "\n")
        for label, weight in zip(cfg.price_segment_labels, cfg.manual_price_weights):
            f.write(f"{label} : {weight}\n")
        f.write(f"clip_min : {cfg.weight_clip_min}\n")
        f.write(f"clip_max : {cfg.weight_clip_max}\n\n")

        f.write("Métriques CV OOF - price_euros\n")
        f.write("-" * 100 + "\n")
        for k, v in cv_results["oof_metrics_price"].items():
            f.write(f"{k:30s}: {v:.6f}\n")
        f.write("\n")

        f.write("Métriques test final - log_price\n")
        f.write("-" * 100 + "\n")
        f.write(f"MAE log : {test_log_mae:.4f}\n")
        f.write(f"RMSE log : {test_log_rmse:.4f}\n")
        f.write(f"R2 log : {test_log_r2:.4f}\n\n")

        f.write("Métriques test final - price_euros\n")
        f.write("-" * 100 + "\n")
        f.write(f"MAE : {test_mae:.4f} €\n")
        f.write(f"RMSE : {test_rmse:.4f} €\n")
        f.write(f"R2 : {test_r2:.4f}\n")
        f.write(f"Biais moyen : {test_bias:.4f} €\n\n")

        f.write("Segments test final\n")
        f.write("-" * 100 + "\n")

        cols = [
            "segment",
            "n",
            "MAE",
            "RMSE",
            "MedAE",
            "Mean_Error",
            "Underestimation_Rate_pct",
            "Abs_Error_P90",
            "Abs_Error_P95",
        ]

        available = [c for c in cols if c in seg_test.columns]
        f.write(seg_test[available].to_string(index=False))
        f.write("\n\n")

        f.write("Lecture critique\n")
        f.write("-" * 100 + "\n")
        f.write(
            "Cette version est plus solide méthodologiquement que la version avec un simple split interne, "
            "car le choix du nombre d'arbres est estimé sur plusieurs folds du train. "
            "Le test final reste isolé jusqu'à la dernière étape. "
            "Si les résultats test final sont proches des résultats OOF, le modèle est plutôt stable. "
            "Si le test final est beaucoup moins bon que l'OOF, il faudra vérifier la distribution du test, "
            "la représentativité des logements chers et la cohérence du split texte.\n"
        )


def main():
    paths = get_paths(CFG)

    print("\n" + "=" * 100)
    print("FUSION TABULAIRE + EMBEDDINGS TEXTE - K-FOLD CV - SANS text_pred_final")
    print("=" * 100)
    print("Dossier projet :", paths["project_dir"])
    print("Dossier data :", paths["data_dir"])
    print("Dossier texte :", paths["text_dir"])
    print("Dossier sorties :", paths["output_dir"])

    tab, text_train, text_test, split_df = load_inputs(paths, CFG)

    merged_train, merged_test = prepare_and_merge(
        tab=tab,
        text_train=text_train,
        text_test=text_test,
        split_df=split_df,
        cfg=CFG,
        paths=paths,
    )

    data = prepare_features_for_catboost(
        merged_train=merged_train,
        merged_test=merged_test,
        cfg=CFG,
        paths=paths,
    )

    with open(paths["reports_dir"] / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable({
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "fusion_tabular_embeddings_only_kfold_without_text_pred_final",
                "config": asdict(CFG),
                "important": (
                    "Fusion tabulaire + embeddings texte. "
                    "text_pred_final exclu. "
                    "K-fold uniquement sur train. "
                    "Test final isolé."
                ),
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    cv_results = run_kfold_cv(
        data=data,
        cfg=CFG,
        paths=paths,
    )

    train_final_model_and_evaluate(
        data=data,
        cfg=CFG,
        paths=paths,
        final_iterations=cv_results["final_iterations"],
        cv_results=cv_results,
    )

    print("\n" + "=" * 100)
    print("FIN - FUSION TABULAIRE + EMBEDDINGS TEXTE - K-FOLD CV")
    print("=" * 100)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])

    print("\nFichiers importants à m'envoyer pour analyse :")
    print(paths["reports_dir"] / "00_merge_report.json")
    print(paths["reports_dir"] / "01_features_utilisees_fusion_embeddings_only.csv")
    print(paths["reports_dir"] / "10_cv_fold_metrics_log_price.csv")
    print(paths["reports_dir"] / "11_cv_fold_metrics_price_euros.csv")
    print(paths["reports_dir"] / "15_cv_oof_metrics_global.csv")
    print(paths["reports_dir"] / "16_cv_oof_segments_global.csv")
    print(paths["reports_dir"] / "17_cv_summary.json")
    print(paths["reports_dir"] / "21_final_metrics_train_test.csv")
    print(paths["reports_dir"] / "22_final_segments_train_test.csv")
    print(paths["reports_dir"] / "23_final_metrics_pivot.csv")
    print(paths["reports_dir"] / "24_final_feature_importance_catboost.csv")
    print(paths["reports_dir"] / "30_resume_final_fusion_embeddings_only_kfold_catboost.txt")
    print(paths["predictions_dir"] / "10_oof_predictions_all_folds.csv")
    print(paths["predictions_dir"] / "20_predictions_test_final.csv")


if __name__ == "__main__":
    main()