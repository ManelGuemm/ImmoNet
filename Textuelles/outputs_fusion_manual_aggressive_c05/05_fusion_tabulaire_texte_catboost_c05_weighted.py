# -*- coding: utf-8 -*-
"""
Fusion tabulaire + texte pour la prédiction du prix Airbnb.

Modèle retenu :
manual_aggressive__C05_premium_oriented

Objectif :
- lire le fichier tabulaire final,
- lire les features texte produites par E5-LoRA,
- utiliser le même split train/test que la branche texte,
- fusionner avec listing_id_clean,
- exclure les colonnes interdites,
- garder les catégorielles en texte pour CatBoost,
- entraîner CatBoost avec la logique manual_aggressive + C05_premium_oriented,
- évaluer une seule fois sur le test final,
- sauvegarder les métriques, les prédictions, les graphiques, les importances et SHAP.

"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

    # Colonnes à exclure des features.
    # price et log_price sont des cibles donc interdites comme variables explicatives.
    cols_to_exclude: Tuple[str, ...] = (
        "id",
        "id_clean",
        "listing_id_clean",
        "listing_url",
        "picture_url",

        "price",
        "price_clean",
        "log_price",

        "split",
        "split_txt",

        "nights_range_is_incoherent",
        "has_reviews",
        "nb_avis_textuels_bert",
        "bert_stars_moyen",

        # Colonnes split-spécifiques du texte.
        # On garde uniquement text_pred_final.
        "text_pred_oof",
        "text_pred_test",
    )

    # Catégorielles principales attendues côté tabulaire
    base_cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # Validation interne pour early stopping
    validation_size: float = 0.15
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

    # Validation non pondérée, comme dans ton tuning
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

    # CPU pour rester proche de ton tuning.
    task_type: str = "CPU"
    thread_count: int = -1
    used_ram_limit: Optional[str] = "16gb"
    allow_writing_files: bool = False
    verbose_eval: int = 200

    # Interprétabilité
    compute_shap_final: bool = True
    shap_max_rows_final: int = 3000
    shap_top_n: int = 30

    # Sécurité
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
    """
    Convertit une colonne identifiant en string propre.
    Important pour éviter les problèmes entre int, float, string, et .0.
    """
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

    output_dir = project_dir / "outputs_fusion_manual_aggressive_c05"
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

# 5. FUSION

def prepare_and_merge(tab, text_train, text_test, split_df, cfg: Config, paths: Dict[str, Path]):
    required_tab_cols = [cfg.id_tabular_col, cfg.target_log, cfg.target_real]
    missing_tab = [c for c in required_tab_cols if c not in tab.columns]
    if missing_tab:
        raise ValueError(f"Colonnes manquantes dans le tabulaire : {missing_tab}")

    required_text_cols = [
        cfg.id_text_col,
        "split",
        "text_pred_final",
        "name_n_words",
        "description_n_words",
        "description_is_missing",
        "txt_e5_000",
        "txt_e5_099",
    ]

    text_all = pd.concat([text_train, text_test], axis=0, ignore_index=True)
    missing_text = [c for c in required_text_cols if c not in text_all.columns]
    if missing_text:
        raise ValueError(f"Colonnes manquantes dans les features texte : {missing_text}")

    tab = tab.copy()
    text_train = text_train.copy()
    text_test = text_test.copy()
    split_df = split_df.copy()

    tab[cfg.id_text_col] = clean_id_series(tab[cfg.id_tabular_col])
    text_train[cfg.id_text_col] = clean_id_series(text_train[cfg.id_text_col])
    text_test[cfg.id_text_col] = clean_id_series(text_test[cfg.id_text_col])
    split_df[cfg.id_text_col] = clean_id_series(split_df[cfg.id_text_col])

    print("\n================ VÉRIFICATION CLÉS ================")

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
        print("Attention : doublons tabulaires détectés. Suppression en gardant la première occurrence.")
        tab = tab.drop_duplicates(subset=[cfg.id_text_col], keep="first").copy()

    train_ids = set(text_train[cfg.id_text_col])
    test_ids = set(text_test[cfg.id_text_col])
    inter = train_ids & test_ids
    assert len(inter) == 0, f"Fuite : {len(inter)} IDs présents à la fois dans text_train et text_test."

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
            f"Trop de lignes perdues lors de l'alignement avec split_listing_ids.csv : "
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

    assert len(merged_train) == len(tab_train), "Perte de lignes dans la fusion train."
    assert len(merged_test) == len(tab_test), "Perte de lignes dans la fusion test."

    merged_train.to_parquet(paths["merged_dir"] / "merged_train_tab_text.parquet", index=False)
    merged_test.to_parquet(paths["merged_dir"] / "merged_test_tab_text.parquet", index=False)

    merge_report = {
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
    }

    with open(paths["reports_dir"] / "00_merge_report.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(merge_report), f, indent=2, ensure_ascii=False)

    return merged_train, merged_test

# 6. FEATURES CATBOOST

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

    cols_to_exclude_present_train = [c for c in cfg.cols_to_exclude if c in merged_train.columns]
    cols_to_exclude_present_test = [c for c in cfg.cols_to_exclude if c in merged_test.columns]
    cols_to_exclude_present = sorted(set(cols_to_exclude_present_train + cols_to_exclude_present_test))

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

    object_cols = X_train_full.select_dtypes(include=["object", "category"]).columns.tolist()

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
        "split",
        "split_txt",
        "nights_range_is_incoherent",
        "has_reviews",
        "nb_avis_textuels_bert",
        "bert_stars_moyen",
        "text_pred_oof",
        "text_pred_test",
    ]

    for forbidden in forbidden_features:
        assert forbidden not in feature_names, f"FUITE OU COLONNE INTERDITE DANS X : {forbidden}"

    assert "text_pred_final" in feature_names, "text_pred_final absent des features."
    assert "name_n_words" in feature_names, "name_n_words absent des features."
    assert "description_n_words" in feature_names, "description_n_words absent des features."
    assert "description_is_missing" in feature_names, "description_is_missing absent des features."
    assert "txt_e5_000" in feature_names, "txt_e5_000 absent des features."
    assert "txt_e5_099" in feature_names, "txt_e5_099 absent des features."

    txt_cols = [c for c in feature_names if c.startswith("txt_e5_")]
    assert len(txt_cols) == 100, f"Nombre de composantes PCA texte inattendu : {len(txt_cols)} au lieu de 100."

    assert list(X_train_full.columns) == list(X_test.columns), "Colonnes train/test non alignées."

    print("Train features :", X_train_full.shape)
    print("Test features  :", X_test.shape)
    print("Nombre variables :", len(feature_names))
    print("Catégorielles :", cat_features)
    print("Colonnes exclues présentes :", cols_to_exclude_present)
    print("Colonnes constantes supprimées :", constant_cols)

    print("\nContrôle fuite :")
    for c in forbidden_features:
        print(f"{c} dans X :", c in X_train_full.columns)
    print("text_pred_final dans X :", "text_pred_final" in X_train_full.columns)
    print("Nombre de colonnes txt_e5_* :", len(txt_cols))

    pd.DataFrame({"feature": feature_names}).to_csv(
        paths["reports_dir"] / "01_features_utilisees_fusion.csv",
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

# 7. PONDÉRATION MANUAL_AGGRESSIVE

def build_price_segments(y_price, cfg: Config):
    return pd.cut(
        y_price,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    )


def normalize_and_clip_weights(weights, clip_min, clip_max):
    w = pd.Series(weights, dtype=float).copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w = w.clip(lower=clip_min, upper=clip_max)

    if w.mean() <= 0 or np.isnan(w.mean()):
        return pd.Series(np.ones(len(w)), index=w.index, dtype=float)

    return (w / w.mean()).astype(float)


def make_manual_aggressive_weights(y_real, cfg: Config):
    y_real = pd.Series(y_real).reset_index(drop=True).astype(float)

    segments = pd.cut(
        y_real,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    ).astype(str)

    mapping = dict(zip(cfg.price_segment_labels, cfg.manual_price_weights))
    weights = segments.map(mapping).astype(float)

    return normalize_and_clip_weights(
        weights,
        cfg.weight_clip_min,
        cfg.weight_clip_max,
    )


def compute_weight_diagnostics(y_real, weights, cfg: Config, paths: Dict[str, Path], split_name: str):
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
            "segment": seg,
            "n": int(len(g)),
            "price_mean": float(g["y_real"].mean()) if len(g) else np.nan,
            "price_median": float(g["y_real"].median()) if len(g) else np.nan,
            "weight_mean": float(g["weight"].mean()) if len(g) else np.nan,
            "weight_min": float(g["weight"].min()) if len(g) else np.nan,
            "weight_max": float(g["weight"].max()) if len(g) else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(
        paths["reports_dir"] / f"05_weight_diagnostics_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out

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
        "Abs_Error_P90": float(np.percentile(abs_errors, 90)),
        "Abs_Error_P95": float(np.percentile(abs_errors, 95)),
    }


def make_predictions_df(ids, split_name, y_true_log, pred_log, y_true_price):
    pred_price = np.maximum(np.expm1(pred_log), 0)

    pred_df = pd.DataFrame({
        "listing_id_clean": ids.values,
        "split": split_name,
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

# 9. CATBOOST

def make_catboost_params(cfg: Config, seed: int, iterations_override: Optional[int] = None, with_early_stopping: bool = True):
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


def fit_catboost_with_validation(
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    w_val,
    cat_features,
    cfg: Config,
):
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features,
        weight=w_train.values,
    )

    if cfg.weight_validation_pool:
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_features,
            weight=w_val.values,
        )
    else:
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_features,
        )

    model = CatBoostRegressor(
        **make_catboost_params(
            cfg=cfg,
            seed=cfg.random_state + 2026,
            with_early_stopping=True,
        )
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=cfg.verbose_eval,
    )

    return model


def fit_catboost_production(
    X_train_full,
    y_train_full,
    y_train_real_full,
    cat_features,
    cfg: Config,
    tree_count: int,
):
    w_full = make_manual_aggressive_weights(y_train_real_full, cfg)

    train_pool = Pool(
        data=X_train_full,
        label=y_train_full,
        cat_features=cat_features,
        weight=w_full.values,
    )

    model = CatBoostRegressor(
        **make_catboost_params(
            cfg=cfg,
            seed=cfg.random_state + 3030,
            iterations_override=max(1, int(tree_count)),
            with_early_stopping=False,
        )
    )

    model.fit(train_pool, verbose=cfg.verbose_eval)

    return model

# 10. ENTRAÎNEMENT ET ÉVALUATION

def train_and_evaluate(data: Dict[str, Any], cfg: Config, paths: Dict[str, Path]):
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

    bins = make_regression_strat_bins(
        y=y_train_log,
        n_splits=2,
        max_bins=cfg.strat_bins,
    )

    idx_all = np.arange(len(X_train_full))

    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=cfg.validation_size,
        random_state=cfg.random_state + 999,
        stratify=bins,
    )

    X_train = X_train_full.iloc[idx_train].reset_index(drop=True)
    X_val = X_train_full.iloc[idx_val].reset_index(drop=True)

    y_train_log_part = y_train_log.iloc[idx_train].reset_index(drop=True)
    y_val_log = y_train_log.iloc[idx_val].reset_index(drop=True)

    y_train_real_part = y_train_real.iloc[idx_train].reset_index(drop=True)
    y_val_real = y_train_real.iloc[idx_val].reset_index(drop=True)

    ids_train_part = ids_train.iloc[idx_train].reset_index(drop=True)
    ids_val = ids_train.iloc[idx_val].reset_index(drop=True)

    w_train = make_manual_aggressive_weights(y_train_real_part, cfg)
    w_val = make_manual_aggressive_weights(y_val_real, cfg)

    compute_weight_diagnostics(
        y_real=y_train_real_part,
        weights=w_train,
        cfg=cfg,
        paths=paths,
        split_name="internal_train",
    )

    compute_weight_diagnostics(
        y_real=y_val_real,
        weights=w_val,
        cfg=cfg,
        paths=paths,
        split_name="internal_validation",
    )

    print("\n================ ENTRAÎNEMENT OFFICIEL CATBOOST ================")
    print("Candidat : manual_aggressive__C05_premium_oriented")
    print("Train interne :", X_train.shape)
    print("Validation interne :", X_val.shape)
    print("Test final :", X_test.shape)

    eval_model = fit_catboost_with_validation(
        X_train=X_train,
        y_train=y_train_log_part,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val_log,
        w_val=w_val,
        cat_features=cat_features,
        cfg=cfg,
    )

    eval_model_path = paths["models_dir"] / "catboost_eval_manual_aggressive_c05.cbm"
    eval_model.save_model(str(eval_model_path))

    best_iteration = eval_model.get_best_iteration()
    tree_count = int(eval_model.tree_count_)

    print("Modèle officiel sauvegardé :", eval_model_path)
    print("Best iteration :", best_iteration)
    print("Tree count :", tree_count)

    preds = {}
    metrics_rows = []
    segment_rows = []

    split_items = [
        ("internal_train", X_train, y_train_log_part, y_train_real_part, ids_train_part),
        ("internal_validation", X_val, y_val_log, y_val_real, ids_val),
        ("test_final", X_test, y_test_log, y_test_real, ids_test),
    ]

    for split_name, X_part, y_log_part, y_real_part, ids_part in split_items:
        pool = Pool(data=X_part, label=y_log_part, cat_features=cat_features)
        pred_log = eval_model.predict(pool)

        pred_df = make_predictions_df(
            ids=ids_part,
            split_name=split_name,
            y_true_log=y_log_part,
            pred_log=pred_log,
            y_true_price=y_real_part,
        )

        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)

        pred_path = paths["predictions_dir"] / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        preds[split_name] = pred_df

        m_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        m_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])

        for metric, value in m_log.items():
            metrics_rows.append({
                "model": "eval_model_with_internal_validation",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "split": split_name,
                "scale": "log_price",
                "metric": metric,
                "value": value,
            })

        for metric, value in m_price.items():
            metrics_rows.append({
                "model": "eval_model_with_internal_validation",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "split": split_name,
                "scale": "price_euros",
                "metric": metric,
                "value": value,
            })

        seg = compute_segment_metrics(pred_df, cfg, split_name)
        segment_rows.append(seg)

    metrics_df = pd.DataFrame(metrics_rows)
    segments_df = pd.concat(segment_rows, ignore_index=True)

    metrics_df.to_csv(
        paths["reports_dir"] / "20_metrics_train_val_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segments_df.to_csv(
        paths["reports_dir"] / "21_segments_train_val_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = metrics_df.pivot_table(
        index=["scale", "metric"],
        columns="split",
        values="value",
    ).reset_index()

    pivot.to_csv(
        paths["reports_dir"] / "22_metrics_pivot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n================ MÉTRIQUES OFFICIELLES ================")
    print(pivot.to_string(index=False))

    save_feature_importance(eval_model, feature_names, paths)
    plot_final_outputs(preds, segments_df, paths)
    write_final_report(eval_model, metrics_df, segments_df, cfg, paths)

    if cfg.compute_shap_final:
        save_shap_final(eval_model, X_train, y_train_log_part, cat_features, cfg, paths)

    print("\n================ ENTRAÎNEMENT MODÈLE PRODUCTION ================")
    print("Ce modèle utilise tout le train avec tree_count fixé par le modèle officiel.")
    print("Il est sauvegardé pour réutilisation, mais les métriques officielles restent celles du modèle avec validation interne.")

    production_model = fit_catboost_production(
        X_train_full=X_train_full,
        y_train_full=y_train_log,
        y_train_real_full=y_train_real,
        cat_features=cat_features,
        cfg=cfg,
        tree_count=tree_count,
    )

    production_model_path = paths["models_dir"] / "catboost_production_full_train_manual_aggressive_c05.cbm"
    production_model.save_model(str(production_model_path))

    print("Modèle production sauvegardé :", production_model_path)

    config = {
        "candidate": "manual_aggressive__C05_premium_oriented",
        "config": asdict(cfg),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "constant_cols": data["constant_cols"],
        "excluded_cols_present": data["cols_to_exclude_present"],
        "official_eval_model": {
            "model_path": str(eval_model_path),
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "tree_count": tree_count,
            "official_metrics": "20_metrics_train_val_test.csv",
        },
        "production_model": {
            "model_path": str(production_model_path),
            "iterations": tree_count,
            "note": "Entraîné sur tout le train. Ne sert pas aux métriques officielles du mémoire.",
        },
        "important": "Le test final ne sert ni à entraîner, ni à faire l'early stopping, ni à choisir les hyperparamètres.",
    }

    with open(paths["models_dir"] / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(config), f, indent=2, ensure_ascii=False)

# 11. IMPORTANCE, SHAP, GRAPHIQUES

def save_feature_importance(model, feature_names, paths: Dict[str, Path]):
    importance = model.get_feature_importance(type="FeatureImportance")

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        paths["reports_dir"] / "23_feature_importance_catboost.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = fi.head(30).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance CatBoost")
    plt.title("Fusion tabulaire + texte - Top 30 variables")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "20_feature_importance_top30.png", dpi=160)
    plt.close()


def save_shap_final(model, X_train, y_train_log, cat_features, cfg: Config, paths: Dict[str, Path]):
    sample_size = min(cfg.shap_max_rows_final, len(X_train))

    X_sample = X_train.sample(sample_size, random_state=cfg.random_state)
    y_sample = y_train_log.loc[X_sample.index]

    shap_pool = Pool(data=X_sample, label=y_sample, cat_features=cat_features)

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
    plt.title("Fusion tabulaire + texte - Top 30 SHAP")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "21_shap_top30.png", dpi=160)
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
    plt.title("Test final - Prix prédit vs réel\nFusion tabulaire + texte")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "22_test_pred_vs_true.png", dpi=160)
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
    plt.savefig(paths["plots_dir"] / "23_test_residuals.png", dpi=160)
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
    plt.savefig(paths["plots_dir"] / "24_test_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_test["segment"].astype(str), seg_test["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("Test final - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "25_test_bias_by_segment.png", dpi=160)
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


def write_final_report(model, metrics_df, segments_df, cfg: Config, paths: Dict[str, Path]):
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

    report_path = paths["reports_dir"] / "30_resume_final_fusion_catboost.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Résumé final - Fusion tabulaire + texte avec CatBoost\n")
        f.write("=" * 90 + "\n\n")

        f.write("Candidat retenu\n")
        f.write("-" * 90 + "\n")
        f.write("manual_aggressive__C05_premium_oriented\n\n")

        f.write("Protocole\n")
        f.write("-" * 90 + "\n")
        f.write("Le split train/test vient de la branche texte via split_listing_ids.csv.\n")
        f.write("Aucun nouveau test split aléatoire n'a été créé.\n")
        f.write("La validation interne sert uniquement à l'early stopping.\n")
        f.write("Les poids manual_aggressive sont calculés uniquement sur le train interne.\n")
        f.write("Les colonnes price, log_price, les identifiants et les variables interdites sont exclues des features.\n")
        f.write("La cible est log_price.\n")
        f.write("Un modèle production est aussi entraîné sur tout le train avec tree_count_ fixé, mais il ne sert pas aux métriques officielles.\n\n")

        f.write("Configuration CatBoost\n")
        f.write("-" * 90 + "\n")
        f.write(f"iterations : {cfg.iterations}\n")
        f.write(f"learning_rate : {cfg.learning_rate}\n")
        f.write(f"depth : {cfg.depth}\n")
        f.write(f"l2_leaf_reg : {cfg.l2_leaf_reg}\n")
        f.write(f"random_strength : {cfg.random_strength}\n")
        f.write(f"bagging_temperature : {cfg.bagging_temperature}\n")
        f.write(f"rsm : {cfg.rsm}\n")
        f.write(f"early_stopping_rounds : {cfg.early_stopping_rounds}\n")
        f.write(f"task_type : {cfg.task_type}\n")
        f.write(f"best_iteration : {int(model.get_best_iteration())}\n")
        f.write(f"tree_count : {int(model.tree_count_)}\n\n")

        f.write("Pondération manual_aggressive\n")
        f.write("-" * 90 + "\n")
        for label, weight in zip(cfg.price_segment_labels, cfg.manual_price_weights):
            f.write(f"{label} : {weight}\n")
        f.write(f"clip_min : {cfg.weight_clip_min}\n")
        f.write(f"clip_max : {cfg.weight_clip_max}\n\n")

        f.write("Métriques test final - log_price\n")
        f.write("-" * 90 + "\n")
        f.write(f"MAE log : {test_log_mae:.4f}\n")
        f.write(f"RMSE log : {test_log_rmse:.4f}\n")
        f.write(f"R2 log : {test_log_r2:.4f}\n\n")

        f.write("Métriques test final - price_euros\n")
        f.write("-" * 90 + "\n")
        f.write(f"MAE : {test_mae:.4f} €\n")
        f.write(f"RMSE : {test_rmse:.4f} €\n")
        f.write(f"R2 : {test_r2:.4f}\n")
        f.write(f"Biais moyen : {test_bias:.4f} €\n\n")

        f.write("Segments test final\n")
        f.write("-" * 90 + "\n")

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
        f.write("-" * 90 + "\n")
        f.write(
            "Ce modèle fusionne les variables tabulaires avec les sorties de la branche texte. "
            "La variable text_pred_final apporte une prédiction texte-only issue d'un stacking propre. "
            "Les composantes txt_e5_000 à txt_e5_099 apportent une représentation sémantique réduite par PCA. "
            "Les performances doivent être comparées au tabulaire seul et au texte seul, en particulier par segments de prix.\n"
        )


def main():
    paths = get_paths(CFG)

    print("\n" + "=" * 100)
    print("FUSION TABULAIRE + TEXTE - manual_aggressive__C05_premium_oriented")
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
                "config": asdict(CFG),
                "important": "Fusion tabulaire + texte avec split texte existant. Test final isolé.",
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    train_and_evaluate(
        data=data,
        cfg=CFG,
        paths=paths,
    )

    print("\n" + "=" * 100)
    print("FIN - FUSION TABULAIRE + TEXTE - manual_aggressive__C05_premium_oriented")
    print("=" * 100)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])
    print("\nFichiers importants à m'envoyer pour analyse :")
    print(paths["reports_dir"] / "00_merge_report.json")
    print(paths["reports_dir"] / "01_features_utilisees_fusion.csv")
    print(paths["reports_dir"] / "02_cat_features_fusion.csv")
    print(paths["reports_dir"] / "03_colonnes_exclues.csv")
    print(paths["reports_dir"] / "20_metrics_train_val_test.csv")
    print(paths["reports_dir"] / "21_segments_train_val_test.csv")
    print(paths["reports_dir"] / "22_metrics_pivot.csv")
    print(paths["reports_dir"] / "23_feature_importance_catboost.csv")
    print(paths["reports_dir"] / "30_resume_final_fusion_catboost.txt")
    print(paths["predictions_dir"] / "predictions_test_final.csv")


if __name__ == "__main__":
    main()
