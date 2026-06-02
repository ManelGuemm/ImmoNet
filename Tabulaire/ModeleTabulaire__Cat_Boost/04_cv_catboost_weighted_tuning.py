# -*- coding: utf-8 -*-
"""
04_cv_catboost_weighted_tuning.py

Objectif :
- tester plusieurs hyperparamètres CatBoost 
- tester plusieurs stratégies de pondération 
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
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

    input_csv: str = "airbnb_tabulaire_final_catboost_rates_corriges.csv"
    input_xlsx: str = "airbnb_tabulaire_final_catboost_rates_corriges.xlsx"

    target_log: str = "log_price"
    target_real: str = "price"

    cols_to_exclude: Tuple[str, ...] = (
        "id_clean",
        "price",
        "log_price",
        "nights_range_is_incoherent",
    )

    cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # Test final jamais utilisé pour choisir les hyperparamètres
    final_test_size: float = 0.15

    # CV sur le dev set
    n_splits: int = 5
    strat_bins: int = 20
    inner_validation_size: float = 0.15

    # Validation finale pour early stopping du meilleur modèle
    final_validation_size: float = 0.15

    # Validation non pondérée : choix plus neutre et plus comparable
    weight_validation_pool: bool = False

    # Segments métier
    price_segment_bins: Tuple[float, ...] = (0, 100, 200, 400, 800, np.inf)
    price_segment_labels: Tuple[str, ...] = (
        "< 100 €",
        "100-200 €",
        "200-400 €",
        "400-800 €",
        "> 800 €",
    )

    # CatBoost commun
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    allow_writing_files: bool = False
    task_type: str = "CPU"
    thread_count: int = -1
    used_ram_limit: Optional[str] = "8gb"
    verbose_eval: int = 200

    # Interprétabilité finale
    compute_shap_final: bool = True
    shap_max_rows_final: int = 3000
    shap_top_n: int = 30


@dataclass
class WeightExperiment:
    name: str
    strategy: str
    manual_price_weights: Tuple[float, ...] = (1.0, 1.0, 1.1, 1.5, 2.5)
    weight_power: float = 1.0
    weight_clip_min: float = 0.5
    weight_clip_max: float = 5.0


@dataclass
class HyperConfig:
    name: str
    description: str
    params: Dict[str, Any]


CFG = Config()

# 2. STRATÉGIES DE PONDÉRATION
# 3 stratégies x 5 configs x 5 folds = 75 entraînements.

WEIGHT_EXPERIMENTS = [
    WeightExperiment(
        name="manual_moderate",
        strategy="manual_price_segments",
        manual_price_weights=(1.00, 1.00, 1.10, 1.50, 2.50),
        weight_clip_min=0.5,
        weight_clip_max=5.0,
    ),
    WeightExperiment(
        name="manual_aggressive",
        strategy="manual_price_segments",
        manual_price_weights=(1.00, 1.00, 1.15, 2.00, 3.50),
        weight_clip_min=0.5,
        weight_clip_max=6.0,
    ),
    WeightExperiment(
        name="sqrt_price",
        strategy="sqrt_price",
        weight_power=0.5,
        weight_clip_min=0.5,
        weight_clip_max=5.0,
    ),
]


# 3. CONFIGURATIONS CATBOOST À TESTER
# - référence
# - régularisée
# - peu profonde
# - plus profonde
# - orientée premium

HYPER_CONFIGS = [
    HyperConfig(
        name="C01_reference",
        description="Configuration proche du CatBoost déjà utilisé.",
        params={
            "iterations": 3500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "rsm": 0.90,
            "border_count": 254,
            "leaf_estimation_iterations": 10,
            "early_stopping_rounds": 150,
        },
    ),
    HyperConfig(
        name="C02_balanced_regularized",
        description="Apprentissage plus lent avec régularisation plus forte.",
        params={
            "iterations": 6000,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 20.0,
            "random_strength": 2.0,
            "bagging_temperature": 0.8,
            "rsm": 0.85,
            "border_count": 254,
            "leaf_estimation_iterations": 10,
            "early_stopping_rounds": 250,
        },
    ),
    HyperConfig(
        name="C03_shallow_robust",
        description="Moins profond, plus prudent contre le surapprentissage.",
        params={
            "iterations": 7000,
            "learning_rate": 0.02,
            "depth": 5,
            "l2_leaf_reg": 25.0,
            "random_strength": 2.5,
            "bagging_temperature": 1.2,
            "rsm": 0.80,
            "border_count": 254,
            "leaf_estimation_iterations": 10,
            "early_stopping_rounds": 250,
        },
    ),
    HyperConfig(
        name="C04_deeper_interactions",
        description="Plus profond pour capter davantage d'interactions.",
        params={
            "iterations": 6000,
            "learning_rate": 0.02,
            "depth": 7,
            "l2_leaf_reg": 15.0,
            "random_strength": 1.5,
            "bagging_temperature": 0.8,
            "rsm": 0.90,
            "border_count": 254,
            "leaf_estimation_iterations": 10,
            "early_stopping_rounds": 250,
        },
    ),
    HyperConfig(
        name="C05_premium_oriented",
        description="Plus flexible, orienté correction des logements chers.",
        params={
            "iterations": 8000,
            "learning_rate": 0.015,
            "depth": 7,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.6,
            "rsm": 0.95,
            "border_count": 254,
            "leaf_estimation_iterations": 10,
            "early_stopping_rounds": 300,
        },
    ),
]



def get_project_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    tabulaire_dir = script_dir.parent
    data_dir = tabulaire_dir / "Donnees_Tabulaires"

    output_dir = script_dir / "Resultats_CatBoost_Weighted_Tuning_Propre"
    reports_dir = output_dir / "rapports"
    plots_dir = output_dir / "graphiques"
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "modeles"
    best_dir = output_dir / "best_model"
    shap_dir = output_dir / "shap"

    for p in [
        output_dir,
        reports_dir,
        plots_dir,
        predictions_dir,
        models_dir,
        best_dir,
        shap_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "tabulaire_dir": tabulaire_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "predictions_dir": predictions_dir,
        "models_dir": models_dir,
        "best_dir": best_dir,
        "shap_dir": shap_dir,
    }


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

# 5. CHARGEMENT ET PRÉPARATION

def load_data(paths: Dict[str, Path], cfg: Config) -> Tuple[pd.DataFrame, Path]:
    csv_path = paths["data_dir"] / cfg.input_csv
    xlsx_path = paths["data_dir"] / cfg.input_xlsx

    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"id_clean": str}, low_memory=False)
        input_path = csv_path
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path, dtype={"id_clean": str}, engine="openpyxl")
        input_path = xlsx_path
    else:
        raise FileNotFoundError(
            f"Fichier introuvable :\n- {csv_path}\n- {xlsx_path}"
        )

    print("\n================ CHARGEMENT ================")
    print("Fichier :", input_path)
    print("Dimensions :", df.shape)

    return df, input_path


def prepare_features(
    df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
):
    if cfg.target_log not in df.columns:
        raise ValueError(f"Cible absente : {cfg.target_log}")

    if cfg.target_real not in df.columns:
        raise ValueError(f"Cible réelle absente : {cfg.target_real}")

    ids = (
        df["id_clean"].astype(str)
        if "id_clean" in df.columns
        else pd.Series(df.index.astype(str), index=df.index)
    )

    feature_cols = [c for c in df.columns if c not in cfg.cols_to_exclude]
    X = df[feature_cols].copy()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    y_log = pd.to_numeric(df[cfg.target_log], errors="coerce")
    y_real = pd.to_numeric(df[cfg.target_real], errors="coerce")

    valid_mask = y_log.notna() & y_real.notna()

    X = X.loc[valid_mask].reset_index(drop=True)
    y_log = y_log.loc[valid_mask].reset_index(drop=True)
    y_real = y_real.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)

    cat_features = [c for c in cfg.cat_features if c in X.columns]

    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in object_cols:
        if col not in cat_features:
            cat_features.append(col)

    for col in cat_features:
        X[col] = X[col].fillna("missing").astype(str)

    feature_names = list(X.columns)

    pd.DataFrame({"feature": feature_names}).to_csv(
        paths["reports_dir"] / "00_features_utilisees.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"cat_feature": cat_features}).to_csv(
        paths["reports_dir"] / "00_cat_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"colonne_constante_supprimee": constant_cols}).to_csv(
        paths["reports_dir"] / "00_colonnes_constantes_supprimees.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n================ PRÉPARATION ================")
    print("Lignes utilisées :", len(X))
    print("Variables utilisées :", len(feature_names))
    print("Variables catégorielles :", cat_features)
    print("Colonnes constantes supprimées :", constant_cols)
    print("price dans X :", "price" in X.columns)
    print("log_price dans X :", "log_price" in X.columns)
    print("id_clean dans X :", "id_clean" in X.columns)

    return X, y_log, y_real, ids, cat_features, feature_names, constant_cols

# 6. SPLIT ET STRATIFICATION

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


def make_dev_test_split(X, y_log, y_real, ids, cfg, paths):
    indices = np.arange(len(X))

    bins = make_regression_strat_bins(
        y=y_log,
        n_splits=2,
        max_bins=cfg.strat_bins,
    )

    idx_dev, idx_test = train_test_split(
        indices,
        test_size=cfg.final_test_size,
        random_state=cfg.random_state,
        stratify=bins,
    )

    data = {
        "idx_dev": idx_dev,
        "idx_test": idx_test,

        "X_dev": X.iloc[idx_dev].reset_index(drop=True),
        "X_test": X.iloc[idx_test].reset_index(drop=True),

        "y_dev_log": y_log.iloc[idx_dev].reset_index(drop=True),
        "y_test_log": y_log.iloc[idx_test].reset_index(drop=True),

        "y_dev_real": y_real.iloc[idx_dev].reset_index(drop=True),
        "y_test_real": y_real.iloc[idx_test].reset_index(drop=True),

        "ids_dev": ids.iloc[idx_dev].reset_index(drop=True),
        "ids_test": ids.iloc[idx_test].reset_index(drop=True),
    }

    split_report = pd.DataFrame({
        "split": ["dev_tuning_cv", "test_final"],
        "n_lignes": [len(idx_dev), len(idx_test)],
        "prix_moyen": [data["y_dev_real"].mean(), data["y_test_real"].mean()],
        "prix_median": [data["y_dev_real"].median(), data["y_test_real"].median()],
        "prix_min": [data["y_dev_real"].min(), data["y_test_real"].min()],
        "prix_max": [data["y_dev_real"].max(), data["y_test_real"].max()],
        "log_price_moyen": [data["y_dev_log"].mean(), data["y_test_log"].mean()],
    })

    split_report.to_csv(
        paths["reports_dir"] / "01_split_dev_test_final.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n================ SPLIT FINAL ================")
    print(split_report)

    return data

# 7. MÉTRIQUES

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


def build_price_segments(y_price, cfg):
    return pd.cut(
        y_price,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    )


def make_predictions_df(
    ids,
    row_index,
    y_true_log,
    pred_log,
    y_true_price,
    experiment_name,
    hyper_name,
    fold_name,
):
    pred_price = np.maximum(np.expm1(pred_log), 0)

    pred_df = pd.DataFrame({
        "row_index": row_index,
        "id_clean": ids.values,
        "fold": fold_name,
        "experiment": experiment_name,
        "hyper_config": hyper_name,
        "y_true_log": y_true_log.values,
        "y_pred_log": pred_log,
        "y_true_price": y_true_price.values,
        "y_pred_price": pred_price,
    })

    pred_df["residual_log"] = pred_df["y_pred_log"] - pred_df["y_true_log"]
    pred_df["residual_price"] = pred_df["y_pred_price"] - pred_df["y_true_price"]
    pred_df["abs_error_price"] = np.abs(pred_df["residual_price"])

    return pred_df


def compute_segment_metrics(pred_df, cfg, split_name, experiment_name, hyper_name):
    tmp = pred_df.copy()
    tmp["price_segment"] = build_price_segments(tmp["y_true_price"], cfg).astype(str)

    rows = []

    for seg in cfg.price_segment_labels:
        g = tmp[tmp["price_segment"] == seg]

        metrics = compute_metrics(g["y_true_price"].values, g["y_pred_price"].values)

        rows.append({
            "experiment": experiment_name,
            "hyper_config": hyper_name,
            "split": split_name,
            "segment": seg,
            "price_mean": float(g["y_true_price"].mean()) if len(g) else np.nan,
            "price_median": float(g["y_true_price"].median()) if len(g) else np.nan,
            **metrics,
        })

    return pd.DataFrame(rows)

# 8. PONDÉRATION

def normalize_and_clip_weights(weights, clip_min, clip_max):
    w = pd.Series(weights, dtype=float).copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w = w.clip(lower=clip_min, upper=clip_max)

    if w.mean() <= 0 or np.isnan(w.mean()):
        return pd.Series(np.ones(len(w)), index=w.index, dtype=float)

    return (w / w.mean()).astype(float)


def fit_weight_scheme(y_train_real, cfg, exp):
    y_train_real = pd.Series(y_train_real).reset_index(drop=True).astype(float)

    if exp.strategy == "manual_price_segments":
        return {
            "strategy": "manual_price_segments",
            "price_bins": list(cfg.price_segment_bins),
            "price_labels": list(cfg.price_segment_labels),
            "price_weights": list(exp.manual_price_weights),
            "clip_min": exp.weight_clip_min,
            "clip_max": exp.weight_clip_max,
        }

    if exp.strategy == "sqrt_price":
        median_price = float(y_train_real.median())
        if median_price <= 0 or np.isnan(median_price):
            median_price = 1.0

        return {
            "strategy": "sqrt_price",
            "median_price": median_price,
            "weight_power": exp.weight_power,
            "clip_min": exp.weight_clip_min,
            "clip_max": exp.weight_clip_max,
        }

    raise ValueError(f"Stratégie inconnue : {exp.strategy}")


def apply_weight_scheme(y_real, scheme):
    y_real = pd.Series(y_real).reset_index(drop=True).astype(float)

    if scheme["strategy"] == "manual_price_segments":
        segments = pd.cut(
            y_real,
            bins=scheme["price_bins"],
            labels=scheme["price_labels"],
            include_lowest=True,
            right=False,
        ).astype(str)

        mapping = dict(zip(scheme["price_labels"], scheme["price_weights"]))
        weights = segments.map(mapping).astype(float)

        return normalize_and_clip_weights(
            weights,
            scheme["clip_min"],
            scheme["clip_max"],
        )

    if scheme["strategy"] == "sqrt_price":
        safe_price = y_real.clip(lower=1.0)

        weights = (safe_price / float(scheme["median_price"])) ** float(
            scheme["weight_power"]
        )

        return normalize_and_clip_weights(
            weights,
            scheme["clip_min"],
            scheme["clip_max"],
        )

    raise ValueError(f"Stratégie inconnue : {scheme['strategy']}")


def compute_weight_diagnostics(y_real, weights, cfg, experiment, hyper_config, fold):
    df_w = pd.DataFrame({
        "y_real": pd.Series(y_real).reset_index(drop=True),
        "weight": pd.Series(weights).reset_index(drop=True),
    })

    df_w["segment"] = build_price_segments(df_w["y_real"], cfg).astype(str)

    rows = []

    for seg in cfg.price_segment_labels:
        g = df_w[df_w["segment"] == seg]

        rows.append({
            "experiment": experiment,
            "hyper_config": hyper_config,
            "fold": fold,
            "segment": seg,
            "n": int(len(g)),
            "weight_mean": float(g["weight"].mean()) if len(g) else np.nan,
            "weight_min": float(g["weight"].min()) if len(g) else np.nan,
            "weight_max": float(g["weight"].max()) if len(g) else np.nan,
            "price_mean": float(g["y_real"].mean()) if len(g) else np.nan,
        })

    return pd.DataFrame(rows)

# 9. CATBOOST

def make_catboost_params(cfg, hyper, seed):
    params = dict(hyper.params)

    params.update({
        "loss_function": cfg.loss_function,
        "eval_metric": cfg.eval_metric,
        "random_seed": seed,
        "allow_writing_files": cfg.allow_writing_files,
        "task_type": cfg.task_type,
        "thread_count": cfg.thread_count,
        "verbose": cfg.verbose_eval,
    })

    if cfg.used_ram_limit:
        params["used_ram_limit"] = cfg.used_ram_limit

    return params


def fit_catboost(
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    w_val,
    cat_features,
    cfg,
    hyper,
    seed,
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

    model = CatBoostRegressor(**make_catboost_params(cfg, hyper, seed))

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=cfg.verbose_eval,
    )

    return model

# 10. SCORE DE SÉLECTION

def segment_value(seg_df, segment, metric):
    row = seg_df[seg_df["segment"] == segment]
    if len(row) == 0:
        return np.nan
    return float(row[metric].iloc[0])


def compute_selection_score(global_metrics, segment_df):
    mae = global_metrics["MAE"]
    rmse = global_metrics["RMSE"]
    bias_global = global_metrics["Mean_Error"]

    mae_400_800 = segment_value(segment_df, "400-800 €", "MAE")
    mae_gt_800 = segment_value(segment_df, "> 800 €", "MAE")
    bias_gt_800 = segment_value(segment_df, "> 800 €", "Mean_Error")

    score = (
        1.00 * mae
        + 0.15 * rmse
        + 0.10 * mae_400_800
        + 0.05 * mae_gt_800
        + 0.03 * abs(bias_gt_800)
        + 0.02 * abs(bias_global)
    )

    return float(score)

# 11. CV D'UN CANDIDAT

def evaluate_candidate_cv(data, cat_features, cfg, exp, hyper, paths):
    X_dev = data["X_dev"]
    y_dev_log = data["y_dev_log"]
    y_dev_real = data["y_dev_real"]
    ids_dev = data["ids_dev"]

    candidate = f"{exp.name}__{hyper.name}"

    print("\n" + "=" * 100)
    print("CANDIDAT :", candidate)
    print("Pondération :", exp.name)
    print("Hyperparamètres :", hyper.name)
    print("=" * 100)

    bins = make_regression_strat_bins(
        y=y_dev_log,
        n_splits=cfg.n_splits,
        max_bins=cfg.strat_bins,
    )

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    oof_parts = []
    fold_rows = []
    segment_rows = []
    weight_rows = []

    for fold_id, (idx_train_full, idx_val_outer) in enumerate(skf.split(X_dev, bins), start=1):
        fold_name = f"fold_{fold_id}"
        seed = cfg.random_state + fold_id

        print("\n" + "-" * 100)
        print(candidate, fold_name)
        print("-" * 100)

        X_train_full = X_dev.iloc[idx_train_full].reset_index(drop=True)
        y_train_full_log = y_dev_log.iloc[idx_train_full].reset_index(drop=True)
        y_train_full_real = y_dev_real.iloc[idx_train_full].reset_index(drop=True)

        X_outer_val = X_dev.iloc[idx_val_outer].reset_index(drop=True)
        y_outer_val_log = y_dev_log.iloc[idx_val_outer].reset_index(drop=True)
        y_outer_val_real = y_dev_real.iloc[idx_val_outer].reset_index(drop=True)
        ids_outer_val = ids_dev.iloc[idx_val_outer].reset_index(drop=True)

        inner_bins = make_regression_strat_bins(
            y=y_train_full_log,
            n_splits=2,
            max_bins=max(4, cfg.strat_bins // 2),
        )

        idx_all_inner = np.arange(len(X_train_full))

        idx_train_inner, idx_es_val = train_test_split(
            idx_all_inner,
            test_size=cfg.inner_validation_size,
            random_state=seed,
            stratify=inner_bins,
        )

        X_train = X_train_full.iloc[idx_train_inner].reset_index(drop=True)
        y_train_log = y_train_full_log.iloc[idx_train_inner].reset_index(drop=True)
        y_train_real = y_train_full_real.iloc[idx_train_inner].reset_index(drop=True)

        X_es_val = X_train_full.iloc[idx_es_val].reset_index(drop=True)
        y_es_val_log = y_train_full_log.iloc[idx_es_val].reset_index(drop=True)
        y_es_val_real = y_train_full_real.iloc[idx_es_val].reset_index(drop=True)

        weight_scheme = fit_weight_scheme(y_train_real, cfg, exp)
        w_train = apply_weight_scheme(y_train_real, weight_scheme)
        w_es_val = apply_weight_scheme(y_es_val_real, weight_scheme)

        weight_rows.append(
            compute_weight_diagnostics(
                y_real=y_train_real,
                weights=w_train,
                cfg=cfg,
                experiment=exp.name,
                hyper_config=hyper.name,
                fold=fold_name,
            )
        )

        model = fit_catboost(
            X_train=X_train,
            y_train=y_train_log,
            w_train=w_train,
            X_val=X_es_val,
            y_val=y_es_val_log,
            w_val=w_es_val,
            cat_features=cat_features,
            cfg=cfg,
            hyper=hyper,
            seed=seed,
        )

        val_pool = Pool(
            data=X_outer_val,
            label=y_outer_val_log,
            cat_features=cat_features,
        )

        pred_log = model.predict(val_pool)

        pred_df = make_predictions_df(
            ids=ids_outer_val,
            row_index=idx_val_outer,
            y_true_log=y_outer_val_log,
            pred_log=pred_log,
            y_true_price=y_outer_val_real,
            experiment_name=exp.name,
            hyper_name=hyper.name,
            fold_name=fold_name,
        )

        oof_parts.append(pred_df)

        metrics_price = compute_metrics(
            pred_df["y_true_price"].values,
            pred_df["y_pred_price"].values,
        )

        fold_rows.append({
            "candidate": candidate,
            "experiment": exp.name,
            "hyper_config": hyper.name,
            "fold": fold_name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            **metrics_price,
        })

        seg_df = compute_segment_metrics(
            pred_df=pred_df,
            cfg=cfg,
            split_name="dev_cv_fold",
            experiment_name=exp.name,
            hyper_name=hyper.name,
        )

        seg_df["candidate"] = candidate
        seg_df["fold"] = fold_name
        segment_rows.append(seg_df)

        print(
            f"{fold_name} terminé | "
            f"MAE € = {metrics_price['MAE']:.2f} | "
            f"RMSE € = {metrics_price['RMSE']:.2f} | "
            f"Biais € = {metrics_price['Mean_Error']:.2f} | "
            f"best_iter = {int(model.get_best_iteration())}"
        )

    oof_df = pd.concat(oof_parts, ignore_index=True)
    oof_df["price_segment"] = build_price_segments(oof_df["y_true_price"], cfg).astype(str)

    oof_path = paths["predictions_dir"] / f"oof_{candidate}.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(fold_rows)
    segment_folds_df = pd.concat(segment_rows, ignore_index=True)
    weight_df = pd.concat(weight_rows, ignore_index=True)

    global_metrics = compute_metrics(
        oof_df["y_true_price"].values,
        oof_df["y_pred_price"].values,
    )

    global_metrics_log = compute_metrics(
        oof_df["y_true_log"].values,
        oof_df["y_pred_log"].values,
    )

    segment_oof = compute_segment_metrics(
        pred_df=oof_df,
        cfg=cfg,
        split_name="dev_cv_oof",
        experiment_name=exp.name,
        hyper_name=hyper.name,
    )

    segment_oof["candidate"] = candidate

    selection_score = compute_selection_score(global_metrics, segment_oof)

    result_row = {
        "candidate": candidate,
        "experiment": exp.name,
        "weight_strategy": exp.strategy,
        "hyper_config": hyper.name,
        "description": hyper.description,
        "selection_score": selection_score,

        "cv_MAE": global_metrics["MAE"],
        "cv_RMSE": global_metrics["RMSE"],
        "cv_R2": global_metrics["R2"],
        "cv_MedAE": global_metrics["MedAE"],
        "cv_MAPE_pct": global_metrics["MAPE_pct"],
        "cv_SMAPE_pct": global_metrics["SMAPE_pct"],
        "cv_Mean_Error": global_metrics["Mean_Error"],
        "cv_Underestimation_Rate_pct": global_metrics["Underestimation_Rate_pct"],

        "cv_log_MAE": global_metrics_log["MAE"],
        "cv_log_RMSE": global_metrics_log["RMSE"],
        "cv_log_R2": global_metrics_log["R2"],

        "cv_400_800_MAE": segment_value(segment_oof, "400-800 €", "MAE"),
        "cv_400_800_Mean_Error": segment_value(segment_oof, "400-800 €", "Mean_Error"),
        "cv_gt_800_MAE": segment_value(segment_oof, "> 800 €", "MAE"),
        "cv_gt_800_Mean_Error": segment_value(segment_oof, "> 800 €", "Mean_Error"),

        "mean_best_iteration": fold_df["best_iteration"].mean(),
        "hyper_params_json": json.dumps(to_jsonable(hyper.params), ensure_ascii=False),
        "weight_params_json": json.dumps(to_jsonable(asdict(exp)), ensure_ascii=False),
        "oof_predictions_file": str(oof_path),
    }

    print("\nRésumé candidat :")
    print(
        pd.DataFrame([result_row])[
            [
                "candidate",
                "selection_score",
                "cv_MAE",
                "cv_RMSE",
                "cv_Mean_Error",
                "cv_400_800_MAE",
                "cv_gt_800_MAE",
                "cv_gt_800_Mean_Error",
            ]
        ].to_string(index=False)
    )

    return {
        "candidate": candidate,
        "experiment": exp,
        "hyper": hyper,
        "result_row": result_row,
        "fold_metrics": fold_df,
        "segments_folds": segment_folds_df,
        "segments_oof": segment_oof,
        "weights": weight_df,
        "oof": oof_df,
    }

# 12. TUNING GLOBAL

def run_tuning(data, cat_features, cfg, paths):
    all_results = []
    all_fold_metrics = []
    all_segments_folds = []
    all_segments_oof = []
    all_weights = []
    all_objects = []

    total = len(WEIGHT_EXPERIMENTS) * len(HYPER_CONFIGS)
    k = 0

    for exp in WEIGHT_EXPERIMENTS:
        for hyper in HYPER_CONFIGS:
            k += 1
            print(f"\n\n########## CANDIDAT {k}/{total} ##########")

            res = evaluate_candidate_cv(
                data=data,
                cat_features=cat_features,
                cfg=cfg,
                exp=exp,
                hyper=hyper,
                paths=paths,
            )

            all_objects.append(res)
            all_results.append(res["result_row"])
            all_fold_metrics.append(res["fold_metrics"])
            all_segments_folds.append(res["segments_folds"])
            all_segments_oof.append(res["segments_oof"])
            all_weights.append(res["weights"])

            pd.DataFrame(all_results).sort_values("selection_score").to_csv(
                paths["reports_dir"] / "10_tuning_summary_progressif.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(all_results).sort_values("selection_score").reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    segments_folds = pd.concat(all_segments_folds, ignore_index=True)
    segments_oof = pd.concat(all_segments_oof, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)

    summary.to_csv(paths["reports_dir"] / "10_tuning_summary_cv_dev.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(paths["reports_dir"] / "11_tuning_fold_metrics_price.csv", index=False, encoding="utf-8-sig")
    segments_folds.to_csv(paths["reports_dir"] / "12_tuning_segments_by_fold.csv", index=False, encoding="utf-8-sig")
    segments_oof.to_csv(paths["reports_dir"] / "13_tuning_segments_oof.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(paths["reports_dir"] / "14_tuning_weight_diagnostics.csv", index=False, encoding="utf-8-sig")

    best_candidate = summary.iloc[0]["candidate"]

    best_object = None
    for obj in all_objects:
        if obj["candidate"] == best_candidate:
            best_object = obj
            break

    if best_object is None:
        raise RuntimeError("Meilleur candidat introuvable.")

    with open(paths["reports_dir"] / "15_best_candidate_cv.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary.iloc[0].to_dict()), f, indent=2, ensure_ascii=False)

    plot_tuning_summary(summary, segments_oof, paths)

    print("\n================ TOP 10 CANDIDATS ================")
    print(summary.head(10).to_string(index=False))

    return {
        "summary": summary,
        "best_object": best_object,
        "segments_oof": segments_oof,
    }

# 13. ENTRAÎNEMENT FINAL

def train_final_model(data, cat_features, feature_names, cfg, paths, best_object):
    exp = best_object["experiment"]
    hyper = best_object["hyper"]
    candidate = best_object["candidate"]

    print("\n" + "=" * 100)
    print("ENTRAÎNEMENT FINAL :", candidate)
    print("=" * 100)

    X_dev = data["X_dev"]
    y_dev_log = data["y_dev_log"]
    y_dev_real = data["y_dev_real"]
    ids_dev = data["ids_dev"]

    X_test = data["X_test"]
    y_test_log = data["y_test_log"]
    y_test_real = data["y_test_real"]
    ids_test = data["ids_test"]

    bins = make_regression_strat_bins(
        y=y_dev_log,
        n_splits=2,
        max_bins=cfg.strat_bins,
    )

    idx_all = np.arange(len(X_dev))

    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=cfg.final_validation_size,
        random_state=cfg.random_state + 999,
        stratify=bins,
    )

    X_train = X_dev.iloc[idx_train].reset_index(drop=True)
    y_train_log = y_dev_log.iloc[idx_train].reset_index(drop=True)
    y_train_real = y_dev_real.iloc[idx_train].reset_index(drop=True)
    ids_train = ids_dev.iloc[idx_train].reset_index(drop=True)

    X_val = X_dev.iloc[idx_val].reset_index(drop=True)
    y_val_log = y_dev_log.iloc[idx_val].reset_index(drop=True)
    y_val_real = y_dev_real.iloc[idx_val].reset_index(drop=True)
    ids_val = ids_dev.iloc[idx_val].reset_index(drop=True)

    weight_scheme = fit_weight_scheme(y_train_real, cfg, exp)
    w_train = apply_weight_scheme(y_train_real, weight_scheme)
    w_val = apply_weight_scheme(y_val_real, weight_scheme)

    model = fit_catboost(
        X_train=X_train,
        y_train=y_train_log,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val_log,
        w_val=w_val,
        cat_features=cat_features,
        cfg=cfg,
        hyper=hyper,
        seed=cfg.random_state + 2026,
    )

    model_path = paths["best_dir"] / "best_catboost_weighted_tuned.cbm"
    model.save_model(str(model_path))

    with open(paths["best_dir"] / "best_model_config.json", "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable({
                "candidate": candidate,
                "experiment": asdict(exp),
                "hyper_config": asdict(hyper),
                "weight_scheme_final": weight_scheme,
                "features": feature_names,
                "cat_features": cat_features,
                "best_iteration": int(model.get_best_iteration()),
                "tree_count": int(model.tree_count_),
                "important": "Le test final n'a pas servi au choix des hyperparamètres.",
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame({"feature": feature_names}).to_csv(
        paths["best_dir"] / "features_final_model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_preds = {}
    metrics_rows = []
    segment_rows = []

    split_items = [
        ("final_train", X_train, y_train_log, y_train_real, ids_train, idx_train),
        ("final_validation", X_val, y_val_log, y_val_real, ids_val, idx_val),
        ("test_final", X_test, y_test_log, y_test_real, ids_test, data["idx_test"]),
    ]

    for split_name, X_part, y_log_part, y_real_part, ids_part, row_idx in split_items:
        pool = Pool(data=X_part, label=y_log_part, cat_features=cat_features)
        pred_log = model.predict(pool)

        pred_df = make_predictions_df(
            ids=ids_part,
            row_index=row_idx,
            y_true_log=y_log_part,
            pred_log=pred_log,
            y_true_price=y_real_part,
            experiment_name=exp.name,
            hyper_name=hyper.name,
            fold_name=split_name,
        )

        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)

        pred_df.to_csv(
            paths["predictions_dir"] / f"final_{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_preds[split_name] = pred_df

        m_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        m_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])

        for metric, value in m_log.items():
            metrics_rows.append({
                "candidate": candidate,
                "split": split_name,
                "scale": "log_price",
                "metric": metric,
                "value": value,
            })

        for metric, value in m_price.items():
            metrics_rows.append({
                "candidate": candidate,
                "split": split_name,
                "scale": "price_euros",
                "metric": metric,
                "value": value,
            })

        seg = compute_segment_metrics(
            pred_df=pred_df,
            cfg=cfg,
            split_name=split_name,
            experiment_name=exp.name,
            hyper_name=hyper.name,
        )

        segment_rows.append(seg)

    metrics_df = pd.DataFrame(metrics_rows)
    segments_df = pd.concat(segment_rows, ignore_index=True)

    metrics_df.to_csv(
        paths["reports_dir"] / "20_final_metrics_train_val_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segments_df.to_csv(
        paths["reports_dir"] / "21_final_segments_train_val_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = metrics_df.pivot_table(
        index=["scale", "metric"],
        columns="split",
        values="value",
    ).reset_index()

    pivot.to_csv(
        paths["reports_dir"] / "22_final_metrics_pivot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n================ MÉTRIQUES FINALES ================")
    print(pivot.to_string(index=False))

    save_feature_importance(model, feature_names, paths)
    plot_final_outputs(all_preds, segments_df, paths, candidate)

    if cfg.compute_shap_final:
        save_shap_final(model, X_train, y_train_log, cat_features, cfg, paths)

    write_final_report(candidate, exp, hyper, model, metrics_df, segments_df, cfg, paths)


# 14. GRAPHIQUES ET INTERPRÉTABILITÉ

def plot_tuning_summary(summary, segments_oof, paths):
    top = summary.sort_values("selection_score").head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(top["candidate"], top["selection_score"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Score de sélection CV dev")
    plt.title("Top 20 candidats - score plus bas = meilleur")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "10_tuning_top20_selection_score.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(top["candidate"], top["cv_MAE"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("MAE CV dev (€)")
    plt.title("Top 20 candidats - MAE CV dev")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "11_tuning_top20_mae_cv.png", dpi=160)
    plt.close()


def save_feature_importance(model, feature_names, paths):
    importance = model.get_feature_importance(type="FeatureImportance")

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        paths["reports_dir"] / "23_final_feature_importance_catboost.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = fi.head(30).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance CatBoost")
    plt.title("Modèle final - Top 30 variables")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "20_final_feature_importance_top30.png", dpi=160)
    plt.close()


def save_shap_final(model, X_train, y_train_log, cat_features, cfg, paths):
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
    plt.title("Modèle final - Top 30 SHAP")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "21_final_shap_top30.png", dpi=160)
    plt.close()


def plot_final_outputs(preds, segments_df, paths, candidate):
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
    plt.title(f"Test final - Prix prédit vs réel\n{candidate}")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "22_final_test_pred_vs_true.png", dpi=160)
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
    plt.savefig(paths["plots_dir"] / "23_final_test_residuals.png", dpi=160)
    plt.close()

    seg_test = segments_df[segments_df["split"] == "test_final"].copy()
    order = ["< 100 €", "100-200 €", "200-400 €", "400-800 €", "> 800 €"]
    seg_test["segment"] = pd.Categorical(seg_test["segment"], categories=order, ordered=True)
    seg_test = seg_test.sort_values("segment")

    plt.figure(figsize=(8, 5))
    plt.bar(seg_test["segment"].astype(str), seg_test["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("Test final - MAE par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "24_final_test_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_test["segment"].astype(str), seg_test["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("Test final - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "25_final_test_bias_by_segment.png", dpi=160)
    plt.close()

# 15. RAPPORT FINAL

def get_metric(metrics_df, split, scale, metric):
    row = metrics_df[
        (metrics_df["split"] == split)
        & (metrics_df["scale"] == scale)
        & (metrics_df["metric"] == metric)
    ]

    if len(row) == 0:
        return np.nan

    return float(row["value"].iloc[0])


def write_final_report(candidate, exp, hyper, model, metrics_df, segments_df, cfg, paths):
    test_mae = get_metric(metrics_df, "test_final", "price_euros", "MAE")
    test_rmse = get_metric(metrics_df, "test_final", "price_euros", "RMSE")
    test_r2 = get_metric(metrics_df, "test_final", "price_euros", "R2")
    test_bias = get_metric(metrics_df, "test_final", "price_euros", "Mean_Error")

    seg_test = segments_df[segments_df["split"] == "test_final"].copy()
    order = list(cfg.price_segment_labels)
    seg_test["segment"] = pd.Categorical(seg_test["segment"], categories=order, ordered=True)
    seg_test = seg_test.sort_values("segment")

    report_path = paths["reports_dir"] / "30_resume_final_experience.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Résumé final - CatBoost pondéré avec tuning\n")
        f.write("=" * 90 + "\n\n")

        f.write("Protocole\n")
        f.write("-" * 90 + "\n")
        f.write("Le test final a été isolé avant le tuning.\n")
        f.write("Le choix des hyperparamètres repose uniquement sur la CV du dev set.\n")
        f.write("Le test final est utilisé une seule fois à la fin.\n")
        f.write("Les poids sont calculés uniquement sur les folds d'entraînement.\n\n")

        f.write("Meilleur candidat\n")
        f.write("-" * 90 + "\n")
        f.write(f"Candidat : {candidate}\n")
        f.write(f"Pondération : {exp.name} / {exp.strategy}\n")
        f.write(f"Hyperparamètres : {hyper.name}\n")
        f.write(f"Description : {hyper.description}\n")
        f.write(f"Best iteration finale : {int(model.get_best_iteration())}\n")
        f.write(f"Tree count finale : {int(model.tree_count_)}\n\n")

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
        f.write("Le meilleur modèle n'est pas forcément celui qui minimise seulement la MAE globale.\n")
        f.write("Il faut regarder la MAE globale, le RMSE, le biais moyen et les segments chers.\n")
        f.write("Une amélioration des logements chers est utile seulement si elle ne dégrade pas trop les petits prix.\n")


def main():
    paths = get_project_paths()

    print("\n" + "=" * 100)
    print("CATBOOST WEIGHTED TUNING PROPRE")
    print("=" * 100)
    print("Dossier données :", paths["data_dir"])
    print("Dossier résultats :", paths["output_dir"])

    df, input_path = load_data(paths, CFG)

    X, y_log, y_real, ids, cat_features, feature_names, constant_cols = prepare_features(
        df=df,
        cfg=CFG,
        paths=paths,
    )

    run_info = {
        "input_path": str(input_path),
        "config": asdict(CFG),
        "weight_experiments": [asdict(e) for e in WEIGHT_EXPERIMENTS],
        "hyper_configs": [asdict(h) for h in HYPER_CONFIGS],
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "constant_cols_removed": constant_cols,
        "important": "Le test final ne sert pas au choix des hyperparamètres.",
    }

    with open(paths["reports_dir"] / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_info), f, indent=2, ensure_ascii=False)

    data = make_dev_test_split(
        X=X,
        y_log=y_log,
        y_real=y_real,
        ids=ids,
        cfg=CFG,
        paths=paths,
    )

    tuning = run_tuning(
        data=data,
        cat_features=cat_features,
        cfg=CFG,
        paths=paths,
    )

    train_final_model(
        data=data,
        cat_features=cat_features,
        feature_names=feature_names,
        cfg=CFG,
        paths=paths,
        best_object=tuning["best_object"],
    )

    print("\n" + "=" * 100)
    print("FIN - CATBOOST WEIGHTED TUNING PROPRE")
    print("=" * 100)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])
    print("\nFichiers importants à m'envoyer pour analyse :")
    print(paths["reports_dir"] / "10_tuning_summary_cv_dev.csv")
    print(paths["reports_dir"] / "13_tuning_segments_oof.csv")
    print(paths["reports_dir"] / "15_best_candidate_cv.json")
    print(paths["reports_dir"] / "20_final_metrics_train_val_test.csv")
    print(paths["reports_dir"] / "21_final_segments_train_val_test.csv")
    print(paths["reports_dir"] / "22_final_metrics_pivot.csv")
    print(paths["reports_dir"] / "30_resume_final_experience.txt")


if __name__ == "__main__":
    main()