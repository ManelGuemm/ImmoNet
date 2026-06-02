# -*- coding: utf-8 -*-
"""
Validation croisée 5-fold pour CatBoost pondéré - Régression Airbnb.

Objectif :
- Comparer plusieurs stratégies de pondération pour mieux prédire
  les logements chers.
- Garder le pipeline propre de la baseline :
    * cible : log_price
    * interprétation : price en euros après expm1
    * CV 5-fold
    * split interne train/validation pour early stopping
    * prédictions OOF
    * métriques globales
    * métriques par segments de prix
    * comparaison avec la baseline CV
    * SHAP optionnel
    * graphiques

Stratégies testées :
1. manual_moderate
   Equivalent de MODERATE_BINS.
   Pondération modérée des logements chers.

2. manual_aggressive
   Equivalent de STRONG_BINS.
   Pondération plus forte des logements chers.

3. sqrt_price
   Equivalent de SQRT_PRICE.
   Poids continu qui augmente progressivement avec le prix.

4. density_equal_width
   Pondération selon la rareté des zones de prix.


Pré-requis :
python -m pip install pandas numpy scikit-learn matplotlib catboost openpyxl
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")


# 1. Configuration générale

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

    n_splits: int = 5
    outer_n_bins: int = 20
    inner_validation_size: float = 0.15

    iterations: int = 3000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 10.0
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    early_stopping_rounds: int = 150
    verbose_eval: int = 200
    use_best_model: bool = True
    allow_writing_files: bool = False
    task_type: str = "CPU"
    thread_count: int = -1
    used_ram_limit: Optional[str] = "8gb"

    price_segment_bins: Tuple[float, ...] = (0, 100, 200, 400, 800, np.inf)
    price_segment_labels: Tuple[str, ...] = (
        "< 100 €",
        "100-200 €",
        "200-400 €",
        "400-800 €",
        "> 800 €",
    )

    compute_shap: bool = True
    shap_max_rows_per_fold: int = 1000
    shap_top_n: int = 30

    # False = validation interne non pondérée.
    # C'est mieux pour comparer proprement avec la baseline.
    weight_validation_pool: bool = False


@dataclass
class WeightExperiment:
    name: str
    strategy: str

    manual_price_weights: Tuple[float, ...] = (
        1.0,
        1.0,
        1.1,
        1.5,
        2.5,
    )

    weight_n_bins: int = 20
    weight_power: float = 1.0

    weight_clip_min: float = 0.5
    weight_clip_max: float = 5.0


CFG = Config()

# 2. Expériences à lancer
# 4 stratégies actives = 4 x 5 folds = 20 entraînements CatBoost.

WEIGHT_EXPERIMENTS = [
    WeightExperiment(
        name="manual_moderate",
        strategy="manual_price_segments",
        manual_price_weights=(
            1.00,   # < 100 €
            1.00,   # 100-200 €
            1.10,   # 200-400 €
            1.50,   # 400-800 €
            2.50,   # > 800 €
        ),
        weight_clip_min=0.5,
        weight_clip_max=5.0,
    ),
    WeightExperiment(
        name="manual_aggressive",
        strategy="manual_price_segments",
        manual_price_weights=(
            1.00,   # < 100 €
            1.00,   # 100-200 €
            1.15,   # 200-400 €
            2.00,   # 400-800 €
            3.50,   # > 800 €
        ),
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
    WeightExperiment(
        name="density_equal_width",
        strategy="density_equal_width",
        weight_n_bins=20,
        weight_power=1.0,
        weight_clip_min=0.5,
        weight_clip_max=5.0,
    ),


]

# 3. Chemins

def get_project_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    tabulaire_dir = script_dir.parent
    data_dir = tabulaire_dir / "Donnees_Tabulaires"

    output_dir = script_dir / "Resultats_CV_Weighted_CatBoost"
    common_reports_dir = output_dir / "rapports_comparaison_globale"
    common_plots_dir = output_dir / "graphiques_comparaison_globale"

    baseline_reports_dir = script_dir / "Resultats_CV_Baseline_CatBoost" / "rapports"

    for p in [output_dir, common_reports_dir, common_plots_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "tabulaire_dir": tabulaire_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "common_reports_dir": common_reports_dir,
        "common_plots_dir": common_plots_dir,
        "baseline_reports_dir": baseline_reports_dir,
    }


def get_experiment_paths(base_paths: Dict[str, Path], exp_name: str) -> Dict[str, Path]:
    exp_dir = base_paths["output_dir"] / exp_name

    reports_dir = exp_dir / "rapports"
    plots_dir = exp_dir / "graphiques"
    models_dir = exp_dir / "modeles"
    predictions_dir = exp_dir / "predictions"
    shap_dir = exp_dir / "shap"

    for p in [exp_dir, reports_dir, plots_dir, models_dir, predictions_dir, shap_dir]:
        p.mkdir(parents=True, exist_ok=True)

    paths = dict(base_paths)
    paths.update({
        "exp_dir": exp_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "models_dir": models_dir,
        "predictions_dir": predictions_dir,
        "shap_dir": shap_dir,
    })

    return paths

# 4. Utilitaires JSON

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        if np.isnan(val):
            return None
        return val

    if isinstance(obj, float):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if np.isnan(obj):
            return None
        return obj

    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

# 5. Chargement et préparation des données

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

    print("\nFichier chargé :")
    print(input_path)
    print("Dimensions :", df.shape)

    return df, input_path


def prepare_features(
    df: pd.DataFrame,
    cfg: Config
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str], List[str], List[str]]:

    if cfg.target_log not in df.columns:
        raise ValueError(f"Cible absente : {cfg.target_log}")

    if cfg.target_real not in df.columns:
        raise ValueError(f"Cible réelle absente : {cfg.target_real}")

    ids = (
        df["id_clean"].astype(str)
        if "id_clean" in df.columns
        else pd.Series(df.index.astype(str), index=df.index)
    )

    feature_cols = [
        c for c in df.columns
        if c not in cfg.cols_to_exclude
    ]

    X = df[feature_cols].copy()

    dropped_constant_cols = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    if dropped_constant_cols:
        X = X.drop(columns=dropped_constant_cols)
        feature_cols = [c for c in feature_cols if c not in dropped_constant_cols]

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

    print("\nPréparation terminée :")
    print("Lignes utilisées :", len(X))
    print("Variables utilisées :", len(feature_names))
    print("Catégorielles CatBoost :", cat_features)
    print("Colonnes constantes supprimées :", dropped_constant_cols)

    return X, y_log, y_real, ids, cat_features, feature_names, dropped_constant_cols

# 6. Métriques

def safe_mape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def build_price_segments(y_price: pd.Series, cfg: Config) -> pd.Categorical:
    return pd.cut(
        y_price,
        bins=list(cfg.price_segment_bins),
        labels=list(cfg.price_segment_labels),
        include_lowest=True,
        right=False,
    )


def compute_segment_metrics(
    df_pred: pd.DataFrame,
    cfg: Config,
    fold_name: str
) -> pd.DataFrame:
    rows = []
    df_local = df_pred.copy()
    df_local["price_segment"] = build_price_segments(df_local["y_true_price"], cfg)

    for seg_label in cfg.price_segment_labels:
        group = df_local[df_local["price_segment"].astype(str) == seg_label]

        metrics = compute_metrics(
            group["y_true_price"].values,
            group["y_pred_price"].values
        )

        rows.append({
            "fold": fold_name,
            "segment": seg_label,
            "price_min": float(group["y_true_price"].min()) if len(group) > 0 else np.nan,
            "price_max": float(group["y_true_price"].max()) if len(group) > 0 else np.nan,
            "price_mean": float(group["y_true_price"].mean()) if len(group) > 0 else np.nan,
            "price_median": float(group["y_true_price"].median()) if len(group) > 0 else np.nan,
            **metrics
        })

    return pd.DataFrame(rows)

# 7. Stratification pour régression

def make_regression_strat_bins(
    y: pd.Series,
    n_splits: int,
    max_bins: int = 20
) -> pd.Series:
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

    fallback = pd.Series(fallback, index=y.index).fillna(0).astype(int)
    return fallback

# 8. Pondération

def normalize_and_clip_weights(
    weights: pd.Series,
    clip_min: float,
    clip_max: float
) -> pd.Series:
    w = pd.Series(weights, dtype=float).copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    w = w.clip(lower=clip_min, upper=clip_max)

    mean_w = w.mean()

    if mean_w <= 0 or np.isnan(mean_w):
        w = pd.Series(np.ones(len(w)), index=w.index, dtype=float)
    else:
        w = w / mean_w

    return w.astype(float)


def fit_manual_price_segments_scheme(
    cfg: Config,
    exp: WeightExperiment
) -> Dict:
    return {
        "strategy": "manual_price_segments",
        "price_bins": list(cfg.price_segment_bins),
        "price_labels": list(cfg.price_segment_labels),
        "price_weights": list(exp.manual_price_weights),
        "clip_min": exp.weight_clip_min,
        "clip_max": exp.weight_clip_max,
    }


def apply_manual_price_segments_scheme(
    y_real: pd.Series,
    scheme: Dict
) -> pd.Series:
    y_real = pd.Series(y_real).reset_index(drop=True)

    segments = pd.cut(
        y_real,
        bins=scheme["price_bins"],
        labels=scheme["price_labels"],
        include_lowest=True,
        right=False,
    ).astype(str)

    mapping = dict(zip(scheme["price_labels"], scheme["price_weights"]))

    weights = segments.map(mapping).astype(float)

    weights = normalize_and_clip_weights(
        weights=weights,
        clip_min=scheme["clip_min"],
        clip_max=scheme["clip_max"],
    )

    return weights


def fit_sqrt_price_scheme(
    y_real_train: pd.Series,
    exp: WeightExperiment
) -> Dict:


    y_real_train = pd.Series(y_real_train).reset_index(drop=True).astype(float)

    median_price = float(y_real_train.median())

    if median_price <= 0 or np.isnan(median_price):
        median_price = 1.0

    return {
        "strategy": "sqrt_price",
        "median_price": median_price,
        "weight_power": exp.weight_power,
        "clip_min": exp.weight_clip_min,
        "clip_max": exp.weight_clip_max,
    }


def apply_sqrt_price_scheme(
    y_real: pd.Series,
    scheme: Dict
) -> pd.Series:
    y_real = pd.Series(y_real).reset_index(drop=True).astype(float)

    median_price = float(scheme["median_price"])
    power = float(scheme["weight_power"])

    safe_price = y_real.clip(lower=1.0)

    weights = (safe_price / median_price) ** power

    weights = normalize_and_clip_weights(
        weights=weights,
        clip_min=scheme["clip_min"],
        clip_max=scheme["clip_max"],
    )

    return weights


def fit_density_equal_width_scheme(
    y_real_train: pd.Series,
    exp: WeightExperiment
) -> Dict:
    y_real_train = pd.Series(y_real_train).reset_index(drop=True)

    hist, bin_edges = np.histogram(y_real_train, bins=exp.weight_n_bins)
    hist = np.where(hist == 0, 1, hist)

    max_density = hist.max()
    raw_bin_weights = (max_density / hist) ** exp.weight_power

    return {
        "strategy": "density_equal_width",
        "bin_edges": bin_edges.tolist(),
        "bin_weights": raw_bin_weights.tolist(),
        "clip_min": exp.weight_clip_min,
        "clip_max": exp.weight_clip_max,
    }


def apply_density_equal_width_scheme(
    y_real: pd.Series,
    scheme: Dict
) -> pd.Series:
    y_real = pd.Series(y_real).reset_index(drop=True)

    edges = np.array(scheme["bin_edges"], dtype=float)
    bin_weights = np.array(scheme["bin_weights"], dtype=float)

    codes = np.digitize(y_real, edges) - 1
    codes = np.clip(codes, 0, len(bin_weights) - 1)

    weights = pd.Series(bin_weights[codes], index=y_real.index)

    weights = normalize_and_clip_weights(
        weights=weights,
        clip_min=scheme["clip_min"],
        clip_max=scheme["clip_max"],
    )

    return weights


def fit_inverse_frequency_qcut_scheme(
    y_real_train: pd.Series,
    exp: WeightExperiment
) -> Dict:
    y_real_train = pd.Series(y_real_train).reset_index(drop=True)

    max_bins = int(min(exp.weight_n_bins, max(2, len(y_real_train) // 5)))

    for q in range(max_bins, 1, -1):
        try:
            codes, bin_edges = pd.qcut(
                y_real_train,
                q=q,
                labels=False,
                retbins=True,
                duplicates="drop"
            )

            codes = pd.Series(codes)
            freq = codes.value_counts(normalize=True).sort_index()

            if len(freq) >= 2:
                raw_weights = (1.0 / freq) ** exp.weight_power
                code_to_weight = {int(k): float(v) for k, v in raw_weights.items()}

                return {
                    "strategy": "inverse_frequency_qcut",
                    "bin_edges": list(bin_edges),
                    "code_to_weight": code_to_weight,
                    "clip_min": exp.weight_clip_min,
                    "clip_max": exp.weight_clip_max,
                }

        except Exception:
            continue

    return {
        "strategy": "inverse_frequency_qcut",
        "bin_edges": None,
        "code_to_weight": {},
        "clip_min": exp.weight_clip_min,
        "clip_max": exp.weight_clip_max,
    }


def apply_inverse_frequency_qcut_scheme(
    y_real: pd.Series,
    scheme: Dict
) -> pd.Series:
    y_real = pd.Series(y_real).reset_index(drop=True)

    if scheme["bin_edges"] is None or len(scheme["code_to_weight"]) == 0:
        return pd.Series(np.ones(len(y_real)), index=y_real.index, dtype=float)

    edges = np.array(scheme["bin_edges"], dtype=float)

    codes = pd.cut(
        y_real,
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )

    codes = pd.Series(codes)

    max_code = max(int(k) for k in scheme["code_to_weight"].keys())

    codes = codes.fillna(
        pd.Series(
            np.where(y_real <= edges[0], 0, max_code),
            index=y_real.index
        )
    )

    codes = codes.astype(int).clip(lower=0, upper=max_code)

    mapping = {int(k): float(v) for k, v in scheme["code_to_weight"].items()}

    weights = codes.map(mapping).astype(float)

    weights = normalize_and_clip_weights(
        weights=weights,
        clip_min=scheme["clip_min"],
        clip_max=scheme["clip_max"],
    )

    return weights


def fit_weight_scheme(
    y_train_inner_real: pd.Series,
    cfg: Config,
    exp: WeightExperiment
) -> Dict:
    if exp.strategy == "manual_price_segments":
        return fit_manual_price_segments_scheme(cfg, exp)

    if exp.strategy == "sqrt_price":
        return fit_sqrt_price_scheme(y_train_inner_real, exp)

    if exp.strategy == "density_equal_width":
        return fit_density_equal_width_scheme(y_train_inner_real, exp)

    if exp.strategy == "inverse_frequency_qcut":
        return fit_inverse_frequency_qcut_scheme(y_train_inner_real, exp)

    raise ValueError(f"Stratégie de pondération inconnue : {exp.strategy}")


def apply_weight_scheme(
    y_real: pd.Series,
    scheme: Dict
) -> pd.Series:
    strategy = scheme["strategy"]

    if strategy == "manual_price_segments":
        return apply_manual_price_segments_scheme(y_real, scheme)

    if strategy == "sqrt_price":
        return apply_sqrt_price_scheme(y_real, scheme)

    if strategy == "density_equal_width":
        return apply_density_equal_width_scheme(y_real, scheme)

    if strategy == "inverse_frequency_qcut":
        return apply_inverse_frequency_qcut_scheme(y_real, scheme)

    raise ValueError(f"Stratégie de pondération inconnue : {strategy}")


def compute_weight_diagnostics(
    y_real: pd.Series,
    weights: pd.Series,
    cfg: Config,
    fold_name: str,
    exp_name: str
) -> pd.DataFrame:
    df_w = pd.DataFrame({
        "y_real": pd.Series(y_real).reset_index(drop=True),
        "weight": pd.Series(weights).reset_index(drop=True),
    })

    df_w["segment"] = build_price_segments(df_w["y_real"], cfg).astype(str)

    rows = []

    for seg in cfg.price_segment_labels:
        g = df_w[df_w["segment"] == seg]

        rows.append({
            "experiment": exp_name,
            "fold": fold_name,
            "segment": seg,
            "n": int(len(g)),
            "price_mean": float(g["y_real"].mean()) if len(g) > 0 else np.nan,
            "weight_mean": float(g["weight"].mean()) if len(g) > 0 else np.nan,
            "weight_min": float(g["weight"].min()) if len(g) > 0 else np.nan,
            "weight_max": float(g["weight"].max()) if len(g) > 0 else np.nan,
        })

    return pd.DataFrame(rows)

# 9. CatBoost

def make_catboost_params(cfg: Config, fold_seed: int) -> Dict:
    params = {
        "loss_function": cfg.loss_function,
        "eval_metric": cfg.eval_metric,
        "iterations": cfg.iterations,
        "learning_rate": cfg.learning_rate,
        "depth": cfg.depth,
        "l2_leaf_reg": cfg.l2_leaf_reg,
        "random_seed": fold_seed,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "use_best_model": cfg.use_best_model,
        "verbose": cfg.verbose_eval,
        "allow_writing_files": cfg.allow_writing_files,
        "task_type": cfg.task_type,
        "thread_count": cfg.thread_count,
    }

    if cfg.used_ram_limit:
        params["used_ram_limit"] = cfg.used_ram_limit

    return params

# 10. SHAP

def sample_for_shap(
    X_fold: pd.DataFrame,
    y_fold_log: pd.Series,
    cfg: Config,
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    if len(X_fold) <= cfg.shap_max_rows_per_fold:
        return X_fold.copy(), y_fold_log.copy()

    strat_bins = make_regression_strat_bins(
        y=y_fold_log,
        n_splits=2,
        max_bins=10
    )

    idx_all = np.arange(len(X_fold))

    idx_sample, _ = train_test_split(
        idx_all,
        train_size=cfg.shap_max_rows_per_fold,
        random_state=random_state,
        stratify=strat_bins
    )

    return (
        X_fold.iloc[idx_sample].copy(),
        y_fold_log.iloc[idx_sample].copy(),
    )

# 11. Entraînement d'une expérience pondérée

def train_one_weighted_experiment(
    cfg: Config,
    exp: WeightExperiment,
    base_paths: Dict[str, Path],
    df: pd.DataFrame,
    input_path: Path
) -> Dict[str, Any]:

    paths = get_experiment_paths(base_paths, exp.name)

    print("\n" + "=" * 90)
    print(f"EXPÉRIENCE : {exp.name}")
    print(f"STRATÉGIE  : {exp.strategy}")
    print("=" * 90)

    X, y_log, y_real, ids, cat_features, feature_names, dropped_constant_cols = prepare_features(df, cfg)

    run_info = {
        "input_path": str(input_path),
        "experiment": asdict(exp),
        "config": asdict(cfg),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "dropped_constant_cols": dropped_constant_cols,
    }

    with open(paths["reports_dir"] / "00_run_info.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_info), f, indent=2, ensure_ascii=False)

    outer_bins = make_regression_strat_bins(
        y=y_log,
        n_splits=cfg.n_splits,
        max_bins=cfg.outer_n_bins
    )

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state
    )

    oof_log_pred = np.full(len(X), np.nan, dtype=float)
    oof_real_pred = np.full(len(X), np.nan, dtype=float)
    oof_fold = np.full(len(X), -1, dtype=int)

    fold_metrics_log_rows = []
    fold_metrics_real_rows = []
    fold_segments_rows = []
    fold_weight_diagnostic_rows = []
    fold_weight_global_rows = []

    shap_weight_sums = np.zeros(len(feature_names), dtype=float)
    shap_total_weight = 0

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, outer_bins), start=1):
        fold_name = f"fold_{fold_id}"
        fold_seed = cfg.random_state + fold_id

        print("\n" + "-" * 90)
        print(f"{exp.name} - {fold_name}")
        print("-" * 90)

        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        y_train_log_full = y_log.iloc[train_idx].reset_index(drop=True)
        y_train_real_full = y_real.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test_log = y_log.iloc[test_idx].reset_index(drop=True)
        y_test_real = y_real.iloc[test_idx].reset_index(drop=True)
        ids_test = ids.iloc[test_idx].reset_index(drop=True)

        inner_bins = make_regression_strat_bins(
            y=y_train_log_full,
            n_splits=2,
            max_bins=max(4, cfg.outer_n_bins // 2)
        )

        idx_all_inner = np.arange(len(X_train_full))

        idx_train_inner, idx_val_inner = train_test_split(
            idx_all_inner,
            test_size=cfg.inner_validation_size,
            random_state=fold_seed,
            stratify=inner_bins
        )

        X_train = X_train_full.iloc[idx_train_inner].reset_index(drop=True)
        y_train_log = y_train_log_full.iloc[idx_train_inner].reset_index(drop=True)
        y_train_real = y_train_real_full.iloc[idx_train_inner].reset_index(drop=True)

        X_val = X_train_full.iloc[idx_val_inner].reset_index(drop=True)
        y_val_log = y_train_log_full.iloc[idx_val_inner].reset_index(drop=True)
        y_val_real = y_train_real_full.iloc[idx_val_inner].reset_index(drop=True)

        weight_scheme = fit_weight_scheme(
            y_train_inner_real=y_train_real,
            cfg=cfg,
            exp=exp
        )

        w_train = apply_weight_scheme(y_train_real, weight_scheme)
        w_val = apply_weight_scheme(y_val_real, weight_scheme)

        fold_weight_global_rows.append({
            "experiment": exp.name,
            "fold": fold_name,
            "strategy": exp.strategy,
            "train_weight_mean": float(w_train.mean()),
            "train_weight_std": float(w_train.std(ddof=1)) if len(w_train) > 1 else 0.0,
            "train_weight_min": float(w_train.min()),
            "train_weight_max": float(w_train.max()),
            "val_weight_mean": float(w_val.mean()),
            "val_weight_std": float(w_val.std(ddof=1)) if len(w_val) > 1 else 0.0,
            "val_weight_min": float(w_val.min()),
            "val_weight_max": float(w_val.max()),
            "weight_validation_pool": cfg.weight_validation_pool,
            "weight_scheme_json": json.dumps(to_jsonable(weight_scheme), ensure_ascii=False),
        })

        fold_weight_diagnostic_rows.append(
            compute_weight_diagnostics(
                y_real=y_train_real,
                weights=w_train,
                cfg=cfg,
                fold_name=fold_name,
                exp_name=exp.name
            )
        )

        train_pool = Pool(
            data=X_train,
            label=y_train_log,
            cat_features=cat_features,
            weight=w_train.values
        )

        if cfg.weight_validation_pool:
            val_pool = Pool(
                data=X_val,
                label=y_val_log,
                cat_features=cat_features,
                weight=w_val.values
            )
        else:
            val_pool = Pool(
                data=X_val,
                label=y_val_log,
                cat_features=cat_features
            )

        test_pool = Pool(
            data=X_test,
            label=y_test_log,
            cat_features=cat_features
        )

        model = CatBoostRegressor(**make_catboost_params(cfg, fold_seed))

        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=cfg.verbose_eval
        )

        model_path = paths["models_dir"] / f"{fold_name}_catboost_weighted_{exp.name}.cbm"
        model.save_model(str(model_path))

        pred_test_log = model.predict(test_pool)
        pred_test_real = np.expm1(pred_test_log)

        oof_log_pred[test_idx] = pred_test_log
        oof_real_pred[test_idx] = pred_test_real
        oof_fold[test_idx] = fold_id

        fold_pred_df = pd.DataFrame({
            "row_index_original": test_idx,
            "id_clean": ids_test.values,
            "fold": fold_id,
            "experiment": exp.name,
            "y_true_log": y_test_log.values,
            "y_pred_log": pred_test_log,
            "y_true_price": y_test_real.values,
            "y_pred_price": pred_test_real,
        })

        fold_pred_df["residual_log"] = fold_pred_df["y_pred_log"] - fold_pred_df["y_true_log"]
        fold_pred_df["residual_price"] = fold_pred_df["y_pred_price"] - fold_pred_df["y_true_price"]
        fold_pred_df["abs_error_price"] = np.abs(fold_pred_df["residual_price"])
        fold_pred_df["price_segment"] = build_price_segments(fold_pred_df["y_true_price"], cfg).astype(str)

        fold_pred_df.to_csv(
            paths["predictions_dir"] / f"{fold_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig"
        )

        metrics_log = compute_metrics(y_test_log.values, pred_test_log)
        metrics_real = compute_metrics(y_test_real.values, pred_test_real)

        fold_metrics_log_rows.append({
            "experiment": exp.name,
            "fold": fold_name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            **metrics_log
        })

        fold_metrics_real_rows.append({
            "experiment": exp.name,
            "fold": fold_name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            **metrics_real
        })

        fold_segments_rows.append(
            compute_segment_metrics(
                df_pred=fold_pred_df,
                cfg=cfg,
                fold_name=fold_name
            ).assign(experiment=exp.name)
        )

        if cfg.compute_shap:
            X_shap, y_shap_log = sample_for_shap(
                X_fold=X_test,
                y_fold_log=y_test_log,
                cfg=cfg,
                random_state=fold_seed
            )

            shap_pool = Pool(
                data=X_shap,
                label=y_shap_log,
                cat_features=cat_features
            )

            shap_values = model.get_feature_importance(
                data=shap_pool,
                type="ShapValues"
            )

            shap_contrib = shap_values[:, :-1]
            shap_mean_abs = np.abs(shap_contrib).mean(axis=0)

            shap_weight_sums += shap_mean_abs * len(X_shap)
            shap_total_weight += len(X_shap)

            fold_shap_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": shap_mean_abs,
                "fold": fold_name,
                "experiment": exp.name,
                "n_rows_shap": len(X_shap)
            }).sort_values("mean_abs_shap", ascending=False)

            fold_shap_df.to_csv(
                paths["shap_dir"] / f"{fold_name}_shap_mean_abs.csv",
                index=False,
                encoding="utf-8-sig"
            )

        print(
            f"{fold_name} terminé | "
            f"MAE € = {metrics_real['MAE']:.2f} | "
            f"RMSE € = {metrics_real['RMSE']:.2f} | "
            f"Biais € = {metrics_real['Mean_Error']:.2f}"
        )

    oof_df = pd.DataFrame({
        "row_index_original": np.arange(len(X)),
        "id_clean": ids.values,
        "fold": oof_fold,
        "experiment": exp.name,
        "y_true_log": y_log.values,
        "y_pred_log": oof_log_pred,
        "y_true_price": y_real.values,
        "y_pred_price": oof_real_pred,
    })

    oof_df["residual_log"] = oof_df["y_pred_log"] - oof_df["y_true_log"]
    oof_df["residual_price"] = oof_df["y_pred_price"] - oof_df["y_true_price"]
    oof_df["abs_error_price"] = np.abs(oof_df["residual_price"])
    oof_df["price_segment"] = build_price_segments(oof_df["y_true_price"], cfg).astype(str)

    oof_df.to_csv(
        paths["predictions_dir"] / "00_oof_predictions_all_folds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fold_metrics_log_df = pd.DataFrame(fold_metrics_log_rows)
    fold_metrics_real_df = pd.DataFrame(fold_metrics_real_rows)
    fold_segments_df = pd.concat(fold_segments_rows, ignore_index=True)
    fold_weight_global_df = pd.DataFrame(fold_weight_global_rows)
    fold_weight_segment_df = pd.concat(fold_weight_diagnostic_rows, ignore_index=True)

    fold_metrics_log_df.to_csv(paths["reports_dir"] / "01_fold_metrics_log_price.csv", index=False, encoding="utf-8-sig")
    fold_metrics_real_df.to_csv(paths["reports_dir"] / "02_fold_metrics_price_euros.csv", index=False, encoding="utf-8-sig")
    fold_segments_df.to_csv(paths["reports_dir"] / "03_fold_segment_metrics_price_euros.csv", index=False, encoding="utf-8-sig")
    fold_weight_global_df.to_csv(paths["reports_dir"] / "04_fold_weight_diagnostics_global.csv", index=False, encoding="utf-8-sig")
    fold_weight_segment_df.to_csv(paths["reports_dir"] / "05_fold_weight_diagnostics_by_segment.csv", index=False, encoding="utf-8-sig")

    def summarize_fold_metrics(df_in: pd.DataFrame, scale_name: str) -> pd.DataFrame:
        ignore_cols = ("experiment", "fold", "best_iteration", "tree_count")
        metrics_cols = [c for c in df_in.columns if c not in ignore_cols]

        rows = []

        for c in metrics_cols:
            rows.append({
                "experiment": exp.name,
                "scale": scale_name,
                "metric": c,
                "fold_mean": float(df_in[c].mean()),
                "fold_std": float(df_in[c].std(ddof=1)) if len(df_in) > 1 else np.nan,
            })

        return pd.DataFrame(rows)

    summary_log = summarize_fold_metrics(fold_metrics_log_df, "log_price")
    summary_real = summarize_fold_metrics(fold_metrics_real_df, "price_euros")

    oof_metrics_log = compute_metrics(oof_df["y_true_log"].values, oof_df["y_pred_log"].values)
    oof_metrics_real = compute_metrics(oof_df["y_true_price"].values, oof_df["y_pred_price"].values)

    oof_rows = []

    for k, v in oof_metrics_log.items():
        oof_rows.append({
            "experiment": exp.name,
            "scale": "log_price",
            "metric": k,
            "oof_global": v
        })

    for k, v in oof_metrics_real.items():
        oof_rows.append({
            "experiment": exp.name,
            "scale": "price_euros",
            "metric": k,
            "oof_global": v
        })

    oof_summary_df = pd.DataFrame(oof_rows)

    aggregate_metrics_df = (
        pd.concat([summary_log, summary_real], ignore_index=True)
        .merge(
            oof_summary_df,
            on=["experiment", "scale", "metric"],
            how="left"
        )
    )

    aggregate_metrics_df.to_csv(
        paths["reports_dir"] / "06_aggregate_metrics_cv.csv",
        index=False,
        encoding="utf-8-sig"
    )

    seg_oof_df = compute_segment_metrics(
        df_pred=oof_df,
        cfg=cfg,
        fold_name="OOF_GLOBAL"
    ).assign(experiment=exp.name)

    seg_oof_df.to_csv(
        paths["reports_dir"] / "07_aggregate_segment_metrics_oof.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if cfg.compute_shap and shap_total_weight > 0:
        global_shap = shap_weight_sums / shap_total_weight

        shap_global_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": global_shap,
            "experiment": exp.name
        }).sort_values("mean_abs_shap", ascending=False)

        shap_global_df.to_csv(
            paths["shap_dir"] / "00_global_mean_abs_shap.csv",
            index=False,
            encoding="utf-8-sig"
        )

        top_shap_df = shap_global_df.head(cfg.shap_top_n).iloc[::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(top_shap_df["feature"], top_shap_df["mean_abs_shap"])
        plt.xlabel("Mean |SHAP|")
        plt.ylabel("Variable")
        plt.title(f"Top variables SHAP - {exp.name}")
        plt.tight_layout()
        plt.savefig(paths["plots_dir"] / "01_shap_top_mean_abs.png", dpi=160)
        plt.close()

    make_experiment_plots(oof_df, seg_oof_df, cfg, paths, exp.name)

    write_experiment_report(
        cfg=cfg,
        exp=exp,
        input_path=input_path,
        X_shape=X.shape,
        cat_features=cat_features,
        oof_metrics_log=oof_metrics_log,
        oof_metrics_real=oof_metrics_real,
        seg_oof_df=seg_oof_df,
        paths=paths
    )

    return {
        "experiment": exp.name,
        "aggregate_metrics": aggregate_metrics_df,
        "segments_oof": seg_oof_df,
        "oof_metrics_log": oof_metrics_log,
        "oof_metrics_real": oof_metrics_real,
        "output_dir": str(paths["exp_dir"]),
    }

# 12. Graphiques d'une expérience

def make_experiment_plots(
    oof_df: pd.DataFrame,
    seg_oof_df: pd.DataFrame,
    cfg: Config,
    paths: Dict[str, Path],
    exp_name: str
) -> None:
    max_value = np.percentile(
        np.concatenate([
            oof_df["y_true_price"].values,
            oof_df["y_pred_price"].values
        ]),
        99
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(oof_df["y_true_price"], oof_df["y_pred_price"], alpha=0.20, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title(f"OOF - Prix prédit vs réel - {exp_name}")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "02_oof_pred_vs_true_price.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(oof_df["y_true_price"], oof_df["abs_error_price"], alpha=0.20, s=8)
    plt.xlim(0, np.percentile(oof_df["y_true_price"], 99))
    plt.ylim(0, np.percentile(oof_df["abs_error_price"], 99))
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Erreur absolue (€)")
    plt.title(f"OOF - Erreur absolue selon prix - {exp_name}")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "03_oof_abs_error_vs_price.png", dpi=160)
    plt.close()

    residuals_clip = np.clip(
        oof_df["residual_price"].values,
        np.percentile(oof_df["residual_price"], 1),
        np.percentile(oof_df["residual_price"], 99),
    )

    plt.figure(figsize=(9, 5))
    plt.hist(residuals_clip, bins=70)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Résidu en euros")
    plt.ylabel("Fréquence")
    plt.title(f"OOF - Distribution résidus P1-P99 - {exp_name}")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "04_oof_residuals_hist.png", dpi=160)
    plt.close()

    seg_order = list(cfg.price_segment_labels)
    seg_plot = seg_oof_df.copy()
    seg_plot["segment"] = pd.Categorical(seg_plot["segment"], categories=seg_order, ordered=True)
    seg_plot = seg_plot.sort_values("segment")

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title(f"OOF - MAE par segment - {exp_name}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "05_oof_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Erreur moyenne (€)")
    plt.title(f"OOF - Biais moyen par segment - {exp_name}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "06_oof_bias_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["Underestimation_Rate_pct"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Sous-estimation (%)")
    plt.title(f"OOF - Taux de sous-estimation - {exp_name}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "07_oof_underestimation_by_segment.png", dpi=160)
    plt.close()

# 13. Rapport texte d'une expérience

def write_experiment_report(
    cfg: Config,
    exp: WeightExperiment,
    input_path: Path,
    X_shape: Tuple[int, int],
    cat_features: List[str],
    oof_metrics_log: Dict[str, float],
    oof_metrics_real: Dict[str, float],
    seg_oof_df: pd.DataFrame,
    paths: Dict[str, Path]
) -> None:
    report_path = paths["reports_dir"] / "08_resume_experience_weighted.txt"

    seg_order = list(cfg.price_segment_labels)
    seg_plot = seg_oof_df.copy()
    seg_plot["segment"] = pd.Categorical(seg_plot["segment"], categories=seg_order, ordered=True)
    seg_plot = seg_plot.sort_values("segment")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"CatBoost pondéré - {exp.name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Fichier d'entrée : {input_path}\n")
        f.write(f"Dimensions X : {X_shape}\n")
        f.write(f"Catégorielles : {cat_features}\n\n")

        f.write("Configuration pondération\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(to_jsonable(asdict(exp)), indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Métriques OOF globales - log_price\n")
        f.write("-" * 80 + "\n")
        for k, v in oof_metrics_log.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\nMétriques OOF globales - price_euros\n")
        f.write("-" * 80 + "\n")
        for k, v in oof_metrics_real.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\nSegments de prix - OOF global\n")
        f.write("-" * 80 + "\n")
        for _, row in seg_plot.iterrows():
            f.write(
                f"{str(row['segment']):>10s} | "
                f"n={int(row['n']):5d} | "
                f"MAE={row['MAE']:.2f}€ | "
                f"RMSE={row['RMSE']:.2f}€ | "
                f"Biais={row['Mean_Error']:.2f}€ | "
                f"Sous-estimation={row['Underestimation_Rate_pct']:.2f}%\n"
            )

# 14. Comparaison globale entre stratégies et baseline

def load_baseline_if_available(base_paths: Dict[str, Path]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    baseline_metrics_path = base_paths["baseline_reports_dir"] / "04_aggregate_metrics_cv.csv"
    baseline_segments_path = base_paths["baseline_reports_dir"] / "05_aggregate_segment_metrics_oof.csv"

    baseline_metrics = None
    baseline_segments = None

    if baseline_metrics_path.exists():
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        baseline_metrics["experiment"] = "baseline_cv"

    if baseline_segments_path.exists():
        baseline_segments = pd.read_csv(baseline_segments_path)
        baseline_segments["experiment"] = "baseline_cv"

    return baseline_metrics, baseline_segments


def make_global_comparisons(
    cfg: Config,
    base_paths: Dict[str, Path],
    experiment_results: List[Dict[str, Any]]
) -> None:
    baseline_metrics, baseline_segments = load_baseline_if_available(base_paths)

    all_metrics = []
    all_segments = []

    if baseline_metrics is not None:
        all_metrics.append(baseline_metrics)

    if baseline_segments is not None:
        all_segments.append(baseline_segments)

    for res in experiment_results:
        all_metrics.append(res["aggregate_metrics"])
        all_segments.append(res["segments_oof"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    segments_df = pd.concat(all_segments, ignore_index=True)

    metrics_df.to_csv(base_paths["common_reports_dir"] / "01_comparison_all_metrics.csv", index=False, encoding="utf-8-sig")
    segments_df.to_csv(base_paths["common_reports_dir"] / "02_comparison_all_segments.csv", index=False, encoding="utf-8-sig")

    metric_names = ["MAE", "RMSE", "MedAE", "R2", "Mean_Error", "Underestimation_Rate_pct"]

    summary_rows = []

    for exp_name in metrics_df["experiment"].unique():
        tmp = metrics_df[
            (metrics_df["experiment"] == exp_name)
            & (metrics_df["scale"] == "price_euros")
            & (metrics_df["metric"].isin(metric_names))
        ]

        row = {"experiment": exp_name}

        for _, r in tmp.iterrows():
            row[r["metric"]] = r.get("oof_global", np.nan)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(base_paths["common_reports_dir"] / "03_summary_global_price_euros.csv", index=False, encoding="utf-8-sig")

    segment_summary = segments_df[
        segments_df["fold"].astype(str).eq("OOF_GLOBAL")
    ].copy()

    segment_summary.to_csv(base_paths["common_reports_dir"] / "04_summary_segments_oof.csv", index=False, encoding="utf-8-sig")

    make_global_comparison_plots(cfg, base_paths, segment_summary, summary_df)

    write_global_report(base_paths, summary_df, segment_summary)


def make_global_comparison_plots(
    cfg: Config,
    base_paths: Dict[str, Path],
    segment_summary: pd.DataFrame,
    summary_df: pd.DataFrame
) -> None:
    seg_order = list(cfg.price_segment_labels)

    df = segment_summary.copy()
    df["segment"] = pd.Categorical(df["segment"], categories=seg_order, ordered=True)
    df = df.sort_values(["segment", "experiment"])

    experiments = list(df["experiment"].dropna().unique())

    x = np.arange(len(seg_order))
    width = 0.8 / max(len(experiments), 1)

    plt.figure(figsize=(11, 6))

    for i, exp_name in enumerate(experiments):
        vals = (
            df[df["experiment"] == exp_name]
            .set_index("segment")["MAE"]
            .reindex(seg_order)
            .values
        )

        plt.bar(
            x + (i - len(experiments) / 2) * width + width / 2,
            vals,
            width=width,
            label=exp_name
        )

    plt.xticks(x, seg_order, rotation=25)
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("Comparaison MAE par segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_paths["common_plots_dir"] / "01_comparison_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))

    for i, exp_name in enumerate(experiments):
        vals = (
            df[df["experiment"] == exp_name]
            .set_index("segment")["Mean_Error"]
            .reindex(seg_order)
            .values
        )

        plt.bar(
            x + (i - len(experiments) / 2) * width + width / 2,
            vals,
            width=width,
            label=exp_name
        )

    plt.axhline(0, linestyle="--")
    plt.xticks(x, seg_order, rotation=25)
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Erreur moyenne (€)")
    plt.title("Comparaison du biais moyen par segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_paths["common_plots_dir"] / "02_comparison_bias_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))

    for i, exp_name in enumerate(experiments):
        vals = (
            df[df["experiment"] == exp_name]
            .set_index("segment")["Underestimation_Rate_pct"]
            .reindex(seg_order)
            .values
        )

        plt.bar(
            x + (i - len(experiments) / 2) * width + width / 2,
            vals,
            width=width,
            label=exp_name
        )

    plt.xticks(x, seg_order, rotation=25)
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Sous-estimation (%)")
    plt.title("Comparaison du taux de sous-estimation par segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_paths["common_plots_dir"] / "03_comparison_underestimation_by_segment.png", dpi=160)
    plt.close()

    if "MAE" in summary_df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(summary_df["experiment"], summary_df["MAE"])
        plt.ylabel("MAE globale (€)")
        plt.title("Comparaison MAE globale OOF")
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(base_paths["common_plots_dir"] / "04_comparison_global_mae.png", dpi=160)
        plt.close()


def write_global_report(
    base_paths: Dict[str, Path],
    summary_df: pd.DataFrame,
    segment_summary: pd.DataFrame
) -> None:
    report_path = base_paths["common_reports_dir"] / "05_resume_comparaison_globale.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Comparaison globale : baseline CV vs CatBoost pondérés\n")
        f.write("=" * 90 + "\n\n")

        f.write("Métriques globales price_euros\n")
        f.write("-" * 90 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("Métriques par segment OOF\n")
        f.write("-" * 90 + "\n")

        cols = [
            "experiment",
            "segment",
            "n",
            "MAE",
            "RMSE",
            "MedAE",
            "Mean_Error",
            "Underestimation_Rate_pct",
        ]

        available_cols = [c for c in cols if c in segment_summary.columns]
        f.write(segment_summary[available_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Lecture attendue :\n")
        f.write("- Si la MAE globale augmente légèrement mais que la MAE et le biais des segments chers diminuent fortement, la pondération peut être utile.\n")
        f.write("- Si les petits prix se dégradent beaucoup, la pondération est trop agressive.\n")
        f.write("- Le meilleur modèle n'est pas forcément celui avec la MAE globale la plus basse ; il faut regarder l'objectif métier.\n")



def main() -> None:
    base_paths = get_project_paths()

    print("\n" + "=" * 90)
    print("CV CATBOOST PONDÉRÉ - COMPARAISON DE STRATÉGIES")
    print("=" * 90)

    print("\nDossier données :")
    print(base_paths["data_dir"])

    print("\nDossier résultats :")
    print(base_paths["output_dir"])

    df, input_path = load_data(base_paths, CFG)

    experiment_results = []

    for exp in WEIGHT_EXPERIMENTS:
        result = train_one_weighted_experiment(
            cfg=CFG,
            exp=exp,
            base_paths=base_paths,
            df=df,
            input_path=input_path
        )

        experiment_results.append(result)

    make_global_comparisons(
        cfg=CFG,
        base_paths=base_paths,
        experiment_results=experiment_results
    )

    print("\n" + "=" * 90)
    print("TERMINÉ - CV CatBoost pondéré")
    print("=" * 90)
    print("Résultats principaux :")
    print(base_paths["output_dir"])
    print("\nComparaison globale :")
    print(base_paths["common_reports_dir"] / "03_summary_global_price_euros.csv")
    print(base_paths["common_reports_dir"] / "04_summary_segments_oof.csv")
    print(base_paths["common_reports_dir"] / "05_resume_comparaison_globale.txt")
    print("\nGraphiques globaux :")
    print(base_paths["common_plots_dir"])


if __name__ == "__main__":
    main()