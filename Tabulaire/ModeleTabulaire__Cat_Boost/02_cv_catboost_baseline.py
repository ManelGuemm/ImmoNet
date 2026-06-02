# -*- coding: utf-8 -*-
"""
02_cv_catboost_baseline.py
Validation croisée 5-fold pour le modèle CatBoost baseline Airbnb.
Objectifs :
- Effectuer une CV 5 folds reproductible avec CatBoostRegressor
- Cible d'entraînement : log_price
- Interprétation finale : price (euros) après np.expm1
- Gérer les variables catégorielles via CatBoost Pool
- Utiliser une stratification "like-regression" via qcut(log_price)
- Utiliser un split interne train/validation dans chaque fold
  pour l'early stopping / use_best_model
- Sauvegarder :
    * métriques par fold
    * métriques agrégées (moyenne, écart-type, OOF global)
    * métriques par segments de prix
    * prédictions OOF globales
    * prédictions de chaque fold
    * modèles de chaque fold
    * SHAP global moyen (CatBoost ShapValues)
    * graphiques principaux

Pré-requis :
pip install catboost pandas numpy scikit-learn matplotlib openpyxl
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# Configuration
# ============================================================

@dataclass
class Config:
    # Graine globale pour reproductibilité
    random_state: int = 42

    # Fichiers d'entrée
    input_csv: str = "airbnb_tabulaire_final_catboost_rates_corriges.csv"
    input_xlsx: str = "airbnb_tabulaire_final_catboost_rates_corriges.xlsx"

    # Cibles
    target_log: str = "log_price"
    target_real: str = "price"

    # Colonnes exclues
    cols_to_exclude: Tuple[str, ...] = (
        "id_clean",
        "price",
        "log_price",
        "nights_range_is_incoherent",
    )

    # Variables catégorielles CatBoost
    cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # Validation croisée externe
    n_splits: int = 5
    outer_n_bins: int = 20
    inner_validation_size: float = 0.15

    # CatBoost
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
    used_ram_limit: Optional[str] = "8gb"  # Ajuster si besoin, ex. "4gb"

    # SHAP / mémoire
    shap_max_rows_per_fold: int = 1200
    shap_top_n: int = 30

    # Segments de prix
    price_segment_bins: Tuple[float, ...] = (0, 100, 200, 400, 800, np.inf)
    price_segment_labels: Tuple[str, ...] = (
        "< 100 €",
        "100-200 €",
        "200-400 €",
        "400-800 €",
        "> 800 €",
    )


CFG = Config()

# Utilitaires généraux

def get_project_paths() -> Dict[str, Path]:
  
    script_dir = Path(__file__).resolve().parent
    tabulaire_dir = script_dir.parent
    data_dir = tabulaire_dir / "Donnees_Tabulaires"

    output_dir = script_dir / "Resultats_CV_Baseline_CatBoost"
    reports_dir = output_dir / "rapports"
    plots_dir = output_dir / "graphiques"
    models_dir = output_dir / "modeles"
    predictions_dir = output_dir / "predictions"
    shap_dir = output_dir / "shap"

    for p in [output_dir, reports_dir, plots_dir, models_dir, predictions_dir, shap_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "tabulaire_dir": tabulaire_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "models_dir": models_dir,
        "predictions_dir": predictions_dir,
        "shap_dir": shap_dir,
    }


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

    return df, input_path


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
        metrics = compute_metrics(group["y_true_price"].values, group["y_pred_price"].values)

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


def make_regression_strat_bins(
    y: pd.Series,
    n_splits: int,
    max_bins: int = 20
) -> pd.Series:
    y = pd.Series(y).reset_index(drop=True)

    # borne maximale raisonnable
    max_bins = int(min(max_bins, max(2, len(y) // n_splits)))

    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
            vc = pd.Series(bins).value_counts(dropna=False)
            if len(vc) >= 2 and vc.min() >= n_splits:
                return pd.Series(bins, index=y.index)
        except Exception:
            continue

    # fallback : 2 bins via médiane
    fallback = pd.cut(
        y,
        bins=np.unique(np.quantile(y, [0.0, 0.5, 1.0])),
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    fallback = pd.Series(fallback, index=y.index).fillna(0).astype(int)
    return fallback


def sample_for_shap(
    X_fold: pd.DataFrame,
    y_fold_log: pd.Series,
    y_fold_real: pd.Series,
    cfg: Config,
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Échantillonnage borné en taille pour SHAP,
    avec légère stratification sur log_price.
    """
    if len(X_fold) <= cfg.shap_max_rows_per_fold:
        return X_fold.copy(), y_fold_log.copy(), y_fold_real.copy()

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
        y_fold_real.iloc[idx_sample].copy(),
    )


def prepare_features(
    df: pd.DataFrame,
    cfg: Config
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str], List[str]]:
    """
    Retourne :
    - X
    - y_log
    - y_real
    - ids
    - cat_features_existantes
    - feature_names
    """
    if cfg.target_log not in df.columns or cfg.target_real not in df.columns:
        raise ValueError(
            f"Les colonnes cibles '{cfg.target_log}' et/ou '{cfg.target_real}' sont absentes."
        )

    ids = df["id_clean"].astype(str) if "id_clean" in df.columns else pd.Series(df.index.astype(str))

    feature_cols = [c for c in df.columns if c not in cfg.cols_to_exclude]
    X = df[feature_cols].copy()

    # Suppression éventuelle de colonnes constantes
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    y_log = pd.to_numeric(df[cfg.target_log], errors="coerce")
    y_real = pd.to_numeric(df[cfg.target_real], errors="coerce")

    # Supprimer les lignes invalides sur la cible
    valid_mask = y_log.notna() & y_real.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y_log = y_log.loc[valid_mask].reset_index(drop=True)
    y_real = y_real.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)

    # Gérer les catégorielles CatBoost
    cat_features = [c for c in cfg.cat_features if c in X.columns]
    for c in cat_features:
        X[c] = X[c].fillna("missing").astype(str)

    return X, y_log, y_real, ids, cat_features, list(X.columns)


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

# Entraînement CV

def train_cv_baseline(cfg: Config) -> None:
    paths = get_project_paths()
    df, input_path = load_data(paths, cfg)
    X, y_log, y_real, ids, cat_features, feature_names = prepare_features(df, cfg)

    # Sauvegarde de la config/exécution
    run_info = {
        "input_path": str(input_path),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "config": asdict(cfg),
    }
    with open(paths["reports_dir"] / "00_run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    # Stratification "proxy" sur la cible continue
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

    # OOF stockages
    oof_log_pred = np.full(shape=len(X), fill_value=np.nan, dtype=float)
    oof_real_pred = np.full(shape=len(X), fill_value=np.nan, dtype=float)
    oof_fold = np.full(shape=len(X), fill_value=-1, dtype=int)

    fold_metrics_log_rows = []
    fold_metrics_real_rows = []
    fold_segments_rows = []

    shap_weight_sums = np.zeros(len(feature_names), dtype=float)
    shap_total_weight = 0

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, outer_bins), start=1):
        fold_name = f"fold_{fold_id}"
        fold_seed = cfg.random_state + fold_id

        print("=" * 80)
        print(f"CV externe - {fold_name}")
        print("=" * 80)

        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        y_train_log_full = y_log.iloc[train_idx].reset_index(drop=True)
        y_train_real_full = y_real.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test_log = y_log.iloc[test_idx].reset_index(drop=True)
        y_test_real = y_real.iloc[test_idx].reset_index(drop=True)
        ids_test = ids.iloc[test_idx].reset_index(drop=True)

        # Split interne apprentissage / validation pour early stopping
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

        X_val = X_train_full.iloc[idx_val_inner].reset_index(drop=True)
        y_val_log = y_train_log_full.iloc[idx_val_inner].reset_index(drop=True)

        train_pool = Pool(
            data=X_train,
            label=y_train_log,
            cat_features=cat_features
        )
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

        params = make_catboost_params(cfg, fold_seed)
        model = CatBoostRegressor(**params)

        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=cfg.verbose_eval
        )

        # Sauvegarde du modèle
        model_path = paths["models_dir"] / f"{fold_name}_catboost_baseline.cbm"
        model.save_model(str(model_path))

        # Prédictions fold test
        pred_test_log = model.predict(test_pool)
        pred_test_real = np.expm1(pred_test_log)

        # Stockage OOF
        oof_log_pred[test_idx] = pred_test_log
        oof_real_pred[test_idx] = pred_test_real
        oof_fold[test_idx] = fold_id

        # DataFrame prédictions fold
        fold_pred_df = pd.DataFrame({
            "row_index_original": test_idx,
            "id_clean": ids_test.values,
            "fold": fold_id,
            "y_true_log": y_test_log.values,
            "y_pred_log": pred_test_log,
            "y_true_price": y_test_real.values,
            "y_pred_price": pred_test_real,
        })
        fold_pred_df["residual_log"] = fold_pred_df["y_pred_log"] - fold_pred_df["y_true_log"]
        fold_pred_df["residual_price"] = fold_pred_df["y_pred_price"] - fold_pred_df["y_true_price"]
        fold_pred_df["abs_error_price"] = np.abs(fold_pred_df["residual_price"])
        fold_pred_df["price_segment"] = build_price_segments(fold_pred_df["y_true_price"], cfg).astype(str)

        fold_pred_path = paths["predictions_dir"] / f"{fold_name}_predictions.csv"
        fold_pred_df.to_csv(fold_pred_path, index=False, encoding="utf-8-sig")

        # Métriques fold
        metrics_log = compute_metrics(y_test_log.values, pred_test_log)
        metrics_real = compute_metrics(y_test_real.values, pred_test_real)

        fold_metrics_log_rows.append({
            "fold": fold_name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            **metrics_log
        })
        fold_metrics_real_rows.append({
            "fold": fold_name,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            **metrics_real
        })

        # Métriques segments fold
        seg_df = compute_segment_metrics(fold_pred_df, cfg, fold_name)
        fold_segments_rows.append(seg_df)

        # SHAP sur un échantillon borné du fold test
        X_shap, y_shap_log, y_shap_real = sample_for_shap(
            X_fold=X_test,
            y_fold_log=y_test_log,
            y_fold_real=y_test_real,
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
        # dernière colonne = expected value
        shap_contrib = shap_values[:, :-1]
        shap_mean_abs = np.abs(shap_contrib).mean(axis=0)

        shap_weight_sums += shap_mean_abs * len(X_shap)
        shap_total_weight += len(X_shap)

        fold_shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": shap_mean_abs,
            "fold": fold_name,
            "n_rows_shap": len(X_shap)
        }).sort_values("mean_abs_shap", ascending=False)

        fold_shap_df.to_csv(
            paths["shap_dir"] / f"{fold_name}_shap_mean_abs.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # Agrégation OOF
    # ========================================================
    oof_df = pd.DataFrame({
        "row_index_original": np.arange(len(X)),
        "id_clean": ids.values,
        "fold": oof_fold,
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

    # DataFrames métriques par fold
    fold_metrics_log_df = pd.DataFrame(fold_metrics_log_rows)
    fold_metrics_real_df = pd.DataFrame(fold_metrics_real_rows)
    fold_segments_df = pd.concat(fold_segments_rows, ignore_index=True)

    fold_metrics_log_df.to_csv(
        paths["reports_dir"] / "01_fold_metrics_log_price.csv",
        index=False,
        encoding="utf-8-sig"
    )
    fold_metrics_real_df.to_csv(
        paths["reports_dir"] / "02_fold_metrics_price_euros.csv",
        index=False,
        encoding="utf-8-sig"
    )
    fold_segments_df.to_csv(
        paths["reports_dir"] / "03_fold_segment_metrics_price_euros.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Agrégations moyenne / std par fold
    def summarize_fold_metrics(df_in: pd.DataFrame, scale_name: str) -> pd.DataFrame:
        metrics_cols = [c for c in df_in.columns if c not in ("fold", "best_iteration", "tree_count")]
        rows = []
        for c in metrics_cols:
            rows.append({
                "scale": scale_name,
                "metric": c,
                "fold_mean": float(df_in[c].mean()),
                "fold_std": float(df_in[c].std(ddof=1)) if len(df_in) > 1 else np.nan,
            })
        return pd.DataFrame(rows)

    summary_log = summarize_fold_metrics(fold_metrics_log_df, "log_price")
    summary_real = summarize_fold_metrics(fold_metrics_real_df, "price_euros")

    # Métriques OOF globales
    oof_metrics_log = compute_metrics(oof_df["y_true_log"].values, oof_df["y_pred_log"].values)
    oof_metrics_real = compute_metrics(oof_df["y_true_price"].values, oof_df["y_pred_price"].values)

    oof_rows = []
    for k, v in oof_metrics_log.items():
        oof_rows.append({
            "scale": "log_price",
            "metric": k,
            "oof_global": v
        })
    for k, v in oof_metrics_real.items():
        oof_rows.append({
            "scale": "price_euros",
            "metric": k,
            "oof_global": v
        })
    oof_summary_df = pd.DataFrame(oof_rows)

    aggregate_metrics_df = (
        pd.concat([summary_log, summary_real], ignore_index=True)
        .merge(oof_summary_df, on=["scale", "metric"], how="left")
    )

    aggregate_metrics_df.to_csv(
        paths["reports_dir"] / "04_aggregate_metrics_cv.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Métriques segments agrégées OOF
    seg_oof_df = compute_segment_metrics(oof_df, cfg, fold_name="OOF_GLOBAL")
    seg_oof_df.to_csv(
        paths["reports_dir"] / "05_aggregate_segment_metrics_oof.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # SHAP global moyen
    global_shap = shap_weight_sums / max(shap_total_weight, 1)
    shap_global_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_shap
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
    plt.title("Top variables selon SHAP moyen absolu - CV baseline")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "01_shap_top_mean_abs.png", dpi=160)
    plt.close()

    # ========================================================
    # Graphiques OOF
    # ========================================================
    # 1. Prix prédit vs réel
    max_value = np.percentile(
        np.concatenate([
            oof_df["y_true_price"].values,
            oof_df["y_pred_price"].values
        ]),
        99
    )

    plt.figure(figsize=(7, 7))
    plt.scatter(
        oof_df["y_true_price"],
        oof_df["y_pred_price"],
        alpha=0.20,
        s=8
    )
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("OOF - Prix prédit vs prix réel")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "02_oof_pred_vs_true_price.png", dpi=160)
    plt.close()

    # 2. Histogramme des résidus
    residuals_clip = np.clip(
        oof_df["residual_price"].values,
        np.percentile(oof_df["residual_price"].values, 1),
        np.percentile(oof_df["residual_price"].values, 99),
    )
    plt.figure(figsize=(9, 5))
    plt.hist(residuals_clip, bins=60)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Résidu en euros")
    plt.ylabel("Fréquence")
    plt.title("OOF - Distribution des résidus (P1-P99)")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "03_oof_residuals_hist.png", dpi=160)
    plt.close()

    # 3. Erreur absolue selon le prix réel
    plt.figure(figsize=(9, 5))
    plt.scatter(
        oof_df["y_true_price"],
        oof_df["abs_error_price"],
        alpha=0.20,
        s=8
    )
    plt.xlim(0, np.percentile(oof_df["y_true_price"], 99))
    plt.ylim(0, np.percentile(oof_df["abs_error_price"], 99))
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Erreur absolue (€)")
    plt.title("OOF - Erreur absolue selon le prix réel")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "04_oof_abs_error_vs_price.png", dpi=160)
    plt.close()

    # 4. MAE par segment (OOF)
    seg_order = list(cfg.price_segment_labels)
    seg_plot = seg_oof_df.copy()
    seg_plot["segment"] = pd.Categorical(seg_plot["segment"], categories=seg_order, ordered=True)
    seg_plot = seg_plot.sort_values("segment")

    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("OOF - MAE par segment de prix")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "05_oof_mae_by_segment.png", dpi=160)
    plt.close()

    # 5. Biais moyen par segment
    plt.figure(figsize=(8, 5))
    plt.bar(seg_plot["segment"].astype(str), seg_plot["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Erreur moyenne (€)")
    plt.title("OOF - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "06_oof_bias_by_segment.png", dpi=160)
    plt.close()

    # Rapport texte synthétique
    with open(paths["reports_dir"] / "06_resume_cv.txt", "w", encoding="utf-8") as f:
        f.write("Validation croisée 5-fold CatBoost baseline\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fichier d'entrée : {input_path}\n")
        f.write(f"Nombre de lignes : {len(X)}\n")
        f.write(f"Nombre de variables : {X.shape[1]}\n")
        f.write(f"Variables catégorielles : {cat_features}\n\n")

        f.write("Métriques OOF globales - log_price\n")
        f.write("-" * 70 + "\n")
        for k, v in oof_metrics_log.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\nMétriques OOF globales - price_euros\n")
        f.write("-" * 70 + "\n")
        for k, v in oof_metrics_real.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\nSegments de prix - OOF global\n")
        f.write("-" * 70 + "\n")
        for _, row in seg_plot.iterrows():
            f.write(
                f"{row['segment']:>9s} | n={int(row['n']):4d} | "
                f"MAE={row['MAE']:.2f}€ | RMSE={row['RMSE']:.2f}€ | "
                f"Biais={row['Mean_Error']:.2f}€ | "
                f"Sous-estimation={row['Underestimation_Rate_pct']:.2f}%\n"
            )

    print("\n" + "=" * 80)
    print("TERMINÉ - Validation croisée baseline")
    print("=" * 80)
    print(f"Résultats : {paths['output_dir']}")
    print("Fichiers clés :")
    print(" - rapports/04_aggregate_metrics_cv.csv")
    print(" - rapports/05_aggregate_segment_metrics_oof.csv")
    print(" - predictions/00_oof_predictions_all_folds.csv")
    print(" - shap/00_global_mean_abs_shap.csv")
    print(" - graphiques/01_shap_top_mean_abs.png")
    print(" - graphiques/02_oof_pred_vs_true_price.png")


if __name__ == "__main__":
    train_cv_baseline(CFG)
