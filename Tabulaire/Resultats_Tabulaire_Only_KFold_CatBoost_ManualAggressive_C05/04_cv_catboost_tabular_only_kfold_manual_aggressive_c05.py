from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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


@dataclass
class Config:
    random_state: int = 42
    input_filenames: Tuple[str, ...] = (
        "airbnb_tabulaire_final_catboost_rates_corriges.csv",
        "airbnb_tabulaire_final_catboost_rates_corriges.xlsx",
        "airbnb_tabulaire_fusionné_sentiment.csv",
        "Airbnb_Final.xlsx",
    )

    # Dossier de sortie
    output_dir_name: str = "Resultats_Tabulaire_Only_KFold_CatBoost_ManualAggressive_C05"

    # Cibles
    target_log: str = "log_price"
    target_real: str = "price"

    # Identifiant principal
    id_col: str = "id_clean"

    # Colonnes à exclure explicitement.
    cols_to_exclude: Tuple[str, ...] = (
        # Identifiants / URLs / images
        "id",
        "id_clean",
        "listing_id_clean",
        "listing_url",
        "picture_url",

        # Cibles et variantes de cible : interdites dans X
        "price",
        "price_clean",
        "log_price",
        "price_txt",
        "price_clean_txt",
        "log_price_txt",

        # Colonne jugée incohérente
        "nights_range_is_incoherent",

        # Colonnes que tu avais choisi de retirer car redondantes / discutables
        "has_reviews",
        "nb_avis_textuels_bert",
        "bert_stars_moyen",

        # Colonnes de split éventuelles
        "split",
        "split_txt",

        # prédictions texte si elles sont présentes dans le fichier
        "text_pred_final",
        "text_pred_oof",
        "text_pred_test",
    )

    # Exclusion par préfixe pour être sûr de rester sur du tabulaire seul.
    forbidden_prefixes: Tuple[str, ...] = (
        "txt_e5_",
        "text_",
        "img_",
        "image_",
        "clip_",
        "resnet_",
        "efficientnet_",
        "bert_embedding_",
        "embedding_",
    )

    # Si des colonnes brutes textuelles existent, on les retire pour rester en tabulaire seul.
    raw_text_cols_to_exclude: Tuple[str, ...] = (
        "name",
        "description",
        "neighborhood_overview",
        "host_about",
        "amenities",
    )

    # Catégorielles principales 
    base_cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # Split final isolé
    train_size: float = 0.80
    n_strat_bins_split: int = 20

    # K-folds sur le train uniquement
    n_splits: int = 5
    n_strat_bins_cv: int = 20

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

    # False = validation non pondérée pendant early stopping.
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

    # Interprétabilité
    compute_shap_final: bool = True
    shap_max_rows_final: int = 3000
    shap_top_n: int = 30


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

# 3. CHEMINS ET CHARGEMENT

def get_paths(cfg: Config) -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent

    candidate_data_dirs = [
        script_dir / "data",
        script_dir,
        script_dir / "Donnees_Tabulaires",
        script_dir.parent / "Donnees_Tabulaires",
        script_dir.parent.parent / "Tabulaire" / "Donnees_Tabulaires",
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "Donnees_Tabulaires",
    ]

    data_dir = None
    input_path = None

    for d in candidate_data_dirs:
        if not d.exists():
            continue
        for filename in cfg.input_filenames:
            p = d / filename
            if p.exists():
                data_dir = d
                input_path = p
                break
        if input_path is not None:
            break

    if input_path is None:
        searched = []
        for d in candidate_data_dirs:
            for filename in cfg.input_filenames:
                searched.append(str(d / filename))
        raise FileNotFoundError(
            "Aucun fichier tabulaire trouvé. Chemins testés :\n" + "\n".join(searched)
        )

    output_dir = script_dir / cfg.output_dir_name

    paths = {
        "script_dir": script_dir,
        "data_dir": data_dir,
        "input_path": input_path,
        "output_dir": output_dir,
        "reports_dir": output_dir / "rapports",
        "plots_dir": output_dir / "graphiques",
        "predictions_dir": output_dir / "predictions",
        "models_dir": output_dir / "modeles",
        "shap_dir": output_dir / "shap",
    }

    for p in paths.values():
        if isinstance(p, Path) and p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)

    return paths


def load_tabular(paths: Dict[str, Path]) -> pd.DataFrame:
    input_path = paths["input_path"]

    print("\n================ CHARGEMENT TABULAIRE ================")
    print("Fichier :", input_path)

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, dtype={CFG.id_col: str}, low_memory=False)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, dtype={CFG.id_col: str}, engine="openpyxl")
    else:
        raise ValueError("Format non supporté. Utilise .csv ou .xlsx.")

    print("Dimensions :", df.shape)
    return df

# 4. AUDIT ET PRÉPARATION FEATURES

def audit_dataset(df: pd.DataFrame, cfg: Config, paths: Dict[str, Path]) -> None:
    missing = pd.DataFrame({
        "colonne": df.columns,
        "type": df.dtypes.astype(str).values,
        "valeurs_manquantes": df.isna().sum().values,
        "pourcentage_manquant": (df.isna().sum().values / len(df) * 100).round(2),
        "nb_valeurs_uniques": df.nunique(dropna=False).values,
    }).sort_values("pourcentage_manquant", ascending=False)

    missing.to_csv(
        paths["reports_dir"] / "01_audit_valeurs_manquantes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    pd.DataFrame({"colonne": constant_cols}).to_csv(
        paths["reports_dir"] / "02_colonnes_constantes_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if cfg.target_real in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(pd.to_numeric(df[cfg.target_real], errors="coerce").dropna(), bins=80)
        plt.title("Distribution de price")
        plt.xlabel("price")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(paths["plots_dir"] / "01_distribution_price.png", dpi=150)
        plt.close()

    if cfg.target_log in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(pd.to_numeric(df[cfg.target_log], errors="coerce").dropna(), bins=80)
        plt.title("Distribution de log_price")
        plt.xlabel("log_price")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(paths["plots_dir"] / "02_distribution_log_price.png", dpi=150)
        plt.close()


def should_exclude_column(col: str, cfg: Config) -> bool:
    if col in cfg.cols_to_exclude:
        return True

    if col in cfg.raw_text_cols_to_exclude:
        return True

    for prefix in cfg.forbidden_prefixes:
        if col.startswith(prefix):
            return True

    return False


def prepare_features(df: pd.DataFrame, cfg: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    print("\n================ PRÉPARATION FEATURES TABULAIRES ================")

    if cfg.target_log not in df.columns:
        raise ValueError(f"Cible absente : {cfg.target_log}")
    if cfg.target_real not in df.columns:
        raise ValueError(f"Cible réelle absente : {cfg.target_real}")

    ids = (
        df[cfg.id_col].astype(str)
        if cfg.id_col in df.columns
        else pd.Series(np.arange(len(df)).astype(str), index=df.index)
    )

    y_log = pd.to_numeric(df[cfg.target_log], errors="coerce")
    y_real = pd.to_numeric(df[cfg.target_real], errors="coerce")

    valid_mask = y_log.notna() & y_real.notna()

    df_valid = df.loc[valid_mask].reset_index(drop=True)
    y_log = y_log.loc[valid_mask].reset_index(drop=True)
    y_real = y_real.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)

    excluded_by_rule = [c for c in df_valid.columns if should_exclude_column(c, cfg)]

    feature_cols = [c for c in df_valid.columns if c not in excluded_by_rule]
    X = df_valid[feature_cols].copy()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    cat_features = [c for c in cfg.base_cat_features if c in X.columns]

    object_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_cols:
        if col not in cat_features:
            cat_features.append(col)

    for col in cat_features:
        X[col] = X[col].fillna("missing").astype(str)

    for col in X.columns:
        if col not in cat_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    forbidden_still_present = [c for c in X.columns if should_exclude_column(c, cfg)]
    if forbidden_still_present:
        raise ValueError(f"Colonnes interdites encore présentes dans X : {forbidden_still_present}")

    print("Lignes utilisées :", len(X))
    print("Variables utilisées :", X.shape[1])
    print("Catégorielles CatBoost :", cat_features)
    print("Colonnes exclues par règle :", len(excluded_by_rule))
    print("Colonnes constantes supprimées :", constant_cols)
    print("price dans X :", cfg.target_real in X.columns)
    print("log_price dans X :", cfg.target_log in X.columns)
    print("id_clean dans X :", cfg.id_col in X.columns)
    print("text_pred_final dans X :", "text_pred_final" in X.columns)
    print("txt_e5_* dans X :", any(c.startswith("txt_e5_") for c in X.columns))

    pd.DataFrame({"feature": X.columns}).to_csv(
        paths["reports_dir"] / "03_features_utilisees_tabulaire_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"cat_feature": cat_features}).to_csv(
        paths["reports_dir"] / "04_cat_features_tabulaire_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"excluded_column": sorted(excluded_by_rule)}).to_csv(
        paths["reports_dir"] / "05_colonnes_exclues_tabulaire_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"constant_column_removed": constant_cols}).to_csv(
        paths["reports_dir"] / "06_colonnes_constantes_supprimees.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "X": X,
        "y_log": y_log,
        "y_real": y_real,
        "ids": ids,
        "cat_features": cat_features,
        "feature_names": list(X.columns),
        "excluded_by_rule": excluded_by_rule,
        "constant_cols": constant_cols,
    }

# 5. SPLIT FINAL ISOLÉ

def make_final_train_test_split(data: Dict[str, Any], cfg: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    X = data["X"]
    y_log = data["y_log"]
    y_real = data["y_real"]
    ids = data["ids"]

    strat_bins = make_regression_strat_bins(
        y=y_log,
        n_splits=2,
        max_bins=cfg.n_strat_bins_split,
    )

    indices = np.arange(len(X))

    idx_train, idx_test = train_test_split(
        indices,
        train_size=cfg.train_size,
        random_state=cfg.random_state,
        stratify=strat_bins,
    )

    split_report = pd.DataFrame({
        "split": ["train_dev", "test_final"],
        "n": [len(idx_train), len(idx_test)],
        "price_mean": [y_real.iloc[idx_train].mean(), y_real.iloc[idx_test].mean()],
        "price_median": [y_real.iloc[idx_train].median(), y_real.iloc[idx_test].median()],
        "price_min": [y_real.iloc[idx_train].min(), y_real.iloc[idx_test].min()],
        "price_max": [y_real.iloc[idx_train].max(), y_real.iloc[idx_test].max()],
        "log_price_mean": [y_log.iloc[idx_train].mean(), y_log.iloc[idx_test].mean()],
    })

    split_report.to_csv(
        paths["reports_dir"] / "07_split_final_train_test_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_ids = pd.DataFrame({
        "id_clean": pd.concat([
            ids.iloc[idx_train].reset_index(drop=True),
            ids.iloc[idx_test].reset_index(drop=True),
        ], ignore_index=True),
        "split": ["train_dev"] * len(idx_train) + ["test_final"] * len(idx_test),
    })

    split_ids.to_csv(
        paths["reports_dir"] / "08_split_final_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n================ SPLIT FINAL ISOLÉ ================")
    print(split_report)

    return {
        "X_train_dev": X.iloc[idx_train].reset_index(drop=True),
        "X_test_final": X.iloc[idx_test].reset_index(drop=True),
        "y_train_dev_log": y_log.iloc[idx_train].reset_index(drop=True),
        "y_test_final_log": y_log.iloc[idx_test].reset_index(drop=True),
        "y_train_dev_real": y_real.iloc[idx_train].reset_index(drop=True),
        "y_test_final_real": y_real.iloc[idx_test].reset_index(drop=True),
        "ids_train_dev": ids.iloc[idx_train].reset_index(drop=True),
        "ids_test_final": ids.iloc[idx_test].reset_index(drop=True),
        "cat_features": data["cat_features"],
        "feature_names": data["feature_names"],
        "excluded_by_rule": data["excluded_by_rule"],
        "constant_cols": data["constant_cols"],
    }


# 6. PONDÉRATION

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
        "id_clean": ids.values,
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

# 8. CATBOOST PARAMS

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


# 9. K-FOLD CV SUR TRAIN UNIQUEMENT

def run_kfold_cv(split_data: Dict[str, Any], cfg: Config, paths: Dict[str, Path]):
    X = split_data["X_train_dev"]
    y_log = split_data["y_train_dev_log"]
    y_real = split_data["y_train_dev_real"]
    ids = split_data["ids_train_dev"]
    cat_features = split_data["cat_features"]

    print("\n" + "=" * 100)
    print("K-FOLD CROSS-VALIDATION SUR TRAIN_DEV UNIQUEMENT")
    print("=" * 100)
    print("Nombre de folds :", cfg.n_splits)
    print("Train_dev utilisé pour CV :", X.shape)
    print("Test final non utilisé ici.")

    strat_bins = make_regression_strat_bins(
        y=y_log,
        n_splits=cfg.n_splits,
        max_bins=cfg.n_strat_bins_cv,
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
        print(fold_name)
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

        model_path = paths["models_dir"] / f"catboost_tabular_cv_{fold_name}.cbm"
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
        "selection_rule": "median_tree_count_from_kfold_cv_on_train_dev_only",
        "test_final_used_in_cv": False,
        "weight_validation_pool": cfg.weight_validation_pool,
    }

    with open(paths["reports_dir"] / "17_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(cv_summary), f, indent=2, ensure_ascii=False)

    with open(paths["reports_dir"] / "18_cv_resume.txt", "w", encoding="utf-8") as f:
        f.write("Résumé CV - CatBoost tabulaire seul\n")
        f.write("=" * 90 + "\n\n")
        f.write("La validation croisée est faite uniquement sur train_dev.\n")
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

# 10. ENTRAÎNEMENT FINAL + TEST FINAL

def train_final_model_and_evaluate(
    split_data: Dict[str, Any],
    cfg: Config,
    paths: Dict[str, Path],
    final_iterations: int,
    cv_results: Dict[str, Any],
):
    X_train = split_data["X_train_dev"]
    X_test = split_data["X_test_final"]

    y_train_log = split_data["y_train_dev_log"]
    y_train_real = split_data["y_train_dev_real"]

    y_test_log = split_data["y_test_final_log"]
    y_test_real = split_data["y_test_final_real"]

    ids_train = split_data["ids_train_dev"]
    ids_test = split_data["ids_test_final"]

    cat_features = split_data["cat_features"]
    feature_names = split_data["feature_names"]

    print("\n" + "=" * 100)
    print("ENTRAÎNEMENT DU MODÈLE FINAL SUR TRAIN_DEV")
    print("=" * 100)
    print("Itérations finales :", final_iterations)
    print("Train_dev :", X_train.shape)
    print("Test final :", X_test.shape)

    w_train, normalizer = make_manual_aggressive_weights(
        y_train_real,
        cfg,
        normalizer=None,
    )

    weight_diag = compute_weight_diagnostics(
        y_real=y_train_real,
        weights=w_train,
        cfg=cfg,
        split_name="train_dev",
        fold_name="final_model",
    )

    weight_diag.to_csv(
        paths["reports_dir"] / "20_final_weight_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_pool = Pool(
        data=X_train,
        label=y_train_log,
        cat_features=cat_features,
        weight=w_train.values,
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

    final_model_path = paths["models_dir"] / "catboost_final_tabular_only_kfold_manual_aggressive_c05.cbm"
    final_model.save_model(str(final_model_path))

    print("Modèle final sauvegardé :", final_model_path)

    split_items = [
        ("train_dev", X_train, y_train_log, y_train_real, ids_train),
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
                "model": "final_model_train_dev",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "tabular_only_kfold",
                "split": split_name,
                "scale": "log_price",
                "metric": metric,
                "value": value,
            })

        for metric, value in m_price.items():
            metrics_rows.append({
                "model": "final_model_train_dev",
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "tabular_only_kfold",
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
        save_shap_final(final_model, X_train, y_train_log, cat_features, cfg, paths)

    write_final_report(
        model=final_model,
        metrics_df=metrics_df,
        segments_df=segments_df,
        cv_results=cv_results,
        split_data=split_data,
        cfg=cfg,
        paths=paths,
        final_model_path=final_model_path,
    )

    model_config = {
        "candidate": "manual_aggressive__C05_premium_oriented",
        "experiment": "tabular_only_kfold",
        "config": asdict(cfg),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "excluded_by_rule": split_data["excluded_by_rule"],
        "constant_cols": split_data["constant_cols"],
        "final_model": {
            "model_path": str(final_model_path),
            "iterations": int(final_iterations),
            "selection_rule": "median tree_count from K-fold CV on train_dev only",
        },
        "important": (
            "Modèle tabulaire seul. "
            "price, log_price, id_clean, nights_range_is_incoherent, has_reviews, "
            "nb_avis_textuels_bert et bert_stars_moyen sont exclus. "
            "Les colonnes texte, image, embeddings et prédictions texte sont aussi exclues si présentes. "
            "La CV est faite uniquement sur train_dev. "
            "Le test final est utilisé une seule fois après le choix du nombre d'arbres."
        ),
    }

    with open(paths["models_dir"] / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(model_config), f, indent=2, ensure_ascii=False)

    return {
        "model": final_model,
        "metrics": metrics_df,
        "segments": segments_df,
        "predictions": preds,
    }

# 11. IMPORTANCE, SHAP, GRAPHIQUES, RAPPORTS

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
    plt.title("CatBoost tabulaire seul - Top 30 variables")
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
    plt.title("CatBoost tabulaire seul - Top 30 SHAP")
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
    plt.title("OOF CV - Prix prédit vs réel\nCatBoost tabulaire seul")
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
    plt.title("Test final - Prix prédit vs réel\nCatBoost tabulaire seul")
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
    split_data,
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

    report_path = paths["reports_dir"] / "30_resume_final_tabular_only_kfold_catboost.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Résumé final - CatBoost tabulaire seul + K-fold CV\n")
        f.write("=" * 100 + "\n\n")

        f.write("Candidat retenu\n")
        f.write("-" * 100 + "\n")
        f.write("manual_aggressive__C05_premium_oriented\n\n")

        f.write("Protocole\n")
        f.write("-" * 100 + "\n")
        f.write("Un split final train_dev/test_final est créé à partir du fichier tabulaire.\n")
        f.write("La validation croisée 5-fold est faite uniquement sur train_dev.\n")
        f.write("Le test final n'est jamais utilisé pendant la CV.\n")
        f.write("Les poids manual_aggressive sont calculés seulement sur le train de chaque fold.\n")
        f.write("La validation du fold reste non pondérée sauf si weight_validation_pool=True.\n")
        f.write("Le nombre d'arbres du modèle final est la médiane des tree_count obtenus en CV.\n")
        f.write("Le modèle final est entraîné sur train_dev puis évalué une seule fois sur test_final.\n")
        f.write("Les colonnes textuelles, embeddings, images et prédictions texte sont exclues si elles existent.\n\n")

        f.write("Colonnes retirées importantes\n")
        f.write("-" * 100 + "\n")
        for c in [
            "id_clean",
            "price",
            "log_price",
            "nights_range_is_incoherent",
            "has_reviews",
            "nb_avis_textuels_bert",
            "bert_stars_moyen",
            "text_pred_final",
            "txt_e5_*",
        ]:
            f.write(f"{c}\n")
        f.write("\n")

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
    


def main():
    paths = get_paths(CFG)

    print("\n" + "=" * 100)
    print("CATBOOST TABULAIRE SEUL - K-FOLD CV - MANUAL AGGRESSIVE C05")
    print("=" * 100)
    print("Dossier script :", paths["script_dir"])
    print("Dossier data :", paths["data_dir"])
    print("Fichier utilisé :", paths["input_path"])
    print("Dossier sorties :", paths["output_dir"])

    df = load_tabular(paths)

    audit_dataset(df, CFG, paths)

    prepared = prepare_features(df, CFG, paths)

    split_data = make_final_train_test_split(prepared, CFG, paths)

    with open(paths["reports_dir"] / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable({
                "candidate": "manual_aggressive__C05_premium_oriented",
                "experiment": "tabular_only_kfold",
                "config": asdict(CFG),
                "important": (
                    "Modèle tabulaire seul. "
                    "Colonnes interdites exclues. "
                    "Split final isolé. "
                    "K-fold uniquement sur train_dev. "
                    "Test final utilisé une seule fois."
                ),
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    cv_results = run_kfold_cv(split_data, CFG, paths)

    train_final_model_and_evaluate(
        split_data=split_data,
        cfg=CFG,
        paths=paths,
        final_iterations=cv_results["final_iterations"],
        cv_results=cv_results,
    )

    print("\n" + "=" * 100)
    print("FIN - CATBOOST TABULAIRE SEUL - K-FOLD CV")
    print("=" * 100)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])

    print("\nFichiers importants à m'envoyer pour analyse :")
    print(paths["reports_dir"] / "03_features_utilisees_tabulaire_only.csv")
    print(paths["reports_dir"] / "05_colonnes_exclues_tabulaire_only.csv")
    print(paths["reports_dir"] / "07_split_final_train_test_report.csv")
    print(paths["reports_dir"] / "10_cv_fold_metrics_log_price.csv")
    print(paths["reports_dir"] / "11_cv_fold_metrics_price_euros.csv")
    print(paths["reports_dir"] / "15_cv_oof_metrics_global.csv")
    print(paths["reports_dir"] / "16_cv_oof_segments_global.csv")
    print(paths["reports_dir"] / "17_cv_summary.json")
    print(paths["reports_dir"] / "21_final_metrics_train_test.csv")
    print(paths["reports_dir"] / "22_final_segments_train_test.csv")
    print(paths["reports_dir"] / "23_final_metrics_pivot.csv")
    print(paths["reports_dir"] / "24_final_feature_importance_catboost.csv")
    print(paths["reports_dir"] / "30_resume_final_tabular_only_kfold_catboost.txt")
    print(paths["predictions_dir"] / "10_oof_predictions_all_folds.csv")
    print(paths["predictions_dir"] / "20_predictions_test_final.csv")


if __name__ == "__main__":
    main()
