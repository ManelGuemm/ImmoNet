import os
import json
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except Exception as exc:
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")

# CONFIGURATION GÉNÉRALE

DATA_PATH = "/workspace/airbnb_tabulaire_fusionné_sentiment.csv"
OUTPUT_DIR = Path("/workspace/resultats_pipeline_complet_pondere_shap_kbest")
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Pour limiter la durée
N_ITER_SKLEARN = 10
N_ITER_CATBOOST = 8

# Sélection et géographie
K_VALUES_SELECTKBEST = [50, 100, 150, "all"]
N_GEO_CLUSTERS = 25
RARE_MIN_FREQUENCY = 50
SHAP_SAMPLE_SIZE = 5000

# Colonnes à retirer des variables explicatives.
DROP_FEATURES = [
    "price",
    "log_price",
    "id_clean",
    "nights_range_is_incoherent",
    # Variables d'avis considérées comme redondantes / peu défendues dans le mémoire.
    "has_reviews",
    "nb_avis_textuels_bert",
    "bert_stars_moyen",
]

PRICE_SEGMENT_BINS = [0, 100, 200, 400, 800, np.inf]
PRICE_SEGMENT_LABELS = ["<100", "100-200", "200-400", "400-800", ">800"]

# Pondération retenue dans le mémoire pour mieux traiter les logements chers.
# Elle est appliquée à tous les modèles de cette expérience pour rendre la comparaison coherente
WEIGHT_STRATEGY_NAME = "manual_aggressive"
MANUAL_AGGRESSIVE_WEIGHTS = {
    "<100": 1.00,
    "100-200": 1.00,
    "200-400": 1.15,
    "400-800": 2.00,
    ">800": 3.50,
}

# Score composite 
# Le modèle n'est pas choisi seulement par MAE globale : on tient aussi compte du RMSE,
# des segments élevés et du biais de sous/surestimation.
PREMIUM_SCORE_WEIGHTS = {
    "mae_global": 0.45,
    "rmse_global": 0.25,
    "mae_400_800": 0.10,
    "mae_gt_800": 0.10,
    "abs_bias_global": 0.05,
    "abs_bias_gt_800": 0.05,
}

# DOSSIERS DE SORTIE

SUBDIRS = {
    "global": OUTPUT_DIR / "00_global",
    "logs": OUTPUT_DIR / "01_logs",
    "plots": OUTPUT_DIR / "02_plots",
    "predictions": OUTPUT_DIR / "03_predictions",
    "reports": OUTPUT_DIR / "04_reports",
    "models": OUTPUT_DIR / "05_models",
    "catboost_all": OUTPUT_DIR / "05_models" / "catboost_weighted_all_features",
    "catboost_top30": OUTPUT_DIR / "05_models" / "catboost_weighted_shap_top30",
    "catboost_top50": OUTPUT_DIR / "05_models" / "catboost_weighted_shap_top50",
    "sklearn": OUTPUT_DIR / "05_models" / "sklearn_models",
}

for folder in SUBDIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SUBDIRS["logs"] / "airbnb_training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def safe_expm1(values):
    """Évite les valeurs négatives après reconversion."""
    arr = np.expm1(values)
    return np.maximum(arr, 0)


def compute_smearing_factor(y_true_log, y_pred_log):
    """
    Smearing estimate adapté à log1p(price).

    Si log_y = log(1 + price), alors :
        1 + price = exp(log_y)
    Correction :
        pred_price = exp(pred_log) * mean(exp(residual_log)) - 1
    """
    residuals_log = np.asarray(y_true_log) - np.asarray(y_pred_log)
    factor = float(np.mean(np.exp(residuals_log)))
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
    return factor


def log_to_price_naive(log_pred):
    return safe_expm1(np.asarray(log_pred))


def log_to_price_smearing(log_pred, smearing_factor):
    price = np.exp(np.asarray(log_pred)) * smearing_factor - 1.0
    return np.maximum(price, 0)


def regression_metrics(y_true_price, y_pred_price, prefix=""):
    y_true_price = np.asarray(y_true_price)
    y_pred_price = np.asarray(y_pred_price)

    return {
        f"{prefix}mae": float(mean_absolute_error(y_true_price, y_pred_price)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        f"{prefix}r2": float(r2_score(y_true_price, y_pred_price)),
        f"{prefix}mape": float(mean_absolute_percentage_error(y_true_price, y_pred_price)),
    }


def log_metrics(y_true_log, y_pred_log, prefix=""):
    return {
        f"{prefix}mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        f"{prefix}rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        f"{prefix}r2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def neg_mae_euro_from_log(y_true_log, y_pred_log):
    """Scorer sklearn : sélection en euros après reconversion naïve."""
    y_true_price = log_to_price_naive(y_true_log)
    y_pred_price = log_to_price_naive(y_pred_log)
    return -mean_absolute_error(y_true_price, y_pred_price)


euro_mae_scorer = make_scorer(neg_mae_euro_from_log, greater_is_better=True)


def compute_manual_aggressive_weights(price_values, normalize=True):
    """
    Poids manual_aggressive utilisés pour donner plus d'importance aux logements chers.
    Les poids sont calculés à partir du prix réel.
    """
    segments = make_price_segments(price_values)
    weights = segments.map(MANUAL_AGGRESSIVE_WEIGHTS).astype(float).values
    if normalize:
        mean_w = np.nanmean(weights)
        if np.isfinite(mean_w) and mean_w > 0:
            weights = weights / mean_w
    return weights


def premium_composite_loss_from_prices(y_true_price, y_pred_price):
    """Score composite à minimiser : global + segments chers + biais."""
    y_true_price = np.asarray(y_true_price)
    y_pred_price = np.asarray(y_pred_price)
    err = y_pred_price - y_true_price

    mae_global = mean_absolute_error(y_true_price, y_pred_price)
    rmse_global = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    abs_bias_global = abs(float(err.mean()))

    segments = make_price_segments(y_true_price)
    mask_400_800 = segments == "400-800"
    mask_gt_800 = segments == ">800"

    mae_400_800 = mean_absolute_error(y_true_price[mask_400_800], y_pred_price[mask_400_800]) if mask_400_800.any() else mae_global
    mae_gt_800 = mean_absolute_error(y_true_price[mask_gt_800], y_pred_price[mask_gt_800]) if mask_gt_800.any() else mae_global
    abs_bias_gt_800 = abs(float((y_pred_price[mask_gt_800] - y_true_price[mask_gt_800]).mean())) if mask_gt_800.any() else abs_bias_global

    return float(
        PREMIUM_SCORE_WEIGHTS["mae_global"] * mae_global
        + PREMIUM_SCORE_WEIGHTS["rmse_global"] * rmse_global
        + PREMIUM_SCORE_WEIGHTS["mae_400_800"] * mae_400_800
        + PREMIUM_SCORE_WEIGHTS["mae_gt_800"] * mae_gt_800
        + PREMIUM_SCORE_WEIGHTS["abs_bias_global"] * abs_bias_global
        + PREMIUM_SCORE_WEIGHTS["abs_bias_gt_800"] * abs_bias_gt_800
    )


def premium_composite_loss_from_log(y_true_log, y_pred_log):
    y_true_price = log_to_price_naive(y_true_log)
    y_pred_price = log_to_price_naive(y_pred_log)
    return premium_composite_loss_from_prices(y_true_price, y_pred_price)


premium_composite_scorer = make_scorer(premium_composite_loss_from_log, greater_is_better=False)


def make_price_segments(price_values):
    return pd.cut(
        price_values,
        bins=PRICE_SEGMENT_BINS,
        labels=PRICE_SEGMENT_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)


def segment_metrics(y_true_price, y_pred_price, model_name, retransform_type):
    df_seg = pd.DataFrame({
        "price_true": np.asarray(y_true_price),
        "price_pred": np.asarray(y_pred_price),
    })
    df_seg["segment"] = make_price_segments(df_seg["price_true"])
    rows = []

    for segment in PRICE_SEGMENT_LABELS:
        sub = df_seg[df_seg["segment"] == segment]
        if len(sub) == 0:
            continue
        err = sub["price_pred"] - sub["price_true"]
        rows.append({
            "model": model_name,
            "retransform": retransform_type,
            "segment": segment,
            "n": int(len(sub)),
            "true_mean": float(sub["price_true"].mean()),
            "pred_mean": float(sub["price_pred"].mean()),
            "mae": float(mean_absolute_error(sub["price_true"], sub["price_pred"])),
            "rmse": float(np.sqrt(mean_squared_error(sub["price_true"], sub["price_pred"]))),
            "bias_pred_minus_true": float(err.mean()),
        })
    return pd.DataFrame(rows)


class BasicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering déterministe, sans apprentissage de cible.
    Ne crée pas reviews_per_month ni quality_score.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        profile_features = [
            col for col in ["has_host_profile_pic", "host_identity_verified", "has_host_about"]
            if col in X.columns
        ]
        if profile_features:
            X["host_completeness"] = X[profile_features].sum(axis=1) / len(profile_features)

        if "minimum_nights_clean" in X.columns and "maximum_nights_clean" in X.columns:
            X["nights_flexibility"] = np.log1p(
                X["maximum_nights_clean"].astype(float) / (X["minimum_nights_clean"].astype(float) + 1.0)
            )

        return X


class GeoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Ajoute :
    - geo_cluster appris par KMeans sur train/fold train seulement
    - neighbourhood_popularity calculée sur train/fold train seulement
    """

    def __init__(self, n_clusters=25, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        X = X.copy()
        self.has_geo_ = "latitude" in X.columns and "longitude" in X.columns
        self.has_neigh_ = "neighbourhood_cleansed_clean" in X.columns

        if self.has_geo_:
            self.lat_median_ = float(pd.to_numeric(X["latitude"], errors="coerce").median())
            self.lon_median_ = float(pd.to_numeric(X["longitude"], errors="coerce").median())
            coords = pd.DataFrame({
                "latitude": pd.to_numeric(X["latitude"], errors="coerce").fillna(self.lat_median_),
                "longitude": pd.to_numeric(X["longitude"], errors="coerce").fillna(self.lon_median_),
            })
            n_clusters = min(self.n_clusters, max(2, len(coords)))
            self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            self.kmeans_.fit(coords)
        else:
            self.lat_median_ = None
            self.lon_median_ = None
            self.kmeans_ = None

        if self.has_neigh_:
            counts = X["neighbourhood_cleansed_clean"].astype(str).fillna("missing").value_counts(normalize=True)
            self.neigh_freq_ = counts.to_dict()
        else:
            self.neigh_freq_ = {}

        return self

    def transform(self, X):
        X = X.copy()

        if self.has_geo_ and self.kmeans_ is not None:
            coords = pd.DataFrame({
                "latitude": pd.to_numeric(X["latitude"], errors="coerce").fillna(self.lat_median_),
                "longitude": pd.to_numeric(X["longitude"], errors="coerce").fillna(self.lon_median_),
            })
            clusters = self.kmeans_.predict(coords)
            X["geo_cluster"] = pd.Series(clusters, index=X.index).astype(str).map(lambda v: f"geo_{v}")
        else:
            X["geo_cluster"] = "geo_missing"

        if self.has_neigh_:
            X["neighbourhood_popularity"] = (
                X["neighbourhood_cleansed_clean"]
                .astype(str)
                .fillna("missing")
                .map(self.neigh_freq_)
                .fillna(0.0)
                .astype(float)
            )
        else:
            X["neighbourhood_popularity"] = 0.0

        return X

# CHARGEMENT ET PRÉPARATION

def load_data(path):
    logger.info("=" * 90)
    logger.info("PHASE 1 - CHARGEMENT DES DONNÉES")
    logger.info("=" * 90)
    df = pd.read_csv(path)
    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    if "price" not in df.columns:
        raise ValueError("La colonne 'price' est absente du fichier.")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)].copy()

    if "log_price" not in df.columns:
        df["log_price"] = np.log1p(df["price"])
        logger.info("log_price créée avec np.log1p(price).")
    else:
        df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")
        missing_log = df["log_price"].isna()
        if missing_log.any():
            df.loc[missing_log, "log_price"] = np.log1p(df.loc[missing_log, "price"])
        logger.info("log_price existante utilisée, valeurs manquantes recalculées si besoin.")

    logger.info(
        f"Price - min={df['price'].min():.2f}, median={df['price'].median():.2f}, "
        f"mean={df['price'].mean():.2f}, max={df['price'].max():.2f}, std={df['price'].std():.2f}"
    )
    logger.info("Aucune suppression d'outliers : tous les prix positifs sont conservés.")

    missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_percent.to_csv(SUBDIRS["global"] / "missing_values_percent.csv")
    logger.info("Analyse des valeurs manquantes sauvegardée.")

    return df


def prepare_X_y(df):
    y_log = df["log_price"].copy()
    y_price = df["price"].copy()

    existing_drop = [col for col in DROP_FEATURES if col in df.columns]
    X = df.drop(columns=existing_drop).copy()

    pd.DataFrame({"dropped_features": existing_drop}).to_csv(
        SUBDIRS["global"] / "dropped_features.csv", index=False
    )
    logger.info(f"Colonnes retirées des features : {existing_drop}")
    logger.info(f"Nombre de features initiales après retrait : {X.shape[1]}")

    return X, y_log, y_price


def stratified_train_test_split(X, y_log, y_price):
    price_bins = make_price_segments(y_price)

    X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test = train_test_split(
        X,
        y_log,
        y_price,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=price_bins,
    )

    logger.info("=" * 90)
    logger.info("PHASE 2 - SPLIT TRAIN/TEST")
    logger.info("=" * 90)
    logger.info(f"Train : {len(X_train)} lignes ({100 * len(X_train) / len(X):.1f}%)")
    logger.info(f"Test  : {len(X_test)} lignes ({100 * len(X_test) / len(X):.1f}%)")

    split_report = pd.DataFrame({
        "split": ["train", "test"],
        "n": [len(X_train), len(X_test)],
        "price_mean": [y_price_train.mean(), y_price_test.mean()],
        "price_median": [y_price_train.median(), y_price_test.median()],
        "price_min": [y_price_train.min(), y_price_test.min()],
        "price_max": [y_price_train.max(), y_price_test.max()],
    })
    split_report.to_csv(SUBDIRS["global"] / "split_train_test_report.csv", index=False)

    return X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test

# PIPELINE SCIKIT-LEARN

def make_onehot_encoder():
    """Compatibilité selon version scikit-learn."""
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=RARE_MIN_FREQUENCY,
            sparse_output=False,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=RARE_MIN_FREQUENCY,
                sparse=False,
            )
        except TypeError:
            logger.warning("OneHotEncoder sans min_frequency disponible : fallback handle_unknown='ignore'.")
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_sklearn_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer, make_column_selector(dtype_include=[object, "category"])),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_sklearn_pipeline(model):
    pipe = Pipeline(steps=[
        ("basic_features", BasicFeatureEngineer()),
        ("geo_features", GeoFeatureEngineer(n_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE)),
        ("preprocessor", build_sklearn_preprocessor()),
        ("feature_selection", SelectKBest(score_func=mutual_info_regression, k=100)),
        ("model", model),
    ])
    return pipe


def get_sklearn_model_spaces():
    return {
        "Ridge": {
            "model": Ridge(random_state=RANDOM_STATE),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
            },
        },
        "Lasso": {
            "model": Lasso(max_iter=10000, random_state=RANDOM_STATE),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            },
        },
        "ElasticNet": {
            "model": ElasticNet(max_iter=10000, random_state=RANDOM_STATE),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [12, 20, 30, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__n_estimators": [100, 150, 250, 350],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__subsample": [0.8, 1.0],
            },
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": {
                "feature_selection__k": K_VALUES_SELECTKBEST,
                "model__max_iter": [100, 200, 350],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        },
    }


def train_sklearn_models(X_train, y_log_train, y_price_train):
    logger.info("=" * 90)
    logger.info("PHASE 3 - MODÈLES SCIKIT-LEARN : PIPELINE + RANDOMIZEDSEARCHCV PONDÉRÉ")
    logger.info("=" * 90)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    strat_bins = make_price_segments(y_price_train)
    sample_weight_train = compute_manual_aggressive_weights(y_price_train, normalize=True)

    pd.DataFrame({
        "segment": make_price_segments(y_price_train),
        "weight": sample_weight_train,
    }).groupby("segment", as_index=False).agg(
        n=("weight", "size"), mean_weight=("weight", "mean"), min_weight=("weight", "min"), max_weight=("weight", "max")
    ).to_csv(SUBDIRS["global"] / "manual_aggressive_weights_train_summary.csv", index=False)

    models_info = get_sklearn_model_spaces()
    trained = {}
    rows = []

    scoring = {
        "premium_composite": premium_composite_scorer,
        "mae_euro_naive": euro_mae_scorer,
    }

    for model_name, cfg in models_info.items():
        start = time.time()
        logger.info(f"\n--- Entraînement sklearn pondéré : {model_name} ---")
        model_dir = SUBDIRS["sklearn"] / f"{model_name}_weighted"
        model_dir.mkdir(parents=True, exist_ok=True)

        pipe = build_sklearn_pipeline(cfg["model"])
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=cfg["params"],
            n_iter=N_ITER_SKLEARN,
            scoring=scoring,
            refit="premium_composite",
            cv=skf.split(X_train, strat_bins),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_log_train, model__sample_weight=sample_weight_train)

        duration = time.time() - start
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(model_dir / "cv_results.csv", index=False)

        best_idx = int(search.best_index_)
        best_cv_composite = float(-search.best_score_)
        best_cv_mae = float(-cv_results.loc[best_idx, "mean_test_mae_euro_naive"])

        logger.info(f"Meilleurs paramètres {model_name} : {search.best_params_}")
        logger.info(f"Meilleur score composite CV : {best_cv_composite:.4f}")
        logger.info(f"MAE euros naïve CV du meilleur candidat : {best_cv_mae:.4f}")

        with open(model_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(search.best_params_, f, indent=2, ensure_ascii=False)

        trained[f"{model_name}_weighted"] = search.best_estimator_
        rows.append({
            "model": f"{model_name}_weighted",
            "base_model": model_name,
            "family": "sklearn",
            "weight_strategy": WEIGHT_STRATEGY_NAME,
            "cv_composite_score": best_cv_composite,
            "cv_mae_euro_naive": best_cv_mae,
            "duration_seconds": float(duration),
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        })

        try:
            selected_df = extract_selected_features_from_sklearn_pipeline(search.best_estimator_)
            selected_df.to_csv(model_dir / "selected_features.csv", index=False)
        except Exception as exc:
            logger.warning(f"Impossible d'extraire les features sélectionnées pour {model_name}: {exc}")

        joblib.dump(search.best_estimator_, model_dir / "best_pipeline.pkl")

    cv_summary = pd.DataFrame(rows)
    cv_summary.to_csv(SUBDIRS["global"] / "sklearn_cv_summary.csv", index=False)
    return trained, cv_summary


def extract_selected_features_from_sklearn_pipeline(pipe):
    preprocessor = pipe.named_steps["preprocessor"]
    selector = pipe.named_steps["feature_selection"]

    feature_names = preprocessor.get_feature_names_out()
    scores = selector.scores_

    if selector.k == "all":
        mask = np.ones(len(feature_names), dtype=bool)
    else:
        mask = selector.get_support()

    df_scores = pd.DataFrame({
        "feature": feature_names,
        "score": scores,
        "selected": mask,
    }).sort_values(["selected", "score"], ascending=[False, False])

    return df_scores

# CATBOOST : PRÉPROCESSING NATIF

class CatBoostPreprocessor:
    """Préprocessing pour CatBoost sans OneHot : catégories natives."""

    def __init__(self, n_geo_clusters=25, random_state=42):
        self.basic = BasicFeatureEngineer()
        self.geo = GeoFeatureEngineer(n_clusters=n_geo_clusters, random_state=random_state)
        self.numeric_medians_ = {}
        self.categorical_cols_ = []
        self.numeric_cols_ = []
        self.columns_ = []

    def fit(self, X, y=None):
        X2 = self.basic.fit_transform(X)
        X2 = self.geo.fit_transform(X2)

        self.categorical_cols_ = X2.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols_ = [c for c in X2.columns if c not in self.categorical_cols_]

        for col in self.numeric_cols_:
            self.numeric_medians_[col] = float(pd.to_numeric(X2[col], errors="coerce").median())
            if not np.isfinite(self.numeric_medians_[col]):
                self.numeric_medians_[col] = 0.0

        self.columns_ = X2.columns.tolist()
        return self

    def transform(self, X):
        X2 = self.basic.transform(X)
        X2 = self.geo.transform(X2)

        # Sécurité : garder les colonnes du fit dans le même ordre.
        for col in self.columns_:
            if col not in X2.columns:
                X2[col] = np.nan
        X2 = X2[self.columns_].copy()

        for col in self.numeric_cols_:
            X2[col] = pd.to_numeric(X2[col], errors="coerce").fillna(self.numeric_medians_[col])

        for col in self.categorical_cols_:
            X2[col] = X2[col].astype(str).replace({"nan": "missing", "None": "missing"}).fillna("missing")

        return X2

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def get_cat_features_indices(X_cat, categorical_cols):
    return [X_cat.columns.get_loc(c) for c in categorical_cols if c in X_cat.columns]


def catboost_candidate_params():
    """
    Candidats CatBoost.
    On force l'inclusion d'une configuration proche du modèle optimisé manual_aggressive__C05
    du mémoire, puis on ajoute des configurations plus rapides/exploratoires.
    """
    forced_candidates = [
        {"name": "C05_manual_aggressive_premium_like", "iterations": 8000, "learning_rate": 0.015, "depth": 7, "l2_leaf_reg": 8, "random_strength": 1.0, "bagging_temperature": 0.6, "rsm": 0.95},
        {"name": "C04_manual_aggressive_premium_like", "iterations": 5000, "learning_rate": 0.02, "depth": 7, "l2_leaf_reg": 8, "random_strength": 1.0, "bagging_temperature": 0.6, "rsm": 0.95},
    ]
    random_space = {
        "iterations": [800, 1200, 2000, 3000],
        "depth": [4, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "l2_leaf_reg": [3, 5, 8, 12],
        "random_strength": [0.5, 1.0, 2.0],
        "bagging_temperature": [0.0, 0.5, 0.8, 1.0],
        "rsm": [0.8, 0.95, 1.0],
    }
    sampled = list(ParameterSampler(random_space, n_iter=max(0, N_ITER_CATBOOST - len(forced_candidates)), random_state=RANDOM_STATE))
    for i, p in enumerate(sampled, start=1):
        p["name"] = f"random_{i:02d}"
    return forced_candidates + sampled


def make_catboost_model(params):
    params = dict(params)
    params.pop("name", None)
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **params,
    )


def sample_df(df, max_size=5000, random_state=42):
    if len(df) <= max_size:
        return df
    return df.sample(n=max_size, random_state=random_state)


def compute_catboost_shap_importance(model, X_processed, cat_features, out_csv=None, out_png=None, title="CatBoost SHAP"):
    X_sample = sample_df(X_processed, SHAP_SAMPLE_SIZE, RANDOM_STATE)
    pool = Pool(X_sample, cat_features=cat_features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")

    # Dernière colonne = expected value
    mean_abs_shap = np.abs(shap_values[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_processed.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    if out_csv is not None:
        shap_df.to_csv(out_csv, index=False)

    if out_png is not None:
        top = shap_df.head(25).sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean |SHAP|")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

    return shap_df


def evaluate_catboost_cv_variant(X_train, y_log_train, y_price_train, params, variant="all", top_k=None):
    """
    CV propre pour CatBoost pondéré.
    Pour top_k : sélection SHAP faite dans chaque fold uniquement sur le fold train.
    Les poids manual_aggressive sont calculés sur le fold train seulement.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    strat_bins = make_price_segments(y_price_train)
    fold_rows = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, strat_bins), start=1):
        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]
        y_tr_log = y_log_train.iloc[tr_idx]
        y_val_log = y_log_train.iloc[val_idx]
        y_tr_price = y_price_train.iloc[tr_idx]
        y_val_price = y_price_train.iloc[val_idx]
        tr_weights = compute_manual_aggressive_weights(y_tr_price, normalize=True)

        prep = CatBoostPreprocessor(n_geo_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE)
        X_tr = prep.fit_transform(X_tr_raw)
        X_val = prep.transform(X_val_raw)
        cat_cols = prep.categorical_cols_
        cat_features = get_cat_features_indices(X_tr, cat_cols)
        selected_features = X_tr.columns.tolist()

        if top_k is not None:
            base_model = make_catboost_model(params)
            base_pool = Pool(X_tr, y_tr_log, cat_features=cat_features, weight=tr_weights)
            base_model.fit(base_pool)
            shap_df = compute_catboost_shap_importance(base_model, X_tr, cat_features, out_csv=None, out_png=None, title=f"Fold {fold_idx} SHAP")
            selected_features = shap_df.head(top_k)["feature"].tolist()
            X_tr_sel = X_tr[selected_features]
            X_val_sel = X_val[selected_features]
            cat_cols_sel = [c for c in cat_cols if c in selected_features]
            cat_features_sel = get_cat_features_indices(X_tr_sel, cat_cols_sel)
        else:
            X_tr_sel = X_tr
            X_val_sel = X_val
            cat_features_sel = cat_features

        model = make_catboost_model(params)
        train_pool = Pool(X_tr_sel, y_tr_log, cat_features=cat_features_sel, weight=tr_weights)
        model.fit(train_pool)

        val_pred_log = model.predict(X_val_sel)
        train_pred_log = model.predict(X_tr_sel)
        smear = compute_smearing_factor(y_tr_log, train_pred_log)
        val_pred_price_naive = log_to_price_naive(val_pred_log)
        val_pred_price_smear = log_to_price_smearing(val_pred_log, smear)

        metrics_naive = regression_metrics(y_val_price, val_pred_price_naive, prefix="cv_naive_")
        metrics_smear = regression_metrics(y_val_price, val_pred_price_smear, prefix="cv_smearing_")
        metrics_log = log_metrics(y_val_log, val_pred_log, prefix="cv_")
        composite_naive = premium_composite_loss_from_prices(y_val_price.values, val_pred_price_naive)
        composite_smear = premium_composite_loss_from_prices(y_val_price.values, val_pred_price_smear)

        row = {"variant": variant, "fold": fold_idx, "top_k": top_k if top_k is not None else "all", "weight_strategy": WEIGHT_STRATEGY_NAME, "smearing_factor": smear, "cv_naive_composite_score": composite_naive, "cv_smearing_composite_score": composite_smear}
        row.update(metrics_log)
        row.update(metrics_naive)
        row.update(metrics_smear)
        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    summary = folds_df.drop(columns=["fold"]).select_dtypes(include=np.number).mean().to_dict()
    summary["variant"] = variant
    summary["top_k"] = top_k if top_k is not None else "all"
    summary["weight_strategy"] = WEIGHT_STRATEGY_NAME
    return pd.DataFrame(fold_rows), summary


def tune_catboost_all_features(X_train, y_log_train, y_price_train):
    logger.info("=" * 90)
    logger.info("PHASE 4 - CATBOOST PONDÉRÉ : RECHERCHE PARAMÈTRES SUR ALL FEATURES")
    logger.info("=" * 90)
    params_list = catboost_candidate_params()
    rows = []

    for i, params in enumerate(params_list, start=1):
        logger.info(f"CatBoost params {i}/{len(params_list)} : {params}")
        folds_df, summary = evaluate_catboost_cv_variant(X_train, y_log_train, y_price_train, params, variant="CatBoost_weighted_all_features", top_k=None)
        row = {"params_id": i, "params_name": params.get("name", f"params_{i}"), "params": json.dumps(params, ensure_ascii=False)}
        row.update(summary)
        rows.append(row)

    search_df = pd.DataFrame(rows)
    search_df.to_csv(SUBDIRS["catboost_all"] / "catboost_param_search_cv.csv", index=False)
    best_idx = search_df["cv_naive_composite_score"].idxmin()
    best_params = json.loads(search_df.loc[best_idx, "params"])
    logger.info(f"Meilleurs paramètres CatBoost pondéré : {best_params}")
    logger.info(f"Meilleur score composite CV CatBoost all : {search_df.loc[best_idx, 'cv_naive_composite_score']:.4f}")
    logger.info(f"MAE euros naïve CV du meilleur CatBoost all : {search_df.loc[best_idx, 'cv_naive_mae']:.4f}")

    with open(SUBDIRS["catboost_all"] / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    return best_params, search_df


def train_final_catboost_variant(X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test, params, variant, top_k=None):
    if variant == "CatBoost_weighted_all_features":
        model_dir = SUBDIRS["catboost_all"]
    elif variant == "CatBoost_weighted_SHAP_top30":
        model_dir = SUBDIRS["catboost_top30"]
    elif variant == "CatBoost_weighted_SHAP_top50":
        model_dir = SUBDIRS["catboost_top50"]
    else:
        model_dir = SUBDIRS["models"] / variant
        model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n--- Entraînement final {variant} ---")
    prep = CatBoostPreprocessor(n_geo_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE)
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)
    train_weights = compute_manual_aggressive_weights(y_price_train, normalize=True)
    cat_cols = prep.categorical_cols_
    cat_features = get_cat_features_indices(X_tr, cat_cols)
    selected_features = X_tr.columns.tolist()

    base_model = make_catboost_model(params)
    base_model.fit(Pool(X_tr, y_log_train, cat_features=cat_features, weight=train_weights))
    shap_all = compute_catboost_shap_importance(base_model, X_tr, cat_features, out_csv=model_dir / f"{variant}_shap_importance_all_features.csv", out_png=model_dir / f"{variant}_shap_importance_all_features.png", title=f"{variant} - SHAP all features")

    if top_k is not None:
        selected_features = shap_all.head(top_k)["feature"].tolist()
        pd.DataFrame({"selected_feature": selected_features}).to_csv(model_dir / f"{variant}_selected_top{top_k}_features.csv", index=False)

    X_tr_sel = X_tr[selected_features]
    X_te_sel = X_te[selected_features]
    cat_cols_sel = [c for c in cat_cols if c in selected_features]
    cat_features_sel = get_cat_features_indices(X_tr_sel, cat_cols_sel)
    final_model = make_catboost_model(params)
    final_model.fit(Pool(X_tr_sel, y_log_train, cat_features=cat_features_sel, weight=train_weights))

    train_pred_log = final_model.predict(X_tr_sel)
    test_pred_log = final_model.predict(X_te_sel)
    smear = compute_smearing_factor(y_log_train, train_pred_log)
    test_pred_naive = log_to_price_naive(test_pred_log)
    test_pred_smear = log_to_price_smearing(test_pred_log, smear)

    metrics = {"model": variant, "family": "catboost", "weight_strategy": WEIGHT_STRATEGY_NAME, "top_k": top_k if top_k is not None else "all", "n_features_used": len(selected_features), "smearing_factor": smear}
    metrics.update(log_metrics(y_log_test, test_pred_log, prefix="test_"))
    metrics.update(regression_metrics(y_price_test, test_pred_naive, prefix="test_naive_"))
    metrics.update(regression_metrics(y_price_test, test_pred_smear, prefix="test_smearing_"))
    metrics["test_naive_composite_score"] = premium_composite_loss_from_prices(y_price_test.values, test_pred_naive)
    metrics["test_smearing_composite_score"] = premium_composite_loss_from_prices(y_price_test.values, test_pred_smear)

    if metrics["test_smearing_mae"] < metrics["test_naive_mae"]:
        chosen_pred = test_pred_smear
        chosen_retransform = "smearing"
        chosen_mae = metrics["test_smearing_mae"]
    else:
        chosen_pred = test_pred_naive
        chosen_retransform = "naive"
        chosen_mae = metrics["test_naive_mae"]
    metrics["chosen_retransform"] = chosen_retransform
    metrics["chosen_test_mae"] = chosen_mae

    pred_df = pd.DataFrame({"model": variant, "chosen_retransform": chosen_retransform, "price_true": y_price_test.values, "log_true": y_log_test.values, "log_pred": test_pred_log, "price_pred_naive": test_pred_naive, "price_pred_smearing": test_pred_smear, "price_pred_chosen": chosen_pred, "segment": make_price_segments(y_price_test).values}, index=X_test.index)
    pred_df.to_csv(model_dir / f"{variant}_test_predictions.csv", index=True)

    seg_naive = segment_metrics(y_price_test, test_pred_naive, variant, "naive")
    seg_smear = segment_metrics(y_price_test, test_pred_smear, variant, "smearing")
    pd.concat([seg_naive, seg_smear], ignore_index=True).to_csv(model_dir / f"{variant}_segment_metrics.csv", index=False)

    joblib.dump({"preprocessor": prep, "model": final_model, "selected_features": selected_features, "cat_features": cat_cols_sel, "cat_features_indices": cat_features_sel, "params": params, "smearing_factor": smear, "chosen_retransform": chosen_retransform, "weight_strategy": WEIGHT_STRATEGY_NAME}, model_dir / f"{variant}_pipeline.pkl")
    with open(model_dir / f"{variant}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics, pred_df, pd.concat([seg_naive, seg_smear], ignore_index=True), shap_all

# ==============================================================================
# ÉVALUATION FINALE SKLEARN
# ==============================================================================

def evaluate_final_sklearn_models(trained_models, X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test):
    logger.info("=" * 90)
    logger.info("PHASE 5 - ÉVALUATION FINALE SCIKIT-LEARN PONDÉRÉ SUR TEST")
    logger.info("=" * 90)
    metrics_rows = []
    pred_dfs = []
    seg_dfs = []

    for model_name, pipe in trained_models.items():
        logger.info(f"Évaluation finale sklearn : {model_name}")
        model_dir = SUBDIRS["sklearn"] / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        train_pred_log = pipe.predict(X_train)
        test_pred_log = pipe.predict(X_test)
        smear = compute_smearing_factor(y_log_train, train_pred_log)
        test_pred_naive = log_to_price_naive(test_pred_log)
        test_pred_smear = log_to_price_smearing(test_pred_log, smear)

        metrics = {"model": model_name, "family": "sklearn", "weight_strategy": WEIGHT_STRATEGY_NAME, "top_k": get_selected_k_from_pipe(pipe), "n_features_used": get_selected_feature_count(pipe), "smearing_factor": smear}
        metrics.update(log_metrics(y_log_test, test_pred_log, prefix="test_"))
        metrics.update(regression_metrics(y_price_test, test_pred_naive, prefix="test_naive_"))
        metrics.update(regression_metrics(y_price_test, test_pred_smear, prefix="test_smearing_"))
        metrics["test_naive_composite_score"] = premium_composite_loss_from_prices(y_price_test.values, test_pred_naive)
        metrics["test_smearing_composite_score"] = premium_composite_loss_from_prices(y_price_test.values, test_pred_smear)

        if metrics["test_smearing_mae"] < metrics["test_naive_mae"]:
            chosen_pred = test_pred_smear
            chosen_retransform = "smearing"
            chosen_mae = metrics["test_smearing_mae"]
        else:
            chosen_pred = test_pred_naive
            chosen_retransform = "naive"
            chosen_mae = metrics["test_naive_mae"]
        metrics["chosen_retransform"] = chosen_retransform
        metrics["chosen_test_mae"] = chosen_mae

        pred_df = pd.DataFrame({"model": model_name, "chosen_retransform": chosen_retransform, "price_true": y_price_test.values, "log_true": y_log_test.values, "log_pred": test_pred_log, "price_pred_naive": test_pred_naive, "price_pred_smearing": test_pred_smear, "price_pred_chosen": chosen_pred, "segment": make_price_segments(y_price_test).values}, index=X_test.index)
        pred_df.to_csv(model_dir / f"{model_name}_test_predictions.csv", index=True)
        seg_naive = segment_metrics(y_price_test, test_pred_naive, model_name, "naive")
        seg_smear = segment_metrics(y_price_test, test_pred_smear, model_name, "smearing")
        seg_df = pd.concat([seg_naive, seg_smear], ignore_index=True)
        seg_df.to_csv(model_dir / f"{model_name}_segment_metrics.csv", index=False)
        with open(model_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        metrics_rows.append(metrics)
        pred_dfs.append(pred_df)
        seg_dfs.append(seg_df)
    return pd.DataFrame(metrics_rows), pred_dfs, seg_dfs


def get_selected_k_from_pipe(pipe):
    try:
        return pipe.named_steps["feature_selection"].k
    except Exception:
        return None


def get_selected_feature_count(pipe):
    try:
        selector = pipe.named_steps["feature_selection"]
        if selector.k == "all":
            return int(len(selector.scores_))
        return int(selector.get_support().sum())
    except Exception:
        return None


# GRAPHIQUES GLOBAUX


def save_global_plots(comparison_df, best_pred_df, segment_df, best_model_name):
    logger.info("=" * 90)
    logger.info("PHASE 6 - GRAPHIQUES")
    logger.info("=" * 90)
    if "chosen_retransform" not in best_pred_df.columns:
        fallback = comparison_df.loc[comparison_df["model"] == best_model_name, "chosen_retransform"]
        best_pred_df = best_pred_df.copy()
        best_pred_df["chosen_retransform"] = fallback.iloc[0] if len(fallback) else "naive"

    df_sorted = comparison_df.sort_values("chosen_test_mae", ascending=True)
    plt.figure(figsize=(12, 7)); plt.barh(df_sorted["model"], df_sorted["chosen_test_mae"]); plt.xlabel("MAE en prix réel"); plt.title("Comparaison des modèles pondérés - MAE test"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "01_model_comparison_mae.png", dpi=250, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(12, 7)); plt.barh(df_sorted["model"], df_sorted["test_naive_r2"]); plt.xlabel("R² en prix réel - reconversion naïve"); plt.title("Comparaison des modèles pondérés - R² test"); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "02_model_comparison_r2.png", dpi=250, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(best_pred_df["price_true"], best_pred_df["price_pred_chosen"], alpha=0.35, s=12)
    min_v = min(best_pred_df["price_true"].min(), best_pred_df["price_pred_chosen"].min()); max_v = max(best_pred_df["price_true"].max(), best_pred_df["price_pred_chosen"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--"); plt.xlabel("Prix réel"); plt.ylabel("Prix prédit"); plt.title(f"Prédictions vs réalité - {best_model_name}"); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "03_best_model_predictions_vs_real.png", dpi=250, bbox_inches="tight"); plt.close()

    residuals = best_pred_df["price_true"] - best_pred_df["price_pred_chosen"]
    plt.figure(figsize=(9, 6)); plt.scatter(best_pred_df["price_pred_chosen"], residuals, alpha=0.35, s=12); plt.axhline(0, linestyle="--"); plt.xlabel("Prix prédit"); plt.ylabel("Résidu = réel - prédit"); plt.title(f"Résidus - {best_model_name}"); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "04_best_model_residuals.png", dpi=250, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(9, 6)); plt.hist(residuals, bins=60, edgecolor="black", alpha=0.8); plt.xlabel("Résidu = réel - prédit"); plt.ylabel("Fréquence"); plt.title(f"Distribution des résidus - {best_model_name}"); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "05_best_model_residual_distribution.png", dpi=250, bbox_inches="tight"); plt.close()

    chosen_retransform = str(best_pred_df["chosen_retransform"].iloc[0]) if len(best_pred_df) else "naive"
    best_seg = segment_df[(segment_df["model"] == best_model_name) & (segment_df["retransform"] == chosen_retransform)].copy()
    if len(best_seg) > 0:
        plt.figure(figsize=(9, 6)); plt.bar(best_seg["segment"], best_seg["mae"]); plt.xlabel("Segment de prix"); plt.ylabel("MAE"); plt.title(f"Erreur par segment - {best_model_name}"); plt.tight_layout(); plt.savefig(SUBDIRS["plots"] / "06_best_model_mae_by_segment.png", dpi=250, bbox_inches="tight"); plt.close()



def write_final_report(comparison_df, segment_df, cv_sklearn_df, cat_search_df, best_model_name):
    best = comparison_df.sort_values("chosen_test_mae").iloc[0]

    report = f"""
================================================================================
RAPPORT FINAL 
Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DONNÉES
- Fichier : {DATA_PATH}
- Outliers : conservés, aucune suppression automatique des logements chers
- Pondération : manual_aggressive appliquée à tous les modèles
- Cible d'entraînement : log_price
- Évaluation finale : prix réel après reconversion
- Reconversions testées : naïve expm1 et smearing estimate de Duan

COLONNES RETIRÉES DES FEATURES
{DROP_FEATURES}

MODÈLES COMPARÉS
- CatBoost_weighted_all_features : catégories natives, pondération manual_aggressive, toutes variables retenues après nettoyage
- CatBoost_weighted_SHAP_top30 : pondération manual_aggressive + sélection SHAP top 30 faite sans utiliser le test
- CatBoost_weighted_SHAP_top50 : pondération manual_aggressive + sélection SHAP top 50 faite sans utiliser le test
- Ridge, Lasso, ElasticNet
- RandomForest, GradientBoosting, HistGradientBoosting

POINTS MÉTHODOLOGIQUES CORRIGÉS
- Split train/test avant transformations apprises
- Pondération manual_aggressive appliquée pendant l’apprentissage
- Sélection par score composite premium-oriented en validation croisée
- Imputation numérique par médiane dans les pipelines
- Imputation catégorielle par mode / missing selon modèle
- OneHotEncoder seulement pour les modèles scikit-learn
- CatBoost avec variables catégorielles natives
- SelectKBest(mutual_info_regression) intégré dans le Pipeline scikit-learn
- Sélection de variables refaite dans chaque fold de validation croisée
- Géographie enrichie : geo_cluster + neighbourhood_popularity
- Analyse par segments de prix
- SHAP CatBoost sauvegardé

MEILLEUR MODÈLE SELON LA MAE TEST CHOISIE
- Modèle : {best_model_name}
- Retransformation retenue : {best['chosen_retransform']}
- MAE : {best['chosen_test_mae']:.4f}
- MAE naïve : {best['test_naive_mae']:.4f}
- RMSE naïve : {best['test_naive_rmse']:.4f}
- R² naïf : {best['test_naive_r2']:.4f}
- MAE smearing : {best['test_smearing_mae']:.4f}
- RMSE smearing : {best['test_smearing_rmse']:.4f}
- R² smearing : {best['test_smearing_r2']:.4f}

FICHIERS IMPORTANTS
- 00_global/model_comparison_results.csv
- 00_global/segment_metrics_by_model.csv
- 00_global/sklearn_cv_summary.csv
- 05_models/catboost_weighted_all_features/catboost_param_search_cv.csv
- 05_models/catboost_all_features/*shap*.csv
- 05_models/catboost_shap_top30/*selected*.csv
- 05_models/catboost_shap_top50/*selected*.csv
- 02_plots/*.png
- 03_predictions/best_model_predictions.csv
================================================================================
"""

    with open(SUBDIRS["reports"] / "training_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(report)


def main():
    if not CATBOOST_AVAILABLE:
        raise ImportError(
            "CatBoost n'est pas disponible. Installe-le avec : pip install catboost\n"
            f"Erreur initiale : {CATBOOST_IMPORT_ERROR}"
        )

    logger.info("Dossier de sortie : %s", OUTPUT_DIR)

    df = load_data(DATA_PATH)
    X, y_log, y_price = prepare_X_y(df)
    X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test = stratified_train_test_split(
        X, y_log, y_price
    )

    # Sauvegarde des dimensions
    pd.DataFrame({
        "item": ["X_train", "X_test", "y_train", "y_test"],
        "n_rows": [len(X_train), len(X_test), len(y_log_train), len(y_log_test)],
    }).to_csv(SUBDIRS["global"] / "dataset_shapes.csv", index=False)

    # 1) Sklearn models
    sklearn_models, sklearn_cv_summary = train_sklearn_models(X_train, y_log_train, y_price_train)
    sklearn_metrics_df, sklearn_pred_dfs, sklearn_seg_dfs = evaluate_final_sklearn_models(
        sklearn_models, X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test
    )

    # 2) CatBoost search + variants
    best_cat_params, cat_search_df = tune_catboost_all_features(X_train, y_log_train, y_price_train)

    cat_cv_summaries = []
    cat_cv_folds_all = []
    for variant, top_k in [
        ("CatBoost_weighted_all_features", None),
        ("CatBoost_weighted_SHAP_top30", 30),
        ("CatBoost_weighted_SHAP_top50", 50),
    ]:
        logger.info(f"CV propre pour variante CatBoost : {variant}")
        folds_df, summary = evaluate_catboost_cv_variant(
            X_train, y_log_train, y_price_train, best_cat_params, variant=variant, top_k=top_k
        )
        cat_cv_folds_all.append(folds_df)
        cat_cv_summaries.append(summary)

    cat_cv_summary_df = pd.DataFrame(cat_cv_summaries)
    cat_cv_summary_df.to_csv(SUBDIRS["global"] / "catboost_variants_cv_summary.csv", index=False)
    pd.concat(cat_cv_folds_all, ignore_index=True).to_csv(
        SUBDIRS["global"] / "catboost_variants_cv_folds.csv", index=False
    )

    cat_metrics = []
    cat_pred_dfs = []
    cat_seg_dfs = []
    cat_shap_dfs = []

    for variant, top_k in [
        ("CatBoost_weighted_all_features", None),
        ("CatBoost_weighted_SHAP_top30", 30),
        ("CatBoost_weighted_SHAP_top50", 50),
    ]:
        metrics, pred_df, seg_df, shap_df = train_final_catboost_variant(
            X_train, X_test, y_log_train, y_log_test, y_price_train, y_price_test,
            best_cat_params, variant=variant, top_k=top_k
        )
        cat_metrics.append(metrics)
        pred_df = pred_df.copy()
        pred_df["model"] = variant
        cat_pred_dfs.append(pred_df)
        cat_seg_dfs.append(seg_df)
        shap_df["variant"] = variant
        cat_shap_dfs.append(shap_df)

    cat_metrics_df = pd.DataFrame(cat_metrics)

    # 3) Comparaison globale
    comparison_df = pd.concat([sklearn_metrics_df, cat_metrics_df], ignore_index=True)

    # Ajouter CV sklearn si disponible
    comparison_df = comparison_df.merge(
        sklearn_cv_summary[["model", "cv_mae_euro_naive"]],
        on="model",
        how="left",
    )

    # Ajouter CV catboost variante
    cat_cv_for_merge = cat_cv_summary_df.rename(columns={
        "variant": "model",
        "cv_naive_mae": "cv_mae_euro_naive",
    })[["model", "cv_mae_euro_naive"]]
    comparison_df = comparison_df.merge(
        cat_cv_for_merge,
        on="model",
        how="left",
        suffixes=("", "_cat")
    )
    comparison_df["cv_mae_euro_naive"] = comparison_df["cv_mae_euro_naive"].fillna(
        comparison_df.get("cv_mae_euro_naive_cat")
    )
    if "cv_mae_euro_naive_cat" in comparison_df.columns:
        comparison_df = comparison_df.drop(columns=["cv_mae_euro_naive_cat"])

    comparison_df = comparison_df.sort_values("chosen_test_mae", ascending=True)
    comparison_df.to_csv(SUBDIRS["global"] / "model_comparison_results.csv", index=False)

    all_seg_df = pd.concat(sklearn_seg_dfs + cat_seg_dfs, ignore_index=True)
    all_seg_df.to_csv(SUBDIRS["global"] / "segment_metrics_by_model.csv", index=False)

    all_predictions = pd.concat(sklearn_pred_dfs + cat_pred_dfs, ignore_index=True)
    all_predictions.to_csv(SUBDIRS["predictions"] / "all_models_test_predictions.csv", index=True)

    if cat_shap_dfs:
        pd.concat(cat_shap_dfs, ignore_index=True).to_csv(
            SUBDIRS["global"] / "catboost_all_variants_shap_importance.csv", index=False
        )

    # Meilleur modèle
    best_model_name = comparison_df.iloc[0]["model"]
    best_pred_df = all_predictions[all_predictions["model"] == best_model_name].copy()
    best_pred_df.to_csv(SUBDIRS["predictions"] / "best_model_predictions.csv", index=True)

    # Pour faciliter l'envoi : résumé léger à la racine
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    all_seg_df.to_csv(OUTPUT_DIR / "segment_metrics_by_model.csv", index=False)
    best_pred_df.to_csv(OUTPUT_DIR / "best_model_predictions.csv", index=True)

    save_global_plots(comparison_df, best_pred_df, all_seg_df, best_model_name)
    write_final_report(comparison_df, all_seg_df, sklearn_cv_summary, cat_search_df, best_model_name)

    logger.info("=" * 90)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("Résultats sauvegardés dans : %s", OUTPUT_DIR)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
