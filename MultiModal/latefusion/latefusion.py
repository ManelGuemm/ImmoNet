# -*- coding: utf-8 -*-

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")



TAB_OOF_PATH = "Resultats_Tabulaire_Only_KFold_CatBoost_ManualAggressive_C05/predictions/10_oof_predictions_all_folds.csv"
TAB_TEST_PATH = "Resultats_Tabulaire_Only_KFold_CatBoost_ManualAggressive_C05/predictions/20_predictions_test_final.csv"

TEXT_TRAIN_PATH = "outputs_text_e5_lora/airbnb_text_audit_train_e5_lora.csv"
TEXT_TEST_PATH = "outputs_text_e5_lora/airbnb_text_audit_test_e5_lora.csv"

TEXT_OOF_NPY_PATH = "outputs_text_e5_lora/oof_pred_train.npy"
TEXT_TEST_NPY_PATH = "outputs_text_e5_lora/test_pred.npy"

IMG_OOF_PATH = "Resultats_ImageOnly_ConvNeXtBase_RTX4500_RapportComplet/09_predictions_oof_for_late_fusion.csv"
IMG_TEST_PATH = "Resultats_ImageOnly_ConvNeXtBase_RTX4500_RapportComplet/10_predictions_test_for_late_fusion.csv"

OUTPUT_DIR = "Resultats_LateFusion_Tabulaire_Texte_Image"



# OUTILS

def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_id(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def inverse_log_price(y_log):
    return np.expm1(np.asarray(y_log, dtype=float))


def check_no_duplicates(df, id_col, name):
    n = len(df)
    n_unique = df[id_col].nunique()
    n_dup = df[id_col].duplicated().sum()

    if n_dup > 0:
        raise ValueError(
            f"{name} contient des identifiants dupliqués : "
            f"{n_dup} doublons sur {n} lignes."
        )

    print(f"{name} : {n} lignes | {n_unique} ids uniques | doublons = {n_dup}")


def metrics_from_log(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[mask]
    y_pred_log = y_pred_log[mask]

    y_true_price = inverse_log_price(y_true_log)
    y_pred_price = inverse_log_price(y_pred_log)

    error_euro = y_pred_price - y_true_price
    abs_error_euro = np.abs(error_euro)

    return {
        "n": int(len(y_true_log)),

        "MAE_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "MedAE_log": float(median_absolute_error(y_true_log, y_pred_log)),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "Biais_log": float(np.mean(y_pred_log - y_true_log)),

        "MAE_euros": float(mean_absolute_error(y_true_price, y_pred_price)),
        "RMSE_euros": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "MedAE_euros": float(median_absolute_error(y_true_price, y_pred_price)),
        "R2_euros": float(r2_score(y_true_price, y_pred_price)),
        "Biais_euros": float(np.mean(error_euro)),

        "Sous_estimation_pct": float(np.mean(y_pred_price < y_true_price) * 100.0),
        "Surestimation_pct": float(np.mean(y_pred_price > y_true_price) * 100.0),
        "Erreur_abs_P90_euros": float(np.percentile(abs_error_euro, 90)),
        "Erreur_abs_P95_euros": float(np.percentile(abs_error_euro, 95)),
    }


def segment_report(y_true_log, y_pred_log):
    y_true_price = inverse_log_price(y_true_log)
    y_pred_price = inverse_log_price(y_pred_log)

    df = pd.DataFrame({
        "price_true": y_true_price,
        "price_pred": y_pred_price,
    })

    df["error_euro"] = df["price_pred"] - df["price_true"]
    df["abs_error_euro"] = df["error_euro"].abs()

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


# CHARGEMENT TABULAIRE


def load_tabular_predictions():
    print("\n" + "=" * 90)
    print("Chargement des prédictions tabulaires")
    print("=" * 90)

    oof = read_csv(TAB_OOF_PATH)
    test = read_csv(TAB_TEST_PATH)

    required_cols = [
        "id_clean",
        "y_true_log",
        "y_pred_log",
        "y_true_price",
        "y_pred_price",
    ]

    for col in required_cols:
        if col not in oof.columns:
            raise ValueError(f"Colonne manquante dans tabulaire OOF : {col}")
        if col not in test.columns:
            raise ValueError(f"Colonne manquante dans tabulaire test : {col}")

    tab_train = pd.DataFrame({
        "listing_id_clean": normalize_id(oof["id_clean"]),
        "log_price_true": to_num(oof["y_true_log"]),
        "price_true": to_num(oof["y_true_price"]),
        "pred_tab_log": to_num(oof["y_pred_log"]),
        "pred_tab_price": to_num(oof["y_pred_price"]),
        "tab_source": "tab_oof",
    })

    tab_test = pd.DataFrame({
        "listing_id_clean": normalize_id(test["id_clean"]),
        "log_price_true": to_num(test["y_true_log"]),
        "price_true": to_num(test["y_true_price"]),
        "pred_tab_log": to_num(test["y_pred_log"]),
        "pred_tab_price": to_num(test["y_pred_price"]),
        "tab_source": "tab_test_final",
    })

    check_no_duplicates(tab_train, "listing_id_clean", "Tabulaire OOF")
    check_no_duplicates(tab_test, "listing_id_clean", "Tabulaire test")

    return tab_train, tab_test


# ==========================================================
# CHARGEMENT TEXTE
# ==========================================================

def load_text_predictions_all():
    print("\n" + "=" * 90)
    print("Chargement des prédictions texte E5-LoRA")
    print("=" * 90)

    train = read_csv(TEXT_TRAIN_PATH)
    test = read_csv(TEXT_TEST_PATH)

    required_cols = [
        "listing_id_clean",
        "price_clean",
        "log_price",
        "text_pred_log",
        "text_pred_price",
    ]

    for col in required_cols:
        if col not in train.columns:
            raise ValueError(f"Colonne manquante dans texte train : {col}")
        if col not in test.columns:
            raise ValueError(f"Colonne manquante dans texte test : {col}")

    text_train = pd.DataFrame({
        "listing_id_clean": normalize_id(train["listing_id_clean"]),
        "log_price_text": to_num(train["log_price"]),
        "price_text": to_num(train["price_clean"]),
        "pred_text_log": to_num(train["text_pred_log"]),
        "pred_text_price": to_num(train["text_pred_price"]),
        "text_source": "text_oof_train",
    })

    text_test = pd.DataFrame({
        "listing_id_clean": normalize_id(test["listing_id_clean"]),
        "log_price_text": to_num(test["log_price"]),
        "price_text": to_num(test["price_clean"]),
        "pred_text_log": to_num(test["text_pred_log"]),
        "pred_text_price": to_num(test["text_pred_price"]),
        "text_source": "text_test",
    })

    # Vérification optionnelle des npy si présents
    if Path(TEXT_OOF_NPY_PATH).exists():
        arr = np.load(TEXT_OOF_NPY_PATH).reshape(-1)
        if len(arr) != len(text_train):
            raise ValueError("oof_pred_train.npy n'a pas la même longueur que le fichier texte train.")
        max_diff = np.max(np.abs(arr.astype(float) - text_train["pred_text_log"].values.astype(float)))
        print(f"Vérification oof_pred_train.npy vs text_pred_log train : max_diff = {max_diff:.10f}")

    if Path(TEXT_TEST_NPY_PATH).exists():
        arr = np.load(TEXT_TEST_NPY_PATH).reshape(-1)
        if len(arr) != len(text_test):
            raise ValueError("test_pred.npy n'a pas la même longueur que le fichier texte test.")
        max_diff = np.max(np.abs(arr.astype(float) - text_test["pred_text_log"].values.astype(float)))
        print(f"Vérification test_pred.npy vs text_pred_log test : max_diff = {max_diff:.10f}")

    text_all = pd.concat([text_train, text_test], ignore_index=True)

    check_no_duplicates(text_all, "listing_id_clean", "Texte complet train+test")

    return text_all


# CHARGEMENT IMAGE


def load_image_predictions_all():
    print("\n" + "=" * 90)
    print("Chargement des prédictions image ConvNeXt")
    print("=" * 90)

    oof = read_csv(IMG_OOF_PATH)
    test = read_csv(IMG_TEST_PATH)

    required_oof_cols = [
        "listing_id_clean",
        "log_price_true",
        "log_price_pred",
        "price_true",
        "price_pred",
    ]

    required_test_cols = [
        "listing_id_clean",
        "log_price_true",
        "log_price_pred",
        "price_true",
        "price_pred",
    ]

    for col in required_oof_cols:
        if col not in oof.columns:
            raise ValueError(f"Colonne manquante dans image OOF : {col}")

    for col in required_test_cols:
        if col not in test.columns:
            raise ValueError(f"Colonne manquante dans image test : {col}")

    img_oof = pd.DataFrame({
        "listing_id_clean": normalize_id(oof["listing_id_clean"]),
        "log_price_image": to_num(oof["log_price_true"]),
        "price_image": to_num(oof["price_true"]),
        "pred_image_log": to_num(oof["log_price_pred"]),
        "pred_image_price": to_num(oof["price_pred"]),
        "image_source": "image_oof_train",
    })

    img_test = pd.DataFrame({
        "listing_id_clean": normalize_id(test["listing_id_clean"]),
        "log_price_image": to_num(test["log_price_true"]),
        "price_image": to_num(test["price_true"]),
        "pred_image_log": to_num(test["log_price_pred"]),
        "pred_image_price": to_num(test["price_pred"]),
        "image_source": "image_test",
    })

    img_all = pd.concat([img_oof, img_test], ignore_index=True)

    check_no_duplicates(img_all, "listing_id_clean", "Image complète OOF+test")

    return img_all

# FUSION DES PRÉDICTIONS

def build_fusion_dataset(tab_df, text_all, img_all, split_name):
    print("\n" + "=" * 90)
    print(f"Fusion des prédictions : {split_name}")
    print("=" * 90)

    merged = tab_df.merge(
        text_all[
            [
                "listing_id_clean",
                "log_price_text",
                "price_text",
                "pred_text_log",
                "pred_text_price",
                "text_source",
            ]
        ],
        on="listing_id_clean",
        how="inner",
    )

    merged = merged.merge(
        img_all[
            [
                "listing_id_clean",
                "log_price_image",
                "price_image",
                "pred_image_log",
                "pred_image_price",
                "image_source",
            ]
        ],
        on="listing_id_clean",
        how="inner",
    )

    print(f"{split_name} : {len(tab_df)} lignes tabulaires -> {len(merged)} lignes après fusion")

    if len(merged) != len(tab_df):
        missing = len(tab_df) - len(merged)
        raise RuntimeError(
            f"Attention : {missing} lignes perdues pendant la fusion {split_name}. "
            "Ce n'est pas normal ici, car texte et image doivent couvrir tous les IDs."
        )

    # Vérification cohérence des cibles
    diff_text = np.abs(merged["log_price_true"] - merged["log_price_text"])
    diff_image = np.abs(merged["log_price_true"] - merged["log_price_image"])

    diff_price_text = np.abs(merged["price_true"] - merged["price_text"])
    diff_price_image = np.abs(merged["price_true"] - merged["price_image"])

    print(f"Max diff log_price tabulaire vs texte : {diff_text.max():.10f}")
    print(f"Max diff log_price tabulaire vs image : {diff_image.max():.10f}")
    print(f"Max diff price tabulaire vs texte : {diff_price_text.max():.10f}")
    print(f"Max diff price tabulaire vs image : {diff_price_image.max():.10f}")

    if diff_text.max() > 1e-4:
        raise RuntimeError("Incohérence log_price entre tabulaire et texte.")

    if diff_image.max() > 1e-4:
        raise RuntimeError("Incohérence log_price entre tabulaire et image.")

    if diff_price_text.max() > 1e-2:
        raise RuntimeError("Incohérence price entre tabulaire et texte.")

    if diff_price_image.max() > 1e-2:
        raise RuntimeError("Incohérence price entre tabulaire et image.")

    needed = [
        "log_price_true",
        "price_true",
        "pred_tab_log",
        "pred_text_log",
        "pred_image_log",
    ]

    before = len(merged)
    merged = merged.dropna(subset=needed).copy()
    after = len(merged)

    if before != after:
        print(f"{before - after} lignes supprimées à cause de valeurs manquantes.")

    return merged


# MODÈLES DE FUSION

def build_meta_models():
    models = {}

    models["mean_simple"] = None

    models["linear_regression"] = LinearRegression()

    models["ridge_cv"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-4, 4, 60)))
    ])

    models["elasticnet_cv"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            alphas=np.logspace(-4, 2, 40),
            l1_ratio=[0.05, 0.10, 0.30, 0.50, 0.70, 0.90],
            cv=5,
            max_iter=50000,
            random_state=42
        ))
    ])

    try:
        from catboost import CatBoostRegressor

        models["catboost_shallow"] = CatBoostRegressor(
            loss_function="RMSE",
            iterations=800,
            learning_rate=0.03,
            depth=2,
            l2_leaf_reg=30,
            random_seed=42,
            verbose=False
        )

    except Exception:
        print("CatBoost n'est pas installé. Le modèle catboost_shallow sera ignoré.")

    return models


def get_model_coefficients(model_name, model, feature_cols):
    rows = []

    if model_name == "linear_regression":
        for col, coef in zip(feature_cols, model.coef_):
            rows.append({
                "model": model_name,
                "feature": col,
                "coef_original_scale": float(coef),
                "coef_scaled_space": float(coef),
            })

        rows.append({
            "model": model_name,
            "feature": "intercept",
            "coef_original_scale": float(model.intercept_),
            "coef_scaled_space": float(model.intercept_),
        })

    elif model_name in ["ridge_cv", "elasticnet_cv"]:
        scaler = model.named_steps["scaler"]
        inner = model.named_steps["model"]

        coef_scaled = inner.coef_
        coef_original = coef_scaled / scaler.scale_
        intercept_original = inner.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

        for col, coef_s, coef_o in zip(feature_cols, coef_scaled, coef_original):
            rows.append({
                "model": model_name,
                "feature": col,
                "coef_original_scale": float(coef_o),
                "coef_scaled_space": float(coef_s),
            })

        rows.append({
            "model": model_name,
            "feature": "intercept",
            "coef_original_scale": float(intercept_original),
            "coef_scaled_space": float(inner.intercept_),
        })

        if hasattr(inner, "alpha_"):
            rows.append({
                "model": model_name,
                "feature": "alpha",
                "coef_original_scale": float(inner.alpha_),
                "coef_scaled_space": float(inner.alpha_),
            })

        if hasattr(inner, "l1_ratio_"):
            rows.append({
                "model": model_name,
                "feature": "l1_ratio",
                "coef_original_scale": float(inner.l1_ratio_),
                "coef_scaled_space": float(inner.l1_ratio_),
            })

    return rows


def fit_and_evaluate_meta_models(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols].values
    y_train = train_df["log_price_true"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["log_price_true"].values

    models = build_meta_models()

    metrics_rows = []
    coef_rows = []

    predictions = test_df[
        [
            "listing_id_clean",
            "log_price_true",
            "price_true",
            "pred_tab_log",
            "pred_text_log",
            "pred_image_log",
            "text_source",
            "image_source",
        ]
    ].copy()

    for name, model in models.items():
        print(f"\nEntraînement / évaluation : {name}")

        if name == "mean_simple":
            pred_train = X_train.mean(axis=1)
            pred_test = X_test.mean(axis=1)
        else:
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            coef_rows.extend(get_model_coefficients(name, model, feature_cols))

        train_metrics = metrics_from_log(y_train, pred_train)
        test_metrics = metrics_from_log(y_test, pred_test)

        row = {
            "model": name,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        metrics_rows.append(row)

        predictions[f"pred_{name}_log"] = pred_test
        predictions[f"pred_{name}_price"] = inverse_log_price(pred_test)
        predictions[f"error_{name}_euro"] = predictions[f"pred_{name}_price"] - predictions["price_true"]
        predictions[f"abs_error_{name}_euro"] = predictions[f"error_{name}_euro"].abs()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("test_MAE_euros", ascending=True).reset_index(drop=True)

    coef_df = pd.DataFrame(coef_rows)

    return metrics_df, predictions, coef_df


# ==========================================================
# RAPPORT
# ==========================================================

def write_report(output_dir, train_fusion, test_fusion, individual_metrics, fusion_metrics, best_segments):
    report_path = output_dir / "09_rapport_late_fusion.txt"

    best_model = fusion_metrics.iloc[0]["model"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Rapport late fusion tabulaire + texte + image\n")
        f.write("=" * 90 + "\n\n")

        f.write("Données utilisées\n")
        f.write("-" * 90 + "\n")
        f.write(f"Train OOF fusionné : {len(train_fusion)} lignes\n")
        f.write(f"Test final fusionné : {len(test_fusion)} lignes\n")
        f.write("Variables de fusion : pred_tab_log, pred_text_log, pred_image_log\n\n")

        f.write("Remarque méthodologique\n")
        f.write("-" * 90 + "\n")
        f.write(
            "Le split maître est celui du modèle tabulaire CatBoost. "
            "Les prédictions texte et image ont été concaténées sur l'ensemble train+test "
            "de leurs expériences respectives, puis réalignées sur les identifiants du split tabulaire. "
            "Cette étape est nécessaire car les splits texte/image ne sont pas exactement alignés avec le split tabulaire.\n\n"
        )

        f.write("Métriques des modèles individuels\n")
        f.write("-" * 90 + "\n")
        f.write(individual_metrics.to_string(index=False))
        f.write("\n\n")

        f.write("Métriques des modèles de late fusion\n")
        f.write("-" * 90 + "\n")
        f.write(fusion_metrics.to_string(index=False))
        f.write("\n\n")

        f.write("Meilleur modèle selon la MAE en euros sur test_final\n")
        f.write("-" * 90 + "\n")
        f.write(str(best_model))
        f.write("\n\n")

        f.write("Segments du meilleur modèle\n")
        f.write("-" * 90 + "\n")
        f.write(best_segments.to_string(index=False))
        f.write("\n")

    return report_path



def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    tab_train, tab_test = load_tabular_predictions()
    text_all = load_text_predictions_all()
    image_all = load_image_predictions_all()

    train_fusion = build_fusion_dataset(
        tab_df=tab_train,
        text_all=text_all,
        img_all=image_all,
        split_name="train OOF"
    )

    test_fusion = build_fusion_dataset(
        tab_df=tab_test,
        text_all=text_all,
        img_all=image_all,
        split_name="test final"
    )

    print("\nRépartition des sources texte dans train :")
    print(train_fusion["text_source"].value_counts())

    print("\nRépartition des sources texte dans test :")
    print(test_fusion["text_source"].value_counts())

    print("\nRépartition des sources image dans train :")
    print(train_fusion["image_source"].value_counts())

    print("\nRépartition des sources image dans test :")
    print(test_fusion["image_source"].value_counts())

    # Sauvegarde des datasets fusionnés
    train_fusion.to_csv(output_dir / "01_train_oof_fusion_dataset.csv", index=False, encoding="utf-8-sig")
    test_fusion.to_csv(output_dir / "02_test_fusion_dataset.csv", index=False, encoding="utf-8-sig")

    feature_cols = ["pred_tab_log", "pred_text_log", "pred_image_log"]

    # Évaluation des modèles individuels
    individual_rows = []

    for pred_col in feature_cols:
        train_metrics = metrics_from_log(train_fusion["log_price_true"].values, train_fusion[pred_col].values)
        test_metrics = metrics_from_log(test_fusion["log_price_true"].values, test_fusion[pred_col].values)

        individual_rows.append({
            "model": pred_col,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

    individual_metrics = pd.DataFrame(individual_rows)
    individual_metrics = individual_metrics.sort_values("test_MAE_euros", ascending=True).reset_index(drop=True)
    individual_metrics.to_csv(output_dir / "03_individual_models_metrics.csv", index=False, encoding="utf-8-sig")

    # Entraînement late fusion
    fusion_metrics, fusion_predictions, coefficients = fit_and_evaluate_meta_models(
        train_df=train_fusion,
        test_df=test_fusion,
        feature_cols=feature_cols
    )

    fusion_metrics.to_csv(output_dir / "04_late_fusion_metrics.csv", index=False, encoding="utf-8-sig")
    fusion_predictions.to_csv(output_dir / "05_late_fusion_test_predictions.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "08_late_fusion_coefficients.csv", index=False, encoding="utf-8-sig")

    # Meilleur modèle
    best_model = fusion_metrics.iloc[0]["model"]
    best_pred_col = f"pred_{best_model}_log"

    best_segments = segment_report(
        test_fusion["log_price_true"].values,
        fusion_predictions[best_pred_col].values
    )

    best_segments.to_csv(output_dir / "06_best_late_fusion_segments.csv", index=False, encoding="utf-8-sig")

    # Comparaison globale : modèles individuels + late fusion
    selected_cols = [
        "model",
        "test_n",
        "test_MAE_log",
        "test_RMSE_log",
        "test_R2_log",
        "test_MAE_euros",
        "test_RMSE_euros",
        "test_MedAE_euros",
        "test_R2_euros",
        "test_Biais_euros",
        "test_Sous_estimation_pct",
        "test_Erreur_abs_P90_euros",
        "test_Erreur_abs_P95_euros",
    ]

    comparison = pd.concat([
        individual_metrics[selected_cols],
        fusion_metrics[selected_cols]
    ], ignore_index=True)

    comparison = comparison.sort_values("test_MAE_euros", ascending=True).reset_index(drop=True)
    comparison.to_csv(output_dir / "07_comparison_individual_and_fusion.csv", index=False, encoding="utf-8-sig")

    # Config JSON
    config = {
        "tab_oof_path": TAB_OOF_PATH,
        "tab_test_path": TAB_TEST_PATH,
        "text_train_path": TEXT_TRAIN_PATH,
        "text_test_path": TEXT_TEST_PATH,
        "image_oof_path": IMG_OOF_PATH,
        "image_test_path": IMG_TEST_PATH,
        "output_dir": OUTPUT_DIR,
        "feature_cols": feature_cols,
        "n_train_fusion": int(len(train_fusion)),
        "n_test_fusion": int(len(test_fusion)),
        "best_model": str(best_model),
    }

    with open(output_dir / "10_config_late_fusion.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    report_path = write_report(
        output_dir=output_dir,
        train_fusion=train_fusion,
        test_fusion=test_fusion,
        individual_metrics=individual_metrics,
        fusion_metrics=fusion_metrics,
        best_segments=best_segments
    )

    print("\n" + "=" * 90)
    print("LATE FUSION TERMINÉE")
    print("=" * 90)
    print(f"Dossier résultats : {output_dir}")
    print(f"Rapport : {report_path}")
    print("\nComparaison finale :")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()