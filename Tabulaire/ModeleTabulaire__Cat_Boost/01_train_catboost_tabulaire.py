
# 01_train_catboost_tabulaire.py
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

from catboost import CatBoostRegressor, Pool


warnings.filterwarnings("ignore")

# 1. Configuration générale

CONFIG = {
    "random_state": 42,

    # Nom du fichier final tabulaire
    "input_filename_csv": "airbnb_tabulaire_final_catboost_rates_corriges.csv",
    "input_filename_xlsx": "airbnb_tabulaire_final_catboost_rates_corriges.xlsx",

    # Cibles
    "target_log": "log_price",
    "target_real": "price",

    # Colonnes à exclure de X
    "cols_to_exclude": [
        "id_clean",
        "price",
        "log_price"
    ],

    # Variables catégorielles finales pour CatBoost
    "cat_features": [
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean"
    ],

    "use_review_features": True,

    # Split train / validation / test
    "train_size": 0.70,
    "validation_size": 0.15,
    "test_size": 0.15,
    "n_strat_bins": 20,

    # Paramètres CatBoost baseline solide
    "catboost_params": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 10,
        "random_seed": 42,
        "early_stopping_rounds": 150,
        "verbose": 100,
        "allow_writing_files": False
    }
}

# 2. Gestion des chemins

def get_project_paths():


    script_dir = Path(__file__).resolve().parent

    # Dossier du script : ModeleTabulaire__Cat_Boost
    model_dir = script_dir

    # Dossier Tabulaire
    tabulaire_dir = model_dir.parent

    # Dossier des données tabulaires
    data_dir = tabulaire_dir / "Donnees_Tabulaires"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Impossible de trouver le dossier Donnees_Tabulaires : {data_dir}"
        )

    # Dossier de résultats dans ModeleTabulaire__Cat_Boost
    output_dir = model_dir / "Resultats_CatBoost_Tabulaire"
    reports_dir = output_dir / "rapports"
    plots_dir = output_dir / "graphiques"
    models_dir = output_dir / "modeles"
    predictions_dir = output_dir / "predictions"

    for path in [output_dir, reports_dir, plots_dir, models_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "model_dir": model_dir,
        "tabulaire_dir": tabulaire_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "models_dir": models_dir,
        "predictions_dir": predictions_dir
    }


PATHS = get_project_paths()

# 3. Chargement des données

def load_data(config, paths):


    csv_path = paths["data_dir"] / config["input_filename_csv"]
    xlsx_path = paths["data_dir"] / config["input_filename_xlsx"]

    if csv_path.exists():
        input_path = csv_path
        df = pd.read_csv(
            input_path,
            dtype={"id_clean": str},
            low_memory=False
        )

    elif xlsx_path.exists():
        input_path = xlsx_path
        df = pd.read_excel(
            input_path,
            dtype={"id_clean": str},
            engine="openpyxl"
        )

    else:
        raise FileNotFoundError(
            "Aucun fichier final trouvé.\n"
            f"Fichier CSV attendu : {csv_path}\n"
            f"Fichier XLSX attendu : {xlsx_path}"
        )

    print("\n================ CHARGEMENT ================")
    print("Fichier chargé :")
    print(input_path)
    print("Dimensions :", df.shape)

    return df, input_path

# 4. Audit général du dataset

def audit_dataset(df, config, paths):

    print("\n================ AUDIT DATASET ================")
    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1])

    print("\nTypes des colonnes :")
    print(df.dtypes.value_counts())

    print("\nDoublons exacts :", df.duplicated().sum())

    if "id_clean" in df.columns:
        print("\nContrôle id_clean :")
        print("id_clean manquants :", df["id_clean"].isna().sum())
        print("id_clean uniques :", df["id_clean"].nunique())
        print("doublons id_clean :", df["id_clean"].duplicated().sum())

    missing_report = pd.DataFrame({
        "colonne": df.columns,
        "type": df.dtypes.astype(str).values,
        "valeurs_manquantes": df.isna().sum().values,
        "pourcentage_manquant": (df.isna().sum().values / len(df) * 100).round(2),
        "nb_valeurs_uniques": df.nunique(dropna=False).values
    }).sort_values("pourcentage_manquant", ascending=False)

    print("\nTop 25 valeurs manquantes :")
    print(missing_report.head(25))

    missing_report.to_csv(
        paths["reports_dir"] / "01_audit_valeurs_manquantes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    constant_cols = []
    quasi_constant_rows = []

    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        vc = df[col].value_counts(dropna=False, normalize=True)

        if nunique <= 1:
            constant_cols.append(col)
        elif len(vc) > 0 and vc.iloc[0] >= 0.99:
            quasi_constant_rows.append({
                "colonne": col,
                "modalite_majoritaire_pct": round(vc.iloc[0] * 100, 4),
                "nb_valeurs_uniques": nunique
            })

    print("\nColonnes constantes détectées :")
    print(constant_cols)

    print("\nColonnes quasi constantes >= 99 % :")
    print(pd.DataFrame(quasi_constant_rows))

    pd.DataFrame({"colonne": constant_cols}).to_csv(
        paths["reports_dir"] / "02_colonnes_constantes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(quasi_constant_rows).to_csv(
        paths["reports_dir"] / "03_colonnes_quasi_constantes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    target_log = config["target_log"]
    target_real = config["target_real"]

    if target_real not in df.columns:
        raise ValueError(f"La colonne cible réelle {target_real} est absente.")

    if target_log not in df.columns:
        raise ValueError(f"La colonne cible log {target_log} est absente.")

    print("\nDistribution de price :")
    print(df[target_real].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    print("\nDistribution de log_price :")
    print(df[target_log].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    plt.figure(figsize=(8, 5))
    plt.hist(df[target_real], bins=80)
    plt.title("Distribution de price")
    plt.xlabel("price")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "01_distribution_price.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df[target_log], bins=80)
    plt.title("Distribution de log_price")
    plt.xlabel("log_price")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "02_distribution_log_price.png", dpi=150)
    plt.close()


# 5. Audit de fuite de données

def leakage_audit(df, config, paths):


    print("\n================ AUDIT FUITE DE DONNÉES ================")

    review_features = [
        "number_of_reviews",
        "has_reviews",
        "review_scores_rating_clean",
        "review_scores_accuracy_clean",
        "review_scores_cleanliness_clean",
        "review_scores_checkin_clean",
        "review_scores_communication_clean",
        "review_scores_location_clean",
        "review_scores_value_clean"
    ]

    rows = []

    for col in df.columns:
        status = "keep"
        reason = "Variable utilisable."

        if col == "id_clean":
            status = "drop"
            reason = (
                "Identifiant unique. Utile pour jointure et suivi, "
                "mais interdit comme variable explicative."
            )

        elif col == "price":
            status = "drop"
            reason = (
                "Cible réelle en euros. Elle ne doit jamais être dans X. "
                "Elle sert seulement à interpréter les erreurs."
            )

        elif col == "log_price":
            status = "target"
            reason = "Cible d'entraînement du modèle."

        elif col == "review_scores_value_clean":
            status = "check_high"
            reason = (
                "Note du rapport qualité/prix. Variable autorisée dans ce projet "
                "car on travaille sur des annonces avec historique mais à surveiller "
                "car elle est liée à la perception du prix."
            )

        elif col in review_features:
            status = "check"
            reason = (
                "Variable issue des avis. Autorisée car l'objectif est de prédire "
                "le prix d'une annonce déjà existante avec historique."
            )

        elif df[col].nunique(dropna=False) <= 1:
            status = "drop"
            reason = "Colonne constante. Elle n'apporte aucune information au modèle."

        rows.append({
            "colonne": col,
            "statut": status,
            "justification": reason,
            "type": str(df[col].dtype),
            "nb_valeurs_uniques": df[col].nunique(dropna=False),
            "pct_manquant": round(df[col].isna().mean() * 100, 2)
        })

    audit = pd.DataFrame(rows)

    print("\nColonnes à supprimer / cible / à surveiller :")
    print(audit[audit["statut"].isin(["drop", "target", "check", "check_high"])])

    audit.to_csv(
        paths["reports_dir"] / "04_audit_fuite_donnees.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return audit

# 6. Préparation des variables X et y

def prepare_features(df, config, paths):

    target_log = config["target_log"]
    target_real = config["target_real"]

    cols_to_exclude = list(config["cols_to_exclude"])

    # Suppression automatique des colonnes constantes
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1 and col not in cols_to_exclude:
            cols_to_exclude.append(col)

    review_features = [
        "number_of_reviews",
        "has_reviews",
        "review_scores_rating_clean",
        "review_scores_accuracy_clean",
        "review_scores_cleanliness_clean",
        "review_scores_checkin_clean",
        "review_scores_communication_clean",
        "review_scores_location_clean",
        "review_scores_value_clean"
    ]

    if not config["use_review_features"]:
        cols_to_exclude.extend([col for col in review_features if col in df.columns])

    cols_to_exclude = sorted(set([col for col in cols_to_exclude if col in df.columns]))

    feature_cols = [
        col for col in df.columns
        if col not in cols_to_exclude
    ]

    X = df[feature_cols].copy()
    y_log = df[target_log].copy()
    y_real = df[target_real].copy()

    cat_features = [
        col for col in config["cat_features"]
        if col in X.columns
    ]

    # Si d'autres colonnes texte apparaissent dans X,
    # elles sont ajoutées aux cat_features.
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        if col not in cat_features:
            cat_features.append(col)

    # CatBoost accepte les catégorielles en texte.
    # On remplace seulement les NaN catégoriels par "missing".
    for col in cat_features:
        X[col] = X[col].fillna("missing").astype(str)

    print("\n================ PRÉPARATION DES FEATURES ================")
    print("Nombre de variables explicatives utilisées :", len(feature_cols))
    print("Variables catégorielles CatBoost :", cat_features)
    print("Colonnes exclues :", cols_to_exclude)

    print("\nVérification anti-fuite :")
    print("price dans X :", "price" in X.columns)
    print("log_price dans X :", "log_price" in X.columns)
    print("id_clean dans X :", "id_clean" in X.columns)

    if "nights_range_is_incoherent" in df.columns:
        print("nights_range_is_incoherent dans fichier :", True)
        print("nights_range_is_incoherent dans X :", "nights_range_is_incoherent" in X.columns)
    else:
        print("nights_range_is_incoherent dans fichier :", False)

    print("\nListe complète des variables utilisées par le modèle :")
    for col in X.columns:
        print("-", col)

    pd.DataFrame({"feature": feature_cols}).to_csv(
        paths["reports_dir"] / "05_features_utilisees_modele.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"cat_feature": cat_features}).to_csv(
        paths["reports_dir"] / "06_cat_features_utilisees.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"colonne_exclue": cols_to_exclude}).to_csv(
        paths["reports_dir"] / "07_colonnes_exclues_modele.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return X, y_log, y_real, cat_features, feature_cols, cols_to_exclude

# 7. Split train / validation / test

def make_regression_split(X, y_log, y_real, df, config, paths):
    """
    Split :
    - train : 70 %
    - validation : 15 %
    - test : 15 %

    Pour une régression, on stratifie approximativement avec des bins
    de log_price afin de conserver une distribution comparable des prix.
    """

    random_state = config["random_state"]

    indices = np.arange(len(X))

    y_bins = pd.qcut(
        y_log,
        q=config["n_strat_bins"],
        duplicates="drop",
        labels=False
    )

    idx_train, idx_temp = train_test_split(
        indices,
        train_size=config["train_size"],
        random_state=random_state,
        stratify=y_bins
    )

    y_bins_temp = y_bins.iloc[idx_temp]

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_bins_temp
    )

    X_train = X.iloc[idx_train].copy()
    X_val = X.iloc[idx_val].copy()
    X_test = X.iloc[idx_test].copy()

    y_train_log = y_log.iloc[idx_train].copy()
    y_val_log = y_log.iloc[idx_val].copy()
    y_test_log = y_log.iloc[idx_test].copy()

    y_train_real = y_real.iloc[idx_train].copy()
    y_val_real = y_real.iloc[idx_val].copy()
    y_test_real = y_real.iloc[idx_test].copy()

    print("\n================ SPLIT ================")
    print("Train :", X_train.shape)
    print("Validation :", X_val.shape)
    print("Test :", X_test.shape)

    split_report = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "n_lignes": [len(X_train), len(X_val), len(X_test)],
        "prix_moyen": [y_train_real.mean(), y_val_real.mean(), y_test_real.mean()],
        "prix_median": [y_train_real.median(), y_val_real.median(), y_test_real.median()],
        "log_price_moyen": [y_train_log.mean(), y_val_log.mean(), y_test_log.mean()],
        "prix_min": [y_train_real.min(), y_val_real.min(), y_test_real.min()],
        "prix_max": [y_train_real.max(), y_val_real.max(), y_test_real.max()]
    })

    print("\nRésumé du split :")
    print(split_report)

    split_report.to_csv(
        paths["reports_dir"] / "08_rapport_split_train_validation_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if "id_clean" in df.columns:
        split_ids = pd.DataFrame({
            "id_clean": pd.concat([
                df["id_clean"].iloc[idx_train],
                df["id_clean"].iloc[idx_val],
                df["id_clean"].iloc[idx_test]
            ]),
            "split": (
                ["train"] * len(idx_train)
                + ["validation"] * len(idx_val)
                + ["test"] * len(idx_test)
            )
        })

        split_ids.to_csv(
            paths["reports_dir"] / "09_split_ids.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_log": y_train_log,
        "y_val_log": y_val_log,
        "y_test_log": y_test_log,
        "y_train_real": y_train_real,
        "y_val_real": y_val_real,
        "y_test_real": y_test_real
    }



def build_pools(split_data, cat_features):
    """
    Construit les Pool CatBoost.
    """

    train_pool = Pool(
        data=split_data["X_train"],
        label=split_data["y_train_log"],
        cat_features=cat_features
    )

    val_pool = Pool(
        data=split_data["X_val"],
        label=split_data["y_val_log"],
        cat_features=cat_features
    )

    test_pool = Pool(
        data=split_data["X_test"],
        label=split_data["y_test_log"],
        cat_features=cat_features
    )

    return train_pool, val_pool, test_pool



def train_catboost(train_pool, val_pool, config, paths):
    """
    Entraîne CatBoostRegressor sur log_price avec early stopping.
    """

    print("\n================ ENTRAÎNEMENT CATBOOST ================")

    model = CatBoostRegressor(**config["catboost_params"])

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True
    )

    print("\nBest iteration :", model.get_best_iteration())
    print("Best score :", model.get_best_score())

    model_path = paths["models_dir"] / "catboost_regressor_airbnb_log_price.cbm"
    model.save_model(model_path)

    print("\nModèle sauvegardé :")
    print(model_path)

    with open(paths["reports_dir"] / "10_hyperparametres_catboost.json", "w", encoding="utf-8") as f:
        json.dump(config["catboost_params"], f, indent=4)

    return model

# 10. Métriques de régression

def smape(y_true, y_pred, eps=1e-8):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, eps)) * 100


def safe_mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
        "MAPE": safe_mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "Erreur_moyenne": np.mean(y_pred - y_true)
    }


def evaluate_model(model, split_data, train_pool, val_pool, test_pool, paths):
    """
    Évalue le modèle sur :
    - train
    - validation
    - test

    Deux échelles :
    - log_price
    - price en euros après expm1
    """

    print("\n================ ÉVALUATION ================")

    pools = {
        "train": train_pool,
        "validation": val_pool,
        "test": test_pool
    }

    y_log_true = {
        "train": split_data["y_train_log"],
        "validation": split_data["y_val_log"],
        "test": split_data["y_test_log"]
    }

    y_real_true = {
        "train": split_data["y_train_real"],
        "validation": split_data["y_val_real"],
        "test": split_data["y_test_real"]
    }

    results = []
    predictions = {}

    for split_name, pool in pools.items():
        pred_log = model.predict(pool)

        pred_real = np.expm1(pred_log)
        pred_real = np.maximum(pred_real, 0)

        metrics_log = regression_metrics(
            y_log_true[split_name].values,
            pred_log
        )

        metrics_real = regression_metrics(
            y_real_true[split_name].values,
            pred_real
        )

        for metric, value in metrics_log.items():
            results.append({
                "split": split_name,
                "echelle": "log_price",
                "metric": metric,
                "value": value
            })

        for metric, value in metrics_real.items():
            results.append({
                "split": split_name,
                "echelle": "price_euros",
                "metric": metric,
                "value": value
            })

        predictions[split_name] = pd.DataFrame({
            "y_true_log": y_log_true[split_name].values,
            "y_pred_log": pred_log,
            "y_true_price": y_real_true[split_name].values,
            "y_pred_price": pred_real,
            "residual_price": pred_real - y_real_true[split_name].values,
            "abs_error_price": np.abs(pred_real - y_real_true[split_name].values)
        })

    results_df = pd.DataFrame(results)

    pivot_metrics = results_df.pivot_table(
        index=["echelle", "metric"],
        columns="split",
        values="value"
    ).reset_index()

    print("\nMétriques :")
    print(pivot_metrics)

    results_df.to_csv(
        paths["reports_dir"] / "11_metrics_long_format.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pivot_metrics.to_csv(
        paths["reports_dir"] / "12_metrics_tableau_comparatif.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for split_name, pred_df in predictions.items():
        pred_df.to_csv(
            paths["predictions_dir"] / f"predictions_{split_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return results_df, pivot_metrics, predictions

# 11. Visualisations d'évaluation

def plot_evaluation(predictions, paths):
    """
    Graphiques :
    - prédictions vs valeurs réelles
    - distribution des résidus
    - erreur absolue selon le prix réel
    """

    test_pred = predictions["test"]

    plt.figure(figsize=(7, 7))
    plt.scatter(
        test_pred["y_true_price"],
        test_pred["y_pred_price"],
        alpha=0.25,
        s=8
    )

    max_value = np.percentile(
        np.concatenate([
            test_pred["y_true_price"].values,
            test_pred["y_pred_price"].values
        ]),
        99
    )

    plt.plot([0, max_value], [0, max_value])
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("Test - Prix prédit vs prix réel")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "03_test_pred_vs_true_price.png", dpi=150)
    plt.close()

    residuals = test_pred["residual_price"]

    plt.figure(figsize=(8, 5))
    residuals_clip = residuals.clip(
        lower=np.percentile(residuals, 1),
        upper=np.percentile(residuals, 99)
    )
    plt.hist(residuals_clip, bins=80)
    plt.xlabel("Résidu en euros")
    plt.ylabel("count")
    plt.title("Distribution des résidus sur test")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "04_test_residus_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        test_pred["y_true_price"],
        test_pred["abs_error_price"],
        alpha=0.25,
        s=8
    )
    plt.xlim(0, np.percentile(test_pred["y_true_price"], 99))
    plt.ylim(0, np.percentile(test_pred["abs_error_price"], 99))
    plt.xlabel("Prix réel")
    plt.ylabel("Erreur absolue")
    plt.title("Erreur absolue selon le prix réel - test")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "05_test_erreur_absolue_selon_prix.png", dpi=150)
    plt.close()

# 12. Importance des variables CatBoost

def save_feature_importance(model, train_pool, feature_cols, paths):
    """
    Sauvegarde l'importance native CatBoost.
    """

    importance = model.get_feature_importance(
        data=train_pool,
        type="FeatureImportance"
    )

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        paths["reports_dir"] / "13_feature_importance_catboost.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop 30 variables importantes :")
    print(fi.head(30))

    top = fi.head(30).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance CatBoost")
    plt.title("Top 30 variables importantes")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "06_feature_importance_top30.png", dpi=150)
    plt.close()

    return fi

# 13. SHAP CatBoost
def save_shap_values(model, split_data, cat_features, paths):
    """
    Calcule les valeurs SHAP sur un échantillon du train.
    """

    print("\n================ SHAP CATBOOST ================")

    X_train = split_data["X_train"]
    y_train_log = split_data["y_train_log"]

    sample_size = min(2000, len(X_train))

    X_sample = X_train.sample(
        sample_size,
        random_state=CONFIG["random_state"]
    )

    y_sample = y_train_log.loc[X_sample.index]

    shap_pool = Pool(
        data=X_sample,
        label=y_sample,
        cat_features=cat_features
    )

    shap_values = model.get_feature_importance(
        data=shap_pool,
        type="ShapValues"
    )

    shap_feature_values = shap_values[:, :-1]

    mean_abs_shap = np.abs(shap_feature_values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        paths["reports_dir"] / "14_shap_mean_abs_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop 30 SHAP :")
    print(shap_importance.head(30))

    top = shap_importance.head(30).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP|")
    plt.title("Top 30 variables selon SHAP")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "07_shap_top30_mean_abs.png", dpi=150)
    plt.close()

    return shap_importance

# 14. Sauvegarde du résumé global

def save_run_summary(df, feature_cols, cat_features, cols_excluded, model, config, paths):

    summary = {
        "dataset_lignes": int(df.shape[0]),
        "dataset_colonnes": int(df.shape[1]),
        "nombre_variables_utilisees": int(len(feature_cols)),
        "variables_categorielles": cat_features,
        "colonnes_exclues": cols_excluded,
        "target_train": config["target_log"],
        "target_real_interpretation": config["target_real"],
        "reviews_utilisees": bool(config["use_review_features"]),
        "best_iteration": int(model.get_best_iteration()),
        "fichier_entree": config["input_filename_csv"]
    }

    with open(paths["reports_dir"] / "15_resume_experience_catboost.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame({
        "feature": feature_cols
    }).to_csv(
        paths["reports_dir"] / "16_liste_finale_variables_modele.csv",
        index=False,
        encoding="utf-8-sig"
    )



def main():
    df, input_path = load_data(CONFIG, PATHS)

    audit_dataset(df, CONFIG, PATHS)

    leakage_audit(df, CONFIG, PATHS)

    X, y_log, y_real, cat_features, feature_cols, cols_excluded = prepare_features(
        df=df,
        config=CONFIG,
        paths=PATHS
    )

    split_data = make_regression_split(
        X=X,
        y_log=y_log,
        y_real=y_real,
        df=df,
        config=CONFIG,
        paths=PATHS
    )

    train_pool, val_pool, test_pool = build_pools(
        split_data=split_data,
        cat_features=cat_features
    )

    model = train_catboost(
        train_pool=train_pool,
        val_pool=val_pool,
        config=CONFIG,
        paths=PATHS
    )

    results_df, pivot_metrics, predictions = evaluate_model(
        model=model,
        split_data=split_data,
        train_pool=train_pool,
        val_pool=val_pool,
        test_pool=test_pool,
        paths=PATHS
    )

    plot_evaluation(
        predictions=predictions,
        paths=PATHS
    )

    fi = save_feature_importance(
        model=model,
        train_pool=train_pool,
        feature_cols=feature_cols,
        paths=PATHS
    )

    shap_importance = save_shap_values(
        model=model,
        split_data=split_data,
        cat_features=cat_features,
        paths=PATHS
    )

    save_run_summary(
        df=df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        cols_excluded=cols_excluded,
        model=model,
        config=CONFIG,
        paths=PATHS
    )

    print("\n================ FIN ================")
    print("Entraînement CatBoost terminé.")
    print("Résultats sauvegardés dans :")
    print(PATHS["output_dir"])


if __name__ == "__main__":
    main()