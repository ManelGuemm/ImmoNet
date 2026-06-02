# ============================================================
# 02_analyse_erreurs_par_segments_prix.py
# Objectif :
# - Analyser les erreurs du modèle CatBoost baseline par segments de prix
# - Utiliser les prédictions déjà générées par 01_train_catboost_tabulaire.py
# - Identifier les zones où le modèle fonctionne bien ou mal
# - Vérifier si les logements chers sont sous-estimés


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

warnings.filterwarnings("ignore")

# 1. Gestion des chemins

def get_project_paths():

    script_dir = Path(__file__).resolve().parent

    resultats_dir = script_dir / "Resultats_CatBoost_Tabulaire"
    predictions_dir = resultats_dir / "predictions"

    if not resultats_dir.exists():
        raise FileNotFoundError(
            f"Le dossier de résultats est introuvable : {resultats_dir}\n"
            "Lance d'abord le script 01_train_catboost_tabulaire.py."
        )

    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Le dossier des prédictions est introuvable : {predictions_dir}\n"
            "Vérifie que le modèle a bien généré les fichiers de prédictions."
        )

    output_dir = resultats_dir / "analyse_segments_prix"
    reports_dir = output_dir / "rapports_segments_prix"
    plots_dir = output_dir / "graphiques_segments_prix"

    for path in [output_dir, reports_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "resultats_dir": resultats_dir,
        "predictions_dir": predictions_dir,
        "output_dir": output_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir
    }


def find_prediction_file(predictions_dir, split_name):
  
    candidates = [
        predictions_dir / f"predictions_{split_name}.csv",
        predictions_dir / f"{split_name}_predictions.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Recherche plus flexible
    possible_files = list(predictions_dir.glob(f"*{split_name}*prediction*.csv"))

    if len(possible_files) > 0:
        return possible_files[0]

    raise FileNotFoundError(
        f"Aucun fichier de prédictions trouvé pour le split : {split_name}\n"
        f"Dossier cherché : {predictions_dir}"
    )


def standardize_prediction_columns(df):
    """
    Harmonise les noms de colonnes si les fichiers ont des noms différents.
    """

    rename_map = {}

    if "y_real_true" in df.columns and "y_true_price" not in df.columns:
        rename_map["y_real_true"] = "y_true_price"

    if "y_real_pred" in df.columns and "y_pred_price" not in df.columns:
        rename_map["y_real_pred"] = "y_pred_price"

    if "y_log_true" in df.columns and "y_true_log" not in df.columns:
        rename_map["y_log_true"] = "y_true_log"

    if "y_log_pred" in df.columns and "y_pred_log" not in df.columns:
        rename_map["y_log_pred"] = "y_pred_log"

    df = df.rename(columns=rename_map)

    required_cols = [
        "y_true_price",
        "y_pred_price"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Colonne obligatoire absente dans le fichier de prédictions : {col}\n"
                f"Colonnes disponibles : {df.columns.tolist()}"
            )

    df["y_true_price"] = pd.to_numeric(df["y_true_price"], errors="coerce")
    df["y_pred_price"] = pd.to_numeric(df["y_pred_price"], errors="coerce")

    df["residual_price"] = df["y_pred_price"] - df["y_true_price"]
    df["abs_error_price"] = np.abs(df["residual_price"])

    if "y_true_log" in df.columns:
        df["y_true_log"] = pd.to_numeric(df["y_true_log"], errors="coerce")

    if "y_pred_log" in df.columns:
        df["y_pred_log"] = pd.to_numeric(df["y_pred_log"], errors="coerce")
        if "y_true_log" in df.columns:
            df["residual_log"] = df["y_pred_log"] - df["y_true_log"]
            df["abs_error_log"] = np.abs(df["residual_log"])

    return df


def load_all_predictions(paths):
    """
    Charge les prédictions train, validation et test.
    """

    predictions = {}

    for split_name in ["train", "validation", "test"]:
        file_path = find_prediction_file(paths["predictions_dir"], split_name)

        df_pred = pd.read_csv(file_path, low_memory=False)
        df_pred = standardize_prediction_columns(df_pred)

        predictions[split_name] = df_pred

        print(f"\nFichier chargé pour {split_name} :")
        print(file_path)
        print("Dimensions :", df_pred.shape)
        print("Prix réel moyen :", round(df_pred["y_true_price"].mean(), 2))
        print("Prix prédit moyen :", round(df_pred["y_pred_price"].mean(), 2))

    return predictions

# 3. Fonctions de métriques

def safe_mape(y_true, y_pred, eps=1e-8):
    """
    MAPE sécurisé, en pourcentage.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    ) * 100


def smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric MAPE, en pourcentage.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    return np.mean(
        np.abs(y_true - y_pred) / np.maximum(denominator, eps)
    ) * 100


def compute_regression_metrics(y_true, y_pred):
    """
    Calcule les métriques principales pour un segment.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)

    if n == 0:
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

    if n >= 2 and len(np.unique(y_true)) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    return {
        "n": n,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2,
        "MAPE_pct": safe_mape(y_true, y_pred),
        "SMAPE_pct": smape(y_true, y_pred),
        "Mean_Error": np.mean(residuals),
        "Underestimation_Rate_pct": np.mean(residuals < 0) * 100,
        "Overestimation_Rate_pct": np.mean(residuals > 0) * 100,
        "Abs_Error_P75": np.percentile(abs_errors, 75),
        "Abs_Error_P90": np.percentile(abs_errors, 90),
        "Abs_Error_P95": np.percentile(abs_errors, 95),
    }


# ============================================================
# 4. Création des segments de prix
# ============================================================

def add_price_segments(df):
    """
    Ajoute des segments fixes de prix.
    Ces segments sont plus lisibles pour le mémoire.
    """

    bins = [0, 100, 200, 400, 800, np.inf]

    labels = [
        "< 100 €",
        "100–200 €",
        "200–400 €",
        "400–800 €",
        "> 800 €"
    ]

    df = df.copy()

    df["price_segment"] = pd.cut(
        df["y_true_price"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    return df


def add_quantile_segments(df):
    """
    Ajoute des segments par quantiles.
    Utile pour voir les erreurs par groupes de taille proche.
    """

    df = df.copy()

    df["price_quantile_segment"] = pd.qcut(
        df["y_true_price"],
        q=4,
        labels=[
            "Q1 - prix les plus bas",
            "Q2",
            "Q3",
            "Q4 - prix les plus élevés"
        ],
        duplicates="drop"
    )

    return df

# 5. Analyse par segments

def analyze_segments_for_split(df_pred, split_name):
    """
    Analyse un split donné par segments fixes et par quantiles.
    """

    df_pred = add_price_segments(df_pred)
    df_pred = add_quantile_segments(df_pred)

    results_fixed = []
    results_quantile = []

    # Segments fixes
    for segment, group in df_pred.groupby("price_segment", observed=False):
        metrics = compute_regression_metrics(
            group["y_true_price"],
            group["y_pred_price"]
        )

        row = {
            "split": split_name,
            "segment_type": "fixed_price_bins",
            "segment": str(segment),
            "price_min": group["y_true_price"].min() if len(group) > 0 else np.nan,
            "price_max": group["y_true_price"].max() if len(group) > 0 else np.nan,
            "price_mean": group["y_true_price"].mean() if len(group) > 0 else np.nan,
            "price_median": group["y_true_price"].median() if len(group) > 0 else np.nan,
            **metrics
        }

        results_fixed.append(row)

    # Segments par quantiles
    for segment, group in df_pred.groupby("price_quantile_segment", observed=False):
        metrics = compute_regression_metrics(
            group["y_true_price"],
            group["y_pred_price"]
        )

        row = {
            "split": split_name,
            "segment_type": "price_quantiles",
            "segment": str(segment),
            "price_min": group["y_true_price"].min() if len(group) > 0 else np.nan,
            "price_max": group["y_true_price"].max() if len(group) > 0 else np.nan,
            "price_mean": group["y_true_price"].mean() if len(group) > 0 else np.nan,
            "price_median": group["y_true_price"].median() if len(group) > 0 else np.nan,
            **metrics
        }

        results_quantile.append(row)

    return pd.DataFrame(results_fixed), pd.DataFrame(results_quantile), df_pred


def analyze_all_splits(predictions, paths):
    """
    Lance l'analyse sur train, validation et test.
    """

    all_fixed = []
    all_quantile = []
    predictions_with_segments = {}

    for split_name, df_pred in predictions.items():
        fixed_df, quantile_df, df_with_segments = analyze_segments_for_split(
            df_pred=df_pred,
            split_name=split_name
        )

        all_fixed.append(fixed_df)
        all_quantile.append(quantile_df)
        predictions_with_segments[split_name] = df_with_segments

    fixed_results = pd.concat(all_fixed, ignore_index=True)
    quantile_results = pd.concat(all_quantile, ignore_index=True)

    fixed_results.to_csv(
        paths["reports_dir"] / "01_metrics_par_segments_prix_fixes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    quantile_results.to_csv(
        paths["reports_dir"] / "02_metrics_par_quantiles_prix.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for split_name, df_seg in predictions_with_segments.items():
        df_seg.to_csv(
            paths["reports_dir"] / f"03_predictions_{split_name}_avec_segments.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return fixed_results, quantile_results, predictions_with_segments

# 6. Graphiques

def plot_segment_metrics_test(fixed_results, paths):
    """
    Graphiques principaux sur le test set.
    """

    test_results = fixed_results[
        (fixed_results["split"] == "test")
        & (fixed_results["segment_type"] == "fixed_price_bins")
    ].copy()

    # Garder l'ordre logique des segments
    segment_order = ["< 100 €", "100–200 €", "200–400 €", "400–800 €", "> 800 €"]
    test_results["segment"] = pd.Categorical(
        test_results["segment"],
        categories=segment_order,
        ordered=True
    )
    test_results = test_results.sort_values("segment")

    # Graphique MAE par segment
    plt.figure(figsize=(9, 5))
    plt.bar(test_results["segment"].astype(str), test_results["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE en euros")
    plt.title("Erreur absolue moyenne par segment de prix - Test")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "01_mae_par_segment_test.png", dpi=150)
    plt.close()

    # Graphique RMSE par segment
    plt.figure(figsize=(9, 5))
    plt.bar(test_results["segment"].astype(str), test_results["RMSE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("RMSE en euros")
    plt.title("RMSE par segment de prix - Test")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "02_rmse_par_segment_test.png", dpi=150)
    plt.close()

    # Biais moyen par segment
    plt.figure(figsize=(9, 5))
    plt.bar(test_results["segment"].astype(str), test_results["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Erreur moyenne en euros")
    plt.title("Biais moyen par segment - négatif = sous-estimation")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "03_biais_moyen_par_segment_test.png", dpi=150)
    plt.close()

    # Taux de sous-estimation
    plt.figure(figsize=(9, 5))
    plt.bar(test_results["segment"].astype(str), test_results["Underestimation_Rate_pct"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Taux de sous-estimation (%)")
    plt.title("Taux de sous-estimation par segment - Test")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "04_taux_sous_estimation_par_segment_test.png", dpi=150)
    plt.close()


def plot_prediction_scatter_by_segment(predictions_with_segments, paths):
    """
    Graphique prédiction vs réel avec segments.
    """

    df_test = predictions_with_segments["test"].copy()

    # Version globale
    plt.figure(figsize=(7, 7))
    plt.scatter(
        df_test["y_true_price"],
        df_test["y_pred_price"],
        alpha=0.25,
        s=8
    )

    max_value = np.percentile(
        np.concatenate([
            df_test["y_true_price"].values,
            df_test["y_pred_price"].values
        ]),
        99
    )

    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("Prix prédit vs prix réel - Test")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "05_pred_vs_true_test_global.png", dpi=150)
    plt.close()

    # Erreur absolue selon le prix
    plt.figure(figsize=(9, 5))
    plt.scatter(
        df_test["y_true_price"],
        df_test["abs_error_price"],
        alpha=0.25,
        s=8
    )

    plt.xlim(0, np.percentile(df_test["y_true_price"], 99))
    plt.ylim(0, np.percentile(df_test["abs_error_price"], 99))
    plt.xlabel("Prix réel")
    plt.ylabel("Erreur absolue")
    plt.title("Erreur absolue selon le prix réel - Test")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "06_erreur_absolue_selon_prix_test.png", dpi=150)
    plt.close()

    # Boxplot des erreurs absolues par segment
    segment_order = ["< 100 €", "100–200 €", "200–400 €", "400–800 €", "> 800 €"]

    data_to_plot = []

    for segment in segment_order:
        values = df_test.loc[
            df_test["price_segment"].astype(str) == segment,
            "abs_error_price"
        ].dropna().values

        # On limite visuellement les valeurs extrêmes pour ne pas écraser le graphe
        if len(values) > 0:
            upper = np.percentile(values, 95)
            values = values[values <= upper]

        data_to_plot.append(values)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_to_plot, labels=segment_order, showfliers=False)
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Erreur absolue en euros")
    plt.title("Distribution des erreurs absolues par segment - Test")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "07_boxplot_erreurs_par_segment_test.png", dpi=150)
    plt.close()

# 7. Rapport texte interprétable

def generate_text_report(fixed_results, paths):


    test_results = fixed_results[
        (fixed_results["split"] == "test")
        & (fixed_results["segment_type"] == "fixed_price_bins")
    ].copy()

    segment_order = ["< 100 €", "100–200 €", "200–400 €", "400–800 €", "> 800 €"]
    test_results["segment"] = pd.Categorical(
        test_results["segment"],
        categories=segment_order,
        ordered=True
    )
    test_results = test_results.sort_values("segment")

    report_path = paths["reports_dir"] / "04_rapport_interpretation_segments_prix.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ANALYSE DES ERREURS PAR SEGMENTS DE PRIX - TEST SET\n")
        f.write("=" * 80 + "\n\n")

        f.write("Objectif :\n")
        f.write(
            "Ce rapport analyse les erreurs du modèle CatBoost baseline selon "
            "différentes gammes de prix réels. L'objectif est de vérifier si le "
            "modèle se comporte de manière homogène ou s'il est moins fiable sur "
            "les logements chers.\n\n"
        )

        f.write("Résultats par segment :\n")
        f.write("-" * 80 + "\n")

        for _, row in test_results.iterrows():
            f.write(f"\nSegment : {row['segment']}\n")
            f.write(f"  Nombre d'annonces : {int(row['n'])}\n")
            f.write(f"  Prix moyen réel : {row['price_mean']:.2f} €\n")
            f.write(f"  Prix médian réel : {row['price_median']:.2f} €\n")
            f.write(f"  MAE : {row['MAE']:.2f} €\n")
            f.write(f"  RMSE : {row['RMSE']:.2f} €\n")
            f.write(f"  MedAE : {row['MedAE']:.2f} €\n")
            f.write(f"  R2 : {row['R2']:.4f}\n")
            f.write(f"  MAPE : {row['MAPE_pct']:.2f} %\n")
            f.write(f"  Erreur moyenne : {row['Mean_Error']:.2f} €\n")
            f.write(f"  Taux de sous-estimation : {row['Underestimation_Rate_pct']:.2f} %\n")
            f.write(f"  Erreur absolue P90 : {row['Abs_Error_P90']:.2f} €\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("LECTURE À FAIRE\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "Si la MAE et la RMSE augmentent fortement dans les segments chers, "
            "cela confirme que le modèle prédit moins bien les logements haut de gamme.\n"
        )
        f.write(
            "Si l'erreur moyenne est négative dans les segments chers, cela signifie "
            "que le modèle sous-estime ces logements.\n"
        )
        f.write(
            "Si le taux de sous-estimation dépasse largement 50 %, cela signifie que "
            "le modèle prédit trop souvent un prix inférieur au prix réel dans ce segment.\n"
        )

        f.write("\nPistes d'amélioration à tester ensuite :\n")
        f.write("- pondération plus forte des logements chers pendant l'entraînement,\n")
        f.write("- Quantile Loss avec alpha > 0.5 pour limiter la sous-estimation,\n")
        f.write("- calibration des prédictions sur validation,\n")
        f.write("- ajout de variables textuelles ou visuelles pour mieux capter le standing,\n")
        f.write("- modèle spécialisé pour logements premium mais sans utiliser le vrai prix au moment de la prédiction.\n")

    print("\nRapport texte généré :")
    print(report_path)



def main():
    paths = get_project_paths()

    print("\n" + "=" * 80)
    print("ANALYSE DES ERREURS PAR SEGMENTS DE PRIX")
    print("=" * 80)

    predictions = load_all_predictions(paths)

    fixed_results, quantile_results, predictions_with_segments = analyze_all_splits(
        predictions=predictions,
        paths=paths
    )

    print("\nRésultats par segments fixes sauvegardés dans :")
    print(paths["reports_dir"] / "01_metrics_par_segments_prix_fixes.csv")

    print("\nRésultats par quantiles sauvegardés dans :")
    print(paths["reports_dir"] / "02_metrics_par_quantiles_prix.csv")

    print("\nAperçu des résultats TEST par segments fixes :")
    test_fixed = fixed_results[
        (fixed_results["split"] == "test")
        & (fixed_results["segment_type"] == "fixed_price_bins")
    ]

    cols_to_show = [
        "segment",
        "n",
        "price_mean",
        "MAE",
        "RMSE",
        "MedAE",
        "MAPE_pct",
        "Mean_Error",
        "Underestimation_Rate_pct"
    ]

    print(test_fixed[cols_to_show].to_string(index=False))

    plot_segment_metrics_test(
        fixed_results=fixed_results,
        paths=paths
    )

    plot_prediction_scatter_by_segment(
        predictions_with_segments=predictions_with_segments,
        paths=paths
    )

    generate_text_report(
        fixed_results=fixed_results,
        paths=paths
    )

    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])


if __name__ == "__main__":
    main()