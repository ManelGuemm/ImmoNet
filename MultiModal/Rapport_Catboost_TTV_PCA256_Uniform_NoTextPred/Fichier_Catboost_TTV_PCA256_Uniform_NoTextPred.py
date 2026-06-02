from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

# 1. CONFIGURATION

@dataclass
class Config:
    random_state: int = 42

    # Fichiers d'entrée
    tabular_path: str = "airbnb_tabulaire_visuel_ready.csv"
    split_path: str = "split_listing_ids.csv"
    visual_embeddings_npy_path: str = "Resultat2/efficientnet_b0_embeddings.npy"
    visual_embeddings_ids_path: str = "Resultat2/efficientnet_b0_embeddings_ids.csv"
    text_features_train_path: str = "outputs_text_e5_lora/airbnb_text_features_train_e5_lora.csv"
    text_features_test_path: str = "outputs_text_e5_lora/airbnb_text_features_test_e5_lora.csv"

    # Dossier de sortie
    output_dir_name: str = "Rapport_Catboost_TTV_PCA256_Uniform_NoTextPred"

    # Cibles et ID
    target_log: str = "log_price"
    target_real: str = "price"
    id_col: str = "id_clean"

    # Texte
    text_embedding_prefix: str = "txt_e5_"
    use_text_embeddings: bool = True
    use_text_length_features: bool = True
    use_text_prediction_feature: bool = False  # Important : toujours False ici.

    # Visuel
    visual_pca_components: int = 256
    pca_svd_solver: str = "randomized"


    drop_price_above: Optional[float] = None

    # Colonnes à exclure du bloc tabulaire
    cols_to_exclude: Tuple[str, ...] = (
        "id", "id_clean", "listing_id_clean", "listing_url", "picture_url", "image_path", "row_npy",
        "price", "price_clean", "log_price", "price_txt", "price_clean_txt", "log_price_txt",
        "split", "split_x", "split_y", "split_txt",
        "nights_range_is_incoherent",
        "has_reviews", "nb_avis_textuels_bert", "bert_stars_moyen",
        "text_pred_final", "text_pred_oof", "text_pred_test",
    )

    forbidden_prefixes_tabular: Tuple[str, ...] = (
        "txt_e5_", "text_", "bert_embedding_", "embedding_",
        "img_emb_", "img_", "image_", "clip_", "resnet_", "efficientnet_",
    )

    raw_text_cols_to_exclude: Tuple[str, ...] = (
        "name", "description", "neighborhood_overview", "host_about", "amenities",
    )

    base_cat_features: Tuple[str, ...] = (
        "host_response_time_clean",
        "room_type_clean",
        "neighbourhood_cleansed_clean",
        "property_type_clean",
    )

    # CV
    n_splits: int = 5
    n_strat_bins_cv: int = 20

    # Segments gardés uniquement pour l'analyse des erreurs
    price_segment_bins: Tuple[float, ...] = (0, 100, 200, 400, 800, np.inf)
    price_segment_labels: Tuple[str, ...] = ("< 100 €", "100-200 €", "200-400 €", "400-800 €", "> 800 €")

    # Pondération 
    use_uniform_weights: bool = True
    uniform_weight_value: float = 1.0
    weight_validation_pool: bool = False

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
    used_ram_limit: Optional[str] = "20gb"
    allow_writing_files: bool = False
    verbose_eval: int = 200

    # Interprétation
    compute_shap_final: bool = True
    shap_max_rows_final: int = 2000
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    return obj


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def normalize_split_value(value: Any) -> str:
    v = str(value).strip().lower()
    if v in {"train", "train_dev", "train-dev", "training", "apprentissage"}:
        return "train_dev"
    if v in {"test", "test_final", "test-final", "final_test", "testing"}:
        return "test_final"
    raise ValueError(f"Valeur de split inconnue : {value}")


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
    return pd.Series(0, index=y.index).astype(int)


def get_paths(cfg: Config) -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg.output_dir_name
    paths = {
        "script_dir": script_dir,
        "tabular_path": script_dir / cfg.tabular_path,
        "split_path": script_dir / cfg.split_path,
        "visual_embeddings_npy_path": script_dir / cfg.visual_embeddings_npy_path,
        "visual_embeddings_ids_path": script_dir / cfg.visual_embeddings_ids_path,
        "text_features_train_path": script_dir / cfg.text_features_train_path,
        "text_features_test_path": script_dir / cfg.text_features_test_path,
        "output_dir": output_dir,
        "reports_dir": output_dir / "rapports",
        "plots_dir": output_dir / "graphiques",
        "predictions_dir": output_dir / "predictions",
        "models_dir": output_dir / "modeles",
        "shap_dir": output_dir / "shap",
        "pca_dir": output_dir / "pca",
    }
    for k, p in paths.items():
        if k.endswith("_dir") or k == "output_dir":
            p.mkdir(parents=True, exist_ok=True)
    return paths

# 3. CHARGEMENT TEXTUEL SANS text_pred

def load_text_features(cfg: Config, paths: Dict[str, Path]) -> pd.DataFrame:
    print("\n================ CHARGEMENT FEATURES TEXTUELLES ================")
    for key in ["text_features_train_path", "text_features_test_path"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Fichier textuel introuvable : {paths[key]}")

    text_train = pd.read_csv(paths["text_features_train_path"], dtype={"listing_id_clean": str}, low_memory=False)
    text_test = pd.read_csv(paths["text_features_test_path"], dtype={"listing_id_clean": str}, low_memory=False)
    text_all = pd.concat([text_train, text_test], ignore_index=True)

    if "listing_id_clean" not in text_all.columns:
        raise ValueError("listing_id_clean absent des fichiers textuels.")

    text_all["listing_id_clean"] = normalize_id_series(text_all["listing_id_clean"])
    if text_all["listing_id_clean"].duplicated().any():
        dup = text_all.loc[text_all["listing_id_clean"].duplicated(), "listing_id_clean"].head(10).tolist()
        raise ValueError(f"IDs textuels dupliqués. Exemples : {dup}")

    text_embedding_cols = [c for c in text_all.columns if c.startswith(cfg.text_embedding_prefix)]

    text_aux_cols = []
    if cfg.use_text_length_features:
        for c in ["name_n_words", "description_n_words", "description_is_missing"]:
            if c in text_all.columns:
                text_aux_cols.append(c)

    # Ici, on retire explicitement toutes les prédictions texte.
    forbidden_text_pred_cols = [c for c in ["text_pred_final", "text_pred_oof", "text_pred_test"] if c in text_all.columns]

    selected_cols = ["listing_id_clean"]
    if cfg.use_text_embeddings:
        selected_cols += text_embedding_cols
    selected_cols += text_aux_cols

    if cfg.use_text_prediction_feature:
        raise ValueError("Ce script doit rester sans text_pred_final. Mets use_text_prediction_feature=False.")

    text_selected = text_all[selected_cols].copy()
    for c in selected_cols:
        if c != "listing_id_clean":
            text_selected[c] = pd.to_numeric(text_selected[c], errors="coerce")

    print("Text train :", text_train.shape)
    print("Text test  :", text_test.shape)
    print("Features textuelles retenues :", len(selected_cols) - 1)
    print("Nombre de colonnes txt_e5_* :", len(text_embedding_cols))
    print("Features auxiliaires texte :", text_aux_cols)
    print("Colonnes text_pred explicitement ignorées :", forbidden_text_pred_cols)

    pd.DataFrame({"text_feature": selected_cols[1:]}).to_csv(
        paths["reports_dir"] / "00_text_features_utilisees_sans_text_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"ignored_text_pred_column": forbidden_text_pred_cols}).to_csv(
        paths["reports_dir"] / "00_text_pred_columns_ignorees.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return text_selected

# 4. ALIGNEMENT MULTIMODAL

def load_and_align_data(cfg: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    print("\n================ CHARGEMENT ET ALIGNEMENT ================")
    required = ["tabular_path", "split_path", "visual_embeddings_npy_path", "visual_embeddings_ids_path"]
    for name in required:
        if not paths[name].exists():
            raise FileNotFoundError(f"Fichier introuvable : {paths[name]}")

    df = pd.read_csv(paths["tabular_path"], dtype={cfg.id_col: str}, low_memory=False)
    split_df = pd.read_csv(paths["split_path"], dtype=str)
    vis_ids = pd.read_csv(paths["visual_embeddings_ids_path"], dtype=str)
    vis_npy = np.load(paths["visual_embeddings_npy_path"], mmap_mode="r")
    text_features = load_text_features(cfg, paths)

    if cfg.id_col not in df.columns:
        raise ValueError(f"Colonne absente dans le tabulaire : {cfg.id_col}")
    if "listing_id_clean" in split_df.columns:
        split_id_col = "listing_id_clean"
    elif "id_clean" in split_df.columns:
        split_id_col = "id_clean"
    else:
        raise ValueError("Aucune colonne ID trouvée dans split_listing_ids.csv")
    if "split" not in split_df.columns:
        raise ValueError("La colonne split est absente de split_listing_ids.csv")

    if {"row_npy", "listing_id_clean"} - set(vis_ids.columns):
        raise ValueError("Le fichier efficientnet_b0_embeddings_ids.csv doit contenir row_npy et listing_id_clean.")
    if len(vis_ids) != vis_npy.shape[0]:
        raise ValueError(f"Alignement visuel incorrect : IDs={len(vis_ids)}, NPY={vis_npy.shape[0]}")
    if vis_npy.shape[1] != 1280:
        raise ValueError(f"Dimension EfficientNet-B0 attendue : 1280, trouvée : {vis_npy.shape[1]}")

    df[cfg.id_col] = normalize_id_series(df[cfg.id_col])
    split_df[split_id_col] = normalize_id_series(split_df[split_id_col])
    vis_ids["listing_id_clean"] = normalize_id_series(vis_ids["listing_id_clean"])
    vis_ids["row_npy"] = pd.to_numeric(vis_ids["row_npy"], errors="raise").astype(int)
    text_features["listing_id_clean"] = normalize_id_series(text_features["listing_id_clean"])

    original_tab_ids = set(df[cfg.id_col])
    visual_ids = set(vis_ids["listing_id_clean"])
    text_ids = set(text_features["listing_id_clean"])

    split_clean = split_df[[split_id_col, "split"]].copy().rename(columns={split_id_col: cfg.id_col})
    split_clean["split"] = split_clean["split"].apply(normalize_split_value)

    for c in ["split", "split_x", "split_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    vis_clean = vis_ids[["listing_id_clean", "row_npy"]].copy().rename(columns={"listing_id_clean": cfg.id_col})
    if "image_path" in vis_ids.columns:
        vis_clean["image_path"] = vis_ids["image_path"]

    text_clean = text_features.rename(columns={"listing_id_clean": cfg.id_col})

    df = df.merge(split_clean, on=cfg.id_col, how="inner", validate="one_to_one")
    df = df.merge(vis_clean, on=cfg.id_col, how="inner", validate="one_to_one")
    before_text_merge = len(df)
    df = df.merge(text_clean, on=cfg.id_col, how="inner", validate="one_to_one")
    after_text_merge = len(df)
    df = df.reset_index(drop=True)

    row_indices = df["row_npy"].to_numpy(dtype=int)
    if row_indices.min() < 0 or row_indices.max() >= vis_npy.shape[0]:
        raise ValueError("row_npy contient des index hors limites.")

    X_img = np.asarray(vis_npy[row_indices], dtype=np.float32)
    text_feature_cols = [c for c in text_clean.columns if c != cfg.id_col]
    X_text = df[text_feature_cols].copy()
    for c in X_text.columns:
        X_text[c] = pd.to_numeric(X_text[c], errors="coerce")

    if len(df) != len(X_text) or len(df) != X_img.shape[0]:
        raise ValueError("Alignement incohérent entre df, X_text et X_img.")

    dropped_by_price_ids = []
    if cfg.drop_price_above is not None:
        price_numeric = pd.to_numeric(df[cfg.target_real], errors="coerce")
        keep_mask = price_numeric <= float(cfg.drop_price_above)
        dropped_by_price_ids = df.loc[~keep_mask, cfg.id_col].astype(str).tolist()

        df = df.loc[keep_mask].reset_index(drop=True)
        X_text = X_text.loc[keep_mask].reset_index(drop=True)
        X_img = X_img[keep_mask.to_numpy()]

        if len(df) != len(X_text) or len(df) != X_img.shape[0]:
            raise ValueError("Erreur : suppression prix appliquée différemment au tabulaire, au textuel ou au visuel.")

    ids_after = set(df[cfg.id_col])
    final_removed_ids = sorted(original_tab_ids - ids_after)
    text_embeddings_ignored = sorted(text_ids - ids_after)

    print("\n================ ALIGNEMENT FINAL ================")
    print("Tabulaire initial :", len(original_tab_ids))
    print("Visuel disponible :", len(visual_ids))
    print("Textuel disponible :", len(text_ids))
    print("Lignes avant merge textuel :", before_text_merge)
    print("Lignes après merge textuel :", after_text_merge)
    print("Drop price above :", cfg.drop_price_above)
    print("Annonces supprimées par seuil prix :", len(dropped_by_price_ids))
    print("Lignes finales :", df.shape)
    print("X_text :", X_text.shape)
    print("X_img  :", X_img.shape)
    print("Split :")
    print(df["split"].value_counts(dropna=False))

    audit = {
        "tabular_rows_initial": int(len(original_tab_ids)),
        "visual_ids_available": int(len(visual_ids)),
        "text_ids_available": int(len(text_ids)),
        "rows_before_text_merge": int(before_text_merge),
        "rows_after_text_merge": int(after_text_merge),
        "drop_price_above": cfg.drop_price_above,
        "n_dropped_by_price_threshold": int(len(dropped_by_price_ids)),
        "rows_after_final_alignment": int(len(df)),
        "visual_matrix_shape": list(X_img.shape),
        "text_matrix_shape": list(X_text.shape),
        "n_train_dev": int((df["split"] == "train_dev").sum()),
        "n_test_final": int((df["split"] == "test_final").sum()),
        "n_final_removed_ids": int(len(final_removed_ids)),
        "n_text_embeddings_ignored_due_to_alignment": int(len(text_embeddings_ignored)),
        "use_text_prediction_feature": False,
        "weight_strategy": "uniform_no_premium_weighting",
    }
    with open(paths["reports_dir"] / "00_audit_alignement_uniform_no_textpred.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(audit), f, indent=2, ensure_ascii=False)

    df[[cfg.id_col, "split", "row_npy"]].to_csv(paths["reports_dir"] / "00_ids_alignement_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id_clean": final_removed_ids}).to_csv(paths["reports_dir"] / "00_annonces_retirees_alignement_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id_clean": dropped_by_price_ids}).to_csv(paths["reports_dir"] / "00_annonces_supprimees_par_seuil_prix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"listing_id_clean": text_embeddings_ignored}).to_csv(paths["reports_dir"] / "00_text_embeddings_ignores_apres_alignement.csv", index=False, encoding="utf-8-sig")

    return {"df": df, "X_img": X_img, "X_text": X_text, "text_feature_cols": text_feature_cols}

# 5. AUDIT ET PREPARATION TABULAIRE

def audit_dataset(df: pd.DataFrame, cfg: Config, paths: Dict[str, Path]) -> None:
    missing = pd.DataFrame({
        "colonne": df.columns,
        "type": df.dtypes.astype(str).values,
        "valeurs_manquantes": df.isna().sum().values,
        "pourcentage_manquant": (df.isna().sum().values / len(df) * 100).round(2),
        "nb_valeurs_uniques": df.nunique(dropna=False).values,
    }).sort_values("pourcentage_manquant", ascending=False)
    missing.to_csv(paths["reports_dir"] / "01_audit_valeurs_manquantes.csv", index=False, encoding="utf-8-sig")

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    pd.DataFrame({"colonne": constant_cols}).to_csv(paths["reports_dir"] / "02_colonnes_constantes_dataset.csv", index=False, encoding="utf-8-sig")

    for col, title, file in [(cfg.target_real, "Distribution de price", "01_distribution_price.png"), (cfg.target_log, "Distribution de log_price", "02_distribution_log_price.png")]:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=80)
            plt.title(title)
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(paths["plots_dir"] / file, dpi=150)
            plt.close()


def should_exclude_column(col: str, cfg: Config, text_feature_cols: List[str]) -> bool:
    if col in text_feature_cols:
        return True
    if col in cfg.cols_to_exclude:
        return True
    if col in cfg.raw_text_cols_to_exclude:
        return True
    for prefix in cfg.forbidden_prefixes_tabular:
        if col.startswith(prefix):
            return True
    return False


def prepare_tabular_features(df: pd.DataFrame, X_text: pd.DataFrame, cfg: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    print("\n================ PREPARATION FEATURES TABULAIRES ================")
    if cfg.target_log not in df.columns or cfg.target_real not in df.columns:
        raise ValueError("price/log_price absents du fichier tabulaire aligné.")

    text_feature_cols = list(X_text.columns)
    ids = df[cfg.id_col].astype(str).reset_index(drop=True)
    split = df["split"].astype(str).reset_index(drop=True)
    y_log = pd.to_numeric(df[cfg.target_log], errors="coerce")
    y_real = pd.to_numeric(df[cfg.target_real], errors="coerce")

    valid_mask = y_log.notna() & y_real.notna()
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    X_text_valid = X_text.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)
    split = split.loc[valid_mask].reset_index(drop=True)
    y_log = y_log.loc[valid_mask].reset_index(drop=True)
    y_real = y_real.loc[valid_mask].reset_index(drop=True)

    excluded_by_rule = [c for c in df_valid.columns if should_exclude_column(c, cfg, text_feature_cols)]
    feature_cols = [c for c in df_valid.columns if c not in excluded_by_rule]
    X_tab = df_valid[feature_cols].copy()

    constant_cols = [c for c in X_tab.columns if X_tab[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X_tab = X_tab.drop(columns=constant_cols)

    cat_features = [c for c in cfg.base_cat_features if c in X_tab.columns]
    object_cols = X_tab.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in object_cols:
        if c not in cat_features:
            cat_features.append(c)

    for c in cat_features:
        X_tab[c] = X_tab[c].fillna("missing").astype(str)
    for c in X_tab.columns:
        if c not in cat_features:
            X_tab[c] = pd.to_numeric(X_tab[c], errors="coerce")

    forbidden_in_tab = [c for c in X_tab.columns if should_exclude_column(c, cfg, text_feature_cols)]
    if forbidden_in_tab:
        raise ValueError(f"Colonnes interdites encore présentes dans X_tab : {forbidden_in_tab}")

    forbidden_text = [c for c in X_text_valid.columns if c in {"text_pred_final", "text_pred_oof", "text_pred_test"} or c.startswith("text_pred")]
    if forbidden_text:
        raise ValueError(f"Colonnes text_pred encore présentes dans X_text : {forbidden_text}")

    print("Lignes utilisées :", len(X_tab))
    print("Variables tabulaires utilisées :", X_tab.shape[1])
    print("Variables textuelles utilisées :", X_text_valid.shape[1])
    print("Catégorielles CatBoost :", cat_features)
    print("Colonnes constantes supprimées :", constant_cols)
    print("text_pred dans X_text :", forbidden_text)

    pd.DataFrame({"feature": X_tab.columns}).to_csv(paths["reports_dir"] / "03_features_tabulaires_utilisees.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"cat_feature": cat_features}).to_csv(paths["reports_dir"] / "04_cat_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"excluded_column": sorted(excluded_by_rule)}).to_csv(paths["reports_dir"] / "05_colonnes_exclues.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"constant_column_removed": constant_cols}).to_csv(paths["reports_dir"] / "06_colonnes_constantes_supprimees.csv", index=False, encoding="utf-8-sig")

    return {
        "X_tab": X_tab.reset_index(drop=True),
        "X_text": X_text_valid.reset_index(drop=True),
        "y_log": y_log.reset_index(drop=True),
        "y_real": y_real.reset_index(drop=True),
        "ids": ids.reset_index(drop=True),
        "split": split.reset_index(drop=True),
        "cat_features": cat_features,
        "feature_names_tabular": list(X_tab.columns),
        "feature_names_text": list(X_text_valid.columns),
        "excluded_by_rule": excluded_by_rule,
        "constant_cols": constant_cols,
        "valid_mask": valid_mask.reset_index(drop=True),
    }

# 6. SPLIT, PCA, FUSION, VERIFICATIONS

def make_split_from_existing_file(data: Dict[str, Any], X_img: np.ndarray, cfg: Config, paths: Dict[str, Path]) -> Dict[str, Any]:
    X_tab, X_text = data["X_tab"], data["X_text"]
    y_log, y_real = data["y_log"], data["y_real"]
    ids, split = data["ids"], data["split"]

    if len(X_tab) != len(X_text) or len(X_tab) != len(X_img):
        raise ValueError("X_tab, X_text et X_img n'ont pas le même nombre de lignes.")

    idx_train = np.where(split.values == "train_dev")[0]
    idx_test = np.where(split.values == "test_final")[0]
    if len(idx_train) == 0 or len(idx_test) == 0:
        raise ValueError("Split invalide : train_dev ou test_final vide.")

    report = pd.DataFrame({
        "split": ["train_dev", "test_final"],
        "n": [len(idx_train), len(idx_test)],
        "price_mean": [y_real.iloc[idx_train].mean(), y_real.iloc[idx_test].mean()],
        "price_median": [y_real.iloc[idx_train].median(), y_real.iloc[idx_test].median()],
        "price_min": [y_real.iloc[idx_train].min(), y_real.iloc[idx_test].min()],
        "price_max": [y_real.iloc[idx_train].max(), y_real.iloc[idx_test].max()],
        "log_price_mean": [y_log.iloc[idx_train].mean(), y_log.iloc[idx_test].mean()],
    })
    report.to_csv(paths["reports_dir"] / "07_split_final_train_test_report.csv", index=False, encoding="utf-8-sig")

    print("\n================ SPLIT FINAL UTILISE ================")
    print(report)

    return {
        "X_tab_train_dev": X_tab.iloc[idx_train].reset_index(drop=True),
        "X_tab_test_final": X_tab.iloc[idx_test].reset_index(drop=True),
        "X_text_train_dev": X_text.iloc[idx_train].reset_index(drop=True),
        "X_text_test_final": X_text.iloc[idx_test].reset_index(drop=True),
        "X_img_train_dev": X_img[idx_train],
        "X_img_test_final": X_img[idx_test],
        "y_train_dev_log": y_log.iloc[idx_train].reset_index(drop=True),
        "y_test_final_log": y_log.iloc[idx_test].reset_index(drop=True),
        "y_train_dev_real": y_real.iloc[idx_train].reset_index(drop=True),
        "y_test_final_real": y_real.iloc[idx_test].reset_index(drop=True),
        "ids_train_dev": ids.iloc[idx_train].reset_index(drop=True),
        "ids_test_final": ids.iloc[idx_test].reset_index(drop=True),
        "cat_features": data["cat_features"],
        "feature_names_tabular": data["feature_names_tabular"],
        "feature_names_text": data["feature_names_text"],
        "excluded_by_rule": data["excluded_by_rule"],
        "constant_cols": data["constant_cols"],
    }


def make_visual_pca_columns(n_components: int) -> list:
    return [f"img_pca_{i}" for i in range(n_components)]


def fit_transform_visual_pca(X_train_img, X_other_img, cfg: Config, seed: int):
    n_components = min(cfg.visual_pca_components, X_train_img.shape[1], X_train_img.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver=cfg.pca_svd_solver, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_img).astype(np.float32)
    X_other_pca = pca.transform(X_other_img).astype(np.float32)
    return pca, X_train_pca, X_other_pca


def build_fusion_dataframe(X_tab: pd.DataFrame, X_text: pd.DataFrame, X_img_pca: np.ndarray, img_pca_columns: list) -> pd.DataFrame:
    X_img_df = pd.DataFrame(X_img_pca, columns=img_pca_columns)
    return pd.concat([X_tab.reset_index(drop=True), X_text.reset_index(drop=True), X_img_df.reset_index(drop=True)], axis=1)


def assert_final_features_clean(X: pd.DataFrame, cfg: Config, context: str) -> None:
    forbidden_exact = set(cfg.cols_to_exclude) | set(cfg.raw_text_cols_to_exclude) | {"text_pred_final", "text_pred_oof", "text_pred_test"}
    bad_exact = [c for c in X.columns if c in forbidden_exact]
    bad_prefix = [c for c in X.columns if c.startswith("text_pred") or c.startswith("img_emb_") or c.startswith("efficientnet_")]
    if bad_exact or bad_prefix:
        raise ValueError(f"Features interdites dans {context}. exact={bad_exact}, prefix={bad_prefix}")

# 7. PONDERATION UNIFORME ET METRIQUES

def build_price_segments(y_price, cfg: Config):
    return pd.cut(y_price, bins=list(cfg.price_segment_bins), labels=list(cfg.price_segment_labels), include_lowest=True, right=False)


def make_uniform_weights(y_real, cfg: Config):
    return pd.Series(np.full(len(y_real), cfg.uniform_weight_value, dtype=float))


def compute_weight_diagnostics(y_real, weights, cfg: Config, split_name: str, fold_name: str) -> pd.DataFrame:
    df_w = pd.DataFrame({"y_real": pd.Series(y_real).reset_index(drop=True), "weight": pd.Series(weights).reset_index(drop=True)})
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


def safe_mape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def smape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MedAE": np.nan, "R2": np.nan, "MAPE_pct": np.nan, "SMAPE_pct": np.nan, "Mean_Error": np.nan, "Underestimation_Rate_pct": np.nan, "Overestimation_Rate_pct": np.nan, "Abs_Error_P75": np.nan, "Abs_Error_P90": np.nan, "Abs_Error_P95": np.nan}
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and len(np.unique(y_true)) > 1 else np.nan
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
    df = pd.DataFrame({
        "id_clean": ids.values,
        "split": split_name,
        "fold": fold_name,
        "y_true_log": y_true_log.values,
        "y_pred_log": pred_log,
        "y_true_price": y_true_price.values,
        "y_pred_price": pred_price,
    })
    df["residual_log"] = df["y_pred_log"] - df["y_true_log"]
    df["residual_price"] = df["y_pred_price"] - df["y_true_price"]
    df["abs_error_log"] = np.abs(df["residual_log"])
    df["abs_error_price"] = np.abs(df["residual_price"])
    return df


def compute_segment_metrics(pred_df, cfg: Config, split_name: str) -> pd.DataFrame:
    tmp = pred_df.copy()
    tmp["price_segment"] = build_price_segments(tmp["y_true_price"], cfg).astype(str)
    rows = []
    for seg in cfg.price_segment_labels:
        g = tmp[tmp["price_segment"] == seg]
        metrics = compute_metrics(g["y_true_price"].values, g["y_pred_price"].values)
        rows.append({"split": split_name, "segment": seg, "price_mean": float(g["y_true_price"].mean()) if len(g) else np.nan, "price_median": float(g["y_true_price"].median()) if len(g) else np.nan, **metrics})
    return pd.DataFrame(rows)


# ============================================================
# 8. CATBOOST
# ============================================================

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


def feature_group_name(feature: str) -> str:
    if feature.startswith("img_pca_"):
        return "visuel_pca"
    if feature.startswith("txt_e5_") or feature in {"name_n_words", "description_n_words", "description_is_missing"}:
        return "textuel"
    return "tabulaire"


# ============================================================
# 9. CV
# ============================================================

def run_kfold_cv(split_data: Dict[str, Any], cfg: Config, paths: Dict[str, Path]):
    X_tab = split_data["X_tab_train_dev"]
    X_text = split_data["X_text_train_dev"]
    X_img = split_data["X_img_train_dev"]
    y_log = split_data["y_train_dev_log"]
    y_real = split_data["y_train_dev_real"]
    ids = split_data["ids_train_dev"]
    cat_features = split_data["cat_features"]

    print("\n" + "=" * 100)
    print("K-FOLD CV - CATBOOST MULTIMODAL - UNIFORM WEIGHTS - SANS TEXT PRED")
    print("=" * 100)
    print("Train_dev tabulaire :", X_tab.shape)
    print("Train_dev textuel :", X_text.shape)
    print("Train_dev visuel :", X_img.shape)
    print("Pondération : uniforme, pas de surpondération des logements chers.")

    strat_bins = make_regression_strat_bins(y_log, cfg.n_splits, cfg.n_strat_bins_cv)
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    oof_predictions = []
    fold_metrics_log_rows = []
    fold_metrics_price_rows = []
    fold_segment_rows = []
    fold_weight_rows = []
    fold_iteration_rows = []
    fold_pca_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_tab, strat_bins), start=1):
        fold_name = f"fold_{fold_id}"
        fold_seed = cfg.random_state + fold_id
        print("\n" + "-" * 100)
        print(fold_name)
        print("-" * 100)

        X_tab_train = X_tab.iloc[train_idx].reset_index(drop=True)
        X_tab_val = X_tab.iloc[val_idx].reset_index(drop=True)
        X_text_train = X_text.iloc[train_idx].reset_index(drop=True)
        X_text_val = X_text.iloc[val_idx].reset_index(drop=True)
        X_img_train = X_img[train_idx]
        X_img_val = X_img[val_idx]
        y_train_log = y_log.iloc[train_idx].reset_index(drop=True)
        y_val_log = y_log.iloc[val_idx].reset_index(drop=True)
        y_train_real = y_real.iloc[train_idx].reset_index(drop=True)
        y_val_real = y_real.iloc[val_idx].reset_index(drop=True)
        ids_val = ids.iloc[val_idx].reset_index(drop=True)

        pca, X_img_train_pca, X_img_val_pca = fit_transform_visual_pca(X_img_train, X_img_val, cfg, seed=fold_seed)
        img_pca_columns = make_visual_pca_columns(X_img_train_pca.shape[1])
        X_train = build_fusion_dataframe(X_tab_train, X_text_train, X_img_train_pca, img_pca_columns)
        X_val = build_fusion_dataframe(X_tab_val, X_text_val, X_img_val_pca, img_pca_columns)

        assert_final_features_clean(X_train, cfg, context=f"X_train {fold_name}")
        assert_final_features_clean(X_val, cfg, context=f"X_val {fold_name}")

        fold_pca_rows.append({"fold": fold_name, "n_components": int(X_img_train_pca.shape[1]), "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))})

        w_train = make_uniform_weights(y_train_real, cfg)
        w_val = make_uniform_weights(y_val_real, cfg)
        fold_weight_rows.append(compute_weight_diagnostics(y_train_real, w_train, cfg, split_name="fold_train", fold_name=fold_name))

        train_pool = Pool(data=X_train, label=y_train_log, cat_features=cat_features, weight=w_train.values)
        if cfg.weight_validation_pool:
            val_pool = Pool(data=X_val, label=y_val_log, cat_features=cat_features, weight=w_val.values)
        else:
            val_pool = Pool(data=X_val, label=y_val_log, cat_features=cat_features)

        model = CatBoostRegressor(**make_catboost_params(cfg=cfg, seed=fold_seed, with_early_stopping=True))
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=cfg.verbose_eval)

        model_path = paths["models_dir"] / f"catboost_ttv_uniform_no_textpred_cv_{fold_name}.cbm"
        model.save_model(str(model_path))
        best_iteration = model.get_best_iteration()
        tree_count = int(model.tree_count_)
        fold_iteration_rows.append({"fold": fold_name, "best_iteration": None if best_iteration is None else int(best_iteration), "tree_count": tree_count, "model_path": str(model_path)})

        pred_val_log = model.predict(val_pool)
        pred_df = make_predictions_df(ids_val, "oof", y_val_log, pred_val_log, y_val_real, fold_name=fold_name)
        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)
        pred_df.to_csv(paths["predictions_dir"] / f"predictions_oof_{fold_name}.csv", index=False, encoding="utf-8-sig")
        oof_predictions.append(pred_df)

        metrics_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        metrics_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])
        fold_metrics_log_rows.append({"fold": fold_name, "best_iteration": best_iteration, "tree_count": tree_count, **metrics_log})
        fold_metrics_price_rows.append({"fold": fold_name, "best_iteration": best_iteration, "tree_count": tree_count, **metrics_price})
        fold_segment_rows.append(compute_segment_metrics(pred_df, cfg, split_name=fold_name))

        print(f"{fold_name} terminé | MAE € = {metrics_price['MAE']:.2f} | RMSE € = {metrics_price['RMSE']:.2f} | Biais € = {metrics_price['Mean_Error']:.2f} | tree_count = {tree_count}")

    oof_df = pd.concat(oof_predictions, ignore_index=True)
    fold_metrics_log_df = pd.DataFrame(fold_metrics_log_rows)
    fold_metrics_price_df = pd.DataFrame(fold_metrics_price_rows)
    fold_segments_df = pd.concat(fold_segment_rows, ignore_index=True)
    fold_weights_df = pd.concat(fold_weight_rows, ignore_index=True)
    fold_iterations_df = pd.DataFrame(fold_iteration_rows)
    fold_pca_df = pd.DataFrame(fold_pca_rows)

    oof_df.to_csv(paths["predictions_dir"] / "10_oof_predictions_all_folds.csv", index=False, encoding="utf-8-sig")
    fold_metrics_log_df.to_csv(paths["reports_dir"] / "10_cv_fold_metrics_log_price.csv", index=False, encoding="utf-8-sig")
    fold_metrics_price_df.to_csv(paths["reports_dir"] / "11_cv_fold_metrics_price_euros.csv", index=False, encoding="utf-8-sig")
    fold_segments_df.to_csv(paths["reports_dir"] / "12_cv_segments_by_fold.csv", index=False, encoding="utf-8-sig")
    fold_weights_df.to_csv(paths["reports_dir"] / "13_cv_weight_diagnostics_uniform_by_fold.csv", index=False, encoding="utf-8-sig")
    fold_iterations_df.to_csv(paths["reports_dir"] / "14_cv_best_iterations.csv", index=False, encoding="utf-8-sig")
    fold_pca_df.to_csv(paths["reports_dir"] / "14b_cv_visual_pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    oof_metrics_log = compute_metrics(oof_df["y_true_log"], oof_df["y_pred_log"])
    oof_metrics_price = compute_metrics(oof_df["y_true_price"], oof_df["y_pred_price"])
    oof_segments = compute_segment_metrics(oof_df, cfg, split_name="oof_global")

    rows = []
    for metric, value in oof_metrics_log.items():
        rows.append({"split": "oof_global", "scale": "log_price", "metric": metric, "value": value})
    for metric, value in oof_metrics_price.items():
        rows.append({"split": "oof_global", "scale": "price_euros", "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(paths["reports_dir"] / "15_cv_oof_metrics_global.csv", index=False, encoding="utf-8-sig")
    oof_segments.to_csv(paths["reports_dir"] / "16_cv_oof_segments_global.csv", index=False, encoding="utf-8-sig")

    tree_counts = fold_iterations_df["tree_count"].dropna().astype(int).values
    final_iterations = cfg.iterations if len(tree_counts) == 0 else max(1, int(np.median(tree_counts)))

    cv_summary = {
        "experiment": "catboost_ttv_pca256_uniform_no_textpred",
        "weight_strategy": "uniform_no_premium_weighting",
        "text_pred_final_used": False,
        "drop_price_above": cfg.drop_price_above,
        "n_splits": cfg.n_splits,
        "visual_pca_components": cfg.visual_pca_components,
        "oof_metrics_log": oof_metrics_log,
        "oof_metrics_price": oof_metrics_price,
        "tree_counts": tree_counts.tolist(),
        "final_iterations_selected_by_median_tree_count": final_iterations,
        "test_final_used_in_cv": False,
        "visual_pca_fitted_inside_each_fold": True,
    }
    with open(paths["reports_dir"] / "17_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(cv_summary), f, indent=2, ensure_ascii=False)

    plot_cv_outputs(oof_df, oof_segments, paths, cfg)

    return {"oof_predictions": oof_df, "fold_metrics_log": fold_metrics_log_df, "fold_metrics_price": fold_metrics_price_df, "oof_metrics_log": oof_metrics_log, "oof_metrics_price": oof_metrics_price, "oof_segments": oof_segments, "fold_iterations": fold_iterations_df, "fold_pca": fold_pca_df, "final_iterations": final_iterations}


# ============================================================
# 10. FINAL + IMPORTANCE
# ============================================================

def train_final_model_and_evaluate(split_data: Dict[str, Any], cfg: Config, paths: Dict[str, Path], final_iterations: int, cv_results: Dict[str, Any]):
    X_tab_train, X_tab_test = split_data["X_tab_train_dev"], split_data["X_tab_test_final"]
    X_text_train, X_text_test = split_data["X_text_train_dev"], split_data["X_text_test_final"]
    X_img_train, X_img_test = split_data["X_img_train_dev"], split_data["X_img_test_final"]
    y_train_log, y_test_log = split_data["y_train_dev_log"], split_data["y_test_final_log"]
    y_train_real, y_test_real = split_data["y_train_dev_real"], split_data["y_test_final_real"]
    ids_train, ids_test = split_data["ids_train_dev"], split_data["ids_test_final"]
    cat_features = split_data["cat_features"]

    print("\n" + "=" * 100)
    print("ENTRAINEMENT FINAL - UNIFORM WEIGHTS - SANS TEXT PRED")
    print("=" * 100)
    print("Itérations finales :", final_iterations)

    final_pca, X_img_train_pca, X_img_test_pca = fit_transform_visual_pca(X_img_train, X_img_test, cfg, seed=cfg.random_state + 999)
    img_pca_columns = make_visual_pca_columns(X_img_train_pca.shape[1])
    X_train = build_fusion_dataframe(X_tab_train, X_text_train, X_img_train_pca, img_pca_columns)
    X_test = build_fusion_dataframe(X_tab_test, X_text_test, X_img_test_pca, img_pca_columns)
    feature_names = list(X_train.columns)

    assert_final_features_clean(X_train, cfg, context="X_train final")
    assert_final_features_clean(X_test, cfg, context="X_test final")

    pca_path = paths["pca_dir"] / "pca_efficientnet_b0_256_train_dev.joblib"
    dump(final_pca, pca_path)
    pd.DataFrame({"component": img_pca_columns, "explained_variance_ratio": final_pca.explained_variance_ratio_, "explained_variance_ratio_cumsum": np.cumsum(final_pca.explained_variance_ratio_)}).to_csv(paths["pca_dir"] / "final_visual_pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_names, "group": [feature_group_name(f) for f in feature_names]}).to_csv(paths["reports_dir"] / "19_features_finales_tabulaire_textuel_visuel.csv", index=False, encoding="utf-8-sig")

    w_train = make_uniform_weights(y_train_real, cfg)
    weight_diag = compute_weight_diagnostics(y_train_real, w_train, cfg, split_name="train_dev", fold_name="final_model")
    weight_diag.to_csv(paths["reports_dir"] / "20_final_weight_diagnostics_uniform.csv", index=False, encoding="utf-8-sig")

    train_pool = Pool(data=X_train, label=y_train_log, cat_features=cat_features, weight=w_train.values)
    final_model = CatBoostRegressor(**make_catboost_params(cfg=cfg, seed=cfg.random_state + 3030, iterations_override=final_iterations, with_early_stopping=False))
    final_model.fit(train_pool, verbose=cfg.verbose_eval)

    final_model_path = paths["models_dir"] / "catboost_final_ttv_pca256_uniform_no_textpred.cbm"
    final_model.save_model(str(final_model_path))

    preds = {}
    metrics_rows = []
    segment_rows = []
    for split_name, X_part, y_log_part, y_real_part, ids_part in [
        ("train_dev", X_train, y_train_log, y_train_real, ids_train),
        ("test_final", X_test, y_test_log, y_test_real, ids_test),
    ]:
        pool = Pool(data=X_part, label=y_log_part, cat_features=cat_features)
        pred_log = final_model.predict(pool)
        pred_df = make_predictions_df(ids_part, split_name, y_log_part, pred_log, y_real_part, fold_name="final_model")
        pred_df["price_segment"] = build_price_segments(pred_df["y_true_price"], cfg).astype(str)
        pred_df.to_csv(paths["predictions_dir"] / f"20_predictions_{split_name}.csv", index=False, encoding="utf-8-sig")
        preds[split_name] = pred_df

        m_log = compute_metrics(pred_df["y_true_log"], pred_df["y_pred_log"])
        m_price = compute_metrics(pred_df["y_true_price"], pred_df["y_pred_price"])
        for metric, value in m_log.items():
            metrics_rows.append({"model": "final_model_train_dev", "candidate": "uniform_no_premium_no_textpred", "experiment": "catboost_ttv_pca256_uniform_no_textpred", "split": split_name, "scale": "log_price", "metric": metric, "value": value})
        for metric, value in m_price.items():
            metrics_rows.append({"model": "final_model_train_dev", "candidate": "uniform_no_premium_no_textpred", "experiment": "catboost_ttv_pca256_uniform_no_textpred", "split": split_name, "scale": "price_euros", "metric": metric, "value": value})
        segment_rows.append(compute_segment_metrics(pred_df, cfg, split_name=split_name))

    metrics_df = pd.DataFrame(metrics_rows)
    segments_df = pd.concat(segment_rows, ignore_index=True)
    metrics_df.to_csv(paths["reports_dir"] / "21_final_metrics_train_test.csv", index=False, encoding="utf-8-sig")
    segments_df.to_csv(paths["reports_dir"] / "22_final_segments_train_test.csv", index=False, encoding="utf-8-sig")
    metrics_df.pivot_table(index=["scale", "metric"], columns="split", values="value").reset_index().to_csv(paths["reports_dir"] / "23_final_metrics_pivot.csv", index=False, encoding="utf-8-sig")

    save_feature_importance(final_model, feature_names, paths)
    save_feature_group_importance(final_model, feature_names, paths)
    plot_final_outputs(preds, segments_df, paths, cfg)

    if cfg.compute_shap_final:
        save_shap_final(final_model, X_train, y_train_log, cat_features, cfg, paths)

    write_final_report(metrics_df, segments_df, cv_results, split_data, cfg, paths, final_model_path, pca_path, final_pca)

    model_config = {
        "candidate": "uniform_no_premium_no_textpred",
        "experiment": "catboost_ttv_pca256_uniform_no_textpred",
        "config": asdict(cfg),
        "feature_names": feature_names,
        "cat_features": cat_features,
        "excluded_by_rule": split_data["excluded_by_rule"],
        "constant_cols": split_data["constant_cols"],
        "final_model": {"model_path": str(final_model_path), "visual_pca_path": str(pca_path), "iterations": int(final_iterations)},
        "important": "Modèle CatBoost multimodal sans text_pred_final et sans pondération premium des logements chers.",
    }
    with open(paths["models_dir"] / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(model_config), f, indent=2, ensure_ascii=False)


# ============================================================
# 11. IMPORTANCE / SHAP / GRAPHES / RAPPORT
# ============================================================

def save_feature_importance(model, feature_names, paths: Dict[str, Path]):
    imp = model.get_feature_importance(type="FeatureImportance")
    fi = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    fi.to_csv(paths["reports_dir"] / "24_final_feature_importance_catboost.csv", index=False, encoding="utf-8-sig")
    top = fi.head(30).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance CatBoost")
    plt.title("CatBoost TTV uniforme sans text_pred - Top 30")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "24_final_feature_importance_top30.png", dpi=160)
    plt.close()


def save_feature_group_importance(model, feature_names, paths: Dict[str, Path]):
    imp = model.get_feature_importance(type="FeatureImportance")
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
    df_imp["group"] = df_imp["feature"].apply(feature_group_name)
    group = df_imp.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    total = group["importance"].sum()
    group["importance_pct"] = group["importance"] / total * 100 if total > 0 else np.nan
    group.to_csv(paths["reports_dir"] / "24b_importance_groupes_tabulaire_textuel_visuel.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 5))
    plt.bar(group["group"], group["importance_pct"])
    plt.ylabel("Importance (%)")
    plt.title("Importance : tabulaire vs textuel vs visuel")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "24b_importance_groupes_tabulaire_textuel_visuel.png", dpi=160)
    plt.close()


def save_shap_final(model, X_train, y_train_log, cat_features, cfg: Config, paths: Dict[str, Path]):
    sample_size = min(cfg.shap_max_rows_final, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=cfg.random_state)
    y_sample = y_train_log.loc[X_sample.index]
    shap_pool = Pool(data=X_sample, label=y_sample, cat_features=cat_features)
    shap_values = model.get_feature_importance(data=shap_pool, type="ShapValues")
    mean_abs = np.abs(shap_values[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(paths["shap_dir"] / "final_mean_abs_shap.csv", index=False, encoding="utf-8-sig")
    shap_df["group"] = shap_df["feature"].apply(feature_group_name)
    shap_group = shap_df.groupby("group", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)
    total = shap_group["mean_abs_shap"].sum()
    shap_group["mean_abs_shap_pct"] = shap_group["mean_abs_shap"] / total * 100 if total > 0 else np.nan
    shap_group.to_csv(paths["shap_dir"] / "final_shap_groupes_tabulaire_textuel_visuel.csv", index=False, encoding="utf-8-sig")
    top = shap_df.head(cfg.shap_top_n).sort_values("mean_abs_shap")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP|")
    plt.title("SHAP - CatBoost TTV uniforme sans text_pred")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "25_final_shap_top30.png", dpi=160)
    plt.close()


def plot_cv_outputs(oof_df, oof_segments, paths: Dict[str, Path], cfg: Config):
    max_value = np.percentile(np.concatenate([oof_df["y_true_price"], oof_df["y_pred_price"]]), 99)
    plt.figure(figsize=(7, 7))
    plt.scatter(oof_df["y_true_price"], oof_df["y_pred_price"], alpha=0.20, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("OOF CV - CatBoost TTV uniforme sans text_pred")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "10_cv_oof_pred_vs_true.png", dpi=160)
    plt.close()

    seg = oof_segments.copy()
    seg["segment"] = pd.Categorical(seg["segment"], categories=list(cfg.price_segment_labels), ordered=True)
    seg = seg.sort_values("segment")
    plt.figure(figsize=(8, 5))
    plt.bar(seg["segment"].astype(str), seg["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("OOF CV - MAE par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "13_cv_oof_mae_by_segment.png", dpi=160)
    plt.close()


def plot_final_outputs(preds, segments_df, paths: Dict[str, Path], cfg: Config):
    test = preds["test_final"]
    max_value = np.percentile(np.concatenate([test["y_true_price"], test["y_pred_price"]]), 99)
    plt.figure(figsize=(7, 7))
    plt.scatter(test["y_true_price"], test["y_pred_price"], alpha=0.25, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("Test final - CatBoost TTV uniforme sans text_pred")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "20_final_test_pred_vs_true.png", dpi=160)
    plt.close()

    residuals = test["residual_price"]
    residuals_clip = np.clip(residuals, np.percentile(residuals, 1), np.percentile(residuals, 99))
    plt.figure(figsize=(9, 5))
    plt.hist(residuals_clip, bins=70)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Résidu en euros")
    plt.ylabel("Fréquence")
    plt.title("Test final - Résidus")
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "21_final_test_residuals.png", dpi=160)
    plt.close()

    seg = segments_df[segments_df["split"] == "test_final"].copy()
    seg["segment"] = pd.Categorical(seg["segment"], categories=list(cfg.price_segment_labels), ordered=True)
    seg = seg.sort_values("segment")
    plt.figure(figsize=(8, 5))
    plt.bar(seg["segment"].astype(str), seg["MAE"])
    plt.xlabel("Segment de prix réel")
    plt.ylabel("MAE (€)")
    plt.title("Test final - MAE par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "22_final_test_mae_by_segment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(seg["segment"].astype(str), seg["Mean_Error"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Segment de prix réel")
    plt.ylabel("Biais moyen (€)")
    plt.title("Test final - Biais moyen par segment")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(paths["plots_dir"] / "23_final_test_bias_by_segment.png", dpi=160)
    plt.close()


def get_metric(metrics_df, split, scale, metric):
    row = metrics_df[(metrics_df["split"] == split) & (metrics_df["scale"] == scale) & (metrics_df["metric"] == metric)]
    if len(row) == 0:
        return np.nan
    return float(row["value"].iloc[0])


def write_final_report(metrics_df, segments_df, cv_results, split_data, cfg: Config, paths: Dict[str, Path], final_model_path: Path, pca_path: Path, final_pca):
    report_path = paths["reports_dir"] / "30_resume_final_catboost_ttv_uniform_no_textpred.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Résumé final - CatBoost tabulaire + textuel + visuel PCA 256\n")
        f.write("Sans pondération premium et sans text_pred_final\n")
        f.write("=" * 100 + "\n\n")
        f.write("Protocole\n")
        f.write("-" * 100 + "\n")
        f.write("Le split train_dev/test_final vient de split_listing_ids.csv.\n")
        f.write("La CV 5-fold est faite uniquement sur train_dev.\n")
        f.write("Le test_final est utilisé une seule fois à la fin.\n")
        f.write("Les embeddings visuels EfficientNet-B0 sont réduits à 256 composantes par PCA.\n")
        f.write("La PCA est apprise uniquement sur le train de chaque fold, puis sur train_dev pour le modèle final.\n")
        f.write("Les features textuelles E5-LoRA sont utilisées sans text_pred_final.\n")
        f.write("La pondération est uniforme : les logements chers ne sont pas surpondérés.\n")
        f.write(f"drop_price_above : {cfg.drop_price_above}\n\n")

        f.write("Données utilisées\n")
        f.write("-" * 100 + "\n")
        f.write(f"Train_dev : {len(split_data['ids_train_dev'])}\n")
        f.write(f"Test_final : {len(split_data['ids_test_final'])}\n")
        f.write(f"Variables tabulaires : {len(split_data['feature_names_tabular'])}\n")
        f.write(f"Variables textuelles : {len(split_data['feature_names_text'])}\n")
        f.write(f"Composantes visuelles PCA : {cfg.visual_pca_components}\n")
        f.write(f"Variance expliquée PCA visuelle finale : {np.sum(final_pca.explained_variance_ratio_):.6f}\n\n")

        f.write("Métriques CV OOF - price_euros\n")
        f.write("-" * 100 + "\n")
        for k, v in cv_results["oof_metrics_price"].items():
            f.write(f"{k:30s}: {v:.6f}\n")
        f.write("\n")

        f.write("Métriques test final - log_price\n")
        f.write("-" * 100 + "\n")
        f.write(f"MAE log : {get_metric(metrics_df, 'test_final', 'log_price', 'MAE'):.6f}\n")
        f.write(f"RMSE log : {get_metric(metrics_df, 'test_final', 'log_price', 'RMSE'):.6f}\n")
        f.write(f"R2 log : {get_metric(metrics_df, 'test_final', 'log_price', 'R2'):.6f}\n\n")

        f.write("Métriques test final - price_euros\n")
        f.write("-" * 100 + "\n")
        f.write(f"MAE : {get_metric(metrics_df, 'test_final', 'price_euros', 'MAE'):.6f} €\n")
        f.write(f"RMSE : {get_metric(metrics_df, 'test_final', 'price_euros', 'RMSE'):.6f} €\n")
        f.write(f"R2 : {get_metric(metrics_df, 'test_final', 'price_euros', 'R2'):.6f}\n")
        f.write(f"Biais moyen : {get_metric(metrics_df, 'test_final', 'price_euros', 'Mean_Error'):.6f} €\n\n")

        f.write("Segments test final\n")
        f.write("-" * 100 + "\n")
        seg_test = segments_df[segments_df["split"] == "test_final"].copy()
        cols = ["segment", "n", "MAE", "RMSE", "MedAE", "Mean_Error", "Underestimation_Rate_pct", "Abs_Error_P90", "Abs_Error_P95"]
        f.write(seg_test[[c for c in cols if c in seg_test.columns]].to_string(index=False))
        f.write("\n")



def main():
    paths = get_paths(CFG)

    print("\n" + "=" * 100)
    print("CATBOOST TTV PCA256 - UNIFORM WEIGHTS - SANS TEXT_PRED")
    print("=" * 100)
    print("Dossier script :", paths["script_dir"])
    print("Dossier sorties :", paths["output_dir"])

    aligned = load_and_align_data(CFG, paths)
    df, X_img, X_text = aligned["df"], aligned["X_img"], aligned["X_text"]

    audit_dataset(df, CFG, paths)
    prepared = prepare_tabular_features(df, X_text, CFG, paths)
    split_data = make_split_from_existing_file(prepared, X_img, CFG, paths)

    with open(paths["reports_dir"] / "00_run_config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"candidate": "uniform_no_premium_no_textpred", "experiment": "catboost_ttv_pca256_uniform_no_textpred", "config": asdict(CFG)}), f, indent=2, ensure_ascii=False)

    cv_results = run_kfold_cv(split_data, CFG, paths)
    train_final_model_and_evaluate(split_data, CFG, paths, cv_results["final_iterations"], cv_results)

    print("\n" + "=" * 100)
    print("FIN - CATBOOST TTV PCA256 - UNIFORM WEIGHTS - SANS TEXT_PRED")
    print("=" * 100)
    print("Résultats sauvegardés dans :")
    print(paths["output_dir"])
    print("\nFichiers importants à m'envoyer pour analyse :")
    print(paths["reports_dir"] / "00_audit_alignement_uniform_no_textpred.json")
    print(paths["reports_dir"] / "07_split_final_train_test_report.csv")
    print(paths["reports_dir"] / "13_cv_weight_diagnostics_uniform_by_fold.csv")
    print(paths["reports_dir"] / "15_cv_oof_metrics_global.csv")
    print(paths["reports_dir"] / "16_cv_oof_segments_global.csv")
    print(paths["reports_dir"] / "21_final_metrics_train_test.csv")
    print(paths["reports_dir"] / "22_final_segments_train_test.csv")
    print(paths["reports_dir"] / "23_final_metrics_pivot.csv")
    print(paths["reports_dir"] / "24b_importance_groupes_tabulaire_textuel_visuel.csv")
    print(paths["reports_dir"] / "30_resume_final_catboost_ttv_uniform_no_textpred.txt")
    print(paths["shap_dir"] / "final_shap_groupes_tabulaire_textuel_visuel.csv")
    print(paths["predictions_dir"] / "20_predictions_test_final.csv")


if __name__ == "__main__":
    main()
