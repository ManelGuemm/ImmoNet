from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = Path(__file__).resolve().parent

fichier_tabulaire = BASE_DIR / "Donnees_Airbnb_Finales_Tabulaire.xlsx"
fichier_sentiment_bert = BASE_DIR / "sentiment_bert_par_annonce_COMPLET.csv"

dossier_sortie = BASE_DIR / "Validation_BERT_Note_Airbnb"
dossier_sortie.mkdir(exist_ok=True)

fichier_fusion = dossier_sortie / "validation_fusion_tabulaire_bert.csv"
fichier_correlations = dossier_sortie / "validation_correlations_bert_notes.csv"
fichier_regression = dossier_sortie / "validation_regression_note_airbnb.csv"
fichier_classification = dossier_sortie / "validation_classification_note_airbnb.csv"
fichier_resume_txt = dossier_sortie / "resume_validation_bert_notes.txt"
fichier_pdf = dossier_sortie / "rapport_validation_bert_notes_airbnb.pdf"


def convertir_numerique(serie):
    return pd.to_numeric(
        serie.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["nan", "None", "", "NaN"], np.nan),
        errors="coerce"
    )


def label_note_3_classes(note):
    if pd.isna(note):
        return np.nan
    if note >= 4.5:
        return "positif"
    elif note >= 4.0:
        return "neutre"
    else:
        return "negatif"


def label_bert_3_classes(bert_stars):
    if pd.isna(bert_stars):
        return np.nan
    if bert_stars >= 4.5:
        return "positif"
    elif bert_stars >= 4.0:
        return "neutre"
    else:
        return "negatif"


def ajouter_page_texte(pdf, titre, lignes):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    texte = titre + "\n\n" + "\n".join(lignes)
    plt.text(
        0.05,
        0.95,
        texte,
        ha="left",
        va="top",
        fontsize=10,
        wrap=True
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


print("Chargement des fichiers...")

if not fichier_tabulaire.exists():
    raise FileNotFoundError(f"Fichier tabulaire introuvable : {fichier_tabulaire}")

if not fichier_sentiment_bert.exists():
    raise FileNotFoundError(f"Fichier BERT introuvable : {fichier_sentiment_bert}")


df_tab = pd.read_excel(fichier_tabulaire, dtype=str)
df_tab.columns = df_tab.columns.str.strip()

df_bert = pd.read_csv(
    fichier_sentiment_bert,
    dtype=str,
    encoding="utf-8-sig",
    on_bad_lines="skip"
)

df_bert.columns = df_bert.columns.str.strip()

if "id_clean" not in df_tab.columns:
    if "listing_url" not in df_tab.columns:
        raise ValueError("Impossible de créer id_clean : listing_url est absente.")
    df_tab["id_clean"] = (
        df_tab["listing_url"]
        .astype(str)
        .str.extract(r"/rooms/(\d+)")[0]
    )

df_tab["id_clean"] = (
    df_tab["id_clean"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

df_bert["id_clean"] = (
    df_bert["id_clean"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)


print("Fusion tabulaire + BERT...")

df = df_tab.merge(
    df_bert,
    on="id_clean",
    how="left",
    indicator=True
)

nb_tabulaire = len(df_tab)
nb_bert = len(df_bert)
nb_fusion_total = len(df)
nb_match_bert = (df["_merge"] == "both").sum()
nb_sans_bert = (df["_merge"] == "left_only").sum()

print("Nombre de lignes tabulaire :", nb_tabulaire)
print("Nombre d'annonces dans BERT :", nb_bert)
print("Nombre de lignes après fusion :", nb_fusion_total)
print("Nombre d'annonces avec sentiment BERT :", nb_match_bert)
print("Nombre d'annonces sans sentiment BERT :", nb_sans_bert)


colonnes_notes = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

colonnes_notes = [c for c in colonnes_notes if c in df.columns]

colonnes_bert = [
    "nb_avis_textuels_bert",
    "bert_stars_moyen",
    "bert_confiance_moyenne",
    "sentiment_bert_moyen",
    "sentiment_bert_min",
    "sentiment_bert_max",
    "sentiment_bert_ecart_type",
    "nb_avis_bert_positifs",
    "nb_avis_bert_neutres",
    "nb_avis_bert_negatifs",
    "ratio_avis_bert_positifs",
    "ratio_avis_bert_neutres",
    "ratio_avis_bert_negatifs"
]

for col in colonnes_notes + colonnes_bert:
    if col in df.columns:
        df[col] = convertir_numerique(df[col])


df["note_airbnb_label"] = df["review_scores_rating"].apply(label_note_3_classes)
df["bert_label_annonce"] = df["bert_stars_moyen"].apply(label_bert_3_classes)

df_validation = df.dropna(
    subset=[
        "review_scores_rating",
        "bert_stars_moyen",
        "sentiment_bert_moyen",
        "ratio_avis_bert_positifs",
        "ratio_avis_bert_negatifs"
    ]
).copy()

df.to_csv(fichier_fusion, index=False, encoding="utf-8-sig")

print("Nombre d'annonces exploitables pour validation :", len(df_validation))


print("Calcul des corrélations...")

variables_bert_a_tester = [
    "bert_stars_moyen",
    "sentiment_bert_moyen",
    "sentiment_bert_min",
    "sentiment_bert_max",
    "sentiment_bert_ecart_type",
    "ratio_avis_bert_positifs",
    "ratio_avis_bert_neutres",
    "ratio_avis_bert_negatifs",
    "nb_avis_bert_positifs",
    "nb_avis_bert_neutres",
    "nb_avis_bert_negatifs",
    "nb_avis_textuels_bert",
    "bert_confiance_moyenne"
]

variables_bert_a_tester = [v for v in variables_bert_a_tester if v in df_validation.columns]

resultats_corr = []

for var in variables_bert_a_tester:
    tmp = df_validation[["review_scores_rating", var]].dropna()

    if len(tmp) > 2:
        pearson_corr, pearson_p = pearsonr(tmp[var], tmp["review_scores_rating"])
        spearman_corr, spearman_p = spearmanr(tmp[var], tmp["review_scores_rating"])

        resultats_corr.append({
            "variable_bert": var,
            "n": len(tmp),
            "pearson_correlation": pearson_corr,
            "pearson_pvalue": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_pvalue": spearman_p
        })

df_corr = pd.DataFrame(resultats_corr).sort_values(
    "spearman_correlation",
    ascending=False
)

df_corr.to_csv(fichier_correlations, index=False, encoding="utf-8-sig")


print("Validation directe : bert_stars_moyen vs review_scores_rating...")

y_rating = df_validation["review_scores_rating"]
y_bert_direct = df_validation["bert_stars_moyen"]

mae_direct = mean_absolute_error(y_rating, y_bert_direct)
rmse_direct = mean_squared_error(y_rating, y_bert_direct) ** 0.5
r2_direct = r2_score(y_rating, y_bert_direct)

metrics_regression = []

metrics_regression.append({
    "modele": "Comparaison directe bert_stars_moyen",
    "MAE": mae_direct,
    "RMSE": rmse_direct,
    "R2": r2_direct,
    "n_test": len(df_validation)
})


print("Validation train/test : prédire review_scores_rating à partir des variables BERT...")

features_bert = [
    "nb_avis_textuels_bert",
    "bert_stars_moyen",
    "bert_confiance_moyenne",
    "sentiment_bert_moyen",
    "sentiment_bert_min",
    "sentiment_bert_max",
    "sentiment_bert_ecart_type",
    "nb_avis_bert_positifs",
    "nb_avis_bert_neutres",
    "nb_avis_bert_negatifs",
    "ratio_avis_bert_positifs",
    "ratio_avis_bert_neutres",
    "ratio_avis_bert_negatifs"
]

features_bert = [c for c in features_bert if c in df_validation.columns]

X = df_validation[features_bert].copy()
y = df_validation["review_scores_rating"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

reg_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

mae_reg = mean_absolute_error(y_test, y_pred)
rmse_reg = mean_squared_error(y_test, y_pred) ** 0.5
r2_reg = r2_score(y_test, y_pred)

metrics_regression.append({
    "modele": "Régression train/test : variables BERT -> review_scores_rating",
    "MAE": mae_reg,
    "RMSE": rmse_reg,
    "R2": r2_reg,
    "n_test": len(y_test)
})

df_metrics_reg = pd.DataFrame(metrics_regression)
df_metrics_reg.to_csv(fichier_regression, index=False, encoding="utf-8-sig")


print("Validation classification 3 classes...")

df_clf = df_validation.dropna(subset=["note_airbnb_label", "bert_label_annonce"]).copy()

labels_ordres = ["negatif", "neutre", "positif"]

acc_direct = accuracy_score(df_clf["note_airbnb_label"], df_clf["bert_label_annonce"])
balanced_acc_direct = balanced_accuracy_score(df_clf["note_airbnb_label"], df_clf["bert_label_annonce"])
f1_macro_direct = f1_score(
    df_clf["note_airbnb_label"],
    df_clf["bert_label_annonce"],
    average="macro"
)

cm_direct = confusion_matrix(
    df_clf["note_airbnb_label"],
    df_clf["bert_label_annonce"],
    labels=labels_ordres
)

classification_results = []

classification_results.append({
    "modele": "Comparaison directe classes BERT vs classes note Airbnb",
    "accuracy": acc_direct,
    "balanced_accuracy": balanced_acc_direct,
    "f1_macro": f1_macro_direct,
    "n_test": len(df_clf)
})


print("Validation train/test classification : variables BERT -> classe note Airbnb...")

X_clf = df_clf[features_bert].copy()
y_clf = df_clf["note_airbnb_label"].copy()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

clf_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
y_proba_c = clf_model.predict_proba(X_test_c)

acc_clf = accuracy_score(y_test_c, y_pred_c)
balanced_acc_clf = balanced_accuracy_score(y_test_c, y_pred_c)
f1_macro_clf = f1_score(y_test_c, y_pred_c, average="macro")

classes_model = list(clf_model.named_steps["logreg"].classes_)

try:
    y_test_bin = label_binarize(y_test_c, classes=classes_model)
    roc_auc_macro = roc_auc_score(
        y_test_bin,
        y_proba_c,
        average="macro",
        multi_class="ovr"
    )
except Exception:
    roc_auc_macro = np.nan

classification_results.append({
    "modele": "Classification train/test : variables BERT -> classe note Airbnb",
    "accuracy": acc_clf,
    "balanced_accuracy": balanced_acc_clf,
    "f1_macro": f1_macro_clf,
    "roc_auc_macro": roc_auc_macro,
    "n_test": len(y_test_c)
})

df_metrics_clf = pd.DataFrame(classification_results)
df_metrics_clf.to_csv(fichier_classification, index=False, encoding="utf-8-sig")


rapport_classification_direct = classification_report(
    df_clf["note_airbnb_label"],
    df_clf["bert_label_annonce"],
    labels=labels_ordres,
    zero_division=0
)

rapport_classification_train_test = classification_report(
    y_test_c,
    y_pred_c,
    labels=labels_ordres,
    zero_division=0
)


print("Validation binaire : détecter les annonces moins bien notées...")

df_bin = df_validation.copy()
df_bin["note_basse"] = (df_bin["review_scores_rating"] < 4.5).astype(int)

X_bin = df_bin[features_bert].copy()
y_bin = df_bin["note_basse"].copy()

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin,
    y_bin,
    test_size=0.20,
    random_state=42,
    stratify=y_bin
)

bin_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

bin_model.fit(X_train_b, y_train_b)

y_pred_b = bin_model.predict(X_test_b)
y_proba_b = bin_model.predict_proba(X_test_b)[:, 1]

binary_metrics = {
    "accuracy": accuracy_score(y_test_b, y_pred_b),
    "balanced_accuracy": balanced_accuracy_score(y_test_b, y_pred_b),
    "f1": f1_score(y_test_b, y_pred_b),
    "roc_auc": roc_auc_score(y_test_b, y_proba_b),
    "average_precision": average_precision_score(y_test_b, y_proba_b),
    "n_test": len(y_test_b)
}

cm_binary = confusion_matrix(y_test_b, y_pred_b)


print("Création des graphiques...")

plt.figure(figsize=(7, 5))
plt.scatter(
    df_validation["bert_stars_moyen"],
    df_validation["review_scores_rating"],
    alpha=0.15,
    s=8
)
plt.xlabel("bert_stars_moyen")
plt.ylabel("review_scores_rating")
plt.title("Relation entre étoiles BERT moyennes et note Airbnb")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dossier_sortie / "scatter_bert_stars_vs_rating.png", dpi=200)
plt.close()


plt.figure(figsize=(7, 5))
plt.scatter(
    df_validation["ratio_avis_bert_negatifs"],
    df_validation["review_scores_rating"],
    alpha=0.15,
    s=8
)
plt.xlabel("ratio_avis_bert_negatifs")
plt.ylabel("review_scores_rating")
plt.title("Relation entre ratio d'avis négatifs BERT et note Airbnb")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dossier_sortie / "scatter_ratio_negatif_vs_rating.png", dpi=200)
plt.close()


plt.figure(figsize=(7, 5))
plt.hist(df_validation["review_scores_rating"], bins=40)
plt.xlabel("review_scores_rating")
plt.ylabel("Nombre d'annonces")
plt.title("Distribution des notes Airbnb")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dossier_sortie / "distribution_review_scores_rating.png", dpi=200)
plt.close()


plt.figure(figsize=(6, 5))
plt.imshow(cm_direct, interpolation="nearest")
plt.title("Matrice de confusion : classes note Airbnb vs classes BERT")
plt.colorbar()
plt.xticks(np.arange(len(labels_ordres)), labels_ordres, rotation=45)
plt.yticks(np.arange(len(labels_ordres)), labels_ordres)
plt.xlabel("Classe prédite par BERT")
plt.ylabel("Classe issue de la note Airbnb")

for i in range(cm_direct.shape[0]):
    for j in range(cm_direct.shape[1]):
        plt.text(j, i, cm_direct[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(dossier_sortie / "matrice_confusion_bert_vs_note.png", dpi=200)
plt.close()


print("Création du résumé texte...")

resume_lignes = []

resume_lignes.append("VALIDATION DU SENTIMENT BERT AVEC LES NOTES AIRBNB")
resume_lignes.append("")
resume_lignes.append(f"Nombre de lignes tabulaire : {nb_tabulaire}")
resume_lignes.append(f"Nombre d'annonces avec sentiment BERT : {nb_match_bert}")
resume_lignes.append(f"Nombre d'annonces exploitables avec review_scores_rating et BERT : {len(df_validation)}")
resume_lignes.append("")
resume_lignes.append("Corrélations principales avec review_scores_rating :")

for _, row in df_corr.head(10).iterrows():
    resume_lignes.append(
        f"- {row['variable_bert']} : Pearson={row['pearson_correlation']:.4f}, "
        f"Spearman={row['spearman_correlation']:.4f}"
    )

resume_lignes.append("")
resume_lignes.append("Validation directe bert_stars_moyen vs review_scores_rating :")
resume_lignes.append(f"- MAE : {mae_direct:.4f}")
resume_lignes.append(f"- RMSE : {rmse_direct:.4f}")
resume_lignes.append(f"- R2 : {r2_direct:.4f}")
resume_lignes.append("")
resume_lignes.append("Régression train/test : variables BERT -> review_scores_rating")
resume_lignes.append(f"- MAE test : {mae_reg:.4f}")
resume_lignes.append(f"- RMSE test : {rmse_reg:.4f}")
resume_lignes.append(f"- R2 test : {r2_reg:.4f}")
resume_lignes.append("")
resume_lignes.append("Classification directe classes BERT vs classes note Airbnb")
resume_lignes.append(f"- Accuracy : {acc_direct:.4f}")
resume_lignes.append(f"- Balanced accuracy : {balanced_acc_direct:.4f}")
resume_lignes.append(f"- F1 macro : {f1_macro_direct:.4f}")
resume_lignes.append("")
resume_lignes.append("Classification train/test : variables BERT -> classe note Airbnb")
resume_lignes.append(f"- Accuracy : {acc_clf:.4f}")
resume_lignes.append(f"- Balanced accuracy : {balanced_acc_clf:.4f}")
resume_lignes.append(f"- F1 macro : {f1_macro_clf:.4f}")
resume_lignes.append(f"- ROC-AUC macro : {roc_auc_macro:.4f}" if not pd.isna(roc_auc_macro) else "- ROC-AUC macro : non calculable")
resume_lignes.append("")
resume_lignes.append("Validation binaire : détecter review_scores_rating < 4.5")
for k, v in binary_metrics.items():
    resume_lignes.append(f"- {k} : {v:.4f}" if isinstance(v, float) else f"- {k} : {v}")

resume_lignes.append("")
resume_lignes.append("Rapport classification direct :")
resume_lignes.append(rapport_classification_direct)
resume_lignes.append("")
resume_lignes.append("Rapport classification train/test :")
resume_lignes.append(rapport_classification_train_test)

with open(fichier_resume_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(resume_lignes))


print("Création du PDF...")

with PdfPages(fichier_pdf) as pdf:
    ajouter_page_texte(
        pdf,
        "Validation du sentiment BERT avec les notes Airbnb",
        resume_lignes[:40]
    )

    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    texte_corr = "Corrélations complètes\n\n" + df_corr.to_string(index=False)
    plt.text(0.05, 0.95, texte_corr, ha="left", va="top", fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    img = plt.imread(dossier_sortie / "scatter_bert_stars_vs_rating.png")
    fig, ax = plt.subplots(figsize=(8.27, 6))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    img = plt.imread(dossier_sortie / "scatter_ratio_negatif_vs_rating.png")
    fig, ax = plt.subplots(figsize=(8.27, 6))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    img = plt.imread(dossier_sortie / "distribution_review_scores_rating.png")
    fig, ax = plt.subplots(figsize=(8.27, 6))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    img = plt.imread(dossier_sortie / "matrice_confusion_bert_vs_note.png")
    fig, ax = plt.subplots(figsize=(8.27, 6))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    ajouter_page_texte(
        pdf,
        "Rapport classification direct",
        rapport_classification_direct.split("\n")
    )

    ajouter_page_texte(
        pdf,
        "Rapport classification train/test",
        rapport_classification_train_test.split("\n")
    )


print("")
print("Validation terminée.")
print("Fichiers créés dans :", dossier_sortie)
print("Fusion :", fichier_fusion)
print("Corrélations :", fichier_correlations)
print("Métriques régression :", fichier_regression)
print("Métriques classification :", fichier_classification)
print("Résumé texte :", fichier_resume_txt)
print("Rapport PDF :", fichier_pdf)