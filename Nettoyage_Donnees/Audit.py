import pandas as pd
from pathlib import Path

FICHIER_CSV = "listings_price_valide.csv"
FICHIER_SORTIE_EXCEL = "audit_colonnes.xlsx"
SEUIL_QUASI_VIDE = 95.0   # en %

df = pd.read_csv(FICHIER_CSV, sep=None, engine="python", encoding="utf-8-sig")

# Nettoyage simple des noms de colonnes
df.columns = df.columns.str.strip()
print(f"Fichier chargé : {len(df)} lignes, {len(df.columns)} colonnes")

# On transforme les chaînes vides ou composées d'espaces en NaN
df = df.replace(r"^\s*$", pd.NA, regex=True)

# AUDIT PAR COLONNE
rapport = []

for col in df.columns:
    serie = df[col]

    nb_lignes = len(serie)
    nb_manquants = serie.isna().sum()
    pct_manquants = (nb_manquants / nb_lignes) * 100

    # Nombre de valeurs uniques non nulles
    nb_uniques = serie.nunique(dropna=True)

    # Type actuel
    type_actuel = str(serie.dtype)

    # Colonne vide = 100% manquante
    colonne_vide = nb_manquants == nb_lignes

    # Colonne quasi vide = % de manquants >= seuil
    colonne_quasi_vide = pct_manquants >= SEUIL_QUASI_VIDE and not colonne_vide

    # Colonne constante :
    # parmi les valeurs non nulles, une seule valeur unique
    non_null = serie.dropna()
    colonne_constante = (len(non_null) > 0) and (non_null.nunique() == 1)

    rapport.append({
        "colonne": col,
        "type_actuel": type_actuel,
        "nb_valeurs_manquantes": nb_manquants,
        "pct_valeurs_manquantes": round(pct_manquants, 2),
        "nb_valeurs_uniques": nb_uniques,
        "colonne_vide": "oui" if colonne_vide else "non",
        "colonne_quasi_vide": "oui" if colonne_quasi_vide else "non",
        "colonne_constante": "oui" if colonne_constante else "non"
    })

df_audit = pd.DataFrame(rapport)


# TABLEAU RÉSUMÉ
colonnes_vides = df_audit[df_audit["colonne_vide"] == "oui"].copy()
colonnes_quasi_vides = df_audit[df_audit["colonne_quasi_vide"] == "oui"].copy()
colonnes_constantes = df_audit[df_audit["colonne_constante"] == "oui"].copy()

print("\n===== RÉSUMÉ AUDIT =====")
print(f"Nombre total de colonnes : {len(df.columns)}")
print(f"Colonnes vides : {len(colonnes_vides)}")
print(f"Colonnes quasi vides (>= {SEUIL_QUASI_VIDE}% manquants) : {len(colonnes_quasi_vides)}")
print(f"Colonnes constantes : {len(colonnes_constantes)}")

if len(colonnes_vides) > 0:
    print("\nColonnes vides :")
    print(colonnes_vides["colonne"].to_string(index=False))

if len(colonnes_quasi_vides) > 0:
    print(f"\nColonnes quasi vides (>= {SEUIL_QUASI_VIDE}% manquants) :")
    print(colonnes_quasi_vides[["colonne", "pct_valeurs_manquantes"]].to_string(index=False))

if len(colonnes_constantes) > 0:
    print("\nColonnes constantes :")
    print(colonnes_constantes["colonne"].to_string(index=False))

# Sauvegearde du rapport sous le nom " rapport_audit"

FICHIER_SORTIE_EXCEL = Path(__file__).resolve().parent / "rapport_audit.xlsx"

with pd.ExcelWriter(FICHIER_SORTIE_EXCEL, engine="openpyxl") as writer:
    df_audit.to_excel(writer, sheet_name="audit_complet", index=False)
    colonnes_vides.to_excel(writer, sheet_name="colonnes_vides", index=False)
    colonnes_quasi_vides.to_excel(writer, sheet_name="colonnes_quasi_vides", index=False)
    colonnes_constantes.to_excel(writer, sheet_name="colonnes_constantes", index=False)

print(f"\nFichier Excel créé : {FICHIER_SORTIE_EXCEL}")