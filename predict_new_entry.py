#!/usr/bin/env python3
"""
predict_new_entry.py
--------------------
Script d'appel des modèles Math Score et Language Mean Score
pour une nouvelle entrée importée d'un CSV.

Pipeline complet :
  Chargement CSV  →  Imputation (most_frequent)
  →  Encodage ordinal  →  Normalisation (StandardScaler)
  →  Score Isolation Forest  →  Prédiction CatBoost
  →  Affichage résultats + zone de confiance

Usage :
  python predict_new_entry.py --input data/new_entry.csv
  python predict_new_entry.py --input data/new_entry.csv --model math
  python predict_new_entry.py --input data/new_entry.csv --model language
  python predict_new_entry.py --input data/new_entry.csv --model both

Colonnes attendues dans le CSV :
  gender, race/ethnicity, parental level of education, lunch,
  test preparation course
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"

EXPECTED_FEATURES = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

ARTIFACT_FILES = {
    "imputer":    "imputer.joblib",
    "encoder":    "ordinal_encoder.joblib",
    "scaler":     "scaler_if.joblib",
    "iso_forest": "isolation_forest.joblib",
    "model_math": "catboost_math_score.joblib",
    "model_lang": "catboost_language_mean_score.joblib",
    "config":     "robustness_config.json",
}


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def load_artifacts() -> dict:
    """Charge tous les artefacts nécessaires depuis le dossier artifacts/."""
    arts = {}
    for key, fname in ARTIFACT_FILES.items():
        path = os.path.join(ARTIFACTS_DIR, fname)
        if not os.path.exists(path):
            print(f"[ERREUR] Artefact manquant : {path}")
            print("  → Veuillez d'abord exécuter tp_isolation_forest_imputation.ipynb")
            sys.exit(1)
        if fname.endswith(".joblib"):
            arts[key] = joblib.load(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                arts[key] = json.load(f)
    return arts


def get_confidence_zone(score: float, config: dict) -> str:
    """Retourne la zone de confiance selon le score d'isolation."""
    thresh = config.get("robustness_threshold", -0.5)
    if score >= thresh:
        return "Zone de robustesse (haute confiance)"
    elif score >= -0.4:
        return "Zone normale (confiance modérée)"
    else:
        return "Zone d'extrapolation (faible confiance — prédiction incertaine)"


def format_row(label: str, value) -> str:
    return f"  {label:<40} {value}"


# ---------------------------------------------------------------------------
# Prédiction principale
# ---------------------------------------------------------------------------

def predict(input_csv: str, model_choice: str = "both") -> list[dict]:
    """
    Effectue la prédiction sur les données d'un CSV.

    Parameters
    ----------
    input_csv    : chemin vers le fichier CSV
    model_choice : 'math', 'language' ou 'both'

    Returns
    -------
    Liste de résultats par entrée (dict).
    """
    # 1. Chargement des artefacts
    arts = load_artifacts()
    config = arts["config"]

    # 2. Lecture du CSV
    if not os.path.exists(input_csv):
        print(f"[ERREUR] Fichier introuvable : {input_csv}")
        sys.exit(1)

    df_raw = pd.read_csv(input_csv)

    # Vérification des colonnes
    missing_cols = [c for c in EXPECTED_FEATURES if c not in df_raw.columns]
    if missing_cols:
        print(f"[ERREUR] Colonnes manquantes dans le CSV : {missing_cols}")
        print(f"         Colonnes attendues : {EXPECTED_FEATURES}")
        sys.exit(1)

    X = df_raw[EXPECTED_FEATURES].copy()
    n_entries = len(X)
    print(f"\n[INFO] {n_entries} entrée(s) chargée(s) depuis '{input_csv}'")

    # 3. Imputation des valeurs manquantes
    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        print(f"[INFO] {n_missing} valeur(s) manquante(s) détectée(s) → imputation (most_frequent)")

    arr_imp = arts["imputer"].transform(X)
    X_imp = pd.DataFrame(arr_imp, columns=EXPECTED_FEATURES)
    for col in EXPECTED_FEATURES:
        X_imp[col] = X_imp[col].astype(str)

    # 4. Encodage + normalisation pour l'Isolation Forest
    X_enc  = arts["encoder"].transform(X_imp)
    X_norm = arts["scaler"].transform(X_enc)

    # 5. Score Isolation Forest
    if_scores  = arts["iso_forest"].score_samples(X_norm)
    if_labels  = arts["iso_forest"].predict(X_norm)  # 1=normal, -1=anomalie

    # 6. Prédictions CatBoost + mise en forme des résultats
    results = []
    for i in range(n_entries):
        entry      = X_imp.iloc[[i]]
        score      = float(if_scores[i])
        is_anomaly = (if_labels[i] == -1)
        zone       = get_confidence_zone(score, config)

        row: dict = {
            "entree"                     : i + 1,
            "score_isolation_forest"     : round(score, 4),
            "anomalie_detectee"          : is_anomaly,
            "zone_de_confiance"          : zone,
        }

        if model_choice in ("math", "both"):
            pred_m = float(arts["model_math"].predict(entry)[0])
            row["prediction_math_score"] = round(pred_m, 2)

        if model_choice in ("language", "both"):
            pred_l = float(arts["model_lang"].predict(entry)[0])
            row["prediction_language_mean_score"] = round(pred_l, 2)

        results.append(row)

    # 7. Affichage
    print(f"\n{'=' * 65}")
    print(f"  RÉSULTATS DE PRÉDICTION  —  {n_entries} entrée(s)")
    print(f"{'=' * 65}")

    for r in results:
        print(f"\n[ Entrée {r['entree']} ]")
        anom_flag = "OUI  " if r["anomalie_detectee"] else "NON ✓"
        print(format_row("Score Isolation Forest :",  r["score_isolation_forest"]))
        print(format_row("Anomalie détectée :",        anom_flag))
        print(format_row("Zone de confiance :",        r["zone_de_confiance"]))
        if "prediction_math_score" in r:
            print(format_row("Math Score prédit :",    r["prediction_math_score"]))
        if "prediction_language_mean_score" in r:
            print(format_row("Language Mean Score prédit :", r["prediction_language_mean_score"]))

    # 8. Sauvegarde CSV
    output_path = input_csv.replace(".csv", "_predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[INFO] Résultats sauvegardés : {output_path}")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction Students Performance — Math & Language Mean Score"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers le fichier CSV d'entrée"
    )
    parser.add_argument(
        "--model",
        choices=["math", "language", "both"],
        default="both",
        help="Modèle(s) à utiliser : 'math', 'language' ou 'both' (défaut : both)"
    )
    args = parser.parse_args()
    predict(args.input, args.model)
