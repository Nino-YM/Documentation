### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests de robustesse — UC3.3–3.4, UC4, UC5
Couvre :
  - Imputation (1 valeur manquante, tout manquant)
  - Bruit catégoriel sur les features les plus corrélées
  - Couverture de la zone de robustesse Isolation Forest
  - Slice testing : entrées à faible vs fort score d'anomalie
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from utils import FEATURE_COLS, impute, run_if_pipeline

MAX_RMSE_INCREASE_PCT = 15.0  # dégradation RMSE max acceptable (%)
NOISE_LEVELS = [5, 10]        # niveaux de bruit testés (% observations perturbées)


# ---------------------------------------------------------------------------
# UC4 — Imputation
# ---------------------------------------------------------------------------

@pytest.mark.robustness
def test_imputation_single_missing_math(artifacts, typical_profile):
    """Une entrée avec 1 valeur manquante doit produire un math score valide."""
    X = typical_profile.copy()
    X.loc[0, "lunch"] = np.nan
    pred = artifacts["model_math"].predict(impute(X, artifacts["imputer"]))[0]
    assert 0 <= pred <= 100, f"Prédiction invalide après imputation lunch=NaN : {pred:.2f}"


@pytest.mark.robustness
def test_imputation_all_missing_math(artifacts):
    """Une entrée entièrement manquante doit toujours donner un math score valide."""
    X = pd.DataFrame([{col: np.nan for col in FEATURE_COLS}])
    pred = artifacts["model_math"].predict(impute(X, artifacts["imputer"]))[0]
    assert 0 <= pred <= 100, f"Prédiction invalide avec toutes valeurs manquantes : {pred:.2f}"


@pytest.mark.robustness
def test_imputation_all_missing_lang(artifacts):
    """Une entrée entièrement manquante doit toujours donner un language score valide."""
    X = pd.DataFrame([{col: np.nan for col in FEATURE_COLS}])
    pred = artifacts["model_lang"].predict(impute(X, artifacts["imputer"]))[0]
    assert 0 <= pred <= 100, f"Prédiction invalide avec toutes valeurs manquantes : {pred:.2f}"


@pytest.mark.robustness
def test_imputation_lunch_rmse_degradation(artifacts, test_split):
    """Imputation complète sur 'lunch' ne doit pas dégrader la RMSE math de plus de 20%."""
    X_test, y_test_m, _ = test_split
    X_imp_clean = impute(X_test, artifacts["imputer"])
    rmse_clean = np.sqrt(mean_squared_error(y_test_m, artifacts["model_math"].predict(X_imp_clean)))

    X_missing = X_test.copy()
    X_missing["lunch"] = np.nan
    X_imp_missing = impute(X_missing, artifacts["imputer"])
    rmse_missing = np.sqrt(mean_squared_error(y_test_m, artifacts["model_math"].predict(X_imp_missing)))

    variation = 100 * (rmse_missing - rmse_clean) / rmse_clean
    assert variation < 20.0, (
        f"Imputation lunch : dégradation RMSE {variation:.2f}% > 20%"
    )


@pytest.mark.robustness
def test_imputation_gender_rmse_degradation(artifacts, test_split):
    """Imputation complète sur 'gender' ne doit pas dégrader la RMSE language de plus de 20%."""
    X_test, _, y_test_l = test_split
    X_imp_clean = impute(X_test, artifacts["imputer"])
    rmse_clean = np.sqrt(mean_squared_error(y_test_l, artifacts["model_lang"].predict(X_imp_clean)))

    X_missing = X_test.copy()
    X_missing["gender"] = np.nan
    X_imp_missing = impute(X_missing, artifacts["imputer"])
    rmse_missing = np.sqrt(mean_squared_error(y_test_l, artifacts["model_lang"].predict(X_imp_missing)))

    variation = 100 * (rmse_missing - rmse_clean) / rmse_clean
    assert variation < 20.0, (
        f"Imputation gender : dégradation RMSE {variation:.2f}% > 20%"
    )


# ---------------------------------------------------------------------------
# UC5 — Bruit catégoriel
# ---------------------------------------------------------------------------

@pytest.mark.robustness
@pytest.mark.parametrize("noise_pct", NOISE_LEVELS)
def test_noise_lunch_math_rmse(artifacts, test_split, noise_pct):
    """Bruit catégoriel sur 'lunch' : dégradation RMSE math < MAX_RMSE_INCREASE_PCT."""
    X_test, y_test_m, _ = test_split
    X_imp_clean = impute(X_test, artifacts["imputer"])
    rmse_clean = np.sqrt(mean_squared_error(y_test_m, artifacts["model_math"].predict(X_imp_clean)))

    np.random.seed(42)
    X_noisy = X_test.copy()
    n_perturb = max(1, int(len(X_noisy) * noise_pct / 100))
    idx = np.random.choice(len(X_noisy), n_perturb, replace=False)
    X_noisy.iloc[idx, X_noisy.columns.get_loc("lunch")] = np.random.choice(
        ["standard", "free/reduced"], n_perturb
    )
    rmse_noisy = np.sqrt(mean_squared_error(y_test_m, artifacts["model_math"].predict(impute(X_noisy, artifacts["imputer"]))))
    variation = 100 * (rmse_noisy - rmse_clean) / rmse_clean
    assert variation < MAX_RMSE_INCREASE_PCT, (
        f"Bruit {noise_pct}% sur lunch : RMSE +{variation:.2f}% > {MAX_RMSE_INCREASE_PCT}%"
    )


@pytest.mark.robustness
@pytest.mark.parametrize("noise_pct", NOISE_LEVELS)
def test_noise_gender_lang_rmse(artifacts, test_split, noise_pct):
    """Bruit catégoriel sur 'gender' : dégradation RMSE language < MAX_RMSE_INCREASE_PCT."""
    X_test, _, y_test_l = test_split
    X_imp_clean = impute(X_test, artifacts["imputer"])
    rmse_clean = np.sqrt(mean_squared_error(y_test_l, artifacts["model_lang"].predict(X_imp_clean)))

    np.random.seed(42)
    X_noisy = X_test.copy()
    n_perturb = max(1, int(len(X_noisy) * noise_pct / 100))
    idx = np.random.choice(len(X_noisy), n_perturb, replace=False)
    X_noisy.iloc[idx, X_noisy.columns.get_loc("gender")] = np.random.choice(
        ["male", "female"], n_perturb
    )
    rmse_noisy = np.sqrt(mean_squared_error(y_test_l, artifacts["model_lang"].predict(impute(X_noisy, artifacts["imputer"]))))
    variation = 100 * (rmse_noisy - rmse_clean) / rmse_clean
    assert variation < MAX_RMSE_INCREASE_PCT, (
        f"Bruit {noise_pct}% sur gender : RMSE +{variation:.2f}% > {MAX_RMSE_INCREASE_PCT}%"
    )


# ---------------------------------------------------------------------------
# UC3.3 — Couverture zone de robustesse
# ---------------------------------------------------------------------------

@pytest.mark.robustness
def test_robustness_zone_coverage(artifacts, test_split):
    """Au moins 85% du jeu de test doit se trouver dans la zone de robustesse."""
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    scores, _ = run_if_pipeline(X_imp, artifacts["encoder"], artifacts["scaler"], artifacts["iso_forest"])
    threshold = artifacts["config"]["robustness_threshold"]
    coverage = np.mean(scores >= threshold) * 100
    assert coverage >= 85.0, f"Couverture zone robustesse : {coverage:.1f}% < 85%"


# ---------------------------------------------------------------------------
# UC3.4 — Slice testing autour du seuil de robustesse
# ---------------------------------------------------------------------------

@pytest.mark.robustness
def test_slice_high_anomaly_higher_error(artifacts, test_split):
    """
    Les entrées à fort score d'anomalie (hors zone de robustesse) doivent
    avoir une erreur de prédiction plus élevée que les entrées typiques.
    """
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    scores, _ = run_if_pipeline(X_imp, artifacts["encoder"], artifacts["scaler"], artifacts["iso_forest"])
    threshold = artifacts["config"]["robustness_threshold"]

    mask_robust = scores >= threshold
    mask_anomaly = scores < threshold

    if mask_anomaly.sum() < 5:
        pytest.skip("Pas assez d'anomalies dans le jeu de test pour ce slice test")

    preds = artifacts["model_math"].predict(X_imp)
    errors_robust  = np.abs(preds[mask_robust]  - y_test_m.values[mask_robust])
    errors_anomaly = np.abs(preds[mask_anomaly] - y_test_m.values[mask_anomaly])

    assert errors_anomaly.mean() >= errors_robust.mean(), (
        f"Erreur anomalies ({errors_anomaly.mean():.2f}) devrait >= erreur robustesse ({errors_robust.mean():.2f})"
    )
