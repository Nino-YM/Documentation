### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests de comparaison — CatBoost vs Second modèle
Fait tourner les mêmes métriques sur les deux modèles et tranche entre eux.

Prérequis : exporter le second modèle sous :
  artifacts/model2_math_score.joblib
  artifacts/model2_language_mean_score.joblib

Les tests de ce fichier sont automatiquement skippés si ces fichiers sont absents.
"""

import time

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import FEATURE_COLS, impute

# Seuils pour que le second modèle soit "acceptable" (même que pour CatBoost)
MATH_MAE_THRESHOLD  = 12.0
MATH_RMSE_THRESHOLD = 15.1
LANG_MAE_THRESHOLD  = 10.9
LANG_RMSE_THRESHOLD = 14.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(model, X_imp, y_true):
    preds = model.predict(X_imp)
    return {
        "mae":  mean_absolute_error(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
    }


def _latency(model, X_imp) -> float:
    t0 = time.time()
    model.predict(X_imp)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Tests unitaires sur le second modèle (plage de valeurs)
# ---------------------------------------------------------------------------

@pytest.mark.compare
def test_model2_math_predictions_in_range(artifacts, model2_math, test_split):
    """Le second modèle math doit prédire dans [0, 100]."""
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = model2_math.predict(X_imp)
    assert np.all(preds >= 0) and np.all(preds <= 100), (
        f"Second modèle math hors [0, 100] : min={preds.min():.2f}, max={preds.max():.2f}"
    )


@pytest.mark.compare
def test_model2_lang_predictions_in_range(artifacts, model2_lang, test_split):
    """Le second modèle language doit prédire dans [0, 100]."""
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = model2_lang.predict(X_imp)
    assert np.all(preds >= 0) and np.all(preds <= 100), (
        f"Second modèle language hors [0, 100] : min={preds.min():.2f}, max={preds.max():.2f}"
    )


# ---------------------------------------------------------------------------
# Tests de performance sur le second modèle
# ---------------------------------------------------------------------------

@pytest.mark.compare
def test_model2_math_mae(artifacts, model2_math, test_split):
    """Le second modèle math doit respecter le seuil MAE."""
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    mae = mean_absolute_error(y_test_m, model2_math.predict(X_imp))
    assert mae < MATH_MAE_THRESHOLD, f"Second modèle Math MAE {mae:.4f} >= seuil {MATH_MAE_THRESHOLD}"


@pytest.mark.compare
def test_model2_math_rmse(artifacts, model2_math, test_split):
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    rmse = np.sqrt(mean_squared_error(y_test_m, model2_math.predict(X_imp)))
    assert rmse < MATH_RMSE_THRESHOLD, f"Second modèle Math RMSE {rmse:.4f} >= seuil {MATH_RMSE_THRESHOLD}"


@pytest.mark.compare
def test_model2_lang_mae(artifacts, model2_lang, test_split):
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    mae = mean_absolute_error(y_test_l, model2_lang.predict(X_imp))
    assert mae < LANG_MAE_THRESHOLD, f"Second modèle Language MAE {mae:.4f} >= seuil {LANG_MAE_THRESHOLD}"


@pytest.mark.compare
def test_model2_lang_rmse(artifacts, model2_lang, test_split):
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    rmse = np.sqrt(mean_squared_error(y_test_l, model2_lang.predict(X_imp)))
    assert rmse < LANG_RMSE_THRESHOLD, f"Second modèle Language RMSE {rmse:.4f} >= seuil {LANG_RMSE_THRESHOLD}"


# ---------------------------------------------------------------------------
# Comparaison directe : quel modèle est meilleur ?
# Les tests échouent si le second modèle est MOINS bon que CatBoost.
# ---------------------------------------------------------------------------

@pytest.mark.compare
def test_compare_math_mae(artifacts, model2_math, test_split):
    """
    Compare les MAE Math des deux modèles et affiche le résultat.
    Échoue si le second modèle dépasse la MAE de CatBoost de plus de 10%.
    """
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    m1 = _compute_metrics(artifacts["model_math"], X_imp, y_test_m)
    m2 = _compute_metrics(model2_math,             X_imp, y_test_m)

    winner = "CatBoost" if m1["mae"] <= m2["mae"] else "Modèle 2"
    print(f"\n[Math MAE] CatBoost={m1['mae']:.4f} | Modèle2={m2['mae']:.4f} → Gagnant : {winner}")

    tolerance = m1["mae"] * 1.10
    assert m2["mae"] <= tolerance, (
        f"Modèle 2 Math MAE ({m2['mae']:.4f}) dépasse CatBoost+10% ({tolerance:.4f})"
    )


@pytest.mark.compare
def test_compare_math_rmse(artifacts, model2_math, test_split):
    """Compare les RMSE Math. Échoue si modèle 2 est plus de 10% moins bon."""
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    m1 = _compute_metrics(artifacts["model_math"], X_imp, y_test_m)
    m2 = _compute_metrics(model2_math,             X_imp, y_test_m)

    winner = "CatBoost" if m1["rmse"] <= m2["rmse"] else "Modèle 2"
    print(f"\n[Math RMSE] CatBoost={m1['rmse']:.4f} | Modèle2={m2['rmse']:.4f} → Gagnant : {winner}")

    tolerance = m1["rmse"] * 1.10
    assert m2["rmse"] <= tolerance, (
        f"Modèle 2 Math RMSE ({m2['rmse']:.4f}) dépasse CatBoost+10% ({tolerance:.4f})"
    )


@pytest.mark.compare
def test_compare_lang_mae(artifacts, model2_lang, test_split):
    """Compare les MAE Language. Échoue si modèle 2 est plus de 10% moins bon."""
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    m1 = _compute_metrics(artifacts["model_lang"], X_imp, y_test_l)
    m2 = _compute_metrics(model2_lang,             X_imp, y_test_l)

    winner = "CatBoost" if m1["mae"] <= m2["mae"] else "Modèle 2"
    print(f"\n[Language MAE] CatBoost={m1['mae']:.4f} | Modèle2={m2['mae']:.4f} → Gagnant : {winner}")

    tolerance = m1["mae"] * 1.10
    assert m2["mae"] <= tolerance, (
        f"Modèle 2 Language MAE ({m2['mae']:.4f}) dépasse CatBoost+10% ({tolerance:.4f})"
    )


@pytest.mark.compare
def test_compare_lang_rmse(artifacts, model2_lang, test_split):
    """Compare les RMSE Language. Échoue si modèle 2 est plus de 10% moins bon."""
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    m1 = _compute_metrics(artifacts["model_lang"], X_imp, y_test_l)
    m2 = _compute_metrics(model2_lang,             X_imp, y_test_l)

    winner = "CatBoost" if m1["rmse"] <= m2["rmse"] else "Modèle 2"
    print(f"\n[Language RMSE] CatBoost={m1['rmse']:.4f} | Modèle2={m2['rmse']:.4f} → Gagnant : {winner}")

    tolerance = m1["rmse"] * 1.10
    assert m2["rmse"] <= tolerance, (
        f"Modèle 2 Language RMSE ({m2['rmse']:.4f}) dépasse CatBoost+10% ({tolerance:.4f})"
    )


@pytest.mark.compare
def test_compare_latency_math(artifacts, model2_math, test_split):
    """Compare la latence de prédiction Math sur le jeu de test."""
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    t1 = _latency(artifacts["model_math"], X_imp)
    t2 = _latency(model2_math,             X_imp)

    winner = "CatBoost" if t1 <= t2 else "Modèle 2"
    print(f"\n[Latence Math] CatBoost={t1:.4f}s | Modèle2={t2:.4f}s → Plus rapide : {winner}")
    # Pas d'assertion stricte sur la latence — informatif


@pytest.mark.compare
def test_compare_latency_lang(artifacts, model2_lang, test_split):
    """Compare la latence de prédiction Language sur le jeu de test."""
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])

    t1 = _latency(artifacts["model_lang"], X_imp)
    t2 = _latency(model2_lang,             X_imp)

    winner = "CatBoost" if t1 <= t2 else "Modèle 2"
    print(f"\n[Latence Language] CatBoost={t1:.4f}s | Modèle2={t2:.4f}s → Plus rapide : {winner}")
