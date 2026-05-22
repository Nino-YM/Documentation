### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests de performance globale — UC1.2, UC2.2
Vérifient que MAE et RMSE restent dans les seuils publiés dans les model cards.
Split identique aux notebooks : random_state=42, test_size=0.2.
"""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import impute

# Seuils = métriques model card + marge de 5 %
MATH_MAE_THRESHOLD  = 12.0   # baseline model card : 11.4542
MATH_RMSE_THRESHOLD = 15.1   # baseline model card : 14.3761
LANG_MAE_THRESHOLD  = 10.9   # baseline model card : 10.3101
LANG_RMSE_THRESHOLD = 14.0   # baseline model card : 13.2784


@pytest.mark.perf
def test_math_mae(artifacts, test_split):
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_math"].predict(X_imp)
    mae = mean_absolute_error(y_test_m, preds)
    assert mae < MATH_MAE_THRESHOLD, f"Math MAE {mae:.4f} >= seuil {MATH_MAE_THRESHOLD}"


@pytest.mark.perf
def test_math_rmse(artifacts, test_split):
    X_test, y_test_m, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_math"].predict(X_imp)
    rmse = np.sqrt(mean_squared_error(y_test_m, preds))
    assert rmse < MATH_RMSE_THRESHOLD, f"Math RMSE {rmse:.4f} >= seuil {MATH_RMSE_THRESHOLD}"


@pytest.mark.perf
def test_lang_mae(artifacts, test_split):
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_lang"].predict(X_imp)
    mae = mean_absolute_error(y_test_l, preds)
    assert mae < LANG_MAE_THRESHOLD, f"Language MAE {mae:.4f} >= seuil {LANG_MAE_THRESHOLD}"


@pytest.mark.perf
def test_lang_rmse(artifacts, test_split):
    X_test, _, y_test_l = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_lang"].predict(X_imp)
    rmse = np.sqrt(mean_squared_error(y_test_l, preds))
    assert rmse < LANG_RMSE_THRESHOLD, f"Language RMSE {rmse:.4f} >= seuil {LANG_RMSE_THRESHOLD}"
