### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests unitaires — UC1.1, UC2.1, UC3.2
Vérifient la forme, le type et la plage des sorties du pipeline.
"""

import numpy as np
import pytest

from utils import impute, run_if_pipeline


@pytest.mark.unit
def test_math_prediction_shape(artifacts, test_split):
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_math"].predict(X_imp)
    assert len(preds) == len(X_test)


@pytest.mark.unit
def test_lang_prediction_shape(artifacts, test_split):
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_lang"].predict(X_imp)
    assert len(preds) == len(X_test)


@pytest.mark.unit
def test_math_predictions_in_range(artifacts, test_split):
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_math"].predict(X_imp)
    assert np.all(preds >= 0) and np.all(preds <= 100), (
        f"Prédictions hors [0, 100] : min={preds.min():.2f}, max={preds.max():.2f}"
    )


@pytest.mark.unit
def test_lang_predictions_in_range(artifacts, test_split):
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    preds = artifacts["model_lang"].predict(X_imp)
    assert np.all(preds >= 0) and np.all(preds <= 100), (
        f"Prédictions hors [0, 100] : min={preds.min():.2f}, max={preds.max():.2f}"
    )


@pytest.mark.unit
def test_isolation_forest_scores_in_range(artifacts, test_split):
    X_test, _, _ = test_split
    X_imp = impute(X_test, artifacts["imputer"])
    scores, _ = run_if_pipeline(X_imp, artifacts["encoder"], artifacts["scaler"], artifacts["iso_forest"])
    assert np.all(scores >= -1) and np.all(scores <= 0), (
        f"Scores IF hors [-1, 0] : min={scores.min():.4f}, max={scores.max():.4f}"
    )


@pytest.mark.unit
def test_pipeline_single_entry_math(artifacts, typical_profile):
    X_imp = impute(typical_profile, artifacts["imputer"])
    pred = artifacts["model_math"].predict(X_imp)
    assert len(pred) == 1
    assert 0 <= pred[0] <= 100


@pytest.mark.unit
def test_pipeline_single_entry_lang(artifacts, typical_profile):
    X_imp = impute(typical_profile, artifacts["imputer"])
    pred = artifacts["model_lang"].predict(X_imp)
    assert len(pred) == 1
    assert 0 <= pred[0] <= 100
