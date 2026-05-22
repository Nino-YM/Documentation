### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests de charge — UC6
Vérifient que le pipeline respecte les SLA de latence sous différents volumes.
"""

import time

import pandas as pd
import pytest

from utils import FEATURE_COLS, impute, run_if_pipeline

BATCH_1000_LIMIT    = 5.0  # secondes pour 1 000 prédictions (2 modèles)
NOMINAL_PRED_LIMIT  = 2.0  # secondes pour le dataset complet (prédictions seules)
FULL_PIPELINE_LIMIT = 5.0  # secondes pour le pipeline complet (IF + prédictions)


@pytest.mark.load
def test_batch_1000_latency(artifacts, dataset):
    """1 000 prédictions (math + language) en moins de BATCH_1000_LIMIT secondes."""
    X_raw = dataset[FEATURE_COLS]
    X = pd.concat([X_raw] * 2, ignore_index=True).iloc[:1000]
    X_imp = impute(X, artifacts["imputer"])

    t0 = time.time()
    artifacts["model_math"].predict(X_imp)
    artifacts["model_lang"].predict(X_imp)
    elapsed = time.time() - t0

    assert elapsed < BATCH_1000_LIMIT, (
        f"1 000 prédictions en {elapsed:.4f}s > limite {BATCH_1000_LIMIT}s"
    )


@pytest.mark.load
def test_nominal_predictions_latency(artifacts, dataset):
    """Dataset complet (~1 000 entrées) — prédictions seules en moins de NOMINAL_PRED_LIMIT secondes."""
    X_imp = impute(dataset[FEATURE_COLS].copy(), artifacts["imputer"])

    t0 = time.time()
    artifacts["model_math"].predict(X_imp)
    artifacts["model_lang"].predict(X_imp)
    elapsed = time.time() - t0

    assert elapsed < NOMINAL_PRED_LIMIT, (
        f"Dataset complet ({len(dataset)} entrées) en {elapsed:.4f}s > limite {NOMINAL_PRED_LIMIT}s"
    )


@pytest.mark.load
def test_full_pipeline_latency(artifacts, dataset):
    """Pipeline complet (imputation + IF + prédictions) sur le dataset entier en moins de FULL_PIPELINE_LIMIT secondes."""
    X = dataset[FEATURE_COLS].copy()

    t0 = time.time()
    X_imp = impute(X, artifacts["imputer"])
    run_if_pipeline(X_imp, artifacts["encoder"], artifacts["scaler"], artifacts["iso_forest"])
    artifacts["model_math"].predict(X_imp)
    artifacts["model_lang"].predict(X_imp)
    elapsed = time.time() - t0

    assert elapsed < FULL_PIPELINE_LIMIT, (
        f"Pipeline complet en {elapsed:.4f}s > limite {FULL_PIPELINE_LIMIT}s"
    )
