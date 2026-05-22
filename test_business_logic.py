### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
Tests de logique métier — UC1.3–1.5, UC2.3–2.4, UC3.1
Vérifient que les relations connues entre features et cibles sont respectées par le modèle.
  - lunch standard > free/reduced  →  math score plus élevé (corr = 0.35)
  - test prep completed > none     →  language et math score plus élevés
  - female > male en moyenne       →  language mean score plus élevé (corr abs = 0.28)
  - profil typique (modes)         →  dans la zone de robustesse IF
"""

import numpy as np
import pytest

from utils import FEATURE_COLS, impute, make_profile, run_if_pipeline


@pytest.mark.business
def test_standard_lunch_higher_math_score(artifacts):
    """Standard lunch prédit un math score plus élevé que free/reduced (corr = 0.35)."""
    pred_std = artifacts["model_math"].predict(impute(make_profile(lunch="standard"),      artifacts["imputer"]))[0]
    pred_red = artifacts["model_math"].predict(impute(make_profile(lunch="free/reduced"),   artifacts["imputer"]))[0]
    assert pred_std > pred_red, (
        f"Lunch standard ({pred_std:.2f}) devrait > free/reduced ({pred_red:.2f})"
    )


@pytest.mark.business
def test_completed_prep_higher_math_score(artifacts):
    """Test prep 'completed' prédit un math score plus élevé que 'none'."""
    pred_comp = artifacts["model_math"].predict(
        impute(make_profile(**{"test preparation course": "completed"}), artifacts["imputer"])
    )[0]
    pred_none = artifacts["model_math"].predict(
        impute(make_profile(**{"test preparation course": "none"}), artifacts["imputer"])
    )[0]
    assert pred_comp > pred_none, (
        f"Test prep completed ({pred_comp:.2f}) devrait > none ({pred_none:.2f})"
    )


@pytest.mark.business
def test_completed_prep_higher_lang_score(artifacts):
    """Test prep 'completed' prédit un language score plus élevé que 'none' (corr = 0.28)."""
    pred_comp = artifacts["model_lang"].predict(
        impute(make_profile(**{"test preparation course": "completed"}), artifacts["imputer"])
    )[0]
    pred_none = artifacts["model_lang"].predict(
        impute(make_profile(**{"test preparation course": "none"}), artifacts["imputer"])
    )[0]
    assert pred_comp > pred_none, (
        f"Test prep completed ({pred_comp:.2f}) devrait > none ({pred_none:.2f})"
    )


@pytest.mark.business
def test_average_standard_lunch_higher_math(artifacts, dataset):
    """En moyenne sur le dataset entier, standard lunch → math score prédit plus élevé."""
    X = dataset[FEATURE_COLS].copy()
    X_imp = impute(X, artifacts["imputer"])
    preds = artifacts["model_math"].predict(X_imp)
    mask_std = (dataset["lunch"] == "standard").values
    mask_red = (dataset["lunch"] == "free/reduced").values
    mean_std = preds[mask_std].mean()
    mean_red = preds[mask_red].mean()
    assert mean_std > mean_red, (
        f"Moy. standard ({mean_std:.2f}) devrait > free/reduced ({mean_red:.2f})"
    )


@pytest.mark.business
def test_average_female_higher_lang_score(artifacts, dataset):
    """En moyenne sur le dataset entier, les femmes ont un language mean score prédit plus élevé."""
    X = dataset[FEATURE_COLS].copy()
    X_imp = impute(X, artifacts["imputer"])
    preds = artifacts["model_lang"].predict(X_imp)
    mask_f = (dataset["gender"] == "female").values
    mask_m = (dataset["gender"] == "male").values
    mean_f = preds[mask_f].mean()
    mean_m = preds[mask_m].mean()
    assert mean_f > mean_m, (
        f"Moy. female ({mean_f:.2f}) devrait > male ({mean_m:.2f})"
    )


@pytest.mark.business
def test_typical_profile_in_robustness_zone(artifacts, typical_profile):
    """Un profil typique (mode de chaque feature) doit être dans la zone de robustesse IF."""
    X_imp = impute(typical_profile, artifacts["imputer"])
    scores, _ = run_if_pipeline(X_imp, artifacts["encoder"], artifacts["scaler"], artifacts["iso_forest"])
    threshold = artifacts["config"]["robustness_threshold"]
    assert scores[0] >= threshold, (
        f"Profil typique hors zone de robustesse : score={scores[0]:.4f} < seuil={threshold:.4f}"
    )
