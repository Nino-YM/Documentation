### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

import json
import os

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from utils import FEATURE_COLS, make_profile

ARTIFACTS_DIR = "artifacts"

# Noms des fichiers du second modèle — à adapter selon le modèle choisi
MODEL2_MATH_PATH = os.path.join(ARTIFACTS_DIR, "model2_math_score.joblib")
MODEL2_LANG_PATH = os.path.join(ARTIFACTS_DIR, "model2_language_mean_score.joblib")


@pytest.fixture(scope="session")
def model2_math():
    """Second modèle Math Score. Skippé automatiquement s'il n'est pas encore exporté."""
    if not os.path.exists(MODEL2_MATH_PATH):
        pytest.skip(f"Second modèle math introuvable : {MODEL2_MATH_PATH}")
    return joblib.load(MODEL2_MATH_PATH)


@pytest.fixture(scope="session")
def model2_lang():
    """Second modèle Language Mean Score. Skippé automatiquement s'il n'est pas encore exporté."""
    if not os.path.exists(MODEL2_LANG_PATH):
        pytest.skip(f"Second modèle language introuvable : {MODEL2_LANG_PATH}")
    return joblib.load(MODEL2_LANG_PATH)


@pytest.fixture(scope="session")
def artifacts():
    arts = {
        "imputer":    joblib.load(os.path.join(ARTIFACTS_DIR, "imputer.joblib")),
        "encoder":    joblib.load(os.path.join(ARTIFACTS_DIR, "ordinal_encoder.joblib")),
        "scaler":     joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_if.joblib")),
        "iso_forest": joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")),
        "model_math": joblib.load(os.path.join(ARTIFACTS_DIR, "catboost_math_score.joblib")),
        "model_lang": joblib.load(os.path.join(ARTIFACTS_DIR, "catboost_language_mean_score.joblib")),
    }
    with open(os.path.join(ARTIFACTS_DIR, "robustness_config.json"), encoding="utf-8") as f:
        arts["config"] = json.load(f)
    return arts


@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv("data/StudentsPerformance_modified.csv")


@pytest.fixture(scope="session")
def test_split(dataset):
    """Reproduit le split exact des notebooks (random_state=42, 20%)."""
    X = dataset[FEATURE_COLS]
    y_math = dataset["math score"]
    y_lang = dataset["language mean score"]
    _, X_test, _, y_test_m = train_test_split(X, y_math, test_size=0.2, random_state=42)
    _, _, _, y_test_l = train_test_split(X, y_lang, test_size=0.2, random_state=42)
    return (
        X_test.reset_index(drop=True),
        y_test_m.reset_index(drop=True),
        y_test_l.reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def typical_profile():
    """Profil typique : mode de chaque feature catégorielle."""
    return make_profile()
