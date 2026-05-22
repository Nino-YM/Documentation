### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]


def impute(X: pd.DataFrame, imputer) -> pd.DataFrame:
    """Applique l'imputer sklearn et retourne un DataFrame avec des valeurs string."""
    arr = imputer.transform(X)
    X_imp = pd.DataFrame(arr, columns=FEATURE_COLS)
    for col in FEATURE_COLS:
        X_imp[col] = X_imp[col].astype(str)
    return X_imp


def run_if_pipeline(X_imp: pd.DataFrame, encoder, scaler, iso_forest):
    """Encodage ordinal + normalisation + scores Isolation Forest."""
    X_enc = encoder.transform(X_imp)
    X_norm = scaler.transform(X_enc)
    scores = iso_forest.score_samples(X_norm)
    labels = iso_forest.predict(X_norm)
    return scores, labels


def make_profile(**overrides) -> pd.DataFrame:
    """Crée un profil d'élève à partir du profil typique (mode), en surchargeant les valeurs voulues."""
    base = {
        "gender": "female",
        "race/ethnicity": "group C",
        "parental level of education": "some college",
        "lunch": "standard",
        "test preparation course": "none",
    }
    return pd.DataFrame([{**base, **overrides}])
