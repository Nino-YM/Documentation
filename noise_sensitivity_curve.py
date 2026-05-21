### Thafat KANEM & Yanis MORIN ###
###   EPSI NANTES 2026          ###

"""
noise_sensitivity_curve.py
--------------------------
Courbe de sensibilité au bruit (Noise Sensitivity Curve) pour les
3 features les plus corrélées aux cibles Math Score et Language Mean Score.

Adaptation pour features CATÉGORIELLES :
  Les features étant toutes catégorielles (genre, groupe ethnique, etc.),
  le bruit gaussien sur valeurs continues n'est pas applicable directement.
  On utilise l'équivalent catégoriel : pour un niveau de bruit x%, on remplace
  aléatoirement x% des valeurs de la feature par une catégorie valide tirée
  au hasard — ce qui simule des erreurs de saisie ou de collecte.

  La RMSE est calculée sur les prédictions des modèles CatBoost (modèles réels
  du projet), et répétée N_TRIALS fois pour moyenner la variabilité aléatoire.

Protocole :
  1. Chargement des données (même split que les notebooks d'entraînement)
  2. Chargement des modèles CatBoost entraînés
  3. Pour chaque feature parmi les 3 plus corrélées :
       Pour chaque niveau de bruit (1%, 3%, 5%, 10%, 15%, 20%) :
         Répéter N_TRIALS fois :
           - Remplacer aléatoirement x% des valeurs de la feature
           - Prédire et calculer la RMSE
         Retenir la variation de RMSE moyenne (%)
  4. Tracé des courbes de sensibilité (un graphe par cible)

3 features les plus corrélées (d'après matrice de corrélation) :
  1. lunch                   – |corr| = 0.351 avec math score
  2. test preparation course – |corr| = 0.281 avec language mean score
  3. gender                  – |corr| = 0.276 avec language mean score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
os.makedirs("artifacts", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Données et split (identique aux notebooks d'entraînement)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

# Top 3 features par corrélation absolue maximale avec les deux cibles
TOP3_FEATURES = ["lunch", "test preparation course", "gender"]

df     = pd.read_csv("data/StudentsPerformance_modified.csv")
X      = df[FEATURE_COLS]
y_math = df["math score"]
y_lang = df["language mean score"]

X_train, X_test, y_train_m, y_test_m = train_test_split(
    X, y_math, test_size=0.2, random_state=42
)
_, _, y_train_l, y_test_l = train_test_split(
    X, y_lang, test_size=0.2, random_state=42
)

print(f"Données d'entraînement : {len(X_train)} lignes")
print(f"Données de test        : {len(X_test)} lignes\n")

# ---------------------------------------------------------------------------
# 2. Chargement des modèles CatBoost entraînés
# ---------------------------------------------------------------------------
model_math = joblib.load("artifacts/catboost_math_score.joblib")
model_lang = joblib.load("artifacts/catboost_language_mean_score.joblib")

# RMSE de base sur le jeu de test (sans bruit)
rmse_m_clean = np.sqrt(mean_squared_error(y_test_m, model_math.predict(X_test)))
rmse_l_clean = np.sqrt(mean_squared_error(y_test_l, model_lang.predict(X_test)))

print(f"RMSE de base — Math Score    : {rmse_m_clean:.4f}")
print(f"RMSE de base — Language Score: {rmse_l_clean:.4f}")


# ---------------------------------------------------------------------------
# 3. Fonction d'évaluation de l'impact du bruit catégoriel
# ---------------------------------------------------------------------------
N_TRIALS = 10  # répétitions pour stabiliser l'estimation stochastique

def evaluate_noise_impact(model, X_df, y_true, feature_name, noise_level, rmse_clean,
                          n_trials=N_TRIALS):
    """
    Équivalent catégoriel du bruit gaussien sur feature numérique :
    remplace aléatoirement noise_level% des valeurs de feature_name
    par une catégorie valide tirée au hasard, répète n_trials fois
    et retourne la variation de RMSE moyenne (%).

    Parameters
    ----------
    noise_level : % des observations à perturber (1 → 1%, 20 → 20%)
    """
    valid_categories = X_df[feature_name].unique().tolist()
    n          = len(X_df)
    n_perturb  = max(1, int(n * noise_level / 100))
    col_idx    = X_df.columns.get_loc(feature_name)

    trial_variations = []
    for _ in range(n_trials):
        X_noisy = X_df.copy()
        idx_perturb = np.random.choice(n, size=n_perturb, replace=False)
        X_noisy.iloc[idx_perturb, col_idx] = np.random.choice(
            valid_categories, size=n_perturb, replace=True
        )
        preds      = model.predict(X_noisy)
        rmse_noisy = np.sqrt(mean_squared_error(y_true, preds))
        trial_variations.append(100.0 * (rmse_noisy - rmse_clean) / rmse_clean)

    return float(np.mean(trial_variations))


# ---------------------------------------------------------------------------
# 4. Calcul des courbes pour les 3 features × 2 cibles
# ---------------------------------------------------------------------------
intensities = np.array([1, 3, 5, 10, 15, 20])

results_math = {}
results_lang = {}

print("\nCalcul des courbes de sensibilité au bruit...")
for feat in TOP3_FEATURES:
    print(f"  -> {feat}")
    results_math[feat] = [
        evaluate_noise_impact(model_math, X_test, y_test_m, feat, n, rmse_m_clean)
        for n in intensities
    ]
    results_lang[feat] = [
        evaluate_noise_impact(model_lang, X_test, y_test_l, feat, n, rmse_l_clean)
        for n in intensities
    ]

# Affichage tabulaire
col_w = 24
for target_label, results, rmse_base in [
    ("Math Score", results_math, rmse_m_clean),
    ("Language Mean Score", results_lang, rmse_l_clean),
]:
    print(f"\n=== Variation RMSE (%) — {target_label} (base : {rmse_base:.4f}) ===")
    header = f"{'Bruit':>7}"
    for feat in TOP3_FEATURES:
        header += f"  {feat[:col_w-2]:<{col_w-2}}"
    print(header)
    print("-" * (7 + (col_w + 2) * len(TOP3_FEATURES)))
    for i, lvl in enumerate(intensities):
        row = f"{lvl:>5}%  "
        for feat in TOP3_FEATURES:
            val = results[feat][i]
            row += f"  {val:>+{col_w-4}.2f}%{' ' * 2}"
        print(row)


# ---------------------------------------------------------------------------
# 5. Visualisation — Noise Sensitivity Curve
# ---------------------------------------------------------------------------
colors  = ["steelblue", "tomato", "seagreen"]
markers = ["o-", "s-", "^-"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Noise Sensitivity Curve — 3 Features les plus corrélées\n"
    "(CatBoost | Bruit = % d'observations avec catégorie remplacée aléatoirement)",
    fontsize=11,
)

for i, feat in enumerate(TOP3_FEATURES):
    axes[0].plot(intensities, results_math[feat],
                 markers[i], color=colors[i], label=feat, lw=2, markersize=7)
axes[0].axhline(0, color="gray", linestyle=":", lw=1)
axes[0].set_title("Math Score")
axes[0].set_xlabel("Niveau de bruit (% des observations dont la catégorie est remplacée)")
axes[0].set_ylabel("Variation de la RMSE (%)")
axes[0].legend()
axes[0].grid(True, alpha=0.4)

for i, feat in enumerate(TOP3_FEATURES):
    axes[1].plot(intensities, results_lang[feat],
                 markers[i], color=colors[i], label=feat, lw=2, markersize=7)
axes[1].axhline(0, color="gray", linestyle=":", lw=1)
axes[1].set_title("Language Mean Score")
axes[1].set_xlabel("Niveau de bruit (% des observations dont la catégorie est remplacée)")
axes[1].set_ylabel("Variation de la RMSE (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("artifacts/noise_sensitivity_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGraphique sauvegardé : artifacts/noise_sensitivity_curve.png")
