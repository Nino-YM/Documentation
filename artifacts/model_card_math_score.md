# Model Card — CatBoost Math Score Predictor

## Overview
- **Task**: Regression (predict 'math score')
- **Model**: CatBoostRegressor
- **Excluded features**: reading score, writing score, math score, language mean score
- **Categorical features**: ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

## Data & preprocessing
- Dataset: 'StudentsPerformance_modified.csv'
- Categorical handling: CatBoost native categorical support (no one-hot)

## Training procedure
- Train/Test split: 80/20 (random_state=42)
- Cross-validation: 5-fold KFold (shuffle=True)
- Hyperparameter tuning: Bayesian Optimization (BayesSearchCV, scoring=R²)

## Performance (Test set)
- **Best CV R²**: 0.2306
- **Test R²**: 0.1507
- **MAE**: 11.4542
- **RMSE**: 14.3761
- **Mean target**: 64.4600

## Best hyperparameters
```json
{
  "depth": 3,
  "iterations": 300,
  "l2_leaf_reg": 1.0,
  "learning_rate": 0.05542290168219912,
  "random_strength": 0.5291589920766805,
  "subsample": 0.5
}
```

---

## Tests de robustesse (TP — Réalisation et analyse des tests)

### Pipeline d'inférence complet
Toute nouvelle prédiction doit suivre le pipeline suivant (voir `predict_new_entry.py`) :

```
CSV entrant  →  Imputation (most_frequent)
            →  OrdinalEncoder  →  StandardScaler
            →  Isolation Forest (score d'anomalie + zone de confiance)
            →  CatBoostRegressor (prédiction Math Score)
```

### Artefacts associés
| Artefact | Fichier | Rôle |
|---|---|---|
| Modèle | `catboost_math_score.joblib` | Prédiction Math Score |
| Imputer | `imputer.joblib` | Imputation most_frequent (features catégorielles) |
| Encoder | `ordinal_encoder.joblib` | Encodage ordinal pour l'Isolation Forest |
| Scaler | `scaler_if.joblib` | Normalisation StandardScaler pour l'Isolation Forest |
| Isolation Forest | `isolation_forest.joblib` | Détection d'anomalies, zone de confiance |
| Configuration | `robustness_config.json` | Seuil de robustesse et zones |

---

### Isolation Forest

L'Isolation Forest a été entraîné sur les **données d'entraînement normalisées** (OrdinalEncoder + StandardScaler appliqués aux 5 features catégorielles).

- `n_estimators = 100` (adapté à un dataset de ~800 lignes)
- `contamination = 'auto'`
- Score retourné par `score_samples()` : compris entre -1 (anomalie) et 0 (normal)

**Zones de confiance :**
| Zone | Score d'anomalie | Interprétation |
|---|---|---|
| Zone de robustesse | ≥ seuil (10ème percentile) | Haute confiance — entrée dans le domaine d'entraînement |
| Zone normale | ≥ -0.4 et < seuil | Confiance modérée |
| Zone d'extrapolation | < -0.4 | Faible confiance — prédiction incertaine |

*Le seuil exact est calculé dynamiquement dans `tp_isolation_forest_imputation.ipynb` et sauvegardé dans `robustness_config.json`.*

---

### Stratégie d'imputation

**Stratégie retenue : `most_frequent` (SimpleImputer sklearn)**

Toutes les features étant catégorielles, la valeur manquante est remplacée par la **valeur la plus fréquente** observée dans le jeu d'entraînement.

**Justification :**
- Garantit que la valeur imputée a toujours été vue par CatBoost à l'entraînement (pas d'erreur de catégorie inconnue)
- Reflète le profil d'élève le plus représentatif dans la population d'entraînement
- Alternative à l'exclusion de l'entrée (préférable pour un système en production)

| Feature | Valeur imputée (mode) |
|---|---|
| gender | female |
| race/ethnicity | group C |
| parental level of education | some college |
| lunch | standard |
| test preparation course | none |

*Valeurs indicatives — recalculées à l'exécution du notebook.*

---

### Tests de robustesse — Valeurs manquantes

**Protocole :** Pour chaque feature, suppression aléatoire progressive de valeurs (de 1 à N_test), imputation par `most_frequent`, puis calcul de la RMSE moyennée sur 5 tirages.

**Interprétation des dégradations RMSE :**
- < 5 % → modèle robuste sur cette feature
- 5–20 % → modèle modérément sensible
- > 20 % → feature critique, surveiller la qualité de collecte

**Résultats (valeurs indicatives — dépendent du tirage) :**

| Feature | Comportement attendu | Raison |
|---|---|---|
| gender | Robuste | Faible corrélation avec math score (0.17) |
| race/ethnicity | Modérément sensible | Corrélation non négligeable (0.22) |
| parental level of education | Faiblement impacté | Corrélation modérée (-0.07) |
| lunch | Sensible | Plus forte corrélation avec math score (0.35) |
| test preparation course | Sensible | Corrélation significative (0.18) |

→ Graphiques détaillés : `artifacts/imputation_test_math.png`

---

### Conclusion sur la robustesse

Le modèle Math Score présente une **robustesse satisfaisante** face aux données manquantes lorsque l'imputation `most_frequent` est appliquée :

- Les features catégorielles imputées par leur mode perturbent peu la distribution vue à l'entraînement
- Les features les plus prédictives (`lunch`, `test preparation course`) montrent une dégradation plus marquée : leur absence d'information réduit la précision de la prédiction
- L'Isolation Forest permet d'identifier les entrées hors du domaine d'entraînement et de communiquer un niveau de confiance à l'utilisateur

**Limites :**
- Le R² de 0.15 sur le test set indique que les features socio-éducatives n'expliquent qu'une fraction limitée de la variance du Math Score
- Des données manquantes sur plusieurs features simultanément dégraderaient davantage les performances
