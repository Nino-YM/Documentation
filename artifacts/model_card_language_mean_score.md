# Model Card — CatBoost Language Mean Score Predictor

## Overview
- **Task**: Regression (predict 'language mean score')
- **Target definition**: '(reading score + writing score) / 2'
- **Model**: CatBoostRegressor
- **Excluded features**: reading score, writing score, math score, language mean score
- **Categorical features**: ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

## Data & preprocessing
- Dataset: 'data/StudentsPerformance_modified.csv'
- Categorical handling: CatBoost native categorical support (no one-hot)

## Training procedure
- Train/Test split: 80/20 (random_state=42)
- Cross-validation: 5-fold KFold (shuffle=True)
- Hyperparameter tuning: Bayesian Optimization (BayesSearchCV, scoring=R²)

## Performance (Test set)
- **Best CV R²**: 0.2571
- **Test R²**: 0.2256
- **MAE**: 10.3101
- **RMSE**: 13.2784
- **Mean target**: 67.0350

## Best hyperparameters
```json
{
  "depth": 3,
  "iterations": 300,
  "l2_leaf_reg": 20.0,
  "learning_rate": 0.028194779313311276,
  "random_strength": 0.0,
  "subsample": 0.5
}
```

---

## Tests de robustesse (TP - Réalisation et analyse des tests)

### Pipeline d'inférence complet
Toute nouvelle prédiction doit suivre le pipeline suivant (voir `predict_new_entry.py`) :

```
CSV entrant  →  Imputation (most_frequent)
            →  OrdinalEncoder  →  StandardScaler
            →  Isolation Forest (score d'anomalie + zone de confiance)
            →  CatBoostRegressor (prédiction Language Mean Score)
```

### Artefacts associés
| Artefact | Fichier | Rôle |
|---|---|---|
| Modèle | `catboost_language_mean_score.joblib` | Prédiction Language Mean Score |
| Imputer | `imputer.joblib` | Imputation most_frequent (features catégorielles) |
| Encoder | `ordinal_encoder.joblib` | Encodage ordinal pour l'Isolation Forest |
| Scaler | `scaler_if.joblib` | Normalisation StandardScaler pour l'Isolation Forest |
| Isolation Forest | `isolation_forest.joblib` | Détection d'anomalies, zone de confiance |
| Configuration | `robustness_config.json` | Seuil de robustesse et zones |

*L'imputer, l'encoder, le scaler et l'Isolation Forest sont **partagés** entre les deux modèles (mêmes features d'entrée).*

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
| gender | Très sensible | Corrélation forte avec language score (-0.28) |
| race/ethnicity | Faiblement impacté | Corrélation limitée (0.16) |
| parental level of education | Faiblement impacté | Corrélation faible (-0.08) |
| lunch | Sensible | Corrélation significative (0.24) |
| test preparation course | Sensible | Plus forte corrélation pour language (0.28) |

→ Graphiques détaillés : `artifacts/imputation_test_language.png`

---

### Conclusion sur la robustesse

Le modèle Language Mean Score présente une **robustesse satisfaisante** face aux données manquantes lorsque l'imputation `most_frequent` est appliquée :

- L'imputation par mode sur des features catégorielles conserve une distribution proche de celle vue à l'entraînement
- La feature `gender` est la plus critique pour ce modèle (corrélation -0.28) : son imputation systématique par "female" peut introduire un biais de genre dans les prédictions
- L'Isolation Forest détecte les profils d'élèves inhabituels et informe sur le niveau de confiance de la prédiction

**Points d'attention :**
- Le R² de 0.23 indique une capacité prédictive limitée uniquement à partir des facteurs socio-éducatifs
- En cas de fort taux de manquants sur `gender` ou `test preparation course`, il est recommandé de signaler la prédiction comme peu fiable
- Le biais de genre potentiel (genre majoritairement imputé à "female") doit être pris en compte dans les décisions métier

**Limites connues :**
- L'imputation par `most_frequent` sur `gender` remplace systématiquement par "female" (valeur la plus fréquente), ce qui peut biaiser les prédictions pour des élèves masculins dont le genre est manquant
