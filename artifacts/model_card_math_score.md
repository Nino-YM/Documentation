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