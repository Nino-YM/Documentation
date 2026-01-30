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
'''json
{
  "depth": 3,
  "iterations": 300,
  "l2_leaf_reg": 20.0,
  "learning_rate": 0.028194779313311276,
  "random_strength": 0.0,
  "subsample": 0.5
}