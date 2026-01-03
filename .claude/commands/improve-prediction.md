---
description: Improve ML price prediction model accuracy.
allowed-tools: Read, Edit, Bash(PYTHONPATH=src python:*)
argument-hint: <improvement-type>
---

# /improve-prediction

Improve the ML prediction model.

## Target files

- `src/pharma_stock/prediction/price_range_predictor.py`
- `src/pharma_stock/prediction/features.py`

## Improvement approaches

### 1. Add features

```python
# In _add_technical_features():
df['new_feature'] = ...
```

### 2. Tune hyperparameters

```python
model_params = {
    'n_estimators': 100,  # Try 200
    'max_depth': 5,       # Try 7
    'learning_rate': 0.1, # Try 0.05
}
```

### 3. Ensemble methods

```python
predictions = [
    xgboost_model.predict(X),
    lightgbm_model.predict(X),
]
ensemble_pred = np.mean(predictions, axis=0)
```

## Validation

```python
from pharma_stock.dev import BacktestRunner
from datetime import date

runner = BacktestRunner()
results = runner.run_all_companies(date(2025, 4, 1))
report = runner.generate_report(results)
print(f"MAE 3M: {report['prediction_accuracy']['mean_error_3m']:.1%}")
print(f"MAE 6M: {report['prediction_accuracy']['mean_error_6m']:.1%}")
```
