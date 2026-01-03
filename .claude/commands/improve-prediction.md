# /improve-prediction - 予測モデル改善タスク

ML価格予測モデルの精度を改善します。

## 使用例

```
/improve-prediction 特徴量追加
/improve-prediction ハイパーパラメータ調整
```

## 改善アプローチ

### 1. 特徴量追加

```python
# src/pharma_stock/prediction/price_range_predictor.py

def _add_technical_features(self, df):
    # 既存の特徴量...

    # 新しい特徴量を追加
    df['feature_name'] = ...

    return df
```

### 2. ハイパーパラメータ調整

```python
model_params = {
    'n_estimators': 100,  # 木の数
    'max_depth': 5,       # 木の深さ
    'learning_rate': 0.1, # 学習率
    'random_state': 42,
}
```

### 3. アンサンブル強化

```python
# 複数モデルの予測を組み合わせ
predictions = [
    xgboost_model.predict(X),
    lightgbm_model.predict(X),
    ridge_model.predict(X),
]
ensemble_pred = np.mean(predictions, axis=0)
```

## バックテストで検証

```python
from pharma_stock.dev import BacktestRunner

runner = BacktestRunner()
results = runner.run_all_companies(date(2025, 4, 1))
report = runner.generate_report(results)

print(f"予測誤差(3M): {report['prediction_accuracy']['mean_error_3m']:.1%}")
print(f"予測誤差(6M): {report['prediction_accuracy']['mean_error_6m']:.1%}")
```

## 変更対象ファイル

- `src/pharma_stock/prediction/price_range_predictor.py`
- `src/pharma_stock/prediction/features.py`
