# /add-indicator - テクニカル指標追加タスク

底値シグナルに新しいテクニカル指標を追加します。

## 使用例

```
/add-indicator MACD
/add-indicator 一目均衡表
```

## 実装手順

1. `src/pharma_stock/analysis/bottom_signal.py` を開く
2. `_calc_technical_score()` メソッドに新しい指標を追加
3. taライブラリを使用して計算
4. スコア計算（0-100に正規化）
5. 理由リストに追加
6. バックテストで効果検証

## 変更対象ファイル

- `src/pharma_stock/analysis/bottom_signal.py`

## taライブラリの使用例

```python
import ta

# MACD
macd = ta.trend.macd_diff(df['close'])

# 一目均衡表
ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], df['close'])
tenkan = ichimoku.ichimoku_conversion_line()
kijun = ichimoku.ichimoku_base_line()

# ストキャスティクス
stoch = ta.momentum.stoch(df['high'], df['low'], df['close'])
```

## スコア計算例

```python
# MACDがシグナルを上抜け（ゴールデンクロス）
if macd.iloc[-1] > 0 and macd.iloc[-2] <= 0:
    score += 15
    reasons.append("MACDゴールデンクロス")
```
