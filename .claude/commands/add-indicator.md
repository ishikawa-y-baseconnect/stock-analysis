---
description: Add new technical indicator to bottom signal detection system.
allowed-tools: Read, Edit, Bash(PYTHONPATH=src python:*)
argument-hint: <indicator-name>
---

# /add-indicator

Add a new technical indicator to the signal system.

## Target file

`src/pharma_stock/analysis/bottom_signal.py`

## Implementation steps

1. Open `_calc_technical_score()` method
2. Add indicator calculation using `ta` library
3. Add score logic (normalize to 0-100)
4. Add reason to reasons list
5. Run backtest to validate

## Example: Adding MACD

```python
import ta

# In _calc_technical_score():
macd = ta.trend.macd_diff(df['close'])
if macd.iloc[-1] > 0 and macd.iloc[-2] <= 0:
    score += 15
    reasons.append("MACD golden cross")
```

## Available ta indicators

- `ta.trend.macd_diff()` - MACD
- `ta.momentum.stoch()` - Stochastic
- `ta.trend.adx()` - ADX
- `ta.volatility.average_true_range()` - ATR
