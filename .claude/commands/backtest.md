---
description: Run backtest to validate signal and prediction accuracy against historical data.
allowed-tools: Bash(source:*), Bash(PYTHONPATH=src python:*)
argument-hint: <ticker> <date>
---

# /backtest

Validate signal accuracy with historical data.

## Usage

```bash
source .venv/bin/activate
PYTHONPATH=src python -c "
from datetime import date
from pharma_stock.dev import quick_backtest
quick_backtest('4506.T', date(2025, 4, 9))
"
```

## Arguments

- `ticker`: Stock ticker (e.g., 4506.T)
- `date`: Test date in YYYY-MM-DD format

## Output

- Signal strength at that date
- Predicted vs actual high/low prices
- Prediction error percentage
