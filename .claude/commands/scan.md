---
description: Scan all pharma stocks for bottom-buying signals. Shows technical, fundamental, ML prediction scores.
allowed-tools: Bash(source:*), Bash(PYTHONPATH=src python:*)
---

# /scan

Scan all stocks for bottom-buying signals.

## Usage

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pharma_stock.cli.main scan bottom
```

## For detail analysis

```bash
PYTHONPATH=src python -m pharma_stock.cli.main scan detail 4506.T
```

## Output columns

| Column | Description |
|--------|-------------|
| Score | Total score 0-100 |
| Tech | RSI, BB, 52w low |
| Fund | PER, PBR, dividend |
| Upside | Phase 3, analyst target |
| Pred | ML predicted high |
| Signal | Strong Buy / Buy / Neutral / Avoid |
