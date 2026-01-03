# /backtest - バックテストスキル

底値シグナル・価格予測の精度を過去データで検証します。

## 使用例

```
/backtest 4506.T 2025-04-09
```

## 実行手順

1. 指定銘柄・日付でバックテストを実行
2. シグナル、予測値、実績を比較
3. 結果をテーブル形式で表示

## コード

```python
from datetime import date
from pharma_stock.dev import quick_backtest

# 引数をパース
ticker = "$ARGS[0]" if len("$ARGS") > 0 else "4506.T"
test_date_str = "$ARGS[1]" if len("$ARGS") > 1 else "2025-04-09"

# 日付をパース
parts = test_date_str.split("-")
test_date = date(int(parts[0]), int(parts[1]), int(parts[2]))

# バックテスト実行
quick_backtest(ticker, test_date)
```
