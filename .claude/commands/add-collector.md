# /add-collector - データコレクター追加タスク

新しいデータソースからデータを収集するコレクターを追加します。

## 使用例

```
/add-collector 決算短信
/add-collector IRニュース
```

## 実装手順

1. `src/pharma_stock/collectors/` に新しいファイルを作成
2. `BaseCollector` を継承
3. `collect()` メソッドを実装
4. `__init__.py` にエクスポートを追加
5. テストを作成
6. 動作確認

## テンプレート

```python
\"\"\"新しいコレクター\"\"\"

from typing import Any
import structlog

from .base import BaseCollector

logger = structlog.get_logger()


class NewCollector(BaseCollector):
    \"\"\"新しいデータを収集するコレクター\"\"\"

    def __init__(self):
        super().__init__()

    def collect(
        self,
        tickers: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        \"\"\"データを収集

        Args:
            tickers: 対象ティッカーリスト

        Returns:
            収集したデータのリスト
        \"\"\"
        logger.info("collection_started", collector=self.__class__.__name__)

        results = []

        for ticker in tickers or []:
            try:
                # データ収集ロジック
                data = self._fetch_data(ticker)
                results.extend(data)
            except Exception as e:
                logger.error("collection_failed", ticker=ticker, error=str(e))

        logger.info("collection_completed", count=len(results))
        return results

    def _fetch_data(self, ticker: str) -> list[Any]:
        \"\"\"個別銘柄のデータを取得\"\"\"
        # 実装
        pass
```

## __init__.py への追加

```python
from .new_collector import NewCollector

__all__ = [
    # ... 既存のエクスポート
    "NewCollector",
]
```
