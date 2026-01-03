---
description: Add new data collector for external data sources.
allowed-tools: Read, Write, Edit, Bash(PYTHONPATH=src python:*)
argument-hint: <collector-name>
---

# /add-collector

Add a new data collector.

## Target directory

`src/pharma_stock/collectors/`

## Implementation steps

1. Create new file in `src/pharma_stock/collectors/`
2. Inherit from `BaseCollector`
3. Implement `collect()` method
4. Add export to `__init__.py`
5. Create test in `tests/collectors/`

## Template

```python
from typing import Any
import structlog
from .base import BaseCollector

logger = structlog.get_logger()

class NewCollector(BaseCollector):
    def collect(
        self,
        tickers: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        logger.info("collection_started", collector=self.__class__.__name__)
        results = []
        # Implementation here
        logger.info("collection_completed", count=len(results))
        return results
```

## Update __init__.py

```python
from .new_collector import NewCollector
__all__ = [..., "NewCollector"]
```
