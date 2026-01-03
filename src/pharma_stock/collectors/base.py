"""データ収集の基底クラス"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Generic, TypeVar

import structlog

from pharma_stock.config import get_settings

T = TypeVar("T")
logger = structlog.get_logger(__name__)


class BaseCollector(ABC, Generic[T]):
    """データ収集の基底クラス"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger.bind(collector=self.__class__.__name__)

    @abstractmethod
    def collect(self, **kwargs: Any) -> list[T]:
        """データを収集する

        Returns:
            収集したデータのリスト
        """
        pass

    def _log_start(self, **context: Any) -> None:
        """収集開始をログ"""
        self.logger.info("collection_started", **context)

    def _log_complete(self, count: int, **context: Any) -> None:
        """収集完了をログ"""
        self.logger.info("collection_completed", count=count, **context)

    def _log_error(self, error: Exception, **context: Any) -> None:
        """エラーをログ"""
        self.logger.error("collection_error", error=str(error), **context)
