"""予測器の基底クラス"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Generic, TypeVar
import pickle

import numpy as np
import pandas as pd
import structlog

from pharma_stock.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class PredictionResult:
    """予測結果"""

    ticker: str
    prediction_date: date  # 予測を行った日
    target_date: date  # 予測対象日
    predicted_price: float
    predicted_change: float  # 変動率(%)
    confidence: float  # 信頼度 (0-1)
    prediction_interval_lower: float | None  # 予測区間下限
    prediction_interval_upper: float | None  # 予測区間上限
    model_name: str
    features_importance: dict[str, float] | None


ModelT = TypeVar("ModelT")


class BasePredictor(ABC, Generic[ModelT]):
    """予測器の基底クラス"""

    model_name: str = "base"

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger.bind(predictor=self.__class__.__name__)
        self.model: ModelT | None = None
        self.is_fitted: bool = False
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> "BasePredictor":
        """モデルを学習

        Args:
            X: 特徴量DataFrame
            y: ターゲット（将来の株価または変動率）

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        """予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測値の配列
        """
        pass

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """予測区間付きで予測

        Args:
            X: 特徴量DataFrame
            confidence_level: 信頼水準

        Returns:
            (予測値, 下限, 上限)のタプル
        """
        predictions = self.predict(X)
        # デフォルト実装：予測値の±10%を予測区間とする
        margin = np.abs(predictions) * 0.1
        lower = predictions - margin
        upper = predictions + margin
        return predictions, lower, upper

    def get_feature_importance(self) -> dict[str, float] | None:
        """特徴量重要度を取得"""
        return None

    def save(self, path: Path | str | None = None) -> Path:
        """モデルを保存

        Args:
            path: 保存先パス（Noneの場合はデフォルトパス）

        Returns:
            保存先パス
        """
        if path is None:
            self.settings.ensure_directories()
            path = self.settings.models_dir / f"{self.model_name}_model.pkl"
        else:
            path = Path(path)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "is_fitted": self.is_fitted,
                    "feature_names": self.feature_names,
                },
                f,
            )

        self.logger.info("model_saved", path=str(path))
        return path

    def load(self, path: Path | str | None = None) -> "BasePredictor":
        """モデルを読み込み

        Args:
            path: 読み込み元パス

        Returns:
            self
        """
        if path is None:
            path = self.settings.models_dir / f"{self.model_name}_model.pkl"
        else:
            path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.is_fitted = data["is_fitted"]
            self.feature_names = data["feature_names"]

        self.logger.info("model_loaded", path=str(path))
        return self

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
    ) -> dict[str, float]:
        """モデルを評価

        Args:
            X: 特徴量DataFrame
            y_true: 真のターゲット値

        Returns:
            評価指標の辞書
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.predict(X)

        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
