"""アンサンブル予測モデル"""

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePredictor, PredictionResult
from .xgboost_predictor import XGBoostPredictor
from .lightgbm_predictor import LightGBMPredictor


class EnsemblePredictor(BasePredictor[list[BasePredictor]]):
    """アンサンブル予測モデル

    複数のモデルの予測を組み合わせて、より安定した予測を行う
    """

    model_name: str = "ensemble"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
    ):
        """
        Args:
            weights: モデルごとの重み（Noneの場合は均等重み）
            use_xgboost: XGBoostを使用するか
            use_lightgbm: LightGBMを使用するか
        """
        super().__init__()
        self.weights = weights
        self.use_xgboost = use_xgboost
        self.use_lightgbm = use_lightgbm
        self.models: dict[str, BasePredictor] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> "EnsemblePredictor":
        """全モデルを学習"""
        self.feature_names = list(X.columns)

        if self.use_xgboost:
            xgb_model = XGBoostPredictor()
            xgb_model.fit(X, y, **kwargs)
            self.models["xgboost"] = xgb_model

        if self.use_lightgbm:
            lgb_model = LightGBMPredictor()
            lgb_model.fit(X, y, **kwargs)
            self.models["lightgbm"] = lgb_model

        # 重みが指定されていない場合は均等重み
        if self.weights is None:
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models}

        self.is_fitted = True
        self.logger.info(
            "ensemble_fitted",
            models=list(self.models.keys()),
            weights=self.weights,
        )

        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """アンサンブル予測を実行"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        weights = []

        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights.get(name, 1.0))

        # 重み付き平均
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 正規化

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """予測区間付きで予測

        各モデルの予測のばらつきから予測区間を推定
        """
        predictions = []

        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # アンサンブル予測
        mean_pred = np.mean(predictions, axis=0)

        # モデル間のばらつきから予測区間を推定
        std_pred = np.std(predictions, axis=0)

        z_score = 1.96 if confidence_level == 0.95 else 2.576
        margin = z_score * std_pred

        lower = mean_pred - margin
        upper = mean_pred + margin

        return mean_pred, lower, upper

    def get_feature_importance(self) -> dict[str, float] | None:
        """各モデルの特徴量重要度を平均"""
        if not self.is_fitted:
            return None

        importance_sum: dict[str, float] = {}
        count = 0

        for model in self.models.values():
            imp = model.get_feature_importance()
            if imp:
                for feature, value in imp.items():
                    importance_sum[feature] = importance_sum.get(feature, 0) + value
                count += 1

        if count == 0:
            return None

        return {k: v / count for k, v in importance_sum.items()}

    def get_model_contributions(
        self, X: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """各モデルの予測貢献度を取得"""
        contributions = {}

        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights.get(name, 1.0) if self.weights else 1.0
            contributions[name] = pred * weight

        return contributions

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, dict[str, list[float]]]:
        """各モデルと全体のクロスバリデーション結果を取得"""
        results = {}

        for name, model in self.models.items():
            if hasattr(model, "cross_validate"):
                results[name] = model.cross_validate(X, y, n_splits)

        # アンサンブル全体のCV
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        ensemble_results = {"mae": [], "rmse": [], "r2": []}

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            temp_ensemble = EnsemblePredictor(
                weights=self.weights,
                use_xgboost=self.use_xgboost,
                use_lightgbm=self.use_lightgbm,
            )
            temp_ensemble.fit(X_train, y_train)

            metrics = temp_ensemble.evaluate(X_val, y_val)
            ensemble_results["mae"].append(metrics["mae"])
            ensemble_results["rmse"].append(metrics["rmse"])
            ensemble_results["r2"].append(metrics["r2"])

        results["ensemble"] = ensemble_results

        return results
