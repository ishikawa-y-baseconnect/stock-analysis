"""LightGBM予測モデル"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .base import BasePredictor


class LightGBMPredictor(BasePredictor[lgb.LGBMRegressor]):
    """LightGBMベースの株価予測モデル"""

    model_name: str = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__()
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "regression",
            "n_jobs": -1,
            "verbose": -1,
            **kwargs,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int | None = 10,
        **kwargs: Any,
    ) -> "LightGBMPredictor":
        """モデルを学習"""
        self.feature_names = list(X.columns)

        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = [(eval_set[0], eval_set[1])]
            fit_params["callbacks"] = callbacks

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        self.logger.info(
            "model_fitted",
            n_samples=len(X),
            n_features=len(self.feature_names),
        )

        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """予測を実行"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        """特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            return None

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """時系列クロスバリデーション"""
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {
            "mae": [],
            "rmse": [],
            "r2": [],
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            temp_model = LightGBMPredictor(**self.params)
            temp_model.fit(X_train, y_train)

            metrics = temp_model.evaluate(X_val, y_val)
            results["mae"].append(metrics["mae"])
            results["rmse"].append(metrics["rmse"])
            results["r2"].append(metrics["r2"])

        self.logger.info(
            "cross_validation_complete",
            n_splits=n_splits,
            mean_mae=np.mean(results["mae"]),
            mean_rmse=np.mean(results["rmse"]),
            mean_r2=np.mean(results["r2"]),
        )

        return results
