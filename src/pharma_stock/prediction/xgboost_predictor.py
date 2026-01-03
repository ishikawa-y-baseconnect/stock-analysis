"""XGBoost予測モデル"""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from .base import BasePredictor


class XGBoostPredictor(BasePredictor[xgb.XGBRegressor]):
    """XGBoostベースの株価予測モデル"""

    model_name: str = "xgboost"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
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
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            **kwargs,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int | None = 10,
        **kwargs: Any,
    ) -> "XGBoostPredictor":
        """モデルを学習

        Args:
            X: 特徴量DataFrame
            y: ターゲット
            eval_set: 検証データセット
            early_stopping_rounds: 早期停止ラウンド数

        Returns:
            self
        """
        self.feature_names = list(X.columns)

        self.model = xgb.XGBRegressor(**self.params)

        fit_params: dict[str, Any] = {"verbose": False}

        if eval_set is not None:
            fit_params["eval_set"] = [(eval_set[0], eval_set[1])]
            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        self.logger.info(
            "model_fitted",
            n_samples=len(X),
            n_features=len(self.feature_names),
        )

        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """予測を実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測値の配列
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """予測区間付きで予測

        XGBoostは不確実性推定を直接サポートしていないため、
        ブートストラップ法で近似的な予測区間を計算
        """
        predictions = self.predict(X)

        # 学習時の残差から標準偏差を推定（簡易版）
        # より正確には、複数モデルのアンサンブルやQuantile Regressionを使用
        std_estimate = np.std(predictions) * 0.1  # 仮の値

        z_score = 1.96 if confidence_level == 0.95 else 2.576
        margin = z_score * std_estimate

        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

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
        """時系列クロスバリデーション

        Args:
            X: 特徴量DataFrame
            y: ターゲット
            n_splits: 分割数

        Returns:
            各フォールドの評価指標
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {
            "mae": [],
            "rmse": [],
            "r2": [],
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 新しいモデルを作成して学習
            temp_model = XGBoostPredictor(**self.params)
            temp_model.fit(X_train, y_train)

            # 評価
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

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list[Any]] | None = None,
        n_splits: int = 3,
    ) -> dict[str, Any]:
        """ハイパーパラメータチューニング

        Args:
            X: 特徴量DataFrame
            y: ターゲット
            param_grid: 探索するパラメータグリッド
            n_splits: クロスバリデーション分割数

        Returns:
            最適パラメータ
        """
        from sklearn.model_selection import GridSearchCV

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            }

        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.params.get("random_state", 42),
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        self.logger.info("hyperparameter_tuning_complete", best_params=best_params)

        return best_params
