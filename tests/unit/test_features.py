"""特徴量エンジニアリングのテスト"""

import pytest

from pharma_stock.prediction.features import FeatureEngineer, FeatureConfig
from pharma_stock.storage.models import StockPrice


class TestFeatureEngineer:
    """特徴量エンジニアリングのテスト"""

    def test_create_features(self, sample_stock_prices: list[StockPrice]):
        """特徴量作成のテスト"""
        engineer = FeatureEngineer()
        X, y = engineer.create_features(sample_stock_prices, target_horizon=5)

        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)

    def test_feature_columns(self, sample_stock_prices: list[StockPrice]):
        """特徴量カラムの確認"""
        engineer = FeatureEngineer()
        X, _ = engineer.create_features(sample_stock_prices, target_horizon=5)

        # 期待される特徴量が含まれているか
        columns = X.columns.tolist()
        assert any("sma" in col for col in columns)
        assert any("rsi" in col for col in columns)
        assert any("return" in col for col in columns)

    def test_custom_config(self, sample_stock_prices: list[StockPrice]):
        """カスタム設定のテスト"""
        config = FeatureConfig(
            use_price_features=True,
            use_technical_features=False,
            use_volatility_features=False,
            use_return_features=False,
            use_volume_features=False,
            use_time_features=False,
            use_pipeline_features=False,
        )
        engineer = FeatureEngineer(config)
        X, _ = engineer.create_features(sample_stock_prices, target_horizon=5)

        # テクニカル指標が含まれていないことを確認
        columns = X.columns.tolist()
        assert not any("sma" in col for col in columns)
        assert not any("rsi" in col for col in columns)

    def test_with_pipeline_events(
        self, sample_stock_prices: list[StockPrice], sample_pipeline_events
    ):
        """パイプラインイベント付きのテスト"""
        engineer = FeatureEngineer()
        X, _ = engineer.create_features(
            sample_stock_prices,
            pipeline_events=sample_pipeline_events,
            target_horizon=5,
        )

        columns = X.columns.tolist()
        assert "recent_pipeline_events" in columns
