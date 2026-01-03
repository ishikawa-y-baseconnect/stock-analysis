"""特徴量エンジニアリング"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import ta

from pharma_stock.storage.models import StockPrice, PipelineEvent


@dataclass
class FeatureConfig:
    """特徴量設定"""

    # 価格特徴量
    use_price_features: bool = True
    price_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)

    # テクニカル指標
    use_technical_features: bool = True
    sma_periods: tuple[int, ...] = (5, 10, 20, 50)
    ema_periods: tuple[int, ...] = (12, 26)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20

    # ボラティリティ特徴量
    use_volatility_features: bool = True
    volatility_windows: tuple[int, ...] = (5, 10, 20)

    # リターン特徴量
    use_return_features: bool = True
    return_periods: tuple[int, ...] = (1, 5, 10, 20)

    # 出来高特徴量
    use_volume_features: bool = True

    # パイプライン特徴量
    use_pipeline_features: bool = True
    pipeline_lookback_days: int = 90

    # 時間特徴量
    use_time_features: bool = True


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def create_features(
        self,
        prices: list[StockPrice],
        pipeline_events: list[PipelineEvent] | None = None,
        target_horizon: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """特徴量とターゲットを作成

        Args:
            prices: 株価データ
            pipeline_events: パイプラインイベント
            target_horizon: 予測対象日数（何日後の株価を予測するか）

        Returns:
            (特徴量DataFrame, ターゲットSeries)のタプル
        """
        df = self._to_dataframe(prices)

        # 各種特徴量を追加
        if self.config.use_price_features:
            df = self._add_price_features(df)

        if self.config.use_technical_features:
            df = self._add_technical_features(df)

        if self.config.use_volatility_features:
            df = self._add_volatility_features(df)

        if self.config.use_return_features:
            df = self._add_return_features(df)

        if self.config.use_volume_features:
            df = self._add_volume_features(df)

        if self.config.use_time_features:
            df = self._add_time_features(df)

        if self.config.use_pipeline_features and pipeline_events:
            df = self._add_pipeline_features(df, pipeline_events)

        # ターゲット作成（n日後の変動率）
        df["target"] = df["close"].shift(-target_horizon) / df["close"] - 1
        df["target"] = df["target"] * 100  # %表記

        # NaNを削除
        df = df.dropna()

        # 特徴量とターゲットを分離
        target = df["target"]
        features = df.drop(columns=["target", "open", "high", "low", "close", "volume"])

        return features, target

    def create_prediction_features(
        self,
        prices: list[StockPrice],
        pipeline_events: list[PipelineEvent] | None = None,
    ) -> pd.DataFrame:
        """予測用の特徴量を作成（最新データのみ）

        Args:
            prices: 株価データ
            pipeline_events: パイプラインイベント

        Returns:
            特徴量DataFrame（最新1行のみ）
        """
        features, _ = self.create_features(
            prices, pipeline_events, target_horizon=1
        )
        # 最新のデータのみ返す（ただしターゲットは使用しない）
        return features.tail(1)

    def _to_dataframe(self, prices: list[StockPrice]) -> pd.DataFrame:
        """株価データをDataFrameに変換"""
        data = [
            {
                "date": p.date,
                "open": float(p.open),
                "high": float(p.high),
                "low": float(p.low),
                "close": float(p.close),
                "volume": p.volume,
            }
            for p in prices
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格特徴量を追加"""
        for lag in self.config.price_lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"close_pct_change_{lag}"] = df["close"].pct_change(lag) * 100

        # 高値・安値からの距離
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"] * 100
        df["close_to_high"] = (df["high"] - df["close"]) / df["close"] * 100
        df["close_to_low"] = (df["close"] - df["low"]) / df["close"] * 100

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加"""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # SMA
        for period in self.config.sma_periods:
            df[f"sma_{period}"] = ta.trend.sma_indicator(close, window=period)
            df[f"close_to_sma_{period}"] = (close - df[f"sma_{period}"]) / df[f"sma_{period}"] * 100

        # EMA
        for period in self.config.ema_periods:
            df[f"ema_{period}"] = ta.trend.ema_indicator(close, window=period)

        # RSI
        df["rsi"] = ta.momentum.rsi(close, window=self.config.rsi_period)

        # MACD
        macd = ta.trend.MACD(
            close,
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # ボリンジャーバンド
        bb = ta.volatility.BollingerBands(close, window=self.config.bb_period)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ADX
        df["adx"] = ta.trend.adx(high, low, close, window=14)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量を追加"""
        for window in self.config.volatility_windows:
            # ヒストリカルボラティリティ
            df[f"volatility_{window}"] = df["close"].pct_change().rolling(window).std() * np.sqrt(252) * 100

            # ATR
            df[f"atr_{window}"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=window
            )

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """リターン特徴量を追加"""
        for period in self.config.return_periods:
            df[f"return_{period}d"] = df["close"].pct_change(period) * 100
            df[f"return_{period}d_abs"] = np.abs(df[f"return_{period}d"])

        # 累積リターン
        df["cumulative_return_20d"] = (df["close"] / df["close"].shift(20) - 1) * 100

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高特徴量を追加"""
        # 出来高の移動平均
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_sma_20"] = df["volume"].rolling(20).mean()

        # 出来高比率
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["obv_pct_change"] = df["obv"].pct_change(5) * 100

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徴量を追加"""
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

        # 月初・月末フラグ
        df["is_month_start"] = (df.index.day <= 5).astype(int)
        df["is_month_end"] = (df.index.day >= 25).astype(int)

        return df

    def _add_pipeline_features(
        self, df: pd.DataFrame, events: list[PipelineEvent]
    ) -> pd.DataFrame:
        """パイプラインイベント特徴量を追加"""
        # イベントをDataFrameに変換
        event_data = []
        for e in events:
            event_data.append(
                {
                    "date": pd.Timestamp(e.event_date),
                    "phase": e.phase,
                    "event_type": e.event_type,
                }
            )

        if not event_data:
            df["recent_pipeline_events"] = 0
            df["has_phase3_event"] = 0
            return df

        events_df = pd.DataFrame(event_data)
        events_df.set_index("date", inplace=True)

        # 各日付に対して、過去N日間のイベント数を計算
        lookback = self.config.pipeline_lookback_days

        recent_events = []
        phase3_events = []

        for idx in df.index:
            start_date = idx - pd.Timedelta(days=lookback)
            mask = (events_df.index >= start_date) & (events_df.index <= idx)
            count = mask.sum()
            recent_events.append(count)

            # Phase 3イベントの有無
            phase3_mask = mask & (events_df["phase"].str.contains("3", na=False))
            phase3_events.append(1 if phase3_mask.any() else 0)

        df["recent_pipeline_events"] = recent_events
        df["has_phase3_event"] = phase3_events

        return df

    def get_feature_names(self) -> list[str]:
        """使用される特徴量名のリストを取得"""
        # ダミーデータで特徴量を生成してカラム名を取得
        dummy_prices = [
            StockPrice(
                ticker="TEST",
                date=date.today() - timedelta(days=i),
                open=100,
                high=105,
                low=95,
                close=100,
                volume=1000000,
                adjusted_close=100,
            )
            for i in range(300)
        ]
        features, _ = self.create_features(dummy_prices)
        return list(features.columns)
