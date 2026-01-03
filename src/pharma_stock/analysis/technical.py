"""テクニカル分析モジュール"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import ta

from pharma_stock.storage.models import StockPrice


class TrendSignal(str, Enum):
    """トレンドシグナル"""

    STRONG_BUY = "強い買い"
    BUY = "買い"
    NEUTRAL = "中立"
    SELL = "売り"
    STRONG_SELL = "強い売り"


@dataclass
class TechnicalIndicators:
    """テクニカル指標の集約"""

    ticker: str
    date: date

    # 移動平均
    sma_5: float | None
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    ema_12: float | None
    ema_26: float | None

    # モメンタム
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None

    # ボリンジャーバンド
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_width: float | None

    # ボラティリティ
    atr_14: float | None

    # トレンド
    adx_14: float | None

    # シグナル
    trend_signal: TrendSignal
    signal_strength: float  # 0-100


class TechnicalAnalyzer:
    """テクニカル分析クラス"""

    def analyze(self, prices: list[StockPrice]) -> TechnicalIndicators | None:
        """株価データからテクニカル指標を計算

        Args:
            prices: 株価データのリスト（時系列順）

        Returns:
            テクニカル指標
        """
        if len(prices) < 50:
            return None

        df = self._to_dataframe(prices)
        indicators = self._calculate_indicators(df)

        return indicators

    def analyze_batch(
        self, prices_by_ticker: dict[str, list[StockPrice]]
    ) -> dict[str, TechnicalIndicators | None]:
        """複数銘柄の一括分析

        Args:
            prices_by_ticker: ティッカーをキーとする株価データの辞書

        Returns:
            ティッカーをキーとするテクニカル指標の辞書
        """
        results = {}
        for ticker, prices in prices_by_ticker.items():
            results[ticker] = self.analyze(prices)
        return results

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
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """テクニカル指標を計算"""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 移動平均
        sma_5 = ta.trend.sma_indicator(close, window=5)
        sma_20 = ta.trend.sma_indicator(close, window=20)
        sma_50 = ta.trend.sma_indicator(close, window=50)
        sma_200 = ta.trend.sma_indicator(close, window=200) if len(df) >= 200 else pd.Series([None] * len(df))
        ema_12 = ta.trend.ema_indicator(close, window=12)
        ema_26 = ta.trend.ema_indicator(close, window=26)

        # RSI
        rsi = ta.momentum.rsi(close, window=14)

        # MACD
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_hist = macd_indicator.macd_diff()

        # ボリンジャーバンド
        bb = ta.volatility.BollingerBands(close)
        bb_upper = bb.bollinger_hband()
        bb_middle = bb.bollinger_mavg()
        bb_lower = bb.bollinger_lband()
        bb_width = bb.bollinger_wband()

        # ATR
        atr = ta.volatility.average_true_range(high, low, close, window=14)

        # ADX
        adx = ta.trend.adx(high, low, close, window=14)

        # 最新値を取得
        latest_idx = df.index[-1]

        # シグナル判定
        trend_signal, signal_strength = self._calculate_signal(
            close.iloc[-1],
            sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
            sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
            rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None,
            macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else None,
        )

        ticker = df.index.name or "UNKNOWN"

        return TechnicalIndicators(
            ticker=ticker,
            date=latest_idx,
            sma_5=self._safe_float(sma_5.iloc[-1]),
            sma_20=self._safe_float(sma_20.iloc[-1]),
            sma_50=self._safe_float(sma_50.iloc[-1]),
            sma_200=self._safe_float(sma_200.iloc[-1]) if len(df) >= 200 else None,
            ema_12=self._safe_float(ema_12.iloc[-1]),
            ema_26=self._safe_float(ema_26.iloc[-1]),
            rsi_14=self._safe_float(rsi.iloc[-1]),
            macd=self._safe_float(macd.iloc[-1]),
            macd_signal=self._safe_float(macd_signal.iloc[-1]),
            macd_histogram=self._safe_float(macd_hist.iloc[-1]),
            bb_upper=self._safe_float(bb_upper.iloc[-1]),
            bb_middle=self._safe_float(bb_middle.iloc[-1]),
            bb_lower=self._safe_float(bb_lower.iloc[-1]),
            bb_width=self._safe_float(bb_width.iloc[-1]),
            atr_14=self._safe_float(atr.iloc[-1]),
            adx_14=self._safe_float(adx.iloc[-1]),
            trend_signal=trend_signal,
            signal_strength=signal_strength,
        )

    def _safe_float(self, value: Any) -> float | None:
        """安全にfloatに変換"""
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _calculate_signal(
        self,
        current_price: float,
        sma_20: float | None,
        sma_50: float | None,
        rsi: float | None,
        macd: float | None,
        macd_signal: float | None,
    ) -> tuple[TrendSignal, float]:
        """トレンドシグナルを計算"""
        score = 0
        factors = 0

        # SMA分析
        if sma_20 is not None:
            factors += 1
            if current_price > sma_20:
                score += 1
            else:
                score -= 1

        if sma_50 is not None:
            factors += 1
            if current_price > sma_50:
                score += 1
            else:
                score -= 1

        # RSI分析
        if rsi is not None:
            factors += 1
            if rsi < 30:
                score += 2  # 売られすぎ -> 買い
            elif rsi > 70:
                score -= 2  # 買われすぎ -> 売り
            elif rsi < 50:
                score -= 0.5
            else:
                score += 0.5

        # MACD分析
        if macd is not None and macd_signal is not None:
            factors += 1
            if macd > macd_signal:
                score += 1
            else:
                score -= 1

        if factors == 0:
            return TrendSignal.NEUTRAL, 50.0

        # スコアを正規化 (-2 to 2 -> 0 to 100)
        normalized_score = (score / factors + 2) / 4 * 100
        normalized_score = max(0, min(100, normalized_score))

        # シグナル判定
        if normalized_score >= 75:
            signal = TrendSignal.STRONG_BUY
        elif normalized_score >= 60:
            signal = TrendSignal.BUY
        elif normalized_score >= 40:
            signal = TrendSignal.NEUTRAL
        elif normalized_score >= 25:
            signal = TrendSignal.SELL
        else:
            signal = TrendSignal.STRONG_SELL

        return signal, normalized_score

    def get_support_resistance(
        self, prices: list[StockPrice], window: int = 20
    ) -> dict[str, float]:
        """サポート・レジスタンスレベルを計算

        Args:
            prices: 株価データ
            window: 計算ウィンドウ

        Returns:
            サポート・レジスタンスレベル
        """
        df = self._to_dataframe(prices)

        recent = df.tail(window)

        return {
            "resistance_1": float(recent["high"].max()),
            "resistance_2": float(recent["high"].nlargest(2).iloc[-1]) if len(recent) >= 2 else None,
            "support_1": float(recent["low"].min()),
            "support_2": float(recent["low"].nsmallest(2).iloc[-1]) if len(recent) >= 2 else None,
            "pivot": float((recent["high"].max() + recent["low"].min() + recent["close"].iloc[-1]) / 3),
        }
