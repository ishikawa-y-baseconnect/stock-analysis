"""パイプラインイベントの株価影響分析"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pharma_stock.storage.models import PipelineEvent, StockPrice


class ImpactLevel(str, Enum):
    """影響レベル"""

    HIGH_POSITIVE = "大幅上昇"
    POSITIVE = "上昇"
    NEUTRAL = "中立"
    NEGATIVE = "下落"
    HIGH_NEGATIVE = "大幅下落"


@dataclass
class EventImpactAnalysis:
    """イベント影響分析結果"""

    event: PipelineEvent
    event_date: date

    # 株価変動
    price_before: float  # イベント前終値
    price_after: float  # イベント後終値
    price_change: float  # 変動額
    price_change_pct: float  # 変動率(%)

    # 短期影響（1-5日）
    return_1d: float | None
    return_3d: float | None
    return_5d: float | None

    # 中期影響（1-4週）
    return_1w: float | None
    return_2w: float | None
    return_4w: float | None

    # ボリューム分析
    volume_ratio: float  # イベント日の出来高/平均出来高

    # 統計的有意性
    abnormal_return: float  # 異常リターン
    t_statistic: float | None
    is_significant: bool  # 有意水準5%

    # 総合評価
    impact_level: ImpactLevel


class PipelineImpactAnalyzer:
    """パイプラインイベントの株価影響を分析"""

    def __init__(self, lookback_days: int = 60, event_window: int = 5):
        """
        Args:
            lookback_days: 正常リターン計算のための期間
            event_window: イベント前後のウィンドウ
        """
        self.lookback_days = lookback_days
        self.event_window = event_window

    def analyze_event(
        self,
        event: PipelineEvent,
        prices: list[StockPrice],
    ) -> EventImpactAnalysis | None:
        """単一イベントの影響を分析

        Args:
            event: パイプラインイベント
            prices: 株価データ（時系列順）

        Returns:
            イベント影響分析結果
        """
        df = self._to_dataframe(prices)

        # イベント日のインデックスを検索
        event_date = event.event_date
        if event_date not in df.index:
            # 最も近い営業日を探す
            event_date = self._find_nearest_trading_day(df.index, event_date)
            if event_date is None:
                return None

        event_idx = df.index.get_loc(event_date)

        # 十分なデータがあるか確認
        if event_idx < self.lookback_days or event_idx >= len(df) - 20:
            return None

        # 株価・リターン計算
        price_before = df.iloc[event_idx - 1]["close"]
        price_after = df.iloc[event_idx]["close"]
        price_change = price_after - price_before
        price_change_pct = (price_change / price_before) * 100

        # 短期・中期リターン
        return_1d = self._calculate_return(df, event_idx, 1)
        return_3d = self._calculate_return(df, event_idx, 3)
        return_5d = self._calculate_return(df, event_idx, 5)
        return_1w = self._calculate_return(df, event_idx, 5)
        return_2w = self._calculate_return(df, event_idx, 10)
        return_4w = self._calculate_return(df, event_idx, 20)

        # ボリューム分析
        avg_volume = df.iloc[event_idx - self.lookback_days : event_idx]["volume"].mean()
        event_volume = df.iloc[event_idx]["volume"]
        volume_ratio = event_volume / avg_volume if avg_volume > 0 else 1.0

        # 異常リターン計算（マーケットモデル簡易版）
        normal_returns = df.iloc[event_idx - self.lookback_days : event_idx]["close"].pct_change().dropna()
        expected_return = normal_returns.mean()
        std_return = normal_returns.std()

        actual_return = price_change_pct / 100
        abnormal_return = actual_return - expected_return

        # t統計量
        t_stat = abnormal_return / std_return if std_return > 0 else 0
        is_significant = abs(t_stat) > 1.96  # 5%有意水準

        # 影響レベル判定
        impact_level = self._determine_impact_level(price_change_pct, t_stat, volume_ratio)

        return EventImpactAnalysis(
            event=event,
            event_date=event_date,
            price_before=price_before,
            price_after=price_after,
            price_change=price_change,
            price_change_pct=price_change_pct,
            return_1d=return_1d,
            return_3d=return_3d,
            return_5d=return_5d,
            return_1w=return_1w,
            return_2w=return_2w,
            return_4w=return_4w,
            volume_ratio=volume_ratio,
            abnormal_return=abnormal_return * 100,
            t_statistic=t_stat,
            is_significant=is_significant,
            impact_level=impact_level,
        )

    def analyze_events(
        self,
        events: list[PipelineEvent],
        prices: list[StockPrice],
    ) -> list[EventImpactAnalysis]:
        """複数イベントの影響を分析

        Args:
            events: パイプラインイベントのリスト
            prices: 株価データ

        Returns:
            イベント影響分析結果のリスト
        """
        results = []
        for event in events:
            analysis = self.analyze_event(event, prices)
            if analysis:
                results.append(analysis)
        return results

    def calculate_phase_impact_stats(
        self,
        analyses: list[EventImpactAnalysis],
    ) -> dict[str, dict[str, float]]:
        """フェーズ別の影響統計を計算

        Returns:
            フェーズをキーとする統計情報の辞書
        """
        phase_groups: dict[str, list[EventImpactAnalysis]] = {}

        for analysis in analyses:
            phase = analysis.event.phase
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(analysis)

        stats_by_phase = {}
        for phase, group in phase_groups.items():
            returns = [a.price_change_pct for a in group]
            abnormal_returns = [a.abnormal_return for a in group]

            stats_by_phase[phase] = {
                "count": len(group),
                "avg_return": np.mean(returns),
                "median_return": np.median(returns),
                "std_return": np.std(returns),
                "avg_abnormal_return": np.mean(abnormal_returns),
                "positive_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
                "significant_count": sum(1 for a in group if a.is_significant),
            }

        return stats_by_phase

    def calculate_event_type_impact_stats(
        self,
        analyses: list[EventImpactAnalysis],
    ) -> dict[str, dict[str, float]]:
        """イベントタイプ別の影響統計を計算

        Returns:
            イベントタイプをキーとする統計情報の辞書
        """
        type_groups: dict[str, list[EventImpactAnalysis]] = {}

        for analysis in analyses:
            event_type = analysis.event.event_type
            if event_type not in type_groups:
                type_groups[event_type] = []
            type_groups[event_type].append(analysis)

        stats_by_type = {}
        for event_type, group in type_groups.items():
            returns = [a.price_change_pct for a in group]

            stats_by_type[event_type] = {
                "count": len(group),
                "avg_return": np.mean(returns),
                "median_return": np.median(returns),
                "std_return": np.std(returns),
                "positive_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
            }

        return stats_by_type

    def _to_dataframe(self, prices: list[StockPrice]) -> pd.DataFrame:
        """株価データをDataFrameに変換"""
        data = [
            {
                "date": p.date,
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

    def _find_nearest_trading_day(
        self, index: pd.DatetimeIndex, target_date: date
    ) -> date | None:
        """最も近い取引日を探す"""
        target = pd.Timestamp(target_date)

        # 前後3日以内で探す
        for delta in range(4):
            for sign in [1, -1]:
                check_date = target + pd.Timedelta(days=delta * sign)
                if check_date in index:
                    return check_date.date()

        return None

    def _calculate_return(
        self, df: pd.DataFrame, event_idx: int, days: int
    ) -> float | None:
        """指定日数後のリターンを計算"""
        if event_idx + days >= len(df):
            return None

        price_at_event = df.iloc[event_idx]["close"]
        price_after = df.iloc[event_idx + days]["close"]

        return ((price_after - price_at_event) / price_at_event) * 100

    def _determine_impact_level(
        self, price_change_pct: float, t_stat: float, volume_ratio: float
    ) -> ImpactLevel:
        """影響レベルを判定"""
        # 出来高が通常の2倍以上で、かつ統計的に有意な場合は影響大
        high_volume = volume_ratio >= 2.0
        significant = abs(t_stat) > 1.96

        if price_change_pct >= 5 or (price_change_pct >= 3 and high_volume and significant):
            return ImpactLevel.HIGH_POSITIVE
        elif price_change_pct >= 2:
            return ImpactLevel.POSITIVE
        elif price_change_pct <= -5 or (price_change_pct <= -3 and high_volume and significant):
            return ImpactLevel.HIGH_NEGATIVE
        elif price_change_pct <= -2:
            return ImpactLevel.NEGATIVE
        else:
            return ImpactLevel.NEUTRAL
