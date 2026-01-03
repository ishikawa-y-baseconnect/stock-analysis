"""カタリスト検知モジュール

住友ファーマのiPS細胞パーキンソン病治療薬のような
株価変動カタリストを事前に検知するための分析

問題点と解決策:
1. テクニカル指標だけでは「タイミング」を予測できない
2. IRニュース・パイプラインイベントの「内容」評価が必要
3. 「承認申請」「iPS細胞」などの重要キーワードの監視が必要
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np

from pharma_stock.storage.models import StockPrice, PipelineEvent


class CatalystType(str, Enum):
    """カタリストの種類"""

    # 重大（株価に大きく影響）
    APPROVAL_FILING = "承認申請"
    APPROVAL_GRANTED = "承認取得"
    BREAKTHROUGH = "ブレークスルーセラピー指定"
    IPS_CELL = "iPS細胞関連"
    GENE_THERAPY = "遺伝子治療関連"

    # 高影響
    PHASE3_POSITIVE = "Phase 3 良好結果"
    PHASE3_NEGATIVE = "Phase 3 不良結果"
    MAJOR_PARTNERSHIP = "大型提携"

    # 中影響
    PHASE2_POSITIVE = "Phase 2 良好結果"
    INDICATION_EXPANSION = "適応拡大"

    # テクニカル
    OVERSOLD = "売られすぎ"
    VOLUME_SPIKE = "出来高急増"
    GOLDEN_CROSS = "ゴールデンクロス"


@dataclass
class CatalystSignal:
    """カタリストシグナル"""

    ticker: str
    detected_date: date
    catalyst_type: CatalystType
    confidence: float  # 0-1
    description: str
    source: str
    expected_impact: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"


class CatalystDetector:
    """カタリスト検知クラス

    住友ファーマのケースで必要だった検知能力:
    1. RSI売られすぎ → 底値圏の可能性（検知済み）
    2. iPS細胞承認申請ニュース → 上昇トリガー（検知できず）

    改善策:
    - IRニュース・適時開示の自動監視
    - パイプラインカレンダー（予定イベント）の追跡
    - 重要キーワードのアラート設定
    """

    # 株価に大きく影響するイベントのパターン
    CRITICAL_PATTERNS = {
        "iPS細胞": CatalystType.IPS_CELL,
        "iPS": CatalystType.IPS_CELL,
        "再生医療": CatalystType.IPS_CELL,
        "承認申請": CatalystType.APPROVAL_FILING,
        "製造販売承認申請": CatalystType.APPROVAL_FILING,
        "承認取得": CatalystType.APPROVAL_GRANTED,
        "薬事承認": CatalystType.APPROVAL_GRANTED,
        "遺伝子治療": CatalystType.GENE_THERAPY,
        "CAR-T": CatalystType.GENE_THERAPY,
        "ブレークスルー": CatalystType.BREAKTHROUGH,
    }

    def __init__(self):
        pass

    def detect_technical_catalysts(
        self,
        prices: list[StockPrice],
    ) -> list[CatalystSignal]:
        """テクニカル指標ベースのカタリストを検知

        住友ファーマ4月のケースで検知できた：
        - RSI売られすぎ（4/7-16）→ 底値シグナル

        検知できなかった：
        - 上昇開始のタイミング（4/18）
        """
        signals: list[CatalystSignal] = []

        df = self._to_dataframe(prices)
        df = self._add_indicators(df)

        latest = df.iloc[-1]
        latest_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]

        # RSI売られすぎ検知
        if latest['RSI'] < 30:
            signals.append(CatalystSignal(
                ticker=prices[0].ticker,
                detected_date=latest_date,
                catalyst_type=CatalystType.OVERSOLD,
                confidence=0.6,  # テクニカルのみなので信頼度は中程度
                description=f"RSI={latest['RSI']:.1f} 売られすぎ水準",
                source="technical",
                expected_impact="BUY",  # 底値の可能性、ただしタイミング不明
            ))

        # 出来高急増検知
        if latest['Volume_Ratio'] > 3.0:
            signals.append(CatalystSignal(
                ticker=prices[0].ticker,
                detected_date=latest_date,
                catalyst_type=CatalystType.VOLUME_SPIKE,
                confidence=0.7,
                description=f"出来高が通常の{latest['Volume_Ratio']:.1f}倍",
                source="technical",
                expected_impact="NEUTRAL",  # 方向は不明
            ))

        # ゴールデンクロス検知
        if (latest['SMA_5'] > latest['SMA_20'] and
            df.iloc[-2]['SMA_5'] <= df.iloc[-2]['SMA_20']):
            signals.append(CatalystSignal(
                ticker=prices[0].ticker,
                detected_date=latest_date,
                catalyst_type=CatalystType.GOLDEN_CROSS,
                confidence=0.65,
                description="5日移動平均が20日移動平均を上抜け",
                source="technical",
                expected_impact="BUY",
            ))

        return signals

    def detect_news_catalyst(
        self,
        news_title: str,
        ticker: str,
        news_date: date,
    ) -> CatalystSignal | None:
        """ニュースタイトルからカタリストを検知

        住友ファーマのケースで必要だった：
        「iPS細胞由来パーキンソン病治療薬 承認申請へ」
        → CatalystType.IPS_CELL + APPROVAL_FILING として検知
        """
        for pattern, catalyst_type in self.CRITICAL_PATTERNS.items():
            if pattern in news_title:
                # iPS細胞 + 承認申請は最重要
                if "iPS" in news_title and "承認" in news_title:
                    return CatalystSignal(
                        ticker=ticker,
                        detected_date=news_date,
                        catalyst_type=CatalystType.IPS_CELL,
                        confidence=0.95,
                        description=f"【最重要】{news_title}",
                        source="news",
                        expected_impact="STRONG_BUY",
                    )

                # 承認申請は重要
                if catalyst_type == CatalystType.APPROVAL_FILING:
                    return CatalystSignal(
                        ticker=ticker,
                        detected_date=news_date,
                        catalyst_type=catalyst_type,
                        confidence=0.9,
                        description=news_title,
                        source="news",
                        expected_impact="STRONG_BUY",
                    )

                return CatalystSignal(
                    ticker=ticker,
                    detected_date=news_date,
                    catalyst_type=catalyst_type,
                    confidence=0.8,
                    description=news_title,
                    source="news",
                    expected_impact="BUY",
                )

        return None

    def analyze_sumitomo_case(
        self,
        prices: list[StockPrice],
    ) -> dict[str, Any]:
        """住友ファーマのケーススタディ

        実際に検知できたもの、できなかったものを分析
        """
        signals = self.detect_technical_catalysts(prices)

        result = {
            "ticker": "4506.T",
            "company": "住友ファーマ",
            "analysis_period": "2025年4月",
            "actual_event": "iPS細胞パーキンソン病治療薬 承認申請発表（4/18）",
            "stock_move": "500円 → 2,700円（+440%）",
            "detected_signals": [
                {
                    "type": s.catalyst_type.value,
                    "date": str(s.detected_date),
                    "confidence": s.confidence,
                    "impact": s.expected_impact,
                }
                for s in signals
            ],
            "detection_gaps": [
                "IRニュース監視が未実装 → 4/18のニュースを検知できず",
                "iPS細胞治療の重要性を事前に評価できず",
                "テクニカル指標は底値を示唆したが、上昇タイミングは不明",
            ],
            "recommendations": [
                "IRニュースの自動収集・解析を実装",
                "パイプラインカレンダー（承認申請予定など）の追跡",
                "重要キーワード（iPS、承認申請）のアラート設定",
                "医師主導治験の結果発表日程の監視",
            ],
        }

        return result

    def _to_dataframe(self, prices: list[StockPrice]) -> pd.DataFrame:
        """株価データをDataFrameに変換"""
        import ta

        data = [{
            "date": p.date,
            "close": float(p.close),
            "volume": p.volume,
        } for p in prices]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加"""
        import ta

        df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['Volume_SMA'] = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']

        return df


# 住友ファーマのケースをシミュレート
def simulate_sumitomo_detection():
    """
    もしIRニュース監視が実装されていたら、
    2025年4月18日に以下のシグナルを検知できた：

    ニュースタイトル:
    「iPS細胞由来ドパミン神経前駆細胞を用いたパーキンソン病治療
     製造販売承認申請について」

    検知結果:
    - catalyst_type: IPS_CELL
    - confidence: 0.95
    - expected_impact: STRONG_BUY
    - detected_date: 2025-04-18

    + テクニカル指標（4/7-16のRSI売られすぎ）

    → 総合シグナル: STRONG_BUY
    """
    detector = CatalystDetector()

    # 実際のニュースタイトル（想定）
    news_title = "iPS細胞由来ドパミン神経前駆細胞を用いたパーキンソン病治療 製造販売承認申請について"

    signal = detector.detect_news_catalyst(
        news_title=news_title,
        ticker="4506.T",
        news_date=date(2025, 4, 18),
    )

    if signal:
        print(f"【検知】{signal.catalyst_type.value}")
        print(f"  信頼度: {signal.confidence:.0%}")
        print(f"  予想インパクト: {signal.expected_impact}")
        print(f"  説明: {signal.description}")

    return signal
