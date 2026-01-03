"""底値買いシグナル検知システム

目標:
- 3〜6ヶ月で+20%のキャピタルゲイン
- 最大損失-10%（損切りライン）
- リスクリワード比 2:1

シグナル条件:
1. テクニカル底値スコア（0-100）
2. ファンダメンタル割安スコア（0-100）
3. 上昇ポテンシャルスコア（0-100）
4. ML予測スコア（0-100）- 3-6ヶ月後の予測高値・安値
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from pharma_stock.storage.models import StockPrice, PipelineEvent


class SignalStrength(str, Enum):
    """シグナル強度"""

    STRONG_BUY = "強い買い"
    BUY = "買い"
    NEUTRAL = "様子見"
    AVOID = "見送り"


@dataclass
class BottomSignal:
    """底値買いシグナル"""

    ticker: str
    company_name: str
    signal_date: date
    current_price: float

    # スコア（各0-100）
    technical_score: float
    fundamental_score: float
    upside_potential_score: float
    prediction_score: float  # ML予測スコア
    total_score: float

    # シグナル
    signal_strength: SignalStrength

    # ターゲット（固定20%ではなく、予測ベースも併記）
    target_price: float  # +20%目標
    stop_loss_price: float  # -10%損切り
    risk_reward_ratio: float

    # ML予測値
    predicted_high_3m: float | None  # 3ヶ月後予測高値
    predicted_low_3m: float | None  # 3ヶ月後予測安値
    predicted_high_6m: float | None  # 6ヶ月後予測高値
    predicted_low_6m: float | None  # 6ヶ月後予測安値
    expected_return_3m: float | None  # 3ヶ月期待リターン
    expected_return_6m: float | None  # 6ヶ月期待リターン
    prediction_confidence: float | None  # 予測信頼度

    # 詳細
    technical_reasons: list[str]
    fundamental_reasons: list[str]
    upside_reasons: list[str]
    prediction_reasons: list[str]  # 予測理由
    risks: list[str]


@dataclass
class TechnicalBottomMetrics:
    """テクニカル底値指標"""

    rsi_14: float | None
    price_vs_52w_low: float  # 52週安値からの乖離率（%）
    price_vs_52w_high: float  # 52週高値からの乖離率（%）
    bb_position: float | None  # ボリンジャーバンド位置（0-1）
    volume_ratio: float  # 出来高比率
    price_vs_sma200: float | None  # 200日移動平均からの乖離率（%）
    consecutive_down_days: int  # 連続下落日数


@dataclass
class FundamentalMetrics:
    """ファンダメンタル指標"""

    pe_ratio: float | None
    pb_ratio: float | None
    ps_ratio: float | None
    dividend_yield: float | None
    market_cap: float | None

    # 業界比較
    pe_vs_industry: float | None  # 業界平均比（マイナスなら割安）
    pb_vs_industry: float | None


@dataclass
class UpsidePotentialMetrics:
    """上昇ポテンシャル指標"""

    phase3_trial_count: int  # Phase 3試験数
    active_trial_count: int  # 進行中の試験数
    has_approval_pipeline: bool  # 承認申請予定があるか
    analyst_target_upside: float | None  # アナリスト目標株価の上昇余地（%）
    recent_insider_buying: bool  # 最近のインサイダー買い


class BottomSignalDetector:
    """底値買いシグナル検知クラス"""

    # 重み付け（予測スコアを追加）
    WEIGHTS = {
        "technical": 0.25,  # 35→25
        "fundamental": 0.20,  # 30→20
        "upside": 0.25,  # 35→25
        "prediction": 0.30,  # 新規：ML予測
    }

    # 閾値（住友ファーマのバックテスト結果を反映して調整）
    THRESHOLDS = {
        "strong_buy": 60,  # 75→60 より敏感に
        "buy": 45,  # 60→45 より敏感に
        "neutral": 30,  # 45→30 より敏感に
    }

    def __init__(
        self,
        target_return: float = 0.20,  # 目標リターン +20%
        stop_loss: float = -0.10,  # 損切り -10%
        use_prediction: bool = True,  # ML予測を使用するか
    ):
        self.target_return = target_return
        self.stop_loss = stop_loss
        self.use_prediction = use_prediction
        self._predictor = None

    @property
    def predictor(self):
        """価格予測器を遅延初期化"""
        if self._predictor is None and self.use_prediction:
            from pharma_stock.prediction.price_range_predictor import PriceRangePredictor
            self._predictor = PriceRangePredictor()
        return self._predictor

    def detect(
        self,
        prices: list[StockPrice],
        company_name: str,
        pipeline_events: list[PipelineEvent] | None = None,
        fundamental_data: dict[str, Any] | None = None,
    ) -> BottomSignal:
        """底値シグナルを検知

        Args:
            prices: 株価データ（1年分推奨）
            company_name: 企業名
            pipeline_events: パイプラインイベント
            fundamental_data: ファンダメンタルデータ（yfinance info）

        Returns:
            底値シグナル
        """
        ticker = prices[0].ticker
        current_price = float(prices[-1].close)

        # 各スコアを計算
        tech_score, tech_reasons = self._calc_technical_score(prices)
        fund_score, fund_reasons = self._calc_fundamental_score(fundamental_data)
        upside_score, upside_reasons = self._calc_upside_score(
            pipeline_events, fundamental_data
        )

        # ML予測スコア
        pred_score, pred_reasons, prediction = self._calc_prediction_score(
            prices, pipeline_events, fundamental_data
        )

        # 総合スコア
        total_score = (
            tech_score * self.WEIGHTS["technical"]
            + fund_score * self.WEIGHTS["fundamental"]
            + upside_score * self.WEIGHTS["upside"]
            + pred_score * self.WEIGHTS["prediction"]
        )

        # シグナル判定
        if total_score >= self.THRESHOLDS["strong_buy"]:
            signal = SignalStrength.STRONG_BUY
        elif total_score >= self.THRESHOLDS["buy"]:
            signal = SignalStrength.BUY
        elif total_score >= self.THRESHOLDS["neutral"]:
            signal = SignalStrength.NEUTRAL
        else:
            signal = SignalStrength.AVOID

        # ターゲット・損切り価格
        target_price = current_price * (1 + self.target_return)
        stop_loss_price = current_price * (1 + self.stop_loss)

        # リスクリワード比
        upside = target_price - current_price
        downside = current_price - stop_loss_price
        risk_reward = upside / downside if downside > 0 else 0

        # リスク要因
        risks = self._identify_risks(prices, fundamental_data)

        # 予測値を取得
        pred_high_3m = prediction.predicted_high_3m if prediction else None
        pred_low_3m = prediction.predicted_low_3m if prediction else None
        pred_high_6m = prediction.predicted_high_6m if prediction else None
        pred_low_6m = prediction.predicted_low_6m if prediction else None
        exp_return_3m = prediction.expected_return_3m if prediction else None
        exp_return_6m = prediction.expected_return_6m if prediction else None
        pred_confidence = prediction.confidence if prediction else None

        return BottomSignal(
            ticker=ticker,
            company_name=company_name,
            signal_date=prices[-1].date,
            current_price=current_price,
            technical_score=tech_score,
            fundamental_score=fund_score,
            upside_potential_score=upside_score,
            prediction_score=pred_score,
            total_score=total_score,
            signal_strength=signal,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            risk_reward_ratio=risk_reward,
            predicted_high_3m=pred_high_3m,
            predicted_low_3m=pred_low_3m,
            predicted_high_6m=pred_high_6m,
            predicted_low_6m=pred_low_6m,
            expected_return_3m=exp_return_3m,
            expected_return_6m=exp_return_6m,
            prediction_confidence=pred_confidence,
            technical_reasons=tech_reasons,
            fundamental_reasons=fund_reasons,
            upside_reasons=upside_reasons,
            prediction_reasons=pred_reasons,
            risks=risks,
        )

    def _calc_technical_score(
        self, prices: list[StockPrice]
    ) -> tuple[float, list[str]]:
        """テクニカル底値スコアを計算

        高スコア条件:
        - RSI < 30（売られすぎ）
        - 52週安値付近
        - ボリンジャーバンド下限以下
        - 出来高減少（セリングクライマックス後）
        """
        import ta

        df = self._to_dataframe(prices)
        score = 0
        reasons = []

        # RSI
        rsi = ta.momentum.rsi(df["close"], window=14).iloc[-1]
        if rsi < 25:
            score += 30
            reasons.append(f"RSI={rsi:.1f} 極度の売られすぎ")
        elif rsi < 30:
            score += 25
            reasons.append(f"RSI={rsi:.1f} 売られすぎ")
        elif rsi < 40:
            score += 15
            reasons.append(f"RSI={rsi:.1f} やや売られすぎ")
        elif rsi > 70:
            score -= 10
            reasons.append(f"RSI={rsi:.1f} 買われすぎ（マイナス）")

        # 52週安値からの位置
        high_52w = df["close"].rolling(252).max().iloc[-1]
        low_52w = df["close"].rolling(252).min().iloc[-1]
        current = df["close"].iloc[-1]

        if high_52w and low_52w:
            range_52w = high_52w - low_52w
            position = (current - low_52w) / range_52w if range_52w > 0 else 0.5

            if position < 0.1:
                score += 25
                reasons.append("52週安値付近（下位10%）")
            elif position < 0.2:
                score += 20
                reasons.append("52週レンジ下位20%")
            elif position < 0.3:
                score += 10
                reasons.append("52週レンジ下位30%")
            elif position > 0.8:
                score -= 5
                reasons.append("52週高値付近（マイナス）")

        # ボリンジャーバンド
        bb = ta.volatility.BollingerBands(df["close"])
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_upper = bb.bollinger_hband().iloc[-1]

        if current < bb_lower:
            score += 20
            reasons.append("ボリンジャーバンド下限を下回る")
        elif current < (bb_lower + (bb_upper - bb_lower) * 0.2):
            score += 10
            reasons.append("ボリンジャーバンド下限付近")

        # 連続下落日数（5日以上なら反発期待）
        consecutive_down = 0
        for i in range(1, min(10, len(df))):
            if df["close"].iloc[-i] < df["close"].iloc[-i - 1]:
                consecutive_down += 1
            else:
                break

        if consecutive_down >= 7:
            score += 15
            reasons.append(f"{consecutive_down}日連続下落（反発期待）")
        elif consecutive_down >= 5:
            score += 10
            reasons.append(f"{consecutive_down}日連続下落")

        # 200日移動平均からの乖離
        if len(df) >= 200:
            sma200 = df["close"].rolling(200).mean().iloc[-1]
            deviation = (current - sma200) / sma200 * 100

            if deviation < -30:
                score += 15
                reasons.append(f"200日線から-{abs(deviation):.0f}%乖離")
            elif deviation < -20:
                score += 10
                reasons.append(f"200日線から-{abs(deviation):.0f}%乖離")

        return min(100, max(0, score)), reasons

    def _calc_fundamental_score(
        self, data: dict[str, Any] | None
    ) -> tuple[float, list[str]]:
        """ファンダメンタル割安スコアを計算"""
        if data is None:
            return 50, ["ファンダメンタルデータなし"]

        score = 50  # ベーススコア
        reasons = []

        # PER
        pe = data.get("trailingPE")
        forward_pe = data.get("forwardPE")

        if pe and pe < 10:
            score += 20
            reasons.append(f"PER={pe:.1f}倍 割安")
        elif pe and pe < 15:
            score += 10
            reasons.append(f"PER={pe:.1f}倍 適正〜割安")
        elif pe and pe > 30:
            score -= 10
            reasons.append(f"PER={pe:.1f}倍 割高")

        # PBR
        pb = data.get("priceToBook")
        if pb and pb < 1.0:
            score += 15
            reasons.append(f"PBR={pb:.2f}倍 割安")
        elif pb and pb < 1.5:
            score += 5
            reasons.append(f"PBR={pb:.2f}倍 適正")
        elif pb and pb > 3.0:
            score -= 5

        # 配当利回り
        div_yield = data.get("dividendYield")
        if div_yield and div_yield > 0.04:
            score += 10
            reasons.append(f"配当利回り{div_yield*100:.1f}% 高配当")
        elif div_yield and div_yield > 0.03:
            score += 5

        # 時価総額の変化（下落していれば割安の可能性）
        # ※ yfinance では直接取れないので省略

        return min(100, max(0, score)), reasons

    def _calc_upside_score(
        self,
        events: list[PipelineEvent] | None,
        fundamental_data: dict[str, Any] | None,
    ) -> tuple[float, list[str]]:
        """上昇ポテンシャルスコアを計算"""
        score = 40  # ベーススコア
        reasons = []

        if events:
            # Phase 3試験の数
            phase3_count = sum(1 for e in events if "3" in e.phase)
            if phase3_count >= 5:
                score += 25
                reasons.append(f"Phase 3試験{phase3_count}本（豊富）")
            elif phase3_count >= 3:
                score += 15
                reasons.append(f"Phase 3試験{phase3_count}本")
            elif phase3_count >= 1:
                score += 5
                reasons.append(f"Phase 3試験{phase3_count}本")

            # 進行中の試験
            active_count = sum(
                1 for e in events if e.event_type in ["進行中", "募集中", "RECRUITING"]
            )
            if active_count >= 10:
                score += 15
                reasons.append(f"進行中の試験{active_count}本（活発）")
            elif active_count >= 5:
                score += 10
                reasons.append(f"進行中の試験{active_count}本")

        # アナリスト目標株価
        if fundamental_data:
            target = fundamental_data.get("targetMeanPrice")
            current = fundamental_data.get("currentPrice")
            if target and current and target > current:
                upside = (target - current) / current * 100
                if upside >= 30:
                    score += 20
                    reasons.append(f"アナリスト目標+{upside:.0f}%")
                elif upside >= 20:
                    score += 15
                    reasons.append(f"アナリスト目標+{upside:.0f}%")
                elif upside >= 10:
                    score += 5

        return min(100, max(0, score)), reasons

    def _calc_prediction_score(
        self,
        prices: list[StockPrice],
        pipeline_events: list[PipelineEvent] | None,
        fundamental_data: dict[str, Any] | None,
    ) -> tuple[float, list[str], Any]:
        """ML予測スコアを計算

        3-6ヶ月後の予測高値・安値に基づくスコア

        Returns:
            score: 予測スコア（0-100）
            reasons: 理由リスト
            prediction: PriceRangePrediction オブジェクト
        """
        if not self.use_prediction or self.predictor is None:
            return 50, ["予測未使用"], None

        try:
            prediction = self.predictor.predict(
                prices, pipeline_events, fundamental_data
            )
        except Exception as e:
            return 50, [f"予測エラー: {e}"], None

        score = 50  # ベーススコア
        reasons = []

        # 3ヶ月期待リターン
        ret_3m = prediction.expected_return_3m
        if ret_3m >= 0.30:
            score += 25
            reasons.append(f"3ヶ月予測リターン+{ret_3m*100:.0f}%（高い）")
        elif ret_3m >= 0.20:
            score += 20
            reasons.append(f"3ヶ月予測リターン+{ret_3m*100:.0f}%")
        elif ret_3m >= 0.10:
            score += 10
            reasons.append(f"3ヶ月予測リターン+{ret_3m*100:.0f}%")
        elif ret_3m < 0:
            score -= 15
            reasons.append(f"3ヶ月予測リターン{ret_3m*100:.0f}%（マイナス）")

        # 6ヶ月期待リターン
        ret_6m = prediction.expected_return_6m
        if ret_6m >= 0.40:
            score += 20
            reasons.append(f"6ヶ月予測リターン+{ret_6m*100:.0f}%（高い）")
        elif ret_6m >= 0.25:
            score += 15
            reasons.append(f"6ヶ月予測リターン+{ret_6m*100:.0f}%")
        elif ret_6m >= 0.15:
            score += 5

        # リスクリワード比
        rr_3m = prediction.risk_reward_3m
        if rr_3m >= 3.0:
            score += 15
            reasons.append(f"3ヶ月リスクリワード比{rr_3m:.1f}:1（優秀）")
        elif rr_3m >= 2.0:
            score += 10
            reasons.append(f"3ヶ月リスクリワード比{rr_3m:.1f}:1")
        elif rr_3m < 1.0:
            score -= 10
            reasons.append(f"3ヶ月リスクリワード比{rr_3m:.1f}:1（不利）")

        # 最大ドローダウンが小さい
        dd_3m = abs(prediction.max_drawdown_3m)
        if dd_3m < 0.05:
            score += 10
            reasons.append(f"3ヶ月最大下落-{dd_3m*100:.0f}%（低リスク）")
        elif dd_3m > 0.20:
            score -= 10
            reasons.append(f"3ヶ月最大下落-{dd_3m*100:.0f}%（高リスク）")

        # 信頼度による調整
        confidence = prediction.confidence
        if confidence < 0.5:
            score = score * 0.8  # 信頼度が低い場合はスコアを下げる
            reasons.append(f"予測信頼度{confidence*100:.0f}%（参考値）")
        else:
            reasons.append(f"予測信頼度{confidence*100:.0f}%")

        return min(100, max(0, score)), reasons, prediction

    def _identify_risks(
        self,
        prices: list[StockPrice],
        fundamental_data: dict[str, Any] | None,
    ) -> list[str]:
        """リスク要因を特定"""
        risks = []

        df = self._to_dataframe(prices)

        # 下落トレンドの継続
        if len(df) >= 60:
            sma60 = df["close"].rolling(60).mean().iloc[-1]
            if df["close"].iloc[-1] < sma60:
                risks.append("60日移動平均を下回り、下落トレンド継続中")

        # ボラティリティの高さ
        volatility = df["close"].pct_change().std() * np.sqrt(252) * 100
        if volatility > 50:
            risks.append(f"年率ボラティリティ{volatility:.0f}%（高い）")

        # 業績懸念
        if fundamental_data:
            profit_margin = fundamental_data.get("profitMargins")
            if profit_margin and profit_margin < 0:
                risks.append("赤字企業")
            elif profit_margin and profit_margin < 0.05:
                risks.append("利益率が低い")

        return risks

    def _to_dataframe(self, prices: list[StockPrice]) -> pd.DataFrame:
        """株価データをDataFrameに変換"""
        data = [
            {"date": p.date, "close": float(p.close), "volume": p.volume}
            for p in prices
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df


def scan_all_companies() -> list[BottomSignal]:
    """全企業をスキャンして買いシグナルを検出

    使用例:
    ```
    signals = scan_all_companies()
    for s in signals:
        if s.signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            print(f"{s.company_name}: {s.total_score:.1f}点 {s.signal_strength.value}")
    ```
    """
    from pharma_stock.collectors import StockCollector, PipelineCollector
    from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES
    import yfinance as yf

    detector = BottomSignalDetector()
    stock_collector = StockCollector()
    pipeline_collector = PipelineCollector()

    signals = []

    for company in TOP_TIER_PHARMA_COMPANIES:
        try:
            # 株価データ
            prices = stock_collector.collect(
                tickers=[company.ticker], period="1y"
            )
            if len(prices) < 100:
                continue

            # パイプライン
            events = pipeline_collector.collect(tickers=[company.ticker])

            # ファンダメンタル
            stock = yf.Ticker(company.ticker)
            fundamental = stock.info

            signal = detector.detect(
                prices=prices,
                company_name=company.name,
                pipeline_events=events,
                fundamental_data=fundamental,
            )

            signals.append(signal)

        except Exception:
            continue

    # スコア順にソート
    signals.sort(key=lambda x: x.total_score, reverse=True)

    return signals
