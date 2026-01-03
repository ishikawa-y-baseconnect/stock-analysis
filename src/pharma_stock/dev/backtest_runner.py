"""バックテストランナー

シグナル・予測の精度検証を簡単に実行するためのユーティリティ
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from pharma_stock.analysis.bottom_signal import BottomSignalDetector, SignalStrength
from pharma_stock.collectors import StockCollector
from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES, get_company_by_ticker


@dataclass
class BacktestResult:
    """バックテスト結果"""

    ticker: str
    company_name: str
    test_date: date
    entry_price: float

    # シグナル
    signal_strength: SignalStrength
    total_score: float
    technical_score: float
    fundamental_score: float
    upside_score: float
    prediction_score: float

    # 予測
    predicted_high_3m: float | None
    predicted_high_6m: float | None
    expected_return_3m: float | None
    expected_return_6m: float | None

    # 実績
    actual_high_3m: float | None
    actual_low_3m: float | None
    actual_high_6m: float | None
    actual_low_6m: float | None
    actual_return_3m: float | None
    actual_return_6m: float | None

    # 評価
    signal_correct: bool | None  # シグナルが正しかったか
    prediction_error_3m: float | None  # 予測誤差
    prediction_error_6m: float | None


class BacktestRunner:
    """バックテストランナー"""

    def __init__(
        self,
        target_return: float = 0.20,
        stop_loss: float = -0.10,
        use_prediction: bool = True,
    ):
        self.detector = BottomSignalDetector(
            target_return=target_return,
            stop_loss=stop_loss,
            use_prediction=use_prediction,
        )
        self.stock_collector = StockCollector()

    def run_single(
        self,
        ticker: str,
        test_date: date,
        horizon_3m: int = 63,
        horizon_6m: int = 126,
    ) -> BacktestResult | None:
        """単一銘柄・単一日付でバックテストを実行

        Args:
            ticker: ティッカーシンボル
            test_date: テスト日付
            horizon_3m: 3ヶ月の営業日数
            horizon_6m: 6ヶ月の営業日数
        """
        import yfinance as yf

        company = get_company_by_ticker(ticker)
        if not company:
            return None

        # 株価データを取得（テスト日より前のデータのみ使用）
        prices = self.stock_collector.collect(tickers=[ticker], period="2y")

        # テスト日のインデックスを探す
        test_idx = None
        for i, p in enumerate(prices):
            if p.date == test_date:
                test_idx = i
                break

        if test_idx is None:
            # 最も近い日付を探す
            for i, p in enumerate(prices):
                if p.date >= test_date:
                    test_idx = i
                    break

        if test_idx is None:
            return None

        # テスト日までのデータでシグナル計算
        prices_until = prices[:test_idx + 1]
        entry_price = float(prices_until[-1].close)

        stock = yf.Ticker(ticker)
        fundamental = stock.info

        signal = self.detector.detect(
            prices=prices_until,
            company_name=company.name,
            pipeline_events=None,
            fundamental_data=fundamental,
        )

        # 実績を計算
        actual_high_3m = None
        actual_low_3m = None
        actual_high_6m = None
        actual_low_6m = None

        if len(prices) > test_idx + horizon_3m:
            future_3m = prices[test_idx + 1:test_idx + 1 + horizon_3m]
            actual_high_3m = max(float(p.high) for p in future_3m)
            actual_low_3m = min(float(p.low) for p in future_3m)

        if len(prices) > test_idx + horizon_6m:
            future_6m = prices[test_idx + 1:test_idx + 1 + horizon_6m]
            actual_high_6m = max(float(p.high) for p in future_6m)
            actual_low_6m = min(float(p.low) for p in future_6m)

        # リターン計算
        actual_return_3m = (actual_high_3m / entry_price - 1) if actual_high_3m else None
        actual_return_6m = (actual_high_6m / entry_price - 1) if actual_high_6m else None

        # シグナル評価
        signal_correct = None
        if actual_return_6m is not None:
            if signal.signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
                signal_correct = actual_return_6m >= 0.10  # 10%以上上昇で正解
            elif signal.signal_strength == SignalStrength.AVOID:
                signal_correct = actual_return_6m < 0.10  # 10%未満で正解

        # 予測誤差
        prediction_error_3m = None
        prediction_error_6m = None

        if signal.predicted_high_3m and actual_high_3m:
            prediction_error_3m = (signal.predicted_high_3m - actual_high_3m) / actual_high_3m

        if signal.predicted_high_6m and actual_high_6m:
            prediction_error_6m = (signal.predicted_high_6m - actual_high_6m) / actual_high_6m

        return BacktestResult(
            ticker=ticker,
            company_name=company.name,
            test_date=prices_until[-1].date,
            entry_price=entry_price,
            signal_strength=signal.signal_strength,
            total_score=signal.total_score,
            technical_score=signal.technical_score,
            fundamental_score=signal.fundamental_score,
            upside_score=signal.upside_potential_score,
            prediction_score=signal.prediction_score,
            predicted_high_3m=signal.predicted_high_3m,
            predicted_high_6m=signal.predicted_high_6m,
            expected_return_3m=signal.expected_return_3m,
            expected_return_6m=signal.expected_return_6m,
            actual_high_3m=actual_high_3m,
            actual_low_3m=actual_low_3m,
            actual_high_6m=actual_high_6m,
            actual_low_6m=actual_low_6m,
            actual_return_3m=actual_return_3m,
            actual_return_6m=actual_return_6m,
            signal_correct=signal_correct,
            prediction_error_3m=prediction_error_3m,
            prediction_error_6m=prediction_error_6m,
        )

    def run_period(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval_days: int = 5,
    ) -> list[BacktestResult]:
        """期間内で定期的にバックテストを実行"""
        results = []
        current = start_date

        while current <= end_date:
            result = self.run_single(ticker, current)
            if result:
                results.append(result)
            current += timedelta(days=interval_days)

        return results

    def run_all_companies(
        self,
        test_date: date,
    ) -> list[BacktestResult]:
        """全銘柄でバックテストを実行"""
        results = []

        for company in TOP_TIER_PHARMA_COMPANIES:
            result = self.run_single(company.ticker, test_date)
            if result:
                results.append(result)

        return results

    def generate_report(
        self,
        results: list[BacktestResult],
    ) -> dict[str, Any]:
        """バックテスト結果のレポートを生成"""
        if not results:
            return {"error": "No results"}

        # 統計計算
        total = len(results)
        buy_signals = [r for r in results if r.signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]]
        correct_signals = [r for r in results if r.signal_correct is True]

        # 予測誤差
        errors_3m = [r.prediction_error_3m for r in results if r.prediction_error_3m is not None]
        errors_6m = [r.prediction_error_6m for r in results if r.prediction_error_6m is not None]

        return {
            "summary": {
                "total_tests": total,
                "buy_signals": len(buy_signals),
                "signal_accuracy": len(correct_signals) / total if total > 0 else 0,
            },
            "prediction_accuracy": {
                "mean_error_3m": sum(errors_3m) / len(errors_3m) if errors_3m else None,
                "mean_error_6m": sum(errors_6m) / len(errors_6m) if errors_6m else None,
            },
            "results": [
                {
                    "ticker": r.ticker,
                    "date": str(r.test_date),
                    "signal": r.signal_strength.value,
                    "score": r.total_score,
                    "predicted_return_6m": f"+{r.expected_return_6m*100:.0f}%" if r.expected_return_6m else "-",
                    "actual_return_6m": f"+{r.actual_return_6m*100:.0f}%" if r.actual_return_6m else "-",
                    "correct": r.signal_correct,
                }
                for r in results
            ],
        }

    def to_dataframe(self, results: list[BacktestResult]) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        data = []
        for r in results:
            data.append({
                "ticker": r.ticker,
                "company": r.company_name,
                "date": r.test_date,
                "price": r.entry_price,
                "signal": r.signal_strength.value,
                "score": r.total_score,
                "tech": r.technical_score,
                "fund": r.fundamental_score,
                "upside": r.upside_score,
                "pred": r.prediction_score,
                "pred_high_3m": r.predicted_high_3m,
                "pred_high_6m": r.predicted_high_6m,
                "actual_high_3m": r.actual_high_3m,
                "actual_high_6m": r.actual_high_6m,
                "return_3m": r.actual_return_3m,
                "return_6m": r.actual_return_6m,
                "correct": r.signal_correct,
                "error_3m": r.prediction_error_3m,
                "error_6m": r.prediction_error_6m,
            })
        return pd.DataFrame(data)


def quick_backtest(ticker: str, test_date: date) -> None:
    """簡易バックテスト実行（デバッグ用）

    使用例:
    ```python
    from datetime import date
    from pharma_stock.dev.backtest_runner import quick_backtest
    quick_backtest("4506.T", date(2025, 4, 9))
    ```
    """
    runner = BacktestRunner()
    result = runner.run_single(ticker, test_date)

    if result:
        print(f"=== {result.company_name} ({result.ticker}) ===")
        print(f"日付: {result.test_date}")
        print(f"価格: ¥{result.entry_price:,.0f}")
        print(f"シグナル: {result.signal_strength.value} ({result.total_score:.0f}点)")
        print()
        print("【スコア内訳】")
        print(f"  テクニカル: {result.technical_score:.0f}")
        print(f"  ファンダメンタル: {result.fundamental_score:.0f}")
        print(f"  上昇ポテンシャル: {result.upside_score:.0f}")
        print(f"  ML予測: {result.prediction_score:.0f}")

        if result.predicted_high_6m:
            print()
            print("【予測 vs 実績】")
            print(f"  6M予測高値: ¥{result.predicted_high_6m:,.0f}")
            if result.actual_high_6m:
                print(f"  6M実績高値: ¥{result.actual_high_6m:,.0f}")
                print(f"  予測誤差: {result.prediction_error_6m*100:+.1f}%")

        if result.signal_correct is not None:
            print()
            print(f"シグナル正否: {'✓ 正解' if result.signal_correct else '✗ 不正解'}")
    else:
        print(f"バックテスト失敗: {ticker}")
