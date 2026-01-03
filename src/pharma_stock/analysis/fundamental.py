"""ファンダメンタル分析モジュール"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import yfinance as yf

from pharma_stock.config.companies import PharmaCompany, TOP_TIER_PHARMA_COMPANIES


@dataclass
class FundamentalMetrics:
    """ファンダメンタル指標"""

    ticker: str
    date: date

    # バリュエーション
    market_cap: float | None  # 時価総額
    enterprise_value: float | None  # 企業価値
    pe_ratio: float | None  # PER
    forward_pe: float | None  # 予想PER
    pb_ratio: float | None  # PBR
    ps_ratio: float | None  # PSR
    ev_ebitda: float | None  # EV/EBITDA

    # 収益性
    profit_margin: float | None  # 利益率
    operating_margin: float | None  # 営業利益率
    roe: float | None  # ROE
    roa: float | None  # ROA

    # 成長性
    revenue_growth: float | None  # 売上高成長率
    earnings_growth: float | None  # 利益成長率

    # 財務健全性
    debt_to_equity: float | None  # D/Eレシオ
    current_ratio: float | None  # 流動比率
    quick_ratio: float | None  # 当座比率

    # 配当
    dividend_yield: float | None  # 配当利回り
    payout_ratio: float | None  # 配当性向

    # 製薬特有
    rd_expense_ratio: float | None  # R&D費用比率（推定）


@dataclass
class PeerComparison:
    """同業他社比較"""

    ticker: str
    name: str
    metrics: FundamentalMetrics

    # ランキング（業界内順位）
    rank_market_cap: int
    rank_pe_ratio: int
    rank_profit_margin: int
    rank_roe: int

    # 業界平均との比較
    pe_vs_industry: float  # 業界平均比（%）
    margin_vs_industry: float
    roe_vs_industry: float


class FundamentalAnalyzer:
    """ファンダメンタル分析クラス"""

    def get_metrics(self, ticker: str) -> FundamentalMetrics | None:
        """ティッカーのファンダメンタル指標を取得

        Args:
            ticker: ティッカーシンボル

        Returns:
            ファンダメンタル指標
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return FundamentalMetrics(
                ticker=ticker,
                date=date.today(),
                market_cap=info.get("marketCap"),
                enterprise_value=info.get("enterpriseValue"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                ev_ebitda=info.get("enterpriseToEbitda"),
                profit_margin=self._to_percent(info.get("profitMargins")),
                operating_margin=self._to_percent(info.get("operatingMargins")),
                roe=self._to_percent(info.get("returnOnEquity")),
                roa=self._to_percent(info.get("returnOnAssets")),
                revenue_growth=self._to_percent(info.get("revenueGrowth")),
                earnings_growth=self._to_percent(info.get("earningsGrowth")),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio"),
                dividend_yield=self._to_percent(info.get("dividendYield")),
                payout_ratio=self._to_percent(info.get("payoutRatio")),
                rd_expense_ratio=self._estimate_rd_ratio(stock),
            )
        except Exception:
            return None

    def get_all_companies_metrics(self) -> dict[str, FundamentalMetrics | None]:
        """全トップティア企業のファンダメンタル指標を取得

        Returns:
            ティッカーをキーとするファンダメンタル指標の辞書
        """
        results = {}
        for company in TOP_TIER_PHARMA_COMPANIES:
            results[company.ticker] = self.get_metrics(company.ticker)
        return results

    def compare_peers(self) -> list[PeerComparison]:
        """同業他社比較を実行

        Returns:
            同業他社比較結果のリスト
        """
        all_metrics = self.get_all_companies_metrics()

        # 有効なメトリクスのみ抽出
        valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}

        if not valid_metrics:
            return []

        # 業界平均を計算
        industry_avg = self._calculate_industry_average(list(valid_metrics.values()))

        # ランキング計算
        comparisons = []
        for ticker, metrics in valid_metrics.items():
            company = next(
                (c for c in TOP_TIER_PHARMA_COMPANIES if c.ticker == ticker), None
            )
            if not company:
                continue

            comparison = PeerComparison(
                ticker=ticker,
                name=company.name,
                metrics=metrics,
                rank_market_cap=self._calculate_rank(
                    valid_metrics, "market_cap", ticker, reverse=True
                ),
                rank_pe_ratio=self._calculate_rank(
                    valid_metrics, "pe_ratio", ticker, reverse=False
                ),
                rank_profit_margin=self._calculate_rank(
                    valid_metrics, "profit_margin", ticker, reverse=True
                ),
                rank_roe=self._calculate_rank(
                    valid_metrics, "roe", ticker, reverse=True
                ),
                pe_vs_industry=self._vs_industry(
                    metrics.pe_ratio, industry_avg.get("pe_ratio")
                ),
                margin_vs_industry=self._vs_industry(
                    metrics.profit_margin, industry_avg.get("profit_margin")
                ),
                roe_vs_industry=self._vs_industry(
                    metrics.roe, industry_avg.get("roe")
                ),
            )
            comparisons.append(comparison)

        # 時価総額順にソート
        comparisons.sort(key=lambda x: x.rank_market_cap)

        return comparisons

    def _to_percent(self, value: float | None) -> float | None:
        """小数を%に変換"""
        if value is None:
            return None
        return value * 100

    def _estimate_rd_ratio(self, stock: yf.Ticker) -> float | None:
        """R&D費用比率を推定（財務諸表から）"""
        try:
            financials = stock.financials
            if financials is None or financials.empty:
                return None

            # Research And Developmentの行を探す
            rd_row = None
            for idx in financials.index:
                if "research" in idx.lower() and "development" in idx.lower():
                    rd_row = idx
                    break

            if rd_row is None:
                return None

            revenue_row = None
            for idx in financials.index:
                if "total revenue" in idx.lower():
                    revenue_row = idx
                    break

            if revenue_row is None:
                return None

            rd_expense = financials.loc[rd_row].iloc[0]
            revenue = financials.loc[revenue_row].iloc[0]

            if revenue and revenue > 0:
                return (rd_expense / revenue) * 100

            return None
        except Exception:
            return None

    def _calculate_industry_average(
        self, metrics_list: list[FundamentalMetrics]
    ) -> dict[str, float | None]:
        """業界平均を計算"""
        avg = {}

        fields = ["pe_ratio", "forward_pe", "pb_ratio", "profit_margin", "roe", "roa"]

        for field in fields:
            values = [
                getattr(m, field) for m in metrics_list if getattr(m, field) is not None
            ]
            avg[field] = sum(values) / len(values) if values else None

        return avg

    def _calculate_rank(
        self,
        metrics_dict: dict[str, FundamentalMetrics],
        field: str,
        ticker: str,
        reverse: bool = True,
    ) -> int:
        """ランキングを計算"""
        values = []
        for t, m in metrics_dict.items():
            val = getattr(m, field)
            if val is not None:
                values.append((t, val))

        values.sort(key=lambda x: x[1], reverse=reverse)

        for i, (t, _) in enumerate(values, 1):
            if t == ticker:
                return i

        return len(values)

    def _vs_industry(self, value: float | None, avg: float | None) -> float:
        """業界平均との比較（%）"""
        if value is None or avg is None or avg == 0:
            return 0.0
        return ((value - avg) / abs(avg)) * 100
