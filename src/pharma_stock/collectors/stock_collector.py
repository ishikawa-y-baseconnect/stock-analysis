"""株価データ収集モジュール（yfinance使用）"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import yfinance as yf

from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES, get_all_tickers
from pharma_stock.storage.models import StockPrice

from .base import BaseCollector


class StockCollector(BaseCollector[StockPrice]):
    """株価データ収集クラス"""

    def collect(
        self,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        period: str | None = None,
    ) -> list[StockPrice]:
        """株価データを収集

        Args:
            tickers: ティッカーシンボルのリスト（Noneの場合は全企業）
            start_date: 開始日
            end_date: 終了日
            period: 期間（"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"）

        Returns:
            株価データのリスト
        """
        if tickers is None:
            tickers = get_all_tickers()

        self._log_start(tickers=tickers, start_date=start_date, end_date=end_date, period=period)

        try:
            results = self._fetch_stock_data(tickers, start_date, end_date, period)
            self._log_complete(count=len(results))
            return results
        except Exception as e:
            self._log_error(e, tickers=tickers)
            raise

    def _fetch_stock_data(
        self,
        tickers: list[str],
        start_date: date | None,
        end_date: date | None,
        period: str | None,
    ) -> list[StockPrice]:
        """yfinanceから株価データを取得"""
        results: list[StockPrice] = []

        # 日付指定がない場合はデフォルト期間を設定
        if start_date is None and period is None:
            period = "1y"

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)

                if period:
                    df = stock.history(period=period)
                else:
                    df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    self.logger.warning("no_data_found", ticker=ticker)
                    continue

                for idx, row in df.iterrows():
                    stock_price = StockPrice(
                        ticker=ticker,
                        date=idx.date() if isinstance(idx, pd.Timestamp) else idx,
                        open=Decimal(str(round(row["Open"], 2))),
                        high=Decimal(str(round(row["High"], 2))),
                        low=Decimal(str(round(row["Low"], 2))),
                        close=Decimal(str(round(row["Close"], 2))),
                        volume=int(row["Volume"]),
                        adjusted_close=Decimal(str(round(row["Close"], 2))),
                    )
                    results.append(stock_price)

                self.logger.debug("ticker_fetched", ticker=ticker, count=len(df))

            except Exception as e:
                self.logger.warning("ticker_fetch_failed", ticker=ticker, error=str(e))
                continue

        return results

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        """企業情報を取得

        Args:
            ticker: ティッカーシンボル

        Returns:
            企業情報の辞書
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "employees": info.get("fullTimeEmployees", 0),
            "website": info.get("website", ""),
            "description": info.get("longBusinessSummary", ""),
        }

    def get_latest_price(self, ticker: str) -> StockPrice | None:
        """最新の株価を取得

        Args:
            ticker: ティッカーシンボル

        Returns:
            最新の株価データ
        """
        prices = self.collect(tickers=[ticker], period="1d")
        return prices[-1] if prices else None

    def get_all_companies_latest(self) -> dict[str, StockPrice | None]:
        """全企業の最新株価を取得

        Returns:
            ティッカーをキーとする最新株価の辞書
        """
        results: dict[str, StockPrice | None] = {}

        for company in TOP_TIER_PHARMA_COMPANIES:
            price = self.get_latest_price(company.ticker)
            results[company.ticker] = price

        return results
