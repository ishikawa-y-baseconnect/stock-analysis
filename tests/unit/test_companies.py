"""企業データのテスト"""

import pytest

from pharma_stock.config.companies import (
    TOP_TIER_PHARMA_COMPANIES,
    get_company_by_ticker,
    get_company_by_name,
    get_all_tickers,
)


class TestCompanies:
    """企業データのテスト"""

    def test_top_tier_companies_count(self):
        """トップティア企業数の確認"""
        assert len(TOP_TIER_PHARMA_COMPANIES) == 10

    def test_get_company_by_ticker(self):
        """ティッカーで企業を取得"""
        company = get_company_by_ticker("4502.T")
        assert company is not None
        assert company.name == "武田薬品工業"

    def test_get_company_by_ticker_code_only(self):
        """証券コードのみで企業を取得"""
        company = get_company_by_ticker("4502")
        assert company is not None
        assert company.name == "武田薬品工業"

    def test_get_company_by_ticker_not_found(self):
        """存在しないティッカー"""
        company = get_company_by_ticker("9999.T")
        assert company is None

    def test_get_company_by_name(self):
        """企業名で検索"""
        company = get_company_by_name("武田")
        assert company is not None
        assert company.ticker == "4502.T"

    def test_get_company_by_name_english(self):
        """英語名で検索"""
        company = get_company_by_name("Takeda")
        assert company is not None
        assert company.ticker == "4502.T"

    def test_get_all_tickers(self):
        """全ティッカー取得"""
        tickers = get_all_tickers()
        assert len(tickers) == 10
        assert "4502.T" in tickers

    def test_company_has_focus_areas(self):
        """企業が研究開発領域を持つ"""
        for company in TOP_TIER_PHARMA_COMPANIES:
            assert len(company.focus_areas) > 0
