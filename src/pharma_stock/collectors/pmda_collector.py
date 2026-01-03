"""PMDA（医薬品医療機器総合機構）からの情報収集モジュール"""

from datetime import date, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES
from pharma_stock.storage.models import PipelineEvent

from .base import BaseCollector


class PMDACollector(BaseCollector[PipelineEvent]):
    """PMDAからの新薬承認情報を収集

    PMDAには公式APIがないため、承認情報ページをスクレイピングする
    """

    # PMDA新薬承認情報ページ
    APPROVAL_URL = "https://www.pmda.go.jp/review-services/drug-reviews/review-information/p-drugs/0028.html"

    # 企業名マッピング（PMDA表記 -> ティッカー）
    COMPANY_NAME_MAP: dict[str, str] = {
        "武田薬品工業": "4502.T",
        "武田薬品": "4502.T",
        "アステラス製薬": "4503.T",
        "第一三共": "4568.T",
        "エーザイ": "4523.T",
        "中外製薬": "4519.T",
        "大塚製薬": "4578.T",
        "住友ファーマ": "4506.T",
        "塩野義製薬": "4507.T",
        "協和キリン": "4151.T",
        "小野薬品工業": "4528.T",
        "小野薬品": "4528.T",
    }

    def collect(
        self,
        tickers: list[str] | None = None,
        year: int | None = None,
    ) -> list[PipelineEvent]:
        """PMDA承認情報を収集

        Args:
            tickers: フィルタするティッカーのリスト
            year: 対象年

        Returns:
            パイプラインイベントのリスト
        """
        self._log_start(tickers=tickers, year=year)

        try:
            # 注意: PMDAサイトの構造が変わる可能性があるため、
            # 実運用では定期的な確認が必要
            events = self._fetch_approval_info()

            # ティッカーでフィルタ
            if tickers:
                events = [e for e in events if e.ticker in tickers]

            self._log_complete(count=len(events))
            return events

        except Exception as e:
            self._log_error(e)
            # PMDAスクレイピングが失敗した場合は空リストを返す
            return []

    def _fetch_approval_info(self) -> list[PipelineEvent]:
        """PMDA承認情報ページから情報を取得"""
        events: list[PipelineEvent] = []

        try:
            response = requests.get(
                self.APPROVAL_URL,
                timeout=self.settings.request_timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PharmaStockAnalyzer/1.0)"
                },
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            soup = BeautifulSoup(response.text, "lxml")

            # テーブルから承認情報を抽出
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # ヘッダー行をスキップ
                    event = self._parse_approval_row(row)
                    if event:
                        events.append(event)

        except requests.RequestException as e:
            self.logger.warning("pmda_fetch_failed", error=str(e))

        return events

    def _parse_approval_row(self, row: Any) -> PipelineEvent | None:
        """承認情報の行をパース"""
        try:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                return None

            # セルの内容を取得（PMDAページの構造に依存）
            # 典型的な構造: 承認日 | 薬品名 | 適応症 | 企業名
            approval_date_str = cells[0].get_text(strip=True)
            drug_name = cells[1].get_text(strip=True)
            indication = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            company_name = cells[3].get_text(strip=True) if len(cells) > 3 else ""

            # 企業名からティッカーを取得
            ticker = self._find_ticker(company_name)
            if not ticker:
                return None

            # 日付パース
            event_date = self._parse_date(approval_date_str)

            return PipelineEvent(
                ticker=ticker,
                drug_name=drug_name,
                indication=indication,
                phase="承認済",
                event_type="承認",
                event_date=event_date,
                source_url=self.APPROVAL_URL,
                details={"source": "PMDA", "company_name": company_name},
            )

        except Exception as e:
            self.logger.debug("parse_row_failed", error=str(e))
            return None

    def _find_ticker(self, company_name: str) -> str | None:
        """企業名からティッカーを検索"""
        for name, ticker in self.COMPANY_NAME_MAP.items():
            if name in company_name:
                return ticker
        return None

    def _parse_date(self, date_str: str) -> date:
        """日付文字列をパース（和暦対応）"""
        if not date_str:
            return date.today()

        # 和暦を西暦に変換
        era_map = {
            "令和": 2018,  # 令和1年 = 2019年
            "平成": 1988,  # 平成1年 = 1989年
        }

        for era, base_year in era_map.items():
            if era in date_str:
                try:
                    # "令和5年12月25日" のような形式をパース
                    import re

                    match = re.search(rf"{era}(\d+)年(\d+)月(\d+)日", date_str)
                    if match:
                        year = base_year + int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                        return date(year, month, day)
                except Exception:
                    pass

        # 西暦形式を試行
        formats = ["%Y年%m月%d日", "%Y/%m/%d", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return date.today()


class IRNewsCollector(BaseCollector[PipelineEvent]):
    """企業IRニュースからパイプライン情報を収集

    各企業のIRページから新薬関連のニュースを取得
    """

    # 企業IRページURL
    IR_URLS: dict[str, str] = {
        "4502.T": "https://www.takeda.com/jp/newsroom/",
        "4503.T": "https://www.astellas.com/jp/news",
        "4568.T": "https://www.daiichisankyo.co.jp/news/",
        "4523.T": "https://www.eisai.co.jp/news/index.html",
        "4519.T": "https://www.chugai-pharm.co.jp/news/",
        "4507.T": "https://www.shionogi.com/jp/ja/news.html",
    }

    # パイプライン関連キーワード
    PIPELINE_KEYWORDS = [
        "承認",
        "申請",
        "Phase",
        "臨床試験",
        "治験",
        "FDA",
        "EMA",
        "PMDA",
        "上市",
        "販売開始",
        "適応追加",
        "新薬",
    ]

    def collect(
        self,
        tickers: list[str] | None = None,
    ) -> list[PipelineEvent]:
        """IRニュースからパイプライン情報を収集

        注意: 実装はサイト構造に依存するため、
        本格運用時は各サイトに合わせた個別実装が必要
        """
        self._log_start(tickers=tickers)

        # 現時点ではスタブ実装
        # 実際の実装は各企業サイトの構造に合わせて個別に行う必要がある
        self.logger.info("ir_news_collection_stub", message="IR news collection requires site-specific implementation")

        return []
