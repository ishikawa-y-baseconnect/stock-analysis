"""企業IRニュース・適時開示情報の収集モジュール

住友ファーマのiPS細胞パーキンソン病治療薬のような
「株価に大きく影響するイベント」を検知するための実装

データソース:
1. 東証適時開示システム (TDnet)
2. 各社IRページ
3. PMDA（医薬品医療機器総合機構）承認情報
4. ニュースAPI（外部サービス）
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import requests
from bs4 import BeautifulSoup

from pharma_stock.collectors.base import BaseCollector
from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES


class NewsImpactLevel(str, Enum):
    """ニュースの株価影響度"""

    CRITICAL = "重大"      # 承認申請、承認取得、大型提携
    HIGH = "高"            # Phase 3結果、適応拡大
    MEDIUM = "中"          # Phase 2結果、小規模提携
    LOW = "低"             # その他


@dataclass
class IRNews:
    """IRニュース/適時開示情報"""

    ticker: str
    title: str
    published_date: date
    source: str
    url: str
    impact_level: NewsImpactLevel
    keywords: list[str]
    content_summary: str | None = None


# 株価に大きく影響するキーワード（重要度順）
CRITICAL_KEYWORDS = [
    # 承認関連
    "承認申請", "承認取得", "製造販売承認", "薬事承認",
    "FDA承認", "EMA承認", "PMDA",

    # 先端医療
    "iPS細胞", "再生医療", "遺伝子治療", "CAR-T",

    # 臨床試験の重要結果
    "主要評価項目達成", "有効性確認", "良好な結果",
    "Phase 3 完了", "ピボタル試験",

    # 提携・買収
    "ライセンス契約", "戦略的提携", "買収", "マイルストン",
]

HIGH_IMPACT_KEYWORDS = [
    "Phase 3 開始", "Phase 2 良好", "適応拡大",
    "上市", "販売開始", "ブレークスルーセラピー",
    "優先審査", "希少疾病",
]

NEGATIVE_KEYWORDS = [
    "承認申請取り下げ", "開発中止", "治験中止",
    "有効性未達", "主要評価項目未達",
    "安全性懸念", "副作用", "リコール",
]


class IRNewsCollector(BaseCollector[IRNews]):
    """企業IRニュース収集クラス

    住友ファーマのケースで必要だった情報源:
    1. 企業IRページの適時開示
    2. 京都大学病院のプレスリリース（医師主導治験結果）
    3. 日経新聞等のニュース報道
    """

    # 各社のIRニュースページ
    IR_NEWS_URLS = {
        "4502.T": "https://www.takeda.com/jp/newsroom/newsreleases/",
        "4503.T": "https://www.astellas.com/jp/news",
        "4568.T": "https://www.daiichisankyo.co.jp/news/",
        "4523.T": "https://www.eisai.co.jp/news/",
        "4519.T": "https://www.chugai-pharm.co.jp/news/",
        "4578.T": "https://www.otsuka.co.jp/company/newsreleases/",
        "4506.T": "https://www.sumitomo-pharma.co.jp/news/",  # 住友ファーマ
        "4507.T": "https://www.shionogi.com/jp/ja/news.html",
        "4151.T": "https://www.kyowakirin.co.jp/news/",
        "4528.T": "https://www.ono-pharma.com/ja/news/",
    }

    def collect(
        self,
        tickers: list[str] | None = None,
        days_back: int = 30,
    ) -> list[IRNews]:
        """IRニュースを収集

        Args:
            tickers: 対象ティッカー
            days_back: 過去何日分を取得するか

        Returns:
            IRニュースのリスト
        """
        if tickers is None:
            tickers = list(self.IR_NEWS_URLS.keys())

        self._log_start(tickers=tickers, days_back=days_back)

        all_news: list[IRNews] = []

        for ticker in tickers:
            try:
                news = self._fetch_company_news(ticker, days_back)
                all_news.extend(news)
            except Exception as e:
                self._log_error(e, ticker=ticker)
                continue

        # 重要度でソート
        all_news.sort(
            key=lambda x: (
                list(NewsImpactLevel).index(x.impact_level),
                x.published_date
            ),
            reverse=True
        )

        self._log_complete(count=len(all_news))
        return all_news

    def _fetch_company_news(
        self, ticker: str, days_back: int
    ) -> list[IRNews]:
        """特定企業のニュースを取得"""
        url = self.IR_NEWS_URLS.get(ticker)
        if not url:
            return []

        # 注意: 実際の実装では各サイトの構造に合わせたスクレイピングが必要
        # ここではスケルトン実装

        try:
            response = requests.get(
                url,
                timeout=self.settings.request_timeout,
                headers={"User-Agent": "Mozilla/5.0 (PharmaStockAnalyzer/1.0)"}
            )
            response.raise_for_status()

            # サイト構造に応じたパース（要個別実装）
            news = self._parse_news_page(ticker, response.text, days_back)
            return news

        except requests.RequestException as e:
            self.logger.warning("fetch_failed", ticker=ticker, error=str(e))
            return []

    def _parse_news_page(
        self, ticker: str, html: str, days_back: int
    ) -> list[IRNews]:
        """ニュースページをパース（サイトごとに要カスタマイズ）"""
        soup = BeautifulSoup(html, "lxml")
        news_items: list[IRNews] = []

        # 共通的なパターンで記事を探す
        # 実際には各サイトの構造に合わせた実装が必要

        # 例: <article>, <div class="news-item"> などを探す
        articles = soup.find_all(["article", "li", "div"], class_=lambda x: x and "news" in x.lower() if x else False)

        for article in articles[:20]:  # 最大20件
            try:
                # タイトルを取得
                title_elem = article.find(["h2", "h3", "a", "span"])
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # URLを取得
                link = article.find("a", href=True)
                news_url = link["href"] if link else ""

                # 日付を取得（様々な形式に対応が必要）
                date_elem = article.find(["time", "span", "div"], class_=lambda x: x and "date" in x.lower() if x else False)
                published_date = date.today()  # デフォルト

                # 影響度を判定
                impact_level = self._assess_impact_level(title)
                keywords = self._extract_keywords(title)

                if impact_level in [NewsImpactLevel.CRITICAL, NewsImpactLevel.HIGH]:
                    news_items.append(IRNews(
                        ticker=ticker,
                        title=title,
                        published_date=published_date,
                        source="IR",
                        url=news_url,
                        impact_level=impact_level,
                        keywords=keywords,
                    ))

            except Exception:
                continue

        return news_items

    def _assess_impact_level(self, title: str) -> NewsImpactLevel:
        """ニュースタイトルから影響度を判定"""
        title_lower = title.lower()

        # ネガティブキーワードチェック
        for kw in NEGATIVE_KEYWORDS:
            if kw in title:
                return NewsImpactLevel.CRITICAL  # ネガティブも重大

        # クリティカルキーワードチェック
        for kw in CRITICAL_KEYWORDS:
            if kw in title:
                return NewsImpactLevel.CRITICAL

        # 高影響キーワードチェック
        for kw in HIGH_IMPACT_KEYWORDS:
            if kw in title:
                return NewsImpactLevel.HIGH

        return NewsImpactLevel.LOW

    def _extract_keywords(self, title: str) -> list[str]:
        """タイトルからキーワードを抽出"""
        found = []
        all_keywords = CRITICAL_KEYWORDS + HIGH_IMPACT_KEYWORDS + NEGATIVE_KEYWORDS

        for kw in all_keywords:
            if kw in title:
                found.append(kw)

        return found

    def get_critical_news(
        self, tickers: list[str] | None = None
    ) -> list[IRNews]:
        """重大ニュースのみを取得"""
        all_news = self.collect(tickers)
        return [n for n in all_news if n.impact_level == NewsImpactLevel.CRITICAL]


class TDnetCollector(BaseCollector[IRNews]):
    """東証適時開示システム（TDnet）からの情報収集

    TDnetは日本の上場企業の適時開示情報を集約
    https://www.release.tdnet.info/

    注意: TDnetには公式APIがないため、スクレイピングまたは
    有料データプロバイダの利用が必要
    """

    TDNET_URL = "https://www.release.tdnet.info/inbs/I_main_00.html"

    def collect(
        self,
        tickers: list[str] | None = None,
        days_back: int = 7,
    ) -> list[IRNews]:
        """TDnetから適時開示情報を収集"""
        self._log_start(tickers=tickers, days_back=days_back)

        # TDnetの構造は複雑で、証券コードでのフィルタリングが必要
        # 実装にはサイト構造の詳細な分析が必要

        self.logger.info(
            "tdnet_collection_stub",
            message="TDnet collection requires site-specific implementation"
        )

        return []


# 住友ファーマのケースを検知するための改善版予測機能
def detect_pipeline_catalyst(
    ticker: str,
    news_collector: IRNewsCollector,
) -> dict[str, Any]:
    """パイプラインカタリスト（株価変動要因）を検知

    住友ファーマのiPS細胞パーキンソン病治療薬のような
    重大イベントを検知するための関数

    Returns:
        検知結果の辞書
    """
    critical_news = news_collector.get_critical_news([ticker])

    result = {
        "ticker": ticker,
        "has_critical_catalyst": len(critical_news) > 0,
        "critical_news_count": len(critical_news),
        "catalysts": [],
        "signal": "NEUTRAL",
    }

    for news in critical_news:
        catalyst = {
            "title": news.title,
            "date": news.published_date,
            "impact": news.impact_level.value,
            "keywords": news.keywords,
        }
        result["catalysts"].append(catalyst)

        # シグナル判定
        if any(kw in NEGATIVE_KEYWORDS for kw in news.keywords):
            result["signal"] = "STRONG_SELL"
        elif "承認申請" in news.keywords or "iPS細胞" in news.keywords:
            result["signal"] = "STRONG_BUY"

    return result
