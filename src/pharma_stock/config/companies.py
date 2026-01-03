"""国内トップティア製薬企業の定義"""

from dataclasses import dataclass
from enum import Enum


class Market(Enum):
    """上場市場"""

    PRIME = "プライム"
    STANDARD = "スタンダード"


@dataclass(frozen=True)
class PharmaCompany:
    """製薬企業データ"""

    ticker: str  # 証券コード（Yahoo Finance形式: XXXX.T）
    code: str  # 証券コード（数字のみ）
    name: str  # 企業名
    name_en: str  # 企業名（英語）
    market: Market  # 上場市場
    focus_areas: tuple[str, ...]  # 主要な研究開発領域


# 国内トップティア製薬企業（時価総額上位）
TOP_TIER_PHARMA_COMPANIES: tuple[PharmaCompany, ...] = (
    PharmaCompany(
        ticker="4502.T",
        code="4502",
        name="武田薬品工業",
        name_en="Takeda Pharmaceutical",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "希少疾患", "消化器系", "神経科学"),
    ),
    PharmaCompany(
        ticker="4503.T",
        code="4503",
        name="アステラス製薬",
        name_en="Astellas Pharma",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "泌尿器", "移植", "眼科"),
    ),
    PharmaCompany(
        ticker="4568.T",
        code="4568",
        name="第一三共",
        name_en="Daiichi Sankyo",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "循環器", "希少疾患", "ワクチン"),
    ),
    PharmaCompany(
        ticker="4523.T",
        code="4523",
        name="エーザイ",
        name_en="Eisai",
        market=Market.PRIME,
        focus_areas=("神経科学", "オンコロジー", "認知症"),
    ),
    PharmaCompany(
        ticker="4519.T",
        code="4519",
        name="中外製薬",
        name_en="Chugai Pharmaceutical",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "免疫", "眼科", "血液"),
    ),
    PharmaCompany(
        ticker="4578.T",
        code="4578",
        name="大塚ホールディングス",
        name_en="Otsuka Holdings",
        market=Market.PRIME,
        focus_areas=("精神・神経", "オンコロジー", "循環器"),
    ),
    PharmaCompany(
        ticker="4506.T",
        code="4506",
        name="住友ファーマ",
        name_en="Sumitomo Pharma",
        market=Market.PRIME,
        focus_areas=("精神・神経", "オンコロジー", "再生医療"),
    ),
    PharmaCompany(
        ticker="4507.T",
        code="4507",
        name="塩野義製薬",
        name_en="Shionogi",
        market=Market.PRIME,
        focus_areas=("感染症", "疼痛", "神経科学"),
    ),
    PharmaCompany(
        ticker="4151.T",
        code="4151",
        name="協和キリン",
        name_en="Kyowa Kirin",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "腎臓", "免疫・アレルギー"),
    ),
    PharmaCompany(
        ticker="4528.T",
        code="4528",
        name="小野薬品工業",
        name_en="Ono Pharmaceutical",
        market=Market.PRIME,
        focus_areas=("オンコロジー", "免疫", "神経"),
    ),
)


def get_company_by_ticker(ticker: str) -> PharmaCompany | None:
    """ティッカーから企業情報を取得"""
    for company in TOP_TIER_PHARMA_COMPANIES:
        if company.ticker == ticker or company.code == ticker:
            return company
    return None


def get_company_by_name(name: str) -> PharmaCompany | None:
    """企業名から企業情報を取得（部分一致）"""
    name_lower = name.lower()
    for company in TOP_TIER_PHARMA_COMPANIES:
        if name_lower in company.name.lower() or name_lower in company.name_en.lower():
            return company
    return None


def get_all_tickers() -> list[str]:
    """全企業のティッカーリストを取得"""
    return [c.ticker for c in TOP_TIER_PHARMA_COMPANIES]
