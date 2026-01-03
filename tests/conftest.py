"""pytest設定とフィクスチャ"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from pharma_stock.storage.models import StockPrice, PipelineEvent


@pytest.fixture
def sample_stock_prices() -> list[StockPrice]:
    """テスト用株価データ"""
    base_date = date(2024, 1, 1)
    prices = []

    for i in range(100):
        prices.append(
            StockPrice(
                ticker="4502.T",
                date=base_date + timedelta(days=i),
                open=Decimal("5000") + Decimal(str(i * 10)),
                high=Decimal("5050") + Decimal(str(i * 10)),
                low=Decimal("4950") + Decimal(str(i * 10)),
                close=Decimal("5000") + Decimal(str(i * 10)),
                volume=1000000 + i * 1000,
                adjusted_close=Decimal("5000") + Decimal(str(i * 10)),
            )
        )

    return prices


@pytest.fixture
def sample_pipeline_events() -> list[PipelineEvent]:
    """テスト用パイプラインイベント"""
    return [
        PipelineEvent(
            ticker="4502.T",
            drug_name="Test Drug A",
            indication="Cancer",
            phase="Phase 3",
            event_type="進行中",
            event_date=date(2024, 1, 15),
            source_url="https://example.com",
            details={"nct_id": "NCT12345678"},
        ),
        PipelineEvent(
            ticker="4502.T",
            drug_name="Test Drug B",
            indication="Diabetes",
            phase="Phase 2",
            event_type="完了",
            event_date=date(2024, 2, 1),
            source_url="https://example.com",
            details={"nct_id": "NCT87654321"},
        ),
    ]
