"""データモデル定義"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import String, Numeric, Integer, Date, DateTime, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy ベースクラス"""

    pass


# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================


class StockPriceDB(Base):
    """株価データ（DB）"""

    __tablename__ = "stock_prices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    high: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    low: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    close: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    volume: Mapped[int] = mapped_column(Integer)
    adjusted_close: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PipelineEventDB(Base):
    """パイプラインイベント（DB）"""

    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    drug_name: Mapped[str] = mapped_column(String(200))
    indication: Mapped[str] = mapped_column(String(200))
    phase: Mapped[str] = mapped_column(String(50))
    event_type: Mapped[str] = mapped_column(String(50))
    event_date: Mapped[date] = mapped_column(Date, index=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PredictionDB(Base):
    """予測結果（DB）"""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    prediction_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date)
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    confidence: Mapped[float] = mapped_column(Numeric(5, 4))
    model_name: Mapped[str] = mapped_column(String(100))
    features_used: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# =============================================================================
# Pydantic Models (API/Domain Layer)
# =============================================================================


class StockPrice(BaseModel):
    """株価データ"""

    ticker: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal

    class Config:
        from_attributes = True


class PipelinePhase(str, Enum):
    """開発フェーズ"""

    PRECLINICAL = "前臨床"
    PHASE1 = "Phase 1"
    PHASE2 = "Phase 2"
    PHASE3 = "Phase 3"
    NDA_FILED = "承認申請中"
    APPROVED = "承認済"


class PipelineEventType(str, Enum):
    """イベントタイプ"""

    STARTED = "開始"
    TOPLINE_POSITIVE = "良好な結果"
    TOPLINE_NEGATIVE = "不良な結果"
    COMPLETED = "完了"
    APPROVED = "承認"
    REJECTED = "却下"
    PARTNERSHIP = "提携"


class PipelineEvent(BaseModel):
    """パイプラインイベント"""

    ticker: str
    drug_name: str
    indication: str
    phase: str
    event_type: str
    event_date: date
    source_url: str | None = None
    details: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class Company(BaseModel):
    """企業情報"""

    ticker: str
    name: str
    name_en: str
    market: str
    focus_areas: list[str]


class Prediction(BaseModel):
    """予測結果"""

    ticker: str
    prediction_date: date
    target_date: date
    predicted_price: Decimal
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    features_used: dict[str, Any] | None = None

    class Config:
        from_attributes = True
