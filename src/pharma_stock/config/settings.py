"""アプリケーション設定"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定クラス"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PHARMA_STOCK_",
        case_sensitive=False,
    )

    # データベース設定
    database_url: str = "sqlite:///data/pharma_stock.db"

    # パス設定
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")

    # API設定
    request_timeout: int = 30
    request_retry_count: int = 3
    request_retry_delay: float = 1.0

    # 予測モデル設定
    default_model: str = "xgboost"
    prediction_horizon_days: int = 30

    # ロギング設定
    log_level: str = "INFO"
    log_format: str = "json"

    def ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        for path in [self.data_dir, self.raw_data_dir, self.processed_data_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """設定インスタンスを取得（キャッシュ済み）"""
    return Settings()
