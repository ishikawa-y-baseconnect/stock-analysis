"""データベース接続管理"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from pharma_stock.config import get_settings
from .models import Base


class Database:
    """データベース管理クラス"""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self.database_url = database_url or settings.database_url

        # ディレクトリ作成
        settings.ensure_directories()

        self.engine = create_engine(
            self.database_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {},
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self) -> None:
        """テーブルを作成"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """テーブルを削除"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """セッションを取得"""
        return self.SessionLocal()


@lru_cache
def get_database() -> Database:
    """データベースインスタンスを取得（キャッシュ済み）"""
    return Database()
