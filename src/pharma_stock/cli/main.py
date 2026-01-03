"""CLIメインエントリポイント"""

import typer
from rich.console import Console

from pharma_stock import __version__

from .commands import collect, analyze, predict, companies

app = typer.Typer(
    name="pharma-stock",
    help="国内製薬企業の株価分析・予測ツール",
    no_args_is_help=True,
)

console = Console()

# サブコマンドを登録
app.add_typer(collect.app, name="collect", help="データ収集コマンド")
app.add_typer(analyze.app, name="analyze", help="分析コマンド")
app.add_typer(predict.app, name="predict", help="予測コマンド")
app.add_typer(companies.app, name="companies", help="企業情報コマンド")


@app.command()
def version() -> None:
    """バージョンを表示"""
    console.print(f"pharma-stock version {__version__}")


@app.command()
def init() -> None:
    """プロジェクトを初期化（データベース作成など）"""
    from pharma_stock.config import get_settings
    from pharma_stock.storage import get_database

    console.print("[bold blue]プロジェクトを初期化しています...[/bold blue]")

    # ディレクトリ作成
    settings = get_settings()
    settings.ensure_directories()
    console.print(f"  ✓ データディレクトリを作成: {settings.data_dir}")

    # データベース初期化
    db = get_database()
    db.create_tables()
    console.print(f"  ✓ データベースを初期化: {settings.database_url}")

    console.print("[bold green]初期化完了！[/bold green]")


if __name__ == "__main__":
    app()
