"""企業情報コマンド"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pharma_stock.config.companies import (
    TOP_TIER_PHARMA_COMPANIES,
    get_company_by_ticker,
    get_company_by_name,
)
from pharma_stock.collectors import StockCollector

app = typer.Typer(help="企業情報コマンド")
console = Console()


@app.command("list")
def list_companies() -> None:
    """対象企業一覧を表示"""
    table = Table(title="国内トップティア製薬企業一覧")
    table.add_column("ティッカー", style="cyan")
    table.add_column("証券コード")
    table.add_column("企業名")
    table.add_column("英語名")
    table.add_column("主要領域")

    for company in TOP_TIER_PHARMA_COMPANIES:
        table.add_row(
            company.ticker,
            company.code,
            company.name,
            company.name_en,
            ", ".join(company.focus_areas[:3]),
        )

    console.print(table)
    console.print(f"\n合計: {len(TOP_TIER_PHARMA_COMPANIES)} 社")


@app.command("info")
def company_info(
    query: str = typer.Argument(..., help="ティッカーまたは企業名"),
) -> None:
    """企業詳細情報を表示"""
    # ティッカーで検索
    company = get_company_by_ticker(query)

    # 見つからなければ企業名で検索
    if company is None:
        company = get_company_by_name(query)

    if company is None:
        console.print(f"[red]エラー: '{query}' に一致する企業が見つかりませんでした[/red]")
        raise typer.Exit(1)

    # Yahoo Financeから追加情報を取得
    collector = StockCollector()

    try:
        yf_info = collector.get_company_info(company.ticker)
    except Exception:
        yf_info = {}

    # 表示
    console.print(
        Panel(
            f"[bold]{company.name}[/bold]\n{company.name_en}",
            subtitle=f"ティッカー: {company.ticker}",
            expand=False,
        )
    )

    # 基本情報
    table = Table(title="基本情報")
    table.add_column("項目", style="cyan")
    table.add_column("値")

    table.add_row("証券コード", company.code)
    table.add_row("ティッカー", company.ticker)
    table.add_row("上場市場", company.market.value)

    if yf_info.get("market_cap"):
        table.add_row("時価総額", f"¥{yf_info['market_cap'] / 1e12:.2f}兆")

    if yf_info.get("employees"):
        table.add_row("従業員数", f"{yf_info['employees']:,}人")

    if yf_info.get("website"):
        table.add_row("Webサイト", yf_info["website"])

    console.print(table)

    # 研究開発領域
    console.print("\n[bold]主要研究開発領域:[/bold]")
    for area in company.focus_areas:
        console.print(f"  • {area}")

    # 企業説明
    if yf_info.get("description"):
        console.print(f"\n[bold]企業概要:[/bold]")
        desc = yf_info["description"]
        # 長すぎる場合は省略
        if len(desc) > 500:
            desc = desc[:500] + "..."
        console.print(f"  {desc}")


@app.command("prices")
def show_prices() -> None:
    """全企業の最新株価を表示"""
    console.print("[bold blue]最新株価を取得中...[/bold blue]")

    collector = StockCollector()
    latest_prices = collector.get_all_companies_latest()

    table = Table(title="国内製薬企業 最新株価")
    table.add_column("ティッカー", style="cyan")
    table.add_column("企業名")
    table.add_column("終値", justify="right")
    table.add_column("日付")

    for company in TOP_TIER_PHARMA_COMPANIES:
        price = latest_prices.get(company.ticker)
        if price:
            table.add_row(
                company.ticker,
                company.name,
                f"¥{price.close:,.0f}",
                str(price.date),
            )
        else:
            table.add_row(
                company.ticker,
                company.name,
                "-",
                "-",
            )

    console.print(table)
