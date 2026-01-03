"""データ収集コマンド"""

from datetime import date, datetime
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pharma_stock.collectors import StockCollector, PipelineCollector
from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES, get_all_tickers

app = typer.Typer(help="データ収集コマンド")
console = Console()


@app.command("stock")
def collect_stock(
    ticker: Optional[str] = typer.Option(
        None, "--ticker", "-t", help="ティッカーシンボル（例: 4502.T）。指定なしで全企業"
    ),
    period: str = typer.Option(
        "1y", "--period", "-p", help="取得期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max）"
    ),
    start: Optional[str] = typer.Option(
        None, "--start", "-s", help="開始日（YYYY-MM-DD）"
    ),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="終了日（YYYY-MM-DD）"),
) -> None:
    """株価データを収集"""
    collector = StockCollector()

    tickers = [ticker] if ticker else get_all_tickers()
    start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else None
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else None

    console.print(f"[bold blue]株価データを収集しています...[/bold blue]")
    console.print(f"  対象: {len(tickers)} 銘柄")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("収集中...", total=None)

        if start_date:
            prices = collector.collect(
                tickers=tickers, start_date=start_date, end_date=end_date
            )
        else:
            prices = collector.collect(tickers=tickers, period=period)

        progress.update(task, completed=True)

    # 結果をテーブル表示
    table = Table(title="収集結果サマリー")
    table.add_column("ティッカー", style="cyan")
    table.add_column("企業名")
    table.add_column("データ件数", justify="right")
    table.add_column("最新日付")
    table.add_column("最新終値", justify="right")

    # ティッカーごとに集計
    prices_by_ticker: dict[str, list] = {}
    for p in prices:
        if p.ticker not in prices_by_ticker:
            prices_by_ticker[p.ticker] = []
        prices_by_ticker[p.ticker].append(p)

    for t in tickers:
        ticker_prices = prices_by_ticker.get(t, [])
        if ticker_prices:
            latest = max(ticker_prices, key=lambda x: x.date)
            company = next(
                (c for c in TOP_TIER_PHARMA_COMPANIES if c.ticker == t), None
            )
            name = company.name if company else "-"
            table.add_row(
                t,
                name,
                str(len(ticker_prices)),
                str(latest.date),
                f"¥{latest.close:,.0f}",
            )
        else:
            table.add_row(t, "-", "0", "-", "-")

    console.print(table)
    console.print(f"\n[bold green]合計 {len(prices)} 件のデータを収集しました[/bold green]")


@app.command("pipeline")
def collect_pipeline(
    ticker: Optional[str] = typer.Option(
        None, "--ticker", "-t", help="ティッカーシンボル"
    ),
    phase: Optional[str] = typer.Option(
        None, "--phase", help="フェーズフィルタ（PHASE1, PHASE2, PHASE3）"
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="取得件数上限"),
) -> None:
    """製薬パイプライン情報を収集（ClinicalTrials.gov）"""
    collector = PipelineCollector()

    tickers = [ticker] if ticker else None

    console.print("[bold blue]パイプライン情報を収集しています...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("ClinicalTrials.govから取得中...", total=None)
        events = collector.collect(tickers=tickers, phase=phase, limit=limit)
        progress.update(task, completed=True)

    if not events:
        console.print("[yellow]パイプラインイベントが見つかりませんでした[/yellow]")
        return

    # 結果をテーブル表示
    table = Table(title=f"パイプラインイベント（{len(events)}件）")
    table.add_column("ティッカー", style="cyan")
    table.add_column("薬品名", max_width=40)
    table.add_column("フェーズ")
    table.add_column("適応症", max_width=30)
    table.add_column("ステータス")
    table.add_column("日付")

    for event in events[:20]:  # 最大20件表示
        table.add_row(
            event.ticker,
            event.drug_name[:40] + "..." if len(event.drug_name) > 40 else event.drug_name,
            event.phase,
            event.indication[:30] + "..." if len(event.indication) > 30 else event.indication,
            event.event_type,
            str(event.event_date),
        )

    console.print(table)

    if len(events) > 20:
        console.print(f"\n[dim]... 他 {len(events) - 20} 件[/dim]")


@app.command("all")
def collect_all(
    period: str = typer.Option("1y", "--period", "-p", help="株価データの取得期間"),
) -> None:
    """全データを一括収集"""
    console.print("[bold blue]全データを収集しています...[/bold blue]\n")

    # 株価データ
    console.print("[bold]1. 株価データ[/bold]")
    collect_stock(ticker=None, period=period, start=None, end=None)

    console.print()

    # パイプライン情報
    console.print("[bold]2. パイプライン情報[/bold]")
    collect_pipeline(ticker=None, phase=None, limit=100)

    console.print("\n[bold green]全データの収集が完了しました[/bold green]")
