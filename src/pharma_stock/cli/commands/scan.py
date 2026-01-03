"""底値スキャンコマンド"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="底値買いシグナルスキャン")
console = Console()


@app.command("bottom")
def scan_bottom(
    target_return: float = typer.Option(
        0.20, "--target", "-t", help="目標リターン（0.20 = +20%）"
    ),
    stop_loss: float = typer.Option(
        -0.10, "--stop", "-s", help="損切りライン（-0.10 = -10%）"
    ),
) -> None:
    """全銘柄の底値買いシグナルをスキャン"""
    from pharma_stock.analysis.bottom_signal import BottomSignalDetector, SignalStrength
    from pharma_stock.collectors import StockCollector, PipelineCollector
    from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES
    import yfinance as yf

    console.print(
        Panel(
            f"[bold]底値買いシグナルスキャン[/bold]\n"
            f"目標リターン: +{target_return*100:.0f}%\n"
            f"損切りライン: {stop_loss*100:.0f}%\n"
            f"リスクリワード比: {target_return/abs(stop_loss):.1f}:1",
            expand=False,
        )
    )

    detector = BottomSignalDetector(target_return=target_return, stop_loss=stop_loss)
    stock_collector = StockCollector()
    pipeline_collector = PipelineCollector()

    signals = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("スキャン中...", total=len(TOP_TIER_PHARMA_COMPANIES))

        for company in TOP_TIER_PHARMA_COMPANIES:
            progress.update(task, description=f"{company.name}を分析中...")

            try:
                # 株価データ
                prices = stock_collector.collect(tickers=[company.ticker], period="1y")
                if len(prices) < 100:
                    progress.advance(task)
                    continue

                # パイプライン
                events = pipeline_collector.collect(tickers=[company.ticker], limit=50)

                # ファンダメンタル
                stock = yf.Ticker(company.ticker)
                fundamental = stock.info

                signal = detector.detect(
                    prices=prices,
                    company_name=company.name,
                    pipeline_events=events,
                    fundamental_data=fundamental,
                )

                signals.append(signal)

            except Exception as e:
                console.print(f"[dim]{company.name}: エラー ({e})[/dim]")

            progress.advance(task)

    if not signals:
        console.print("[red]シグナルを取得できませんでした[/red]")
        raise typer.Exit(1)

    # スコア順にソート
    signals.sort(key=lambda x: x.total_score, reverse=True)

    # 結果テーブル
    table = Table(title="底値買いシグナル スキャン結果")
    table.add_column("順位", style="dim", justify="right")
    table.add_column("銘柄", style="cyan")
    table.add_column("現在価格", justify="right")
    table.add_column("総合", justify="right")
    table.add_column("技術", justify="right")
    table.add_column("割安", justify="right")
    table.add_column("上昇", justify="right")
    table.add_column("シグナル")
    table.add_column("目標価格", justify="right")
    table.add_column("損切り", justify="right")

    for i, sig in enumerate(signals, 1):
        # シグナル色
        if sig.signal_strength == SignalStrength.STRONG_BUY:
            signal_str = "[bold green]強い買い[/bold green]"
        elif sig.signal_strength == SignalStrength.BUY:
            signal_str = "[green]買い[/green]"
        elif sig.signal_strength == SignalStrength.NEUTRAL:
            signal_str = "[yellow]様子見[/yellow]"
        else:
            signal_str = "[dim]見送り[/dim]"

        table.add_row(
            str(i),
            sig.company_name,
            f"¥{sig.current_price:,.0f}",
            f"{sig.total_score:.0f}",
            f"{sig.technical_score:.0f}",
            f"{sig.fundamental_score:.0f}",
            f"{sig.upside_potential_score:.0f}",
            signal_str,
            f"¥{sig.target_price:,.0f}",
            f"¥{sig.stop_loss_price:,.0f}",
        )

    console.print(table)

    # 買いシグナルの詳細
    buy_signals = [
        s for s in signals
        if s.signal_strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]
    ]

    if buy_signals:
        console.print("\n[bold]買いシグナル詳細[/bold]")

        for sig in buy_signals[:3]:  # 上位3銘柄
            console.print(
                Panel(
                    f"[bold cyan]{sig.company_name}[/bold cyan] ({sig.ticker})\n\n"
                    f"[bold]テクニカル理由:[/bold]\n"
                    + "\n".join(f"  • {r}" for r in sig.technical_reasons)
                    + f"\n\n[bold]ファンダメンタル理由:[/bold]\n"
                    + "\n".join(f"  • {r}" for r in sig.fundamental_reasons)
                    + f"\n\n[bold]上昇ポテンシャル:[/bold]\n"
                    + "\n".join(f"  • {r}" for r in sig.upside_reasons)
                    + (
                        f"\n\n[bold red]リスク:[/bold red]\n"
                        + "\n".join(f"  • {r}" for r in sig.risks)
                        if sig.risks
                        else ""
                    ),
                    title=f"総合スコア: {sig.total_score:.0f}点",
                    expand=False,
                )
            )
    else:
        console.print("\n[yellow]現在、明確な買いシグナルはありません[/yellow]")


@app.command("detail")
def scan_detail(
    ticker: str = typer.Argument(..., help="ティッカーシンボル（例: 4506.T）"),
) -> None:
    """特定銘柄の底値シグナル詳細分析"""
    from pharma_stock.analysis.bottom_signal import BottomSignalDetector
    from pharma_stock.collectors import StockCollector, PipelineCollector
    from pharma_stock.config.companies import get_company_by_ticker
    import yfinance as yf

    company = get_company_by_ticker(ticker)
    if not company:
        console.print(f"[red]企業が見つかりません: {ticker}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]{company.name} の底値シグナル分析中...[/bold blue]")

    detector = BottomSignalDetector()
    stock_collector = StockCollector()
    pipeline_collector = PipelineCollector()

    # データ収集
    prices = stock_collector.collect(tickers=[ticker], period="1y")
    events = pipeline_collector.collect(tickers=[ticker], limit=50)
    stock = yf.Ticker(ticker)
    fundamental = stock.info

    signal = detector.detect(
        prices=prices,
        company_name=company.name,
        pipeline_events=events,
        fundamental_data=fundamental,
    )

    # 結果表示
    console.print(
        Panel(
            f"[bold]{signal.company_name}[/bold]\n"
            f"現在価格: ¥{signal.current_price:,.0f}\n"
            f"分析日: {signal.signal_date}",
            expand=False,
        )
    )

    # スコアテーブル
    table = Table(title="スコア内訳")
    table.add_column("項目", style="cyan")
    table.add_column("スコア", justify="right")
    table.add_column("重み", justify="right")
    table.add_column("寄与", justify="right")

    table.add_row(
        "テクニカル底値",
        f"{signal.technical_score:.0f}/100",
        "35%",
        f"{signal.technical_score * 0.35:.1f}",
    )
    table.add_row(
        "ファンダメンタル割安",
        f"{signal.fundamental_score:.0f}/100",
        "30%",
        f"{signal.fundamental_score * 0.30:.1f}",
    )
    table.add_row(
        "上昇ポテンシャル",
        f"{signal.upside_potential_score:.0f}/100",
        "35%",
        f"{signal.upside_potential_score * 0.35:.1f}",
    )
    table.add_row(
        "[bold]総合スコア[/bold]",
        f"[bold]{signal.total_score:.0f}/100[/bold]",
        "100%",
        f"[bold]{signal.total_score:.1f}[/bold]",
    )

    console.print(table)

    # シグナル
    console.print(f"\n[bold]シグナル: {signal.signal_strength.value}[/bold]")
    console.print(f"目標価格（+20%）: ¥{signal.target_price:,.0f}")
    console.print(f"損切りライン（-10%）: ¥{signal.stop_loss_price:,.0f}")
    console.print(f"リスクリワード比: {signal.risk_reward_ratio:.1f}:1")

    # 詳細理由
    if signal.technical_reasons:
        console.print("\n[bold]テクニカル分析:[/bold]")
        for r in signal.technical_reasons:
            console.print(f"  • {r}")

    if signal.fundamental_reasons:
        console.print("\n[bold]ファンダメンタル分析:[/bold]")
        for r in signal.fundamental_reasons:
            console.print(f"  • {r}")

    if signal.upside_reasons:
        console.print("\n[bold]上昇ポテンシャル:[/bold]")
        for r in signal.upside_reasons:
            console.print(f"  • {r}")

    if signal.risks:
        console.print("\n[bold red]リスク要因:[/bold red]")
        for r in signal.risks:
            console.print(f"  • {r}")
