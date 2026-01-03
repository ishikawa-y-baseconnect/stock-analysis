"""分析コマンド"""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pharma_stock.collectors import StockCollector
from pharma_stock.analysis import TechnicalAnalyzer, FundamentalAnalyzer
from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES, get_company_by_ticker

app = typer.Typer(help="分析コマンド")
console = Console()


@app.command("technical")
def analyze_technical(
    ticker: str = typer.Argument(..., help="ティッカーシンボル（例: 4502.T）"),
    period: str = typer.Option("1y", "--period", "-p", help="分析期間"),
) -> None:
    """テクニカル分析を実行"""
    console.print(f"[bold blue]{ticker} のテクニカル分析を実行中...[/bold blue]")

    # データ収集
    collector = StockCollector()
    prices = collector.collect(tickers=[ticker], period=period)

    if not prices:
        console.print(f"[red]エラー: {ticker} のデータが取得できませんでした[/red]")
        raise typer.Exit(1)

    # テクニカル分析
    analyzer = TechnicalAnalyzer()
    indicators = analyzer.analyze(prices)

    if indicators is None:
        console.print("[red]エラー: 分析に必要なデータが不足しています[/red]")
        raise typer.Exit(1)

    company = get_company_by_ticker(ticker)
    company_name = company.name if company else ticker

    # 結果表示
    console.print(Panel(f"[bold]{company_name}[/bold] テクニカル分析結果", expand=False))

    # シグナル表示
    signal_color = {
        "強い買い": "green",
        "買い": "cyan",
        "中立": "yellow",
        "売り": "orange1",
        "強い売り": "red",
    }.get(indicators.trend_signal.value, "white")

    console.print(
        f"\n[bold]総合シグナル:[/bold] [{signal_color}]{indicators.trend_signal.value}[/{signal_color}]"
        f" (強度: {indicators.signal_strength:.1f}%)"
    )

    # 移動平均
    table = Table(title="移動平均")
    table.add_column("指標", style="cyan")
    table.add_column("値", justify="right")

    if indicators.sma_5:
        table.add_row("SMA(5)", f"¥{indicators.sma_5:,.0f}")
    if indicators.sma_20:
        table.add_row("SMA(20)", f"¥{indicators.sma_20:,.0f}")
    if indicators.sma_50:
        table.add_row("SMA(50)", f"¥{indicators.sma_50:,.0f}")
    if indicators.sma_200:
        table.add_row("SMA(200)", f"¥{indicators.sma_200:,.0f}")

    console.print(table)

    # モメンタム指標
    table2 = Table(title="モメンタム指標")
    table2.add_column("指標", style="cyan")
    table2.add_column("値", justify="right")
    table2.add_column("判定")

    if indicators.rsi_14:
        rsi_judgment = (
            "売られすぎ" if indicators.rsi_14 < 30
            else "買われすぎ" if indicators.rsi_14 > 70
            else "中立"
        )
        table2.add_row("RSI(14)", f"{indicators.rsi_14:.1f}", rsi_judgment)

    if indicators.macd is not None and indicators.macd_signal is not None:
        macd_judgment = "買いシグナル" if indicators.macd > indicators.macd_signal else "売りシグナル"
        table2.add_row("MACD", f"{indicators.macd:.2f}", macd_judgment)

    if indicators.adx_14:
        adx_judgment = "強いトレンド" if indicators.adx_14 > 25 else "弱いトレンド"
        table2.add_row("ADX(14)", f"{indicators.adx_14:.1f}", adx_judgment)

    console.print(table2)

    # ボリンジャーバンド
    if indicators.bb_upper and indicators.bb_lower:
        table3 = Table(title="ボリンジャーバンド")
        table3.add_column("指標", style="cyan")
        table3.add_column("値", justify="right")

        table3.add_row("上限", f"¥{indicators.bb_upper:,.0f}")
        table3.add_row("中央", f"¥{indicators.bb_middle:,.0f}")
        table3.add_row("下限", f"¥{indicators.bb_lower:,.0f}")
        if indicators.bb_width:
            table3.add_row("幅", f"{indicators.bb_width:.2f}%")

        console.print(table3)


@app.command("fundamental")
def analyze_fundamental(
    ticker: Optional[str] = typer.Option(
        None, "--ticker", "-t", help="ティッカーシンボル（指定なしで全企業比較）"
    ),
) -> None:
    """ファンダメンタル分析を実行"""
    analyzer = FundamentalAnalyzer()

    if ticker:
        # 単一企業の分析
        console.print(f"[bold blue]{ticker} のファンダメンタル分析を実行中...[/bold blue]")

        metrics = analyzer.get_metrics(ticker)
        if metrics is None:
            console.print(f"[red]エラー: {ticker} のデータが取得できませんでした[/red]")
            raise typer.Exit(1)

        company = get_company_by_ticker(ticker)
        company_name = company.name if company else ticker

        console.print(Panel(f"[bold]{company_name}[/bold] ファンダメンタル分析", expand=False))

        # バリュエーション
        table = Table(title="バリュエーション指標")
        table.add_column("指標", style="cyan")
        table.add_column("値", justify="right")

        if metrics.market_cap:
            table.add_row("時価総額", f"¥{metrics.market_cap / 1e12:.2f}兆")
        if metrics.pe_ratio:
            table.add_row("PER", f"{metrics.pe_ratio:.1f}倍")
        if metrics.pb_ratio:
            table.add_row("PBR", f"{metrics.pb_ratio:.2f}倍")
        if metrics.ps_ratio:
            table.add_row("PSR", f"{metrics.ps_ratio:.2f}倍")
        if metrics.ev_ebitda:
            table.add_row("EV/EBITDA", f"{metrics.ev_ebitda:.1f}倍")

        console.print(table)

        # 収益性
        table2 = Table(title="収益性指標")
        table2.add_column("指標", style="cyan")
        table2.add_column("値", justify="right")

        if metrics.profit_margin:
            table2.add_row("利益率", f"{metrics.profit_margin:.1f}%")
        if metrics.operating_margin:
            table2.add_row("営業利益率", f"{metrics.operating_margin:.1f}%")
        if metrics.roe:
            table2.add_row("ROE", f"{metrics.roe:.1f}%")
        if metrics.roa:
            table2.add_row("ROA", f"{metrics.roa:.1f}%")

        console.print(table2)

    else:
        # 同業他社比較
        console.print("[bold blue]国内製薬企業の比較分析を実行中...[/bold blue]")

        comparisons = analyzer.compare_peers()

        if not comparisons:
            console.print("[red]エラー: データが取得できませんでした[/red]")
            raise typer.Exit(1)

        table = Table(title="国内製薬企業 比較分析")
        table.add_column("順位", justify="right")
        table.add_column("企業名", style="cyan")
        table.add_column("時価総額", justify="right")
        table.add_column("PER", justify="right")
        table.add_column("利益率", justify="right")
        table.add_column("ROE", justify="right")

        for i, comp in enumerate(comparisons, 1):
            market_cap = comp.metrics.market_cap
            market_cap_str = f"¥{market_cap / 1e12:.2f}兆" if market_cap else "-"

            table.add_row(
                str(i),
                comp.name,
                market_cap_str,
                f"{comp.metrics.pe_ratio:.1f}" if comp.metrics.pe_ratio else "-",
                f"{comp.metrics.profit_margin:.1f}%" if comp.metrics.profit_margin else "-",
                f"{comp.metrics.roe:.1f}%" if comp.metrics.roe else "-",
            )

        console.print(table)


@app.command("summary")
def analyze_summary(
    ticker: str = typer.Argument(..., help="ティッカーシンボル"),
) -> None:
    """銘柄のサマリー分析（テクニカル＋ファンダメンタル）"""
    console.print(f"\n[bold blue]═══ {ticker} 総合分析 ═══[/bold blue]\n")

    # テクニカル分析
    console.print("[bold]【テクニカル分析】[/bold]")
    analyze_technical(ticker, "1y")

    console.print()

    # ファンダメンタル分析
    console.print("[bold]【ファンダメンタル分析】[/bold]")
    analyze_fundamental(ticker)
