"""予測コマンド"""

from datetime import date, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pharma_stock.collectors import StockCollector, PipelineCollector
from pharma_stock.prediction import (
    FeatureEngineer,
    XGBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor,
)
from pharma_stock.config.companies import get_company_by_ticker, get_all_tickers

app = typer.Typer(help="予測コマンド")
console = Console()


@app.command("train")
def train_model(
    ticker: str = typer.Argument(..., help="ティッカーシンボル"),
    model: str = typer.Option(
        "ensemble", "--model", "-m", help="モデル種類（xgboost, lightgbm, ensemble）"
    ),
    horizon: int = typer.Option(5, "--horizon", "-h", help="予測対象日数（何日後を予測するか）"),
    period: str = typer.Option("2y", "--period", "-p", help="学習データ期間"),
) -> None:
    """予測モデルを学習"""
    console.print(f"[bold blue]{ticker} の予測モデルを学習中...[/bold blue]")
    console.print(f"  モデル: {model}")
    console.print(f"  予測対象: {horizon}日後")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # データ収集
        task = progress.add_task("株価データを収集中...", total=None)
        stock_collector = StockCollector()
        prices = stock_collector.collect(tickers=[ticker], period=period)
        progress.update(task, completed=True)

        if len(prices) < 100:
            console.print("[red]エラー: 学習に十分なデータがありません（100件以上必要）[/red]")
            raise typer.Exit(1)

        # パイプライン情報
        task = progress.add_task("パイプライン情報を収集中...", total=None)
        pipeline_collector = PipelineCollector()
        events = pipeline_collector.collect(tickers=[ticker])
        progress.update(task, completed=True)

        # 特徴量作成
        task = progress.add_task("特徴量を作成中...", total=None)
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.create_features(prices, events, target_horizon=horizon)
        progress.update(task, completed=True)

        console.print(f"  学習データ: {len(X)} サンプル, {len(X.columns)} 特徴量")

        # モデル選択と学習
        task = progress.add_task("モデルを学習中...", total=None)

        if model == "xgboost":
            predictor = XGBoostPredictor()
        elif model == "lightgbm":
            predictor = LightGBMPredictor()
        else:
            predictor = EnsemblePredictor()

        # 学習/検証分割
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        predictor.fit(X_train, y_train, eval_set=(X_val, y_val))
        progress.update(task, completed=True)

        # 評価
        task = progress.add_task("モデルを評価中...", total=None)
        metrics = predictor.evaluate(X_val, y_val)
        progress.update(task, completed=True)

        # モデル保存
        task = progress.add_task("モデルを保存中...", total=None)
        save_path = predictor.save()
        progress.update(task, completed=True)

    # 結果表示
    console.print("\n[bold green]学習完了！[/bold green]")

    table = Table(title="評価指標")
    table.add_column("指標", style="cyan")
    table.add_column("値", justify="right")

    table.add_row("MAE", f"{metrics['mae']:.4f}%")
    table.add_row("RMSE", f"{metrics['rmse']:.4f}%")
    table.add_row("R²", f"{metrics['r2']:.4f}")
    table.add_row("MAPE", f"{metrics['mape']:.2f}%")

    console.print(table)
    console.print(f"\nモデル保存先: {save_path}")

    # 特徴量重要度（トップ10）
    importance = predictor.get_feature_importance()
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        table2 = Table(title="特徴量重要度（トップ10）")
        table2.add_column("特徴量", style="cyan")
        table2.add_column("重要度", justify="right")

        for feature, imp in sorted_imp:
            table2.add_row(feature, f"{imp:.4f}")

        console.print(table2)


@app.command("price")
def predict_price(
    ticker: str = typer.Argument(..., help="ティッカーシンボル"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="予測対象日数"),
    model: str = typer.Option(
        "ensemble", "--model", "-m", help="使用するモデル"
    ),
) -> None:
    """株価を予測"""
    console.print(f"[bold blue]{ticker} の株価予測を実行中...[/bold blue]")

    company = get_company_by_ticker(ticker)
    company_name = company.name if company else ticker

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # データ収集
        task = progress.add_task("最新データを取得中...", total=None)
        stock_collector = StockCollector()
        prices = stock_collector.collect(tickers=[ticker], period="1y")
        progress.update(task, completed=True)

        if not prices:
            console.print(f"[red]エラー: {ticker} のデータが取得できませんでした[/red]")
            raise typer.Exit(1)

        # 現在価格
        current_price = float(prices[-1].close)
        current_date = prices[-1].date

        # 特徴量作成
        task = progress.add_task("特徴量を作成中...", total=None)
        feature_engineer = FeatureEngineer()
        X, _ = feature_engineer.create_features(prices, target_horizon=horizon)
        X_latest = X.tail(1)
        progress.update(task, completed=True)

        # モデル読み込みまたは簡易学習
        task = progress.add_task("予測を実行中...", total=None)

        if model == "xgboost":
            predictor = XGBoostPredictor()
        elif model == "lightgbm":
            predictor = LightGBMPredictor()
        else:
            predictor = EnsemblePredictor()

        # 簡易学習（本番では保存済みモデルを使用）
        X_full, y_full = feature_engineer.create_features(prices, target_horizon=horizon)
        predictor.fit(X_full, y_full)

        # 予測
        pred, lower, upper = predictor.predict_with_interval(X_latest)
        predicted_change = pred[0]
        lower_change = lower[0]
        upper_change = upper[0]

        predicted_price = current_price * (1 + predicted_change / 100)
        lower_price = current_price * (1 + lower_change / 100)
        upper_price = current_price * (1 + upper_change / 100)

        target_date = current_date + timedelta(days=horizon)

        progress.update(task, completed=True)

    # 結果表示
    console.print(
        Panel(
            f"[bold]{company_name}[/bold] 株価予測\n"
            f"予測日: {current_date} → {target_date} ({horizon}営業日後)",
            expand=False,
        )
    )

    table = Table()
    table.add_column("項目", style="cyan")
    table.add_column("価格", justify="right")
    table.add_column("変動率", justify="right")

    table.add_row(
        "現在価格",
        f"¥{current_price:,.0f}",
        "-",
    )

    change_color = "green" if predicted_change > 0 else "red" if predicted_change < 0 else "white"
    table.add_row(
        "[bold]予測価格[/bold]",
        f"[bold]¥{predicted_price:,.0f}[/bold]",
        f"[{change_color}]{predicted_change:+.2f}%[/{change_color}]",
    )

    table.add_row(
        "予測下限（95%CI）",
        f"¥{lower_price:,.0f}",
        f"{lower_change:+.2f}%",
    )

    table.add_row(
        "予測上限（95%CI）",
        f"¥{upper_price:,.0f}",
        f"{upper_change:+.2f}%",
    )

    console.print(table)

    # シグナル
    if predicted_change > 3:
        signal = "[bold green]強い買いシグナル[/bold green]"
    elif predicted_change > 1:
        signal = "[green]買いシグナル[/green]"
    elif predicted_change < -3:
        signal = "[bold red]強い売りシグナル[/bold red]"
    elif predicted_change < -1:
        signal = "[red]売りシグナル[/red]"
    else:
        signal = "[yellow]中立[/yellow]"

    console.print(f"\n予測シグナル: {signal}")
    console.print(
        "\n[dim]※ これは統計的予測であり、投資助言ではありません。"
        "投資判断は自己責任でお願いします。[/dim]"
    )


@app.command("all")
def predict_all(
    horizon: int = typer.Option(5, "--horizon", "-h", help="予測対象日数"),
) -> None:
    """全銘柄の株価予測サマリー"""
    console.print("[bold blue]全銘柄の株価予測を実行中...[/bold blue]")

    from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES

    results = []

    for company in TOP_TIER_PHARMA_COMPANIES:
        try:
            # データ収集
            stock_collector = StockCollector()
            prices = stock_collector.collect(tickers=[company.ticker], period="1y")

            if len(prices) < 100:
                continue

            # 特徴量作成と予測
            feature_engineer = FeatureEngineer()
            X, y = feature_engineer.create_features(prices, target_horizon=horizon)

            predictor = EnsemblePredictor()
            predictor.fit(X, y)

            X_latest = X.tail(1)
            pred, _, _ = predictor.predict_with_interval(X_latest)

            current_price = float(prices[-1].close)
            predicted_change = pred[0]

            results.append({
                "ticker": company.ticker,
                "name": company.name,
                "current_price": current_price,
                "predicted_change": predicted_change,
            })

        except Exception:
            continue

    if not results:
        console.print("[red]エラー: 予測を実行できませんでした[/red]")
        raise typer.Exit(1)

    # 変動率でソート
    results.sort(key=lambda x: x["predicted_change"], reverse=True)

    table = Table(title=f"全銘柄 {horizon}日後予測サマリー")
    table.add_column("ティッカー", style="cyan")
    table.add_column("企業名")
    table.add_column("現在価格", justify="right")
    table.add_column("予測変動率", justify="right")
    table.add_column("シグナル")

    for r in results:
        change_color = (
            "green" if r["predicted_change"] > 0
            else "red" if r["predicted_change"] < 0
            else "white"
        )

        if r["predicted_change"] > 3:
            signal = "強い買い"
        elif r["predicted_change"] > 1:
            signal = "買い"
        elif r["predicted_change"] < -3:
            signal = "強い売り"
        elif r["predicted_change"] < -1:
            signal = "売り"
        else:
            signal = "中立"

        table.add_row(
            r["ticker"],
            r["name"],
            f"¥{r['current_price']:,.0f}",
            f"[{change_color}]{r['predicted_change']:+.2f}%[/{change_color}]",
            signal,
        )

    console.print(table)
