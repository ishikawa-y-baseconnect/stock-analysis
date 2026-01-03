# Claude Code開発ガイド

このプロジェクトは国内製薬企業の株価分析・底値買いシグナル検知システムです。

## プロジェクト構造

```
src/pharma_stock/
├── analysis/          # 分析モジュール
│   ├── bottom_signal.py    # 底値買いシグナル検知（メイン）
│   ├── catalyst_detector.py # カタリスト検知
│   ├── technical.py         # テクニカル分析
│   └── fundamental.py       # ファンダメンタル分析
├── collectors/        # データ収集
│   ├── stock.py            # 株価データ（yfinance）
│   ├── pipeline.py         # 治験データ（ClinicalTrials.gov）
│   ├── pmda.py             # 承認情報（PMDA）
│   └── ir_news.py          # IRニュース
├── prediction/        # ML予測
│   ├── price_range_predictor.py  # 価格レンジ予測
│   ├── xgboost_predictor.py      # XGBoost
│   ├── lightgbm_predictor.py     # LightGBM
│   └── ensemble.py               # アンサンブル
├── cli/               # CLIコマンド
│   ├── main.py              # エントリポイント
│   └── commands/            # サブコマンド
├── config/            # 設定
│   └── companies.py         # 対象企業リスト
├── storage/           # データ永続化
└── dev/               # 開発ツール
    ├── agents.py            # エージェント定義
    ├── tasks.py             # タスクテンプレート
    └── backtest_runner.py   # バックテストユーティリティ
```

## よく使うコマンド

```bash
# 仮想環境の有効化
source .venv/bin/activate

# 全銘柄スキャン
PYTHONPATH=src python -m pharma_stock.cli.main scan bottom

# 個別銘柄分析
PYTHONPATH=src python -m pharma_stock.cli.main scan detail 4506.T

# クイックバックテスト
PYTHONPATH=src python -c "
from datetime import date
from pharma_stock.dev import quick_backtest
quick_backtest('4506.T', date(2025, 4, 9))
"

# テスト実行
PYTHONPATH=src pytest tests/ -v
```

## 開発エージェント

このプロジェクトには開発を効率化するエージェント定義があります。

### エージェント一覧

| タイプ | 名前 | 用途 |
|--------|------|------|
| `data_collector` | データコレクター | 新しいデータソース追加 |
| `technical_analyzer` | テクニカル分析 | 新しい指標追加 |
| `signal_detector` | シグナル検知 | 閾値・重み調整 |
| `price_predictor` | 価格予測 | MLモデル改善 |
| `backtester` | バックテスト | 精度検証 |
| `code_reviewer` | コードレビュー | 品質チェック |

### エージェント使用例

```python
from pharma_stock.dev import get_agent_prompt, AgentType

# データコレクターエージェントのプロンプトを取得
prompt = get_agent_prompt(
    AgentType.DATA_COLLECTOR,
    "決算短信PDFからデータを収集するコレクターを追加"
)
```

## タスクテンプレート

よく使うタスクのテンプレートが定義されています。

### タスク一覧

| キー | タスク名 | 説明 |
|------|----------|------|
| `add_data_source` | 新しいデータソース追加 | コレクター作成 |
| `add_technical_indicator` | テクニカル指標追加 | 分析ロジック拡張 |
| `add_fundamental_metric` | ファンダメンタル指標追加 | 割安評価拡張 |
| `improve_prediction_model` | 予測モデル改善 | ML精度向上 |
| `adjust_signal_thresholds` | シグナル閾値調整 | 感度調整 |
| `adjust_score_weights` | スコア重み調整 | バランス調整 |
| `add_cli_command` | CLIコマンド追加 | 新コマンド作成 |
| `add_catalyst_pattern` | カタリストパターン追加 | 検知パターン拡張 |

### タスク使用例

```python
from pharma_stock.dev import get_task_prompt

# タスクのプロンプトを取得
prompt = get_task_prompt("add_technical_indicator")
print(prompt)
```

## バックテスト

変更の効果を検証するためのバックテストツールがあります。

```python
from datetime import date
from pharma_stock.dev import BacktestRunner

# ランナー作成
runner = BacktestRunner()

# 単一銘柄・日付でテスト
result = runner.run_single("4506.T", date(2025, 4, 9))
print(f"シグナル: {result.signal_strength.value}")
print(f"予測高値: ¥{result.predicted_high_6m:,.0f}")
print(f"実績高値: ¥{result.actual_high_6m:,.0f}")

# 全銘柄でテスト
results = runner.run_all_companies(date(2025, 4, 1))
report = runner.generate_report(results)
print(f"シグナル正解率: {report['summary']['signal_accuracy']:.1%}")

# DataFrameで出力
df = runner.to_dataframe(results)
df.to_csv("backtest_results.csv")
```

## シグナルシステムの仕組み

### スコア構成（合計100点）

| コンポーネント | 重み | 評価内容 |
|----------------|------|----------|
| テクニカル | 25% | RSI、BB、52週安値、連続下落 |
| ファンダメンタル | 20% | PER、PBR、配当利回り |
| 上昇ポテンシャル | 25% | Phase 3試験数、アナリスト目標 |
| ML予測 | 30% | 3-6ヶ月後の期待リターン |

### シグナル閾値

| スコア | シグナル |
|--------|----------|
| 60+ | 強い買い |
| 45-59 | 買い |
| 30-44 | 様子見 |
| <30 | 見送り |

## 新機能追加時のチェックリスト

1. [ ] 実装完了
2. [ ] バックテストで効果検証
3. [ ] 既存機能への影響確認
4. [ ] コミット・プッシュ

## 対象企業

```python
from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES
for c in TOP_TIER_PHARMA_COMPANIES:
    print(f"{c.ticker}: {c.name}")
```

| ティッカー | 企業名 |
|------------|--------|
| 4502.T | 武田薬品工業 |
| 4503.T | アステラス製薬 |
| 4568.T | 第一三共 |
| 4523.T | エーザイ |
| 4519.T | 中外製薬 |
| 4578.T | 大塚ホールディングス |
| 4506.T | 住友ファーマ |
| 4507.T | 塩野義製薬 |
| 4151.T | 協和キリン |
| 4528.T | 小野薬品工業 |
