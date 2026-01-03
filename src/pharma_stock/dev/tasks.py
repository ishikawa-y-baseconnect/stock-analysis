"""開発タスク定義

よく使う開発タスクをテンプレート化
Claude Codeから呼び出して使用
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskCategory(str, Enum):
    """タスクカテゴリ"""

    FEATURE = "feature"  # 新機能追加
    ENHANCEMENT = "enhancement"  # 既存機能改善
    BUGFIX = "bugfix"  # バグ修正
    REFACTOR = "refactor"  # リファクタリング
    TEST = "test"  # テスト
    DOCS = "docs"  # ドキュメント


@dataclass
class TaskTemplate:
    """タスクテンプレート"""

    name: str
    category: TaskCategory
    description: str
    steps: list[str]
    files_to_modify: list[str]
    validation_commands: list[str]


# よく使うタスクテンプレート
TASK_TEMPLATES: dict[str, TaskTemplate] = {
    # データ収集系
    "add_data_source": TaskTemplate(
        name="新しいデータソース追加",
        category=TaskCategory.FEATURE,
        description="新しいデータソースからデータを収集するコレクターを追加",
        steps=[
            "1. src/pharma_stock/collectors/ に新しいコレクターファイルを作成",
            "2. BaseCollectorを継承してクラスを実装",
            "3. collect()メソッドを実装",
            "4. __init__.pyにエクスポートを追加",
            "5. tests/collectors/ にテストを作成",
            "6. バックテストで動作確認",
        ],
        files_to_modify=[
            "src/pharma_stock/collectors/new_collector.py",
            "src/pharma_stock/collectors/__init__.py",
            "tests/collectors/test_new_collector.py",
        ],
        validation_commands=[
            "PYTHONPATH=src pytest tests/collectors/ -v",
            "PYTHONPATH=src python -c \"from pharma_stock.collectors import NewCollector; print('OK')\"",
        ],
    ),

    # 分析系
    "add_technical_indicator": TaskTemplate(
        name="新しいテクニカル指標追加",
        category=TaskCategory.ENHANCEMENT,
        description="底値シグナル検知に新しいテクニカル指標を追加",
        steps=[
            "1. src/pharma_stock/analysis/bottom_signal.py を開く",
            "2. _calc_technical_score()に新しい指標のロジックを追加",
            "3. スコア計算と理由リストを更新",
            "4. 重み付けを調整（必要に応じて）",
            "5. バックテストで効果を検証",
        ],
        files_to_modify=[
            "src/pharma_stock/analysis/bottom_signal.py",
        ],
        validation_commands=[
            "PYTHONPATH=src python -c \"from pharma_stock.analysis.bottom_signal import BottomSignalDetector; d=BottomSignalDetector(); print('OK')\"",
        ],
    ),

    "add_fundamental_metric": TaskTemplate(
        name="新しいファンダメンタル指標追加",
        category=TaskCategory.ENHANCEMENT,
        description="ファンダメンタルスコアに新しい指標を追加",
        steps=[
            "1. yfinanceで取得可能なデータを確認",
            "2. _calc_fundamental_score()に新しい指標を追加",
            "3. スコア計算ロジックを実装",
            "4. 理由リストを更新",
            "5. バックテストで検証",
        ],
        files_to_modify=[
            "src/pharma_stock/analysis/bottom_signal.py",
        ],
        validation_commands=[
            "PYTHONPATH=src python -c \"import yfinance as yf; print(yf.Ticker('4506.T').info.keys())\"",
        ],
    ),

    # 予測系
    "improve_prediction_model": TaskTemplate(
        name="予測モデル改善",
        category=TaskCategory.ENHANCEMENT,
        description="価格予測モデルの精度向上",
        steps=[
            "1. 現在の予測精度をバックテストで確認",
            "2. 新しい特徴量を追加（src/pharma_stock/prediction/features.py）",
            "3. モデルのハイパーパラメータを調整",
            "4. クロスバリデーションで評価",
            "5. 改善前後の精度を比較",
        ],
        files_to_modify=[
            "src/pharma_stock/prediction/price_range_predictor.py",
            "src/pharma_stock/prediction/features.py",
        ],
        validation_commands=[
            "PYTHONPATH=src python -c \"from pharma_stock.prediction import PriceRangePredictor; print('OK')\"",
        ],
    ),

    "add_prediction_model": TaskTemplate(
        name="新しい予測モデル追加",
        category=TaskCategory.FEATURE,
        description="新しいML予測モデルを追加",
        steps=[
            "1. src/pharma_stock/prediction/ に新しいモデルファイルを作成",
            "2. BasePredictorを継承して実装",
            "3. train(), predict()メソッドを実装",
            "4. EnsemblePredictorに組み込み",
            "5. バックテストで精度を検証",
        ],
        files_to_modify=[
            "src/pharma_stock/prediction/new_predictor.py",
            "src/pharma_stock/prediction/__init__.py",
            "src/pharma_stock/prediction/ensemble.py",
        ],
        validation_commands=[
            "PYTHONPATH=src pytest tests/prediction/ -v",
        ],
    ),

    # シグナル調整
    "adjust_signal_thresholds": TaskTemplate(
        name="シグナル閾値調整",
        category=TaskCategory.ENHANCEMENT,
        description="底値買いシグナルの閾値を調整",
        steps=[
            "1. 現在の閾値でバックテストを実行",
            "2. 偽陽性/偽陰性の割合を確認",
            "3. 閾値を調整（THRESHOLDS辞書）",
            "4. 再度バックテストを実行",
            "5. 改善を確認してコミット",
        ],
        files_to_modify=[
            "src/pharma_stock/analysis/bottom_signal.py",
        ],
        validation_commands=[
            # バックテストコマンド
        ],
    ),

    "adjust_score_weights": TaskTemplate(
        name="スコア重み調整",
        category=TaskCategory.ENHANCEMENT,
        description="各スコアコンポーネントの重み付けを調整",
        steps=[
            "1. 現在の重み（WEIGHTS辞書）を確認",
            "2. バックテストで各コンポーネントの寄与を分析",
            "3. 重みを調整（合計100%を維持）",
            "4. バックテストで効果を検証",
            "5. 結果を記録してコミット",
        ],
        files_to_modify=[
            "src/pharma_stock/analysis/bottom_signal.py",
        ],
        validation_commands=[],
    ),

    # CLI系
    "add_cli_command": TaskTemplate(
        name="新しいCLIコマンド追加",
        category=TaskCategory.FEATURE,
        description="pharma-stockコマンドに新しいサブコマンドを追加",
        steps=[
            "1. src/pharma_stock/cli/commands/ に新しいコマンドファイルを作成",
            "2. typer.Typer()でappを作成",
            "3. コマンド関数を実装",
            "4. main.pyにサブコマンドを登録",
            "5. __init__.pyにエクスポートを追加",
            "6. 動作確認",
        ],
        files_to_modify=[
            "src/pharma_stock/cli/commands/new_command.py",
            "src/pharma_stock/cli/commands/__init__.py",
            "src/pharma_stock/cli/main.py",
        ],
        validation_commands=[
            "PYTHONPATH=src python -m pharma_stock.cli.main --help",
        ],
    ),

    # バックテスト
    "run_backtest": TaskTemplate(
        name="バックテスト実行",
        category=TaskCategory.TEST,
        description="過去データでシグナル・予測の精度を検証",
        steps=[
            "1. 対象銘柄と期間を決定",
            "2. 株価データを取得（2年分推奨）",
            "3. 各日付でシグナルを計算",
            "4. 実際の結果と比較",
            "5. 精度レポートを作成",
        ],
        files_to_modify=[],
        validation_commands=[
            # バックテストスクリプト
        ],
    ),

    # カタリスト
    "add_catalyst_pattern": TaskTemplate(
        name="新しいカタリストパターン追加",
        category=TaskCategory.ENHANCEMENT,
        description="カタリスト検知に新しいパターンを追加",
        steps=[
            "1. src/pharma_stock/analysis/catalyst_detector.py を開く",
            "2. CRITICAL_PATTERNSに新しいパターンを追加",
            "3. CatalystTypeにenumを追加（必要に応じて）",
            "4. detect_news_catalyst()を更新",
            "5. テストで動作確認",
        ],
        files_to_modify=[
            "src/pharma_stock/analysis/catalyst_detector.py",
        ],
        validation_commands=[
            "PYTHONPATH=src python -c \"from pharma_stock.analysis.catalyst_detector import CatalystDetector; print('OK')\"",
        ],
    ),
}


def get_task_template(task_name: str) -> TaskTemplate | None:
    """タスクテンプレートを取得"""
    return TASK_TEMPLATES.get(task_name)


def list_tasks() -> list[dict[str, Any]]:
    """利用可能なタスク一覧を返す"""
    return [
        {
            "name": t.name,
            "key": key,
            "category": t.category.value,
            "description": t.description,
        }
        for key, t in TASK_TEMPLATES.items()
    ]


def get_task_prompt(task_name: str) -> str:
    """タスク実行用のプロンプトを生成"""
    template = get_task_template(task_name)
    if not template:
        return f"Unknown task: {task_name}"

    return f"""
# {template.name}

## 説明
{template.description}

## 手順
{chr(10).join(template.steps)}

## 変更対象ファイル
{chr(10).join(f'- {f}' for f in template.files_to_modify)}

## 検証コマンド
```bash
{chr(10).join(template.validation_commands)}
```
"""
