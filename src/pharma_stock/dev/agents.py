"""開発用エージェント定義

各エージェントは特定のタスクに特化したサブエージェントとして動作
Claude Codeのタスクツールと連携して使用
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any


class AgentType(str, Enum):
    """エージェントタイプ"""

    # データ収集系
    DATA_COLLECTOR = "data_collector"
    NEWS_COLLECTOR = "news_collector"

    # 分析系
    TECHNICAL_ANALYZER = "technical_analyzer"
    FUNDAMENTAL_ANALYZER = "fundamental_analyzer"
    SIGNAL_DETECTOR = "signal_detector"

    # 予測系
    PRICE_PREDICTOR = "price_predictor"
    MODEL_TRAINER = "model_trainer"

    # 検証系
    BACKTESTER = "backtester"
    ACCURACY_EVALUATOR = "accuracy_evaluator"

    # 開発系
    CODE_REVIEWER = "code_reviewer"
    TEST_RUNNER = "test_runner"


@dataclass
class AgentPrompt:
    """エージェントプロンプト定義"""

    agent_type: AgentType
    name: str
    description: str
    prompt_template: str
    required_context: list[str]
    output_format: str


# エージェントプロンプト定義
AGENT_PROMPTS: dict[AgentType, AgentPrompt] = {
    AgentType.DATA_COLLECTOR: AgentPrompt(
        agent_type=AgentType.DATA_COLLECTOR,
        name="データコレクター",
        description="新しいデータソースの追加・既存コレクターの改善",
        prompt_template="""
あなたはpharma_stockプロジェクトのデータ収集エージェントです。

## 現在のコレクター
- StockCollector: yfinanceから株価データを収集
- PipelineCollector: ClinicalTrials.govから治験データを収集
- PMDACollector: PMDAから承認情報を収集
- IRNewsCollector: IRニュースを収集

## タスク
{task_description}

## 実装ガイドライン
1. BaseCollectorを継承して新しいコレクターを作成
2. collect()メソッドを実装
3. 適切なエラーハンドリングを追加
4. structlogでログを出力
5. テストを作成

## 出力
- 実装したコードの説明
- 使用例
- テスト結果
""",
        required_context=["src/pharma_stock/collectors/"],
        output_format="markdown",
    ),

    AgentType.TECHNICAL_ANALYZER: AgentPrompt(
        agent_type=AgentType.TECHNICAL_ANALYZER,
        name="テクニカル分析エージェント",
        description="新しいテクニカル指標の追加・分析ロジックの改善",
        prompt_template="""
あなたはpharma_stockプロジェクトのテクニカル分析エージェントです。

## 現在の分析機能
- RSI, ボリンジャーバンド, 移動平均
- 52週高値/安値からの位置
- 連続下落日数
- 出来高分析

## タスク
{task_description}

## 実装ガイドライン
1. taライブラリを活用
2. スコア計算は0-100で正規化
3. 理由リストを返す
4. バックテストで検証

## 出力
- 追加した指標の説明
- スコア計算ロジック
- バックテスト結果
""",
        required_context=["src/pharma_stock/analysis/"],
        output_format="markdown",
    ),

    AgentType.SIGNAL_DETECTOR: AgentPrompt(
        agent_type=AgentType.SIGNAL_DETECTOR,
        name="シグナル検知エージェント",
        description="底値買いシグナルの検知ロジック改善",
        prompt_template="""
あなたはpharma_stockプロジェクトのシグナル検知エージェントです。

## 現在のシグナルシステム
- BottomSignalDetector: 底値買いシグナル検知
- CatalystDetector: カタリスト検知
- スコア重み: テクニカル25%, ファンダメンタル20%, 上昇ポテンシャル25%, ML予測30%

## タスク
{task_description}

## 実装ガイドライン
1. 閾値調整はバックテストで検証
2. 新しいスコア要素追加時は重みを再調整
3. SignalStrength enumに新しいレベルを追加可能

## 出力
- 変更内容の説明
- バックテスト結果（変更前/後の比較）
""",
        required_context=[
            "src/pharma_stock/analysis/bottom_signal.py",
            "src/pharma_stock/analysis/catalyst_detector.py",
        ],
        output_format="markdown",
    ),

    AgentType.PRICE_PREDICTOR: AgentPrompt(
        agent_type=AgentType.PRICE_PREDICTOR,
        name="価格予測エージェント",
        description="ML予測モデルの改善・新しい予測手法の追加",
        prompt_template="""
あなたはpharma_stockプロジェクトの価格予測エージェントです。

## 現在の予測モデル
- PriceRangePredictor: 3-6ヶ月後の高値/安値を予測
- XGBoostPredictor, LightGBMPredictor: アンサンブル予測
- ヒューリスティックベースのフォールバック

## タスク
{task_description}

## 実装ガイドライン
1. 特徴量エンジニアリングはFeatureEngineerクラスを使用
2. 時系列クロスバリデーションで評価
3. 予測の信頼区間を計算

## 出力
- モデルの説明
- 特徴量の重要度
- バックテスト結果（MAE, 方向性精度）
""",
        required_context=["src/pharma_stock/prediction/"],
        output_format="markdown",
    ),

    AgentType.BACKTESTER: AgentPrompt(
        agent_type=AgentType.BACKTESTER,
        name="バックテストエージェント",
        description="過去データでのシグナル・予測精度検証",
        prompt_template="""
あなたはpharma_stockプロジェクトのバックテストエージェントです。

## バックテスト対象
- 底値買いシグナルの精度
- 価格予測の精度
- リスクリワード比の実績

## タスク
{task_description}

## 検証項目
1. シグナル発生時の勝率
2. 予測と実績の誤差
3. 最大ドローダウン
4. リスクリワード比の実績

## 出力フォーマット
| 銘柄 | 分析日 | シグナル | 予測 | 実績 | 結果 |
""",
        required_context=[
            "src/pharma_stock/analysis/bottom_signal.py",
            "src/pharma_stock/prediction/price_range_predictor.py",
        ],
        output_format="table",
    ),

    AgentType.CODE_REVIEWER: AgentPrompt(
        agent_type=AgentType.CODE_REVIEWER,
        name="コードレビューエージェント",
        description="コード品質・セキュリティ・パフォーマンスのレビュー",
        prompt_template="""
あなたはpharma_stockプロジェクトのコードレビューエージェントです。

## レビュー観点
1. コード品質（可読性、保守性）
2. エラーハンドリング
3. パフォーマンス
4. セキュリティ（APIキー露出など）
5. テストカバレッジ

## タスク
{task_description}

## 出力
- 問題点のリスト（重大度付き）
- 改善提案
- 良い点（あれば）
""",
        required_context=["src/pharma_stock/"],
        output_format="markdown",
    ),

    AgentType.TEST_RUNNER: AgentPrompt(
        agent_type=AgentType.TEST_RUNNER,
        name="テスト実行エージェント",
        description="テストの実行と結果レポート",
        prompt_template="""
あなたはpharma_stockプロジェクトのテスト実行エージェントです。

## テスト実行
```bash
cd /Users/yuto_ishikawa/github/stock-analysis
source .venv/bin/activate
PYTHONPATH=src pytest tests/ -v
```

## タスク
{task_description}

## 出力
- テスト結果サマリー
- 失敗したテストの詳細
- カバレッジレポート（あれば）
""",
        required_context=["tests/"],
        output_format="markdown",
    ),
}


def get_agent_prompt(agent_type: AgentType, task_description: str) -> str:
    """エージェントプロンプトを生成"""
    prompt_def = AGENT_PROMPTS.get(agent_type)
    if not prompt_def:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return prompt_def.prompt_template.format(task_description=task_description)


def list_agents() -> list[dict[str, str]]:
    """利用可能なエージェント一覧を返す"""
    return [
        {
            "type": ap.agent_type.value,
            "name": ap.name,
            "description": ap.description,
        }
        for ap in AGENT_PROMPTS.values()
    ]
