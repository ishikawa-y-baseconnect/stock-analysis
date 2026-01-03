# Pharma Stock Analyzer

国内トップティア製薬企業の株価分析・予測システム

## 概要

新薬の研究開発パイプライン情報を活用し、製薬企業の株価を分析・予測するツールです。

### 主な機能

- **データ収集**: 株価データ（yfinance）、パイプライン情報（ClinicalTrials.gov）
- **テクニカル分析**: 移動平均、RSI、MACD、ボリンジャーバンド等
- **ファンダメンタル分析**: PER、PBR、ROE等のバリュエーション指標
- **パイプライン影響分析**: 臨床試験イベントの株価への影響を統計分析
- **株価予測**: XGBoost/LightGBM/アンサンブルモデルによる機械学習予測

### 対象企業（国内トップティア製薬10社）

| コード | 企業名 | 主要領域 |
|--------|--------|----------|
| 4502 | 武田薬品工業 | オンコロジー、希少疾患 |
| 4503 | アステラス製薬 | オンコロジー、泌尿器 |
| 4568 | 第一三共 | オンコロジー、循環器 |
| 4523 | エーザイ | 神経科学、認知症 |
| 4519 | 中外製薬 | オンコロジー、免疫 |
| 4578 | 大塚HD | 精神・神経 |
| 4506 | 住友ファーマ | 精神・神経、再生医療 |
| 4507 | 塩野義製薬 | 感染症、疼痛 |
| 4151 | 協和キリン | オンコロジー、腎臓 |
| 4528 | 小野薬品工業 | オンコロジー、免疫 |

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/ishikawa-y-baseconnect/stock-analysis.git
cd stock-analysis

# 仮想環境を作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係をインストール
pip install -e .

# 開発用依存関係（オプション）
pip install -e ".[dev]"

# プロジェクトを初期化
pharma-stock init
```

## 使い方

### 企業情報の確認

```bash
# 対象企業一覧
pharma-stock companies list

# 企業詳細
pharma-stock companies info 4502.T

# 全企業の最新株価
pharma-stock companies prices
```

### データ収集

```bash
# 特定銘柄の株価データ収集
pharma-stock collect stock --ticker 4502.T --period 1y

# 全銘柄の株価データ収集
pharma-stock collect stock --period 1y

# パイプライン情報収集
pharma-stock collect pipeline --ticker 4502.T

# Phase 3試験のみ
pharma-stock collect pipeline --phase PHASE3

# 全データ一括収集
pharma-stock collect all
```

### 分析

```bash
# テクニカル分析
pharma-stock analyze technical 4502.T

# ファンダメンタル分析（単一銘柄）
pharma-stock analyze fundamental --ticker 4502.T

# 同業他社比較
pharma-stock analyze fundamental

# 総合分析
pharma-stock analyze summary 4502.T
```

### 予測

```bash
# モデル学習
pharma-stock predict train 4502.T --model ensemble --horizon 5

# 株価予測
pharma-stock predict price 4502.T --horizon 5

# 全銘柄予測サマリー
pharma-stock predict all --horizon 5
```

## プロジェクト構造

```
stock-analysis/
├── src/pharma_stock/
│   ├── collectors/      # データ収集モジュール
│   ├── analysis/        # 分析エンジン
│   ├── prediction/      # 予測モデル
│   ├── storage/         # データストレージ
│   ├── config/          # 設定・企業データ
│   ├── cli/             # CLIインターフェース
│   └── utils/           # ユーティリティ
├── tests/               # テスト
├── data/                # データ（gitignore）
└── pyproject.toml       # プロジェクト設定
```

## データソース

- **株価**: Yahoo Finance（yfinance経由）
- **パイプライン**: ClinicalTrials.gov API v2
- **承認情報**: PMDA（医薬品医療機器総合機構）

## 技術スタック

- Python 3.11+
- pandas, numpy（データ処理）
- yfinance（株価データ）
- ta（テクニカル指標）
- scikit-learn, XGBoost, LightGBM（機械学習）
- Typer, Rich（CLI）
- SQLAlchemy（データベース）
- Pydantic（データ検証）

## 注意事項

- 本ツールは投資助言を目的としたものではありません
- 予測結果は統計的な推定であり、将来の株価を保証するものではありません
- 投資判断は自己責任でお願いします

## ライセンス

MIT License
