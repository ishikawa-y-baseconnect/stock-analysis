# /scan - 底値シグナルスキャン

全銘柄の底値買いシグナルをスキャンします。

## 使用例

```
/scan
/scan --detail 4506.T
```

## 実行コマンド

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pharma_stock.cli.main scan bottom
```

## 詳細分析

```bash
PYTHONPATH=src python -m pharma_stock.cli.main scan detail 4506.T
```

## 出力項目

| 項目 | 説明 |
|------|------|
| 総合スコア | 0-100点 |
| テクニカル | RSI, BB, 52週安値 |
| ファンダメンタル | PER, PBR, 配当 |
| 上昇ポテンシャル | Phase 3, アナリスト目標 |
| ML予測 | 3-6ヶ月後の予測高値 |
| シグナル | 強い買い/買い/様子見/見送り |
