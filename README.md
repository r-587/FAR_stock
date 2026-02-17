# 🚀 FAR_stock - 急騰株AIレコメンダー

**Financial AI Recommender for Stock Surge**

LightGBM機械学習 × ルールベーステクニカル分析のアンサンブルで、東証の急騰候補銘柄を検出・推薦するシステムです。

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ 主な機能

| 機能 | 説明 |
|------|------|
| 🤖 **ML急騰確率予測** | LightGBMで64特徴量から5日後+5%急騰確率を推定 |
| 📊 **テクニカル分析** | RSI / SMA / MACD / ボリンジャーバンドによるシグナル検出 |
| 🔍 **セクター別スクリーニング** | 東証全セクターからスコアランキング |
| 🔥 **仕手株検知** | 出来高急増パターンの自動検出 |
| 📈 **ROIバックテスト** | シャープレシオ / 最大DD / 勝率の検証 |
| ⭐ **ウォッチリスト** | 注目銘柄の保存・一括分析 |

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                Streamlit UI (main.py)            │
│   📌個別分析 │ 🔍セクター │ 🚀AI検知 │ 📊BT │ ⭐WL │
├─────────────────────────────────────────────────┤
│          Recommender (recommender.py)            │
│         ML確率 × ルールスコア アンサンブル         │
├──────────────┬──────────────────────────────────┤
│  ML Layer    │     Rule-based Layer             │
│  ml_model.py │     analyzer.py                  │
│  LightGBM    │     RSI/SMA/MACD/BB              │
├──────────────┴──────────────────────────────────┤
│     Feature Engineering (feature_engineering.py) │
│              64特徴量 × 7カテゴリ                 │
├─────────────────────────────────────────────────┤
│  Data Layer: data_loader.py + db.py (SQLite)     │
│  yfinance API → キャッシュ → OHLCV               │
└─────────────────────────────────────────────────┘
```

## 📁 ディレクトリ構成

```
FAR_stock/
├── main.py                  # Streamlit UI (5タブ構成)
├── config.py                # パラメータ一元管理
├── requirements.txt         # 依存ライブラリ
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # JPX銘柄一覧 + yfinance株価取得
│   ├── db.py                # SQLiteデータベース操作
│   ├── feature_engineering.py  # 64特徴量生成エンジン
│   ├── analyzer.py          # ルールベーステクニカル分析
│   ├── ml_model.py          # LightGBM急騰確率分類
│   ├── recommender.py       # アンサンブル推薦エンジン
│   ├── backtester.py        # ROIバックテスト
│   └── utils.py             # チャート描画・ヘルパー
└── models/                  # 学習済みモデル保存先
```

## 🚀 セットアップ

### 前提条件
- Python 3.12+
- pip

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/<your-username>/FAR_stock.git
cd FAR_stock

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 依存ライブラリをインストール
pip install -r requirements.txt
```

### 起動

```bash
source venv/bin/activate
streamlit run main.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 📖 使い方

### 1. モデル学習（初回のみ）
1. サイドバーの **「🧠 モデル学習/再学習」** ボタンをクリック
2. 層化サンプリング＋低位株重点で約200銘柄のデータを自動収集
3. 64特徴量の生成 → LightGBMモデルの学習が自動実行されます

### 2. 急騰候補AI検知
1. **「🚀 急騰候補AI検知」** タブを開く
2. スキャンモード（セクター指定 / 仕手株検知）を選択
3. **「AIスキャン開始」** ボタンで推薦ランキングを取得

### 3. 個別銘柄分析
1. **「📌 個別銘柄分析」** タブでティッカーコード入力（例: `7203.T`）
2. チャート + ML急騰確率 + シグナル分析を表示

## 🧠 MLパイプライン

### 特徴量（64次元 × 7カテゴリ）

| カテゴリ | 特徴量数 | 例 |
|---------|---------|-----|
| 価格系 | 9 | リターン(1/5/10/20日), 対数リターン, 高値安値比 |
| SMA系 | 14 | SMA(5/10/20/60/200), 乖離率, ゴールデンクロス |
| モメンタム系 | 8 | RSI(14), MACD, ストキャスティクス |
| ボラティリティ系 | 7 | BB帯幅, ATR, 実現ボラティリティ |
| 出来高系 | 6 | 出来高比(5/20日), OBV変化率 |
| パターン系 | 6 | 連騰日数, 高値更新, 低位株フラグ |
| ラグ系 | 14 | 上記指標の1〜5日ラグ |

### 急騰ラベル定義
- **5営業日後のリターン ≥ +5%** → ラベル `1`（急騰）

### 評価指標
- **Precision@10**: 上位10銘柄の的中率（ROI直結）
- AUC-ROC, F1-Score

## ⚙️ 設定

`config.py` で全パラメータを一元管理：

```python
class ModelConfig:
    SURGE_THRESHOLD_PCT = 5.0   # 急騰定義 (+5%)
    SURGE_HORIZON_DAYS = 5      # 予測期間 (5営業日)
    TOP_K = 10                  # Precision@K の K

class ScanConfig:
    MAX_PRICE = 1000            # 低位株しきい値 (円)
    MIN_VOLUME = 100000         # 最低出来高
```

## 🛣️ ロードマップ

- [x] **Phase 1 (MVP)**: LightGBM + ルールベース アンサンブル推薦
- [ ] **Phase 2**: Temporal Fusion Transformer (TFT) 多段階予測
- [ ] **Phase 3**: DRL強化学習によるポートフォリオ最適化

## ⚠️ 免責事項

本システムは**学習・研究目的**で開発されたものです。投資判断は自己責任で行ってください。本システムの予測結果に基づく投資によるいかなる損失についても、開発者は責任を負いません。

## 📝 ライセンス

MIT License
