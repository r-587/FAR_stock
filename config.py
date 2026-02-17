"""
FAR_stock 設定管理モジュール
全てのしきい値・パラメータを一元管理する。"""

import subprocess
import shutil


def detect_gpu() -> bool:
    """LightGBMでGPU(OpenCL)が実際に利用可能かをテストする。"""
    # nvidia-smiが無ければ即False
    if not shutil.which('nvidia-smi'):
        return False
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return False
    except Exception:
        return False

    # LightGBMで実際にGPU学習を試行
    try:
        import lightgbm as lgb
        import numpy as np
        _X = np.random.rand(50, 5)
        _y = np.random.randint(0, 2, 50)
        ds = lgb.Dataset(_X, label=_y)
        lgb.train(
            {'objective': 'binary', 'device': 'gpu', 'verbose': -1,
             'num_leaves': 4, 'n_estimators': 2},
            ds, num_boost_round=2,
        )
        return True
    except Exception:
        return False


GPU_AVAILABLE = detect_gpu()


class ModelConfig:
    """LightGBM / ML関連設定"""
    SURGE_THRESHOLD_PCT: float = 5.0       # 急騰定義 (+5%)
    FORECAST_HORIZON_DAYS: int = 5         # 予測対象期間（日）
    TRAIN_PERIOD_DAYS: int = 365           # 学習データ期間（日）
    WALK_FORWARD_STEP: int = 20            # Walk-Forward検証ステップ（日）
    TOP_K: int = 10                        # Precision@K の K
    MIN_TRAIN_SAMPLES: int = 100           # 最低学習サンプル数

    # LightGBM ハイパーパラメータ（GPU自動検出）
    LGBM_PARAMS: dict = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 300,
        'is_unbalance': True,  # 不均衡データ対応
        # GPU設定（利用可能な場合のみ）
        **({
            'device': 'gpu',
            'gpu_use_dp': False,  # FP32で十分、FP64不要
        } if GPU_AVAILABLE else {}),
    }


class ScanConfig:
    """スクリーニング設定"""
    # 価格帯フィルタ
    MIN_PRICE: float = 50
    MAX_PRICE: float = 1000
    ULTRA_LOW_PRICE: float = 300

    # 出来高分析
    VOL_SPIKE_RATIO: float = 3.0
    VOL_EXTREME_RATIO: float = 10.0
    VOL_LOOKBACK_DAYS: int = 20

    # 価格モメンタム
    PRICE_SURGE_THRESHOLD: float = 3.0

    # テクニカル指標
    RSI_OVERSOLD: float = 40
    RSI_STRONG_OVERSOLD: float = 30
    SMA_SHORT: int = 5
    SMA_MEDIUM: int = 20
    SMA_LONG: int = 50
    SMA_VERY_LONG: int = 200
    BB_SQUEEZE: float = 10.0

    # アンサンブル重み
    ENSEMBLE_ALPHA: float = 0.7   # ML確率の重み
    ENSEMBLE_BETA: float = 0.3    # ルールベーススコアの重み


class TFTConfig:
    """TFTモデル設定"""
    MAX_ENCODER_LENGTH = 60  # 過去60日分を入力
    MAX_PREDICTION_LENGTH = 5 # 未来5日分を予測
    BATCH_SIZE = 64
    MAX_EPOCHS = 30
    LEARNING_RATE = 0.03
    HIDDEN_SIZE = 16
    ATTENTION_HEAD_SIZE = 1
    DROPOUT = 0.1
    HIDDEN_CONTINUOUS_SIZE = 8
    LSTM_LAYERS = 1
    MODEL_DIR = "models/tft_checkpoints"

class RLConfig:
    """強化学習 (DRL) 設定"""
    TIMESTEPS = 20000  # 学習ステップ数
    LOOKBACK_WINDOW = 30 # 観測する過去期間
    CASH_INITIAL = 1_000_000
    TRANSACTION_COST_PCT = 0.001 # 取引手数料 0.1%
    MODEL_DIR = "models/rl_agents"

class APIConfig:
    # 既存のコード...
    """API・キャッシュ設定"""
    CHUNK_SIZE: int = 50
    WAIT_SEC: float = 1.0
    CACHE_TTL_HOURS: int = 24
    JPX_URL: str = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    CACHE_FILE: str = "tickers.csv"
    DB_PATH: str = "stocks.db"
