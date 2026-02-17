"""
FAR_stock 特徴量エンジニアリングモジュール
OHLCVデータから50+のML用特徴量を生成する。
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, WilliamsRIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from config import ScanConfig, ModelConfig


class FeatureEngineer:
    """50+特徴量を生成するエンジン"""

    def __init__(self):
        self.feature_columns = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCVデータから全特徴量を生成する。

        Args:
            df: OHLCV DataFrame (columns: Open, High, Low, Close, Volume)
        Returns:
            全特徴量が追加されたDataFrame
        """
        df = df.copy()
        df['Close'] = df['Close'].ffill()

        df = self._price_features(df)
        df = self._sma_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._volume_features(df)
        df = self._pattern_features(df)
        df = self._lag_features(df)

        self.feature_columns = [c for c in df.columns
                                if c not in ['Open', 'High', 'Low', 'Close', 'Volume',
                                             'Dividends', 'Stock Splits', 'Capital Gains']]
        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """リターン系特徴量"""
        # 日次リターン
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)

        # 対数リターン
        df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))

        # High-Low 比率
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']

        # Close-Open 比率
        df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # 上ヒゲ・下ヒゲ比率
        df['upper_shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Close']
        df['lower_shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Close']

        return df

    def _sma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """移動平均系特徴量"""
        for window in [ScanConfig.SMA_SHORT, 10, ScanConfig.SMA_MEDIUM,
                        ScanConfig.SMA_LONG, ScanConfig.SMA_VERY_LONG]:
            col = f'sma_{window}'
            df[col] = SMAIndicator(close=df['Close'], window=window).sma_indicator()
            # SMA乖離率
            df[f'sma_{window}_deviation'] = (df['Close'] - df[col]) / df[col]

        # EMA
        for window in [12, 26]:
            col = f'ema_{window}'
            df[col] = EMAIndicator(close=df['Close'], window=window).ema_indicator()
            df[f'ema_{window}_deviation'] = (df['Close'] - df[col]) / df[col]

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム系特徴量"""
        # RSI
        df['rsi_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

        # MACD
        macd = MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # Williams %R
        df['williams_r'] = WilliamsRIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], lbp=14
        ).williams_r()

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ系特徴量"""
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct'] = bb.bollinger_pband()  # レンジ内の位置 (0-1)

        # ATR
        df['atr_14'] = AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        ).average_true_range()

        # 実現ボラティリティ (20日)
        df['realized_vol_20'] = df['log_return_1d'].rolling(20).std() * np.sqrt(252)

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高系特徴量"""
        # 出来高比 (vs 20日平均)
        vol_mean = df['Volume'].rolling(ScanConfig.VOL_LOOKBACK_DAYS).mean()
        df['volume_ratio'] = df['Volume'] / vol_mean

        # 出来高変化率
        df['volume_change'] = df['Volume'].pct_change(1)

        # OBV
        df['obv'] = OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()

        # OBV変化率
        df['obv_change'] = df['obv'].pct_change(5)

        # VWAP風乖離 (日足の簡易版)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap_approx = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['Close'] - vwap_approx) / vwap_approx

        return df

    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """パターン系特徴量（バイナリフラグ）"""
        # ゴールデンクロス (SMA5 > SMA20)
        sma_s = f'sma_{ScanConfig.SMA_SHORT}'
        sma_m = f'sma_{ScanConfig.SMA_MEDIUM}'
        sma_l = f'sma_{ScanConfig.SMA_LONG}'

        if sma_s in df.columns and sma_m in df.columns:
            df['golden_cross'] = (df[sma_s] > df[sma_m]).astype(int)
            df['dead_cross'] = (df[sma_s] < df[sma_m]).astype(int)

        # BB Squeeze (幅が収縮)
        if 'bb_width' in df.columns:
            df['bb_squeeze'] = (df['bb_width'] < ScanConfig.BB_SQUEEZE).astype(int)

        # BB Breakout (終値 > 上限バンド)
        if 'bb_upper' in df.columns:
            df['bb_breakout'] = (df['Close'] > df['bb_upper']).astype(int)

        # MACD クロス
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

        # 出来高スパイク
        if 'volume_ratio' in df.columns:
            df['volume_spike'] = (df['volume_ratio'] > ScanConfig.VOL_SPIKE_RATIO).astype(int)

        return df

    def _lag_features(self, df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
        """時系列ラグ特徴量"""
        if lags is None:
            lags = [1, 2, 3, 5]

        lag_targets = ['return_1d', 'rsi_14', 'volume_ratio', 'macd_histogram']
        for target in lag_targets:
            if target in df.columns:
                for lag in lags:
                    df[f'{target}_lag{lag}'] = df[target].shift(lag)

        return df

    def create_labels(self, df: pd.DataFrame,
                      horizon: int = None,
                      threshold: float = None) -> pd.Series:
        """
        N日後リターンに基づく急騰ラベルを生成する。

        Args:
            horizon: 予測期間（日）
            threshold: 急騰判定しきい値（%）
        Returns:
            pd.Series: 0/1 ラベル
        """
        if horizon is None:
            horizon = ModelConfig.FORECAST_HORIZON_DAYS
        if threshold is None:
            threshold = ModelConfig.SURGE_THRESHOLD_PCT

        # N日後のリターンを計算
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        labels = (future_return >= threshold / 100).astype(int)
        return labels

    def get_feature_columns(self) -> list:
        """ML用の特徴量カラムリストを取得する。"""
        return self.feature_columns
