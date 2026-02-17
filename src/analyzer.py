"""
FAR_stock ルールベーステクニカル分析モジュール
テクニカル指標に基づくシグナル分析とスコアリング。
"""

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from config import ScanConfig


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrameにテクニカル指標カラムを追加する（UI表示用）。
    """
    df = df.copy()
    df['Close'] = df['Close'].ffill()

    df['SMA_short'] = SMAIndicator(close=df['Close'], window=ScanConfig.SMA_SHORT).sma_indicator()
    df['SMA_medium'] = SMAIndicator(close=df['Close'], window=ScanConfig.SMA_MEDIUM).sma_indicator()
    df['SMA_long'] = SMAIndicator(close=df['Close'], window=ScanConfig.SMA_LONG).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=ScanConfig.SMA_VERY_LONG).sma_indicator()

    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    return df


def analyze_term_signal(df: pd.DataFrame, term: str) -> dict:
    """
    期間別のシグナル分析を実行する。

    Args:
        df: テクニカル指標付きDataFrame
        term: 'Short', 'Medium', 'Long'
    Returns:
        dict: {'score': int, 'reason': str}
    """
    if df.empty or len(df) < 50:
        return {'score': -1, 'reason': 'データ不足'}

    latest = df.iloc[-1]
    score = 0
    reasons = []

    if term == 'Short':
        # RSI 売られすぎ
        rsi = latest.get('RSI', 50)
        if rsi <= ScanConfig.RSI_STRONG_OVERSOLD:
            score += 3
            reasons.append(f"RSI強い売られすぎ ({rsi:.1f})")
        elif rsi <= ScanConfig.RSI_OVERSOLD:
            score += 2
            reasons.append(f"RSI売られすぎ ({rsi:.1f})")

        # ゴールデンクロス
        sma_s = latest.get('SMA_short')
        sma_m = latest.get('SMA_medium')
        if sma_s is not None and sma_m is not None and sma_s > sma_m:
            score += 2
            reasons.append("短期上昇トレンド (SMA_S > SMA_M)")

        # MACD クロス
        macd = latest.get('MACD')
        macd_sig = latest.get('MACD_Signal')
        if macd is not None and macd_sig is not None and macd > macd_sig:
            score += 2
            reasons.append("MACD買いシグナル")

    elif term == 'Medium':
        sma_m = latest.get('SMA_medium')
        sma_l = latest.get('SMA_long')
        if sma_m is not None and sma_l is not None and sma_m > sma_l:
            score += 3
            reasons.append("中期上昇トレンド (SMA_M > SMA_L)")

        # BB Squeeze → 将来のブレイク予兆
        bb_w = latest.get('BB_Width')
        if bb_w is not None and bb_w < ScanConfig.BB_SQUEEZE:
            score += 2
            reasons.append(f"BBスクイーズ ({bb_w:.1f})")

    elif term == 'Long':
        close = latest.get('Close')
        sma_200 = latest.get('SMA_200')
        if close is not None and sma_200 is not None and close > sma_200:
            score += 3
            reasons.append("長期上昇トレンド (200日線より上)")

    return {'score': score, 'reason': ' / '.join(reasons) if reasons else '特になし'}


def analyze_speculative_signal(df: pd.DataFrame) -> dict:
    """
    仕手株・急動意株の可能性を判定する。

    Returns:
        dict: {'is_speculative': bool, 'score': int, 'vol_ratio': float, 'reason': str}
    """
    if df.empty or len(df) < 25:
        return {'is_speculative': False, 'score': 0, 'vol_ratio': 0, 'reason': 'データ不足'}

    latest = df.iloc[-1]

    # 過去20日の平均出来高
    past_20 = df.iloc[-21:-1]
    avg_vol = past_20['Volume'].mean()

    vol_ratio = 0.0
    if avg_vol > 0:
        vol_ratio = latest['Volume'] / avg_vol

    # 価格変動
    price_change_pct = (latest['Close'] - latest['Open']) / latest['Open'] * 100

    score = 0
    reasons = []
    is_speculative = False

    # 出来高急増 + 株価上昇
    if vol_ratio >= ScanConfig.VOL_SPIKE_RATIO and price_change_pct > 0:
        score += int(vol_ratio)
        reasons.append(f"出来高急増 ({vol_ratio:.1f}倍)")
        is_speculative = True

    # 急騰判定
    if price_change_pct >= ScanConfig.PRICE_SURGE_THRESHOLD:
        score += 2
        reasons.append(f"株価急騰 (+{price_change_pct:.1f}%)")

    # 超異常出来高
    if vol_ratio >= ScanConfig.VOL_EXTREME_RATIO:
        score += 5
        reasons.append("異常な出来高 (Attention)")

    # 低位株ボーナス
    if latest['Close'] < ScanConfig.MAX_PRICE and is_speculative:
        score += 1
    if latest['Close'] < ScanConfig.ULTRA_LOW_PRICE and is_speculative:
        score += 1

    return {
        'is_speculative': is_speculative,
        'score': score,
        'vol_ratio': vol_ratio,
        'reason': ' / '.join(reasons) if reasons else '特になし'
    }


def get_rule_based_score(df: pd.DataFrame) -> dict:
    """
    全シグナルを統合したルールベーススコアを算出する。
    推薦エンジンとの統合用。

    Returns:
        dict: {'score': float, 'max_possible': float, 'signals': list[str]}
    """
    if df.empty or len(df) < 50:
        return {'score': 0, 'max_possible': 1, 'signals': []}

    df_tech = add_technical_indicators(df)
    latest = df_tech.iloc[-1]
    score = 0
    signals = []

    # RSI
    rsi = latest.get('RSI', 50)
    if rsi <= ScanConfig.RSI_STRONG_OVERSOLD:
        score += 3
        signals.append(f"RSI({rsi:.0f})")
    elif rsi <= ScanConfig.RSI_OVERSOLD:
        score += 2
        signals.append(f"RSI({rsi:.0f})")

    # ゴールデンクロス
    if latest.get('SMA_short', 0) > latest.get('SMA_medium', float('inf')):
        score += 2
        signals.append("GC")

    # トレンドフォロー
    if latest.get('SMA_medium', 0) > latest.get('SMA_long', float('inf')):
        score += 3
        signals.append("TrendFollow")

    # 200日線
    if latest.get('Close', 0) > latest.get('SMA_200', float('inf')):
        score += 3
        signals.append("200SMA↑")

    # MACD
    if latest.get('MACD', 0) > latest.get('MACD_Signal', float('inf')):
        score += 2
        signals.append("MACD↑")

    # BB
    bb_w = latest.get('BB_Width')
    if bb_w is not None and bb_w < ScanConfig.BB_SQUEEZE:
        score += 2
        signals.append("BBSqueeze")
    if latest.get('Close', 0) > latest.get('BB_Upper', float('inf')):
        score += 3
        signals.append("BBBreakout")

    # 出来高
    past_vol = df['Volume'].iloc[-21:-1].mean()
    if past_vol > 0 and latest['Volume'] / past_vol >= ScanConfig.VOL_SPIKE_RATIO:
        vol_r = latest['Volume'] / past_vol
        score += min(int(vol_r), 10)
        signals.append(f"Vol×{vol_r:.1f}")

    # 低位株
    if latest['Close'] < ScanConfig.MAX_PRICE:
        score += 1
        signals.append("低位株")

    max_possible = 20.0  # 理論上の最大スコア概算
    return {'score': score, 'max_possible': max_possible, 'signals': signals}
