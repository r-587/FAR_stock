"""
FAR_stock 推薦エンジン
LightGBM確率 + ルールベーススコアのアンサンブルによる推薦。
"""

import json
import time
import pandas as pd
import numpy as np
from config import ScanConfig, ModelConfig
from src.data_loader import get_stock_data, fetch_jpx_tickers, get_stock_data_cached
from src.feature_engineering import FeatureEngineer
from src.analyzer import (
    add_technical_indicators, analyze_term_signal,
    analyze_speculative_signal, get_rule_based_score
)
from src.ml_model import SurgePredictor
from src import db


class StockRecommender:
    """ML + ルールベース統合推薦エンジン"""

    def __init__(self, model: SurgePredictor = None):
        self.fe = FeatureEngineer()
        self.model = model
        self.config = ScanConfig()

    def scan_with_ml(self, tickers: list, progress_callback=None) -> pd.DataFrame:
        """
        ML急騰確率 + ルールベースのアンサンブルスキャンを実行する。

        Args:
            tickers: スキャン対象のティッカーリスト
            progress_callback: (current, total) コールバック
        Returns:
            pd.DataFrame: スキャン結果（スコア降順）
        """
        results = []
        total = len(tickers)

        # 銘柄名マッピング
        all_tickers_df = fetch_jpx_tickers()
        name_map = {}
        if not all_tickers_df.empty:
            name_map = dict(zip(all_tickers_df['Ticker'], all_tickers_df['Name']))

        for i, ticker in enumerate(tickers):
            try:
                df = get_stock_data(ticker, period="1y")
                if df is None or df.empty or len(df) < 60:
                    continue

                # ルールベーススコア
                rule_result = get_rule_based_score(df)
                rule_score = rule_result['score']
                signals = rule_result['signals']
                normalized_rule = rule_score / max(rule_result['max_possible'], 1)

                # ML確率
                ml_prob = 0.0
                if self.model is not None and self.model.model is not None:
                    features_df = self.fe.build_features(df)
                    feat_cols = self.fe.get_feature_columns()
                    latest_features = features_df[feat_cols].iloc[-1:]
                    ml_prob = float(self.model.predict_proba(latest_features)[0])

                # アンサンブルスコア
                ensemble = (
                    ScanConfig.ENSEMBLE_ALPHA * ml_prob +
                    ScanConfig.ENSEMBLE_BETA * normalized_rule
                )

                latest = df.iloc[-1]
                result = {
                    'Ticker': ticker,
                    'Name': name_map.get(ticker, ''),
                    'Price': latest['Close'],
                    'ML_Prob': ml_prob,
                    'Rule_Score': rule_score,
                    'Ensemble': ensemble,
                    'Signals': ', '.join(signals) if signals else 'None',
                    'RSI': None,
                }

                # RSI取得（テクニカル指標付き）
                df_tech = add_technical_indicators(df)
                if 'RSI' in df_tech.columns:
                    result['RSI'] = df_tech['RSI'].iloc[-1]

                results.append(result)

                # 予測結果をDB保存
                db.save_prediction(
                    ticker, ml_prob, rule_score, ensemble,
                    json.dumps(signals, ensure_ascii=False)
                )

            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue

            if progress_callback:
                progress_callback(i + 1, total)

            time.sleep(0.5)  # API制限緩和

        if not results:
            return pd.DataFrame()

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('Ensemble', ascending=False).reset_index(drop=True)
        return res_df

    def scan_speculative(self, tickers: list,
                         progress_callback=None) -> pd.DataFrame:
        """
        仕手株特化スキャンを実行する。

        Returns:
            pd.DataFrame: 仕手株候補（スコア降順）
        """
        results = []
        total = len(tickers)

        all_tickers_df = fetch_jpx_tickers()
        name_map = {}
        if not all_tickers_df.empty:
            name_map = dict(zip(all_tickers_df['Ticker'], all_tickers_df['Name']))

        for i, ticker in enumerate(tickers):
            try:
                df = get_stock_data(ticker, period="2mo")
                if df is None or df.empty or len(df) < 25:
                    continue

                res = analyze_speculative_signal(df)

                if res['is_speculative']:
                    # MLもあれば追加情報
                    ml_prob = 0.0
                    if self.model is not None and self.model.model is not None:
                        try:
                            features_df = self.fe.build_features(df)
                            feat_cols = self.fe.get_feature_columns()
                            latest_features = features_df[feat_cols].iloc[-1:]
                            ml_prob = float(self.model.predict_proba(latest_features)[0])
                        except Exception:
                            pass

                    results.append({
                        'Ticker': ticker,
                        'Name': name_map.get(ticker, ''),
                        'Score': res['score'],
                        'ML_Prob': ml_prob,
                        'Volume_Ratio': f"{res['vol_ratio']:.1f}x",
                        'Reason': res['reason'],
                        'Price': df['Close'].iloc[-1],
                    })

            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue

            if progress_callback:
                progress_callback(i + 1, total)

            time.sleep(0.5)

        if not results:
            return pd.DataFrame()

        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('Score', ascending=False).reset_index(drop=True)
        return res_df

    def analyze_single(self, ticker: str, period: str = "1y") -> dict:
        """
        単一銘柄の詳細分析を実行する。

        Returns:
            dict: チャートデータ + シグナル + ML確率
        """
        df = get_stock_data(ticker, period=period)
        if df is None or df.empty:
            return {'error': 'データ取得失敗'}

        df_tech = add_technical_indicators(df)

        # 各期間のシグナル
        signals = {}
        for term in ['Short', 'Medium', 'Long']:
            signals[term] = analyze_term_signal(df_tech, term)

        # 仕手株判定
        spec = analyze_speculative_signal(df)

        # ML確率
        ml_prob = 0.0
        feature_importance = None
        if self.model is not None and self.model.model is not None:
            try:
                features_df = self.fe.build_features(df)
                feat_cols = self.fe.get_feature_columns()
                latest_features = features_df[feat_cols].iloc[-1:]
                ml_prob = float(self.model.predict_proba(latest_features)[0])
                feature_importance = self.model.get_feature_importance()
            except Exception as e:
                print(f"ML prediction error: {e}")

        return {
            'df': df_tech,
            'signals': signals,
            'speculative': spec,
            'ml_prob': ml_prob,
            'feature_importance': feature_importance,
        }
