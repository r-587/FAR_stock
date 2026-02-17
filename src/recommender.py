"""
FAR_stock 推薦エンジン
LightGBM確率 + ルールベーススコアのアンサンブルによる推薦。
"""

import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import joblib
import os
import torch
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from config import ModelConfig, ScanConfig, TFTConfig
from src.ml_model import SurgePredictor

from src.dataset import create_tft_dataset
from src.data_loader import get_stock_data, fetch_jpx_tickers, get_stock_data_cached
from src.feature_engineering import FeatureEngineer
from src.analyzer import (
    add_technical_indicators, analyze_term_signal,
    analyze_speculative_signal, get_rule_based_score
)
from src import db


class StockRecommender:
    """ML + ルールベース統合推薦エンジン"""

    def __init__(self, model: SurgePredictor = None):
        self.fe = FeatureEngineer()
        self.model = model
        self.config = ScanConfig()
        self.tft_model = None

    def load_tft_model(self, model_path: str = None):
        """TFTモデルをロードする"""
        if model_path is None:
            # 最新のチェックポイントを探す
            if os.path.exists(TFTConfig.MODEL_DIR):
                checkpoints = sorted([
                    os.path.join(TFTConfig.MODEL_DIR, f) 
                    for f in os.listdir(TFTConfig.MODEL_DIR) 
                    if f.endswith('.ckpt')
                ])
                if checkpoints:
                    model_path = checkpoints[-1]
        
        if model_path and os.path.exists(model_path):
            try:
                self.tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
                print(f"Loaded TFT model from {model_path}")
                return True
            except Exception as e:
                print(f"Failed to load TFT model: {e}")
                return False
        return False

    def predict_tft(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        指定銘柄の未来5日間の価格推移を予測する
        Args:
            ticker: 銘柄コード
            df: 日足データ (Date, Open, High, Low, Close, Volume)
        Returns:
            予測結果DataFrame (Date, Predicted_Mean, Lower, Upper)
        """
        if self.tft_model is None:
            if not self.load_tft_model():
                return pd.DataFrame()
        
        # データセット作成 (推論用なので直近データのみでOKだが、Encoder長が必要)
        # create_tft_datasetはDataLoadersを返すが、ここではデータセット自体が必要
        # predictメソッドはDataLoaderまたはDataFrameを受け取れる
        
        # 前処理: Date型変換など
        df = df.copy()
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        elif 'Date' not in df.columns:
             # IndexがDateでない、かつDateカラムもない場合はreset_indexしてみる
             df = df.reset_index()
             # それでもDateがない場合はrenameを試みる (index -> Date)
             if 'Date' not in df.columns:
                 df = df.rename(columns={'index': 'Date'})

        df['ticker'] = ticker # ticker列追加
        
        # Dateがタイムゾーン付きの場合、TFTでエラーになる可能性があるためtz_localize(None)する
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
             if df['Date'].dt.tz is not None:
                 df['Date'] = df['Date'].dt.tz_localize(None)
        else:
             df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
        # time_idx作成 (簡易版)
        min_date = df['Date'].min()
        df['time_idx'] = (df['Date'] - min_date).dt.days
        
        # 特徴量 (Date derived)
        df['month'] = df['Date'].dt.month.astype(str).astype('category')
        df['day'] = df['Date'].dt.day.astype(str).astype('category')
        df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype('category')

        # 最新のデータを含むようにフィルタリング (学習時と同じ長さが必要)
        # Predictionには Encoder Length + Prediction Length 分のデータが必要
        # ただし、Unknown Reals (Closeなど) は未来の値は不要 (NaNでよい、または自動補完される)
        
        # ここではcreate_tft_datasetを再利用してDataLoaderを作るのが安全
        try:
             # バッチサイズ1で作成
            _, _, _, predict_dataloader = create_tft_dataset(
                df, 
                max_encoder_length=TFTConfig.MAX_ENCODER_LENGTH,
                max_prediction_length=TFTConfig.MAX_PREDICTION_LENGTH,
                batch_size=1
            )
            
            # 推論実行
            # mode="quantiles" で (batch_size, prediction_length, n_quantiles)
            raw_predictions = self.tft_model.predict(predict_dataloader, mode="quantiles", return_x=True)
            
            # 結果整形
            predictions = raw_predictions.output
            # predictions shape: [1, 5, 7] (1 batch, 5 days, 7 quantiles)
            
            # 予測開始日 (x['decoder_time_idx'] の最初の日付に対応)
            # decoder_time_idxは time_idx の続き
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=TFTConfig.MAX_PREDICTION_LENGTH, freq='B') # 営業日換算が必要だが簡易的に
            
            # Quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98
            # default output_size=7
            
            pred_median = predictions[0, :, 3].numpy() # 0.5 (Median)
            pred_lower = predictions[0, :, 1].numpy()  # 0.1 (Lower bound)
            pred_upper = predictions[0, :, 5].numpy()  # 0.9 (Upper bound)
            
            result_df = pd.DataFrame({
                "Date": future_dates[:len(pred_median)], # 長さが合うように
                "Predicted_Mean": pred_median,
                "Lower_Bound": pred_lower,
                "Upper_Bound": pred_upper
            })
            
            return result_df
            
        except Exception as e:
            print(f"TFT prediction error for {ticker}: {e}")
            return pd.DataFrame()

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
