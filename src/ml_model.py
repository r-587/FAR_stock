"""
FAR_stock LightGBM急騰確率分類モデル
Walk-Forward検証によるROIベースの学習・予測を行う。
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report
)
import joblib
from config import ModelConfig


class SurgePredictor:
    """LightGBMベースの急騰確率予測モデル"""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_importance_ = None
        self.metrics_ = {}

    def train(self, X: pd.DataFrame, y: pd.Series,
              walk_forward: bool = True) -> dict:
        """
        モデルを学習する。

        Args:
            X: 特徴量DataFrame
            y: 急騰ラベル (0/1)
            walk_forward: Walk-Forward検証を行うか
        Returns:
            dict: 評価指標
        """
        # NaN除去
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()

        if len(X) < self.config.MIN_TRAIN_SAMPLES:
            print(f"Warning: 学習データ不足 ({len(X)} samples)")
            return {'error': 'insufficient_data'}

        if walk_forward:
            return self._walk_forward_train(X, y)
        else:
            return self._simple_train(X, y)

    def _simple_train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """シンプルな時系列分割での学習。"""
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        params = self.config.LGBM_PARAMS.copy()
        n_estimators = params.pop('n_estimators', 300)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # 評価
        y_pred_proba = self.model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred_proba)

        # 特徴量重要度
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.metrics_ = metrics
        return metrics

    def _walk_forward_train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Walk-Forward検証で学習する。
        時系列の順序を保持したまま、ウィンドウをスライドして検証。
        """
        step = self.config.WALK_FORWARD_STEP
        min_train = max(self.config.MIN_TRAIN_SAMPLES, int(len(X) * 0.5))

        all_preds = []
        all_actuals = []

        # 最後のモデルを保持するために、最終ステップのモデルを保存
        for start in range(min_train, len(X) - step, step):
            X_train = X.iloc[:start]
            y_train = y.iloc[:start]
            X_val = X.iloc[start:start + step]
            y_val = y.iloc[start:start + step]

            params = self.config.LGBM_PARAMS.copy()
            n_estimators = params.pop('n_estimators', 300)

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0),
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            preds = model.predict(X_val)
            all_preds.extend(preds)
            all_actuals.extend(y_val.values)

            # 最後のモデルを保持
            self.model = model

        # 全体の評価
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        metrics = self._calculate_metrics(all_actuals, all_preds)

        # 特徴量重要度（最終モデル）
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.metrics_ = metrics
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """急騰確率を予測する。"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # NaN処理
        X_clean = X.fillna(0)
        return self.model.predict(X_clean)

    def _calculate_metrics(self, y_true: np.ndarray,
                           y_pred_proba: np.ndarray) -> dict:
        """評価指標を算出する。"""
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            'auc_roc': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'support_positive': int(y_true.sum()),
            'support_total': len(y_true),
        }

        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            pass

        if y_true.sum() > 0:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Precision@K
        k = min(self.config.TOP_K, len(y_pred_proba))
        top_k_idx = np.argsort(y_pred_proba)[-k:]
        y_true_arr = np.asarray(y_true)
        metrics['precision_at_k'] = float(y_true_arr[top_k_idx].mean())

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """特徴量重要度を取得する。"""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_

    def save(self, path: str = "models/lgbm_surge_v1.pkl"):
        """モデルを保存する。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance_,
            'metrics': self.metrics_,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "models/lgbm_surge_v1.pkl"):
        """モデルを読み込む。"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_importance_ = data['feature_importance']
        self.metrics_ = data['metrics']
        print(f"Model loaded from {path}")
