import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config import RLConfig
from src.rl.env import StockTradingEnv

class PortfolioOptimizer:
    """
    ポートフォリオ配分最適化エージェント
    PPO (Proximal Policy Optimization) を使用
    """
    def __init__(self, model_dir=RLConfig.MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.tickers = []

    def train(self, price_df: pd.DataFrame, timesteps=RLConfig.TIMESTEPS):
        """
        指定された価格データでエージェントを学習する
        Args:
            price_df: Index=Date, Columns=Ticker の価格データ
            timesteps: 学習ステップ数
        """
        self.tickers = price_df.columns.tolist()
        
        # 環境構築
        # DummyVecEnvは複数の環境を並列実行するためのラッパーだが、1つでも使える
        # lambda関数で環境生成を遅延評価させるのがSB3の作法
        env_maker = lambda: StockTradingEnv(price_df)
        env = DummyVecEnv([env_maker])
        
        # PPOモデル初期化
        # MlpPolicy: 多層パーセプトロン (画像ではないのでこれ)
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="models/rl_logs"
        )
        
        print(f"Starting PPO training for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        print("Training finished.")
        
        # モデル保存 (ティッカー情報のハッシュなどをファイル名にするのが理想だが、簡易的にlatest)
        self.save("latest_agent")
        return self.model

    def predict(self, price_df: pd.DataFrame) -> dict:
        """
        最新の状態から最適な配分を予測する
        Args:
            price_df: 学習時と同じ銘柄セットを含むデータフレーム (直近データが必要)
        Returns:
            {ticker: weight} の辞書
        """
        if self.model is None:
            if not self.load("latest_agent"):
                raise ValueError("Model not trained or loaded.")
                
        # 環境を一時的に作成して、最新の観測状態を取得する
        # (reset -> fix internal state to latest is hard, so we calculate observation manually)
        
        # 直近のデータを使ってObservationを作成
        # Envの内部ロジックと同じ計算が必要
        
        # 簡易的にEnvを作って、reset後に内部状態を強制的に進めるアプローチも取れるが、
        # Observationの計算ロジックを再実装する方が安全かつ高速
        
        window_size = RLConfig.LOOKBACK_WINDOW
        
        # 入力データチェック
        if len(price_df) < window_size + 1:
             raise ValueError(f"Data length must be at least {window_size + 1}")
             
        # 必要な銘柄が含まれているか確認
        # 学習時とカラム順序を合わせる必要がある
        if list(price_df.columns) != self.tickers:
            # カラム順序を再配置、足りない場合はエラー
            try:
                price_df = price_df[self.tickers]
            except KeyError:
                raise ValueError("Input data does not match trained tickers.")
        
        # 最新の観測を作成
        # 1. 現在の配分 (ここではフラットまたは前回の配分と仮定したいが、初期状態としてゼロまたは均等を仮定)
        # 　　実際は「これから投資する」ので、現在の保有は考慮せず、ゼロからの最適化を行うアクションを求めたい
        # 　　しかしEnvは「現在の配分」を状態に含んでいる。
        # 　　ここでは「キャッシュ100%」の状態からの最適アクションを予測する
        current_weights = np.zeros(len(self.tickers)) 
        
        # 2. 過去のリターン (対数リターン)
        log_returns = np.log(price_df / price_df.shift(1)).fillna(0)
        recent_returns = log_returns.iloc[-window_size:].values.flatten()
        
        obs = np.concatenate([current_weights, recent_returns]).astype(np.float32)
        
        # 推論
        # deterministic=True で確定的（平均）なアクションを取得
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Action -> Weights (Softmax or Normalize)
        # Env側で sum > 0 なら正規化している
        weights = np.array(action)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(self.tickers)) / len(self.tickers)
            
        return dict(zip(self.tickers, weights))

    def save(self, name):
        path = os.path.join(self.model_dir, name)
        self.model.save(path)
        # ティッカーリストも保存しておく必要がある
        # (SB3のsaveはモデルパラメータのみ)
        # 別途 json とかで保存
        import joblib
        joblib.dump(self.tickers, path + "_tickers.pkl")
        
    def load(self, name):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path + ".zip"):
            self.model = PPO.load(path)
            # ティッカーリスト読み込み
            import joblib
            if os.path.exists(path + "_tickers.pkl"):
                self.tickers = joblib.load(path + "_tickers.pkl")
            return True
        return False
