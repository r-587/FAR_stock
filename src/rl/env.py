import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from config import RLConfig

class StockTradingEnv(gym.Env):
    """
    複数銘柄のポートフォリオ最適化を行う強化学習環境
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, price_df: pd.DataFrame, initial_balance=RLConfig.CASH_INITIAL):
        super(StockTradingEnv, self).__init__()
        
        self.price_df = price_df.sort_index()
        self.tickers = price_df.columns.tolist()
        self.n_tickers = len(self.tickers)
        self.initial_balance = initial_balance
        
        # Action Space: ポートフォリオの配分比率 (合計1になるように正規化する)
        # Box(low=0, high=1, shape=(n_tickers + 1,))  +1は現金ポジション
        # ここでは簡易化のためフルインベストメントとし、n_tickersのみとする
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_tickers,), dtype=np.float32
        )
        
        # Observation Space: 
        # [現在の保有比率(n), 過去M日のリターン(n * M)] など
        # ここではシンプルに、
        # 1. 現在の保有比率 (n)
        # 2. 過去LOOKBACK_WINDOW日の対数リターン (n * window)
        self.window_size = RLConfig.LOOKBACK_WINDOW
        self.obs_shape = self.n_tickers + (self.n_tickers * self.window_size)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        # 内部状態
        self.current_step = 0
        self.max_steps = len(self.price_df) - self.window_size - 1
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_tickers) # 保有株数
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.n_tickers) # 現在の配分
        
        # リターン計算用
        self.returns_df = self.price_df.pct_change().fillna(0)
        self.log_returns_df = np.log(self.rice_df / self.price_df.shift(1)).fillna(0) if not self.price_df.empty else pd.DataFrame() # typo fix later: rice -> price
        
        # typo fix for log_returns calculation above
        self.log_returns_df = np.log(self.price_df / self.price_df.shift(1)).fillna(0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.n_tickers)
        # 最初の配分は均等とするか、現金100%から始めるか
        # ここでは現金100%スタート（weights all zero -> 実際はactionで決まる）
        
        return self._next_observation(), {}

    def _next_observation(self):
        # 1. 現在の配分 (n)
        obs_weights = self.weights
        
        # 2. 過去のリターン (n * window)
        # current_stepは現在の日付インデックス
        # 過去 window_size 分のデータを取得
        returns = self.log_returns_df.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        
        obs = np.concatenate([obs_weights, returns])
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Actionの正規化 (Softmax or L1 norm)
        # actionは正の値であることが期待される (Box low=0)
        weights = np.array(action)
        sum_weights = np.sum(weights)
        
        if sum_weights > 0:
            weights = weights / sum_weights
        else:
            weights = np.zeros(self.n_tickers) # 現金ポジション相当だが、今回はフルインベストメント前提の簡易版
            # 現金ポジションを導入しないと、全投げ売りができない。
            # 今回は簡易版として、weightの合計が1になるように強制し、常に投資し続けるモデルとする。
            # 現金退避も学習させたい場合はアクション空間を n+1 にする。
            weights = np.ones(self.n_tickers) / self.n_tickers
            
        # 2. ポートフォリオ価値の更新
        # 前回の配分 weights_update ではなく、
        # 今回提示された weights に基づいてリバランスを行うとする。
        # リバランス前のポートフォリオ価値 = sum(保有株数 * 現在価格) + 現金
        
        current_prices = self.price_df.iloc[self.current_step].values
        next_prices = self.price_df.iloc[self.current_step + 1].values
        
        # 簡易シミュレーション:
        # 手数料を無視すれば、
        # Portfolio_Value_t = Portfolio_Value_{t-1} * (1 + sum(w_i * r_i))
        # r_i は t から t+1 へのリターン
        
        market_returns = (next_prices - current_prices) / current_prices
        portfolio_return = np.sum(weights * market_returns)
        
        # Transaction Cost (簡易)
        # 配分の変化量に応じたコスト
        turnover = np.sum(np.abs(weights - self.weights))
        cost = RLConfig.TRANSACTION_COST_PCT * turnover
        
        # Reward
        # ポートフォリオ実リターン - コスト
        net_return = portfolio_return - cost
        self.portfolio_value *= (1 + net_return)
        
        # 配分更新
        self.weights = weights
        
        # Step進行
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Reward Function
        # 単純なリターン最大化ではなく、シャープレシオを意識させる
        # しかしStepごとの報酬でSharpeは計算しにくい。
        # 一般的には対数リターンまたは差分資産額を用いる。
        reward = net_return * 100 # スケール調整
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': net_return
        }
        
        return self._next_observation(), reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value:.0f}")
