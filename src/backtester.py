"""
FAR_stock ROIベースバックテストモジュール
推薦シグナルに基づくトレードシミュレーションとROI評価。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go


class ROIBacktester:
    """ROIベースの戦略バックテスト"""

    def __init__(self, initial_capital: float = 1_000_000,
                 max_positions: int = 5,
                 holding_days: int = 5,
                 stop_loss_pct: float = -10.0):
        """
        Args:
            initial_capital: 初期資金（円）
            max_positions: 最大同時保有銘柄数
            holding_days: 保有日数
            stop_loss_pct: 損切りライン（%）
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.holding_days = holding_days
        self.stop_loss_pct = stop_loss_pct
        self.trades = []
        self.equity_curve = []
        self.metrics_ = {}

    def run(self, signals_df: pd.DataFrame,
            price_dict: dict) -> dict:
        """
        シグナルに基づくシミュレーションを実行する。

        Args:
            signals_df: columns=['date', 'ticker', 'score'] のシグナルDF
                         dateはシグナル発生日、scoreは推薦スコア
            price_dict: {ticker: DataFrame(OHLCV)} の辞書
        Returns:
            dict: バックテスト結果
        """
        capital = self.initial_capital
        positions = []  # {'ticker': str, 'entry_date', 'entry_price', 'shares'}
        self.trades = []
        equity_history = []

        if signals_df.empty:
            self.metrics_ = self._empty_metrics()
            return self.metrics_

        # 日付でソート
        signals_df = signals_df.sort_values('date').reset_index(drop=True)
        all_dates = sorted(signals_df['date'].unique())

        for current_date in all_dates:
            # 1. ポジションの決済チェック
            positions_to_close = []
            for pos in positions:
                ticker = pos['ticker']
                if ticker not in price_dict:
                    continue

                price_df = price_dict[ticker]
                # 現在の日付以降のデータを取得
                future_prices = price_df[price_df.index >= current_date]
                if future_prices.empty:
                    continue

                days_held = (current_date - pos['entry_date']).days

                current_price = future_prices['Close'].iloc[0]
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100

                # 保有期間超過 or 損切り
                if days_held >= self.holding_days or pnl_pct <= self.stop_loss_pct:
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    capital += current_price * pos['shares']

                    self.trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'reason': 'stop_loss' if pnl_pct <= self.stop_loss_pct else 'holding_period',
                    })
                    positions_to_close.append(pos)

            for p in positions_to_close:
                positions.remove(p)

            # 2. 新規エントリー
            today_signals = signals_df[signals_df['date'] == current_date]
            today_signals = today_signals.sort_values('score', ascending=False)

            available_slots = self.max_positions - len(positions)
            if available_slots > 0 and capital > 0:
                for _, sig in today_signals.head(available_slots).iterrows():
                    ticker = sig['ticker']
                    if ticker not in price_dict:
                        continue

                    price_df = price_dict[ticker]
                    entry_prices = price_df[price_df.index >= current_date]
                    if entry_prices.empty:
                        continue

                    entry_price = entry_prices['Close'].iloc[0]
                    if entry_price <= 0:
                        continue

                    # 均等配分
                    position_size = capital / (available_slots + 1)
                    shares = int(position_size / entry_price)

                    if shares > 0:
                        cost = entry_price * shares
                        capital -= cost
                        positions.append({
                            'ticker': ticker,
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'shares': shares,
                        })

            # 3. 資産額記録
            positions_value = sum(
                self._get_price_on_date(price_dict, p['ticker'], current_date) * p['shares']
                for p in positions
            )
            total_equity = capital + positions_value
            equity_history.append({
                'date': current_date,
                'equity': total_equity,
                'cash': capital,
                'positions': len(positions),
            })

        self.equity_curve = pd.DataFrame(equity_history)
        self.metrics_ = self._calculate_metrics()
        return self.metrics_

    def _get_price_on_date(self, price_dict: dict, ticker: str,
                           date) -> float:
        """指定日の終値を取得する。"""
        if ticker not in price_dict:
            return 0
        df = price_dict[ticker]
        prices_after = df[df.index >= date]
        if prices_after.empty:
            prices_before = df[df.index <= date]
            return prices_before['Close'].iloc[-1] if not prices_before.empty else 0
        return prices_after['Close'].iloc[0]

    def _calculate_metrics(self) -> dict:
        """バックテスト結果の評価指標を算出する。"""
        if not self.trades:
            return self._empty_metrics()

        trades_df = pd.DataFrame(self.trades)

        # 基本指標
        total_pnl = trades_df['pnl'].sum()
        roi = total_pnl / self.initial_capital * 100
        num_trades = len(trades_df)
        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(win_trades) / num_trades * 100

        # リスク指標
        avg_pnl_pct = trades_df['pnl_pct'].mean()
        max_loss = trades_df['pnl_pct'].min()

        # シャープレシオ（簡易版）
        if trades_df['pnl_pct'].std() > 0:
            sharpe = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()
        else:
            sharpe = 0

        # 最大ドローダウン
        max_dd = 0
        if not self.equity_curve.empty:
            equity = self.equity_curve['equity']
            peak = equity.expanding().max()
            dd = (equity - peak) / peak * 100
            max_dd = dd.min()

        return {
            'total_pnl': total_pnl,
            'roi_pct': roi,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl_pct,
            'max_loss_pct': max_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
        }

    def _empty_metrics(self) -> dict:
        return {
            'total_pnl': 0, 'roi_pct': 0, 'num_trades': 0,
            'win_rate': 0, 'avg_pnl_pct': 0, 'max_loss_pct': 0,
            'sharpe_ratio': 0, 'max_drawdown_pct': 0,
        }

    def plot_equity_curve(self) -> go.Figure:
        """資産推移チャートを生成する。"""
        if self.equity_curve.empty:
            fig = go.Figure()
            fig.add_annotation(text="データなし", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.equity_curve['date'],
            y=self.equity_curve['equity'],
            mode='lines',
            name='資産額',
            line=dict(color='#2196F3', width=2),
        ))

        # 初期資金ライン
        fig.add_hline(
            y=self.initial_capital,
            line_dash='dash',
            line_color='gray',
            annotation_text=f"初期資金: ¥{self.initial_capital:,.0f}"
        )

        fig.update_layout(
            title="資産推移 (Equity Curve)",
            xaxis_title="日付",
            yaxis_title="資産額 (¥)",
            template='plotly_white',
            height=400,
        )
        return fig

    def get_trades_df(self) -> pd.DataFrame:
        """トレード履歴をDataFrameで取得する。"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
