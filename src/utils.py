"""
FAR_stock ユーティリティモジュール
共通ヘルパー関数群。
"""

import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def validate_ticker_symbol(ticker: str) -> bool:
    """ティッカーコードのバリデーション。"""
    if not ticker:
        return False
    return bool(re.match(r'^[A-Za-z0-9\.\-]+$', ticker))


def format_large_number(n) -> str:
    """大きな数値を読みやすくフォーマットする。"""
    if n is None or pd.isna(n):
        return 'N/A'
    n = float(n)
    if abs(n) >= 1e12:
        return f"{n/1e12:.1f}兆"
    elif abs(n) >= 1e8:
        return f"{n/1e8:.0f}億"
    elif abs(n) >= 1e4:
        return f"{n/1e4:.0f}万"
    else:
        return f"{n:.0f}"


def plot_stock_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    """
    テクニカル指標付きの株価チャートを生成する。

    Args:
        df: テクニカル指標付きDataFrame
        title: チャートタイトル
    Returns:
        go.Figure: Plotlyチャート
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{title} 株価", "RSI", "出来高"),
        row_heights=[0.6, 0.2, 0.2],
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="OHLC"
    ), row=1, col=1)

    # SMA
    sma_styles = [
        ('SMA_short', 'SMA 5', 'orange'),
        ('SMA_medium', 'SMA 20', 'blue'),
        ('SMA_long', 'SMA 50', 'purple'),
        ('SMA_200', 'SMA 200', 'red'),
    ]
    for col, name, color in sma_styles:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=1),
                name=name
            ), row=1, col=1)

    # BB
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            line=dict(color='rgba(128,128,128,0.3)', width=1),
            name="BB Upper", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            line=dict(color='rgba(128,128,128,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
            name="BB Lower", showlegend=False
        ), row=1, col=1)

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            line=dict(color='black', width=1), name="RSI"
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 出来高
    colors = ['red' if row['Close'] < row['Open'] else 'green'
              for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=colors, name="Volume", showlegend=False
    ), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig
