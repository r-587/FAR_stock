"""
FAR_stock データ取得モジュール
JPX銘柄一覧とyfinance株価データの取得・キャッシュ管理。
"""

import os
import time
import pandas as pd
import yfinance as yf
from typing import Optional
from config import APIConfig
from src import db


def fetch_jpx_tickers(force_update: bool = False) -> pd.DataFrame:
    """
    JPXのWebサイトから銘柄一覧を取得しDataFrameとして返す。
    ローカルにキャッシュがある場合はそれを使用する。

    Returns:
        pd.DataFrame: columns=['Ticker', 'Name', 'Market', 'Sector']
    """
    if not force_update and os.path.exists(APIConfig.CACHE_FILE):
        return pd.read_csv(APIConfig.CACHE_FILE, dtype={'Ticker': str})

    try:
        df = pd.read_excel(APIConfig.JPX_URL)
        df.columns = df.columns.str.strip()

        column_map = {
            'コード': 'Ticker',
            '銘柄名': 'Name',
            '市場・商品区分': 'Market',
            '33業種区分': 'Sector'
        }

        missing_cols = [c for c in column_map if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}. Found: {df.columns.tolist()}")

        df = df.rename(columns=column_map)

        # 東証の主要市場のみ
        target_markets = ['プライム', 'スタンダード', 'グロース']
        mask = df['Market'].astype(str).apply(
            lambda x: any(m in x for m in target_markets)
        )
        df = df[mask].copy()

        # yfinance用にティッカー形式を変換
        df['Ticker'] = df['Ticker'].astype(str).str.zfill(4) + ".T"
        df = df[['Ticker', 'Name', 'Market', 'Sector']].reset_index(drop=True)

        # ハイフン業種を除外
        df = df[df['Sector'] != '-']

        # キャッシュ保存
        df.to_csv(APIConfig.CACHE_FILE, index=False)

        return df

    except Exception as e:
        print(f"Error fetching JPX data: {e}")
        if os.path.exists(APIConfig.CACHE_FILE):
            print("Falling back to cached data.")
            return pd.read_csv(APIConfig.CACHE_FILE, dtype={'Ticker': str})
        return pd.DataFrame()


def get_stock_data(ticker: str, period: str = "1y",
                   interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    指定された銘柄の株価データをyfinanceから取得する。

    Returns:
        Optional[pd.DataFrame]: OHLCV DataFrame. 失敗時はNone.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def update_stock_data(tickers: list, progress_callback=None):
    """
    複数銘柄のデータをyfinanceからバッチ取得してDBに保存する。

    Args:
        tickers: ティッカーコードのリスト
        progress_callback: (current, total) を受け取るコールバック
    """
    if not tickers:
        return

    chunk_size = APIConfig.CHUNK_SIZE
    total = len(tickers)

    for i in range(0, total, chunk_size):
        chunk = tickers[i:i + chunk_size]

        try:
            data = yf.download(
                chunk,
                period="1y",
                group_by='ticker',
                threads=True,
                progress=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                available_tickers = data.columns.get_level_values(0).unique()
                data_dict = {}
                for t in available_tickers:
                    try:
                        df_t = data.xs(t, axis=1, level=0, drop_level=True)
                        df_t = df_t.dropna(how='all')
                        if not df_t.empty:
                            data_dict[t] = df_t
                    except KeyError:
                        continue
                db.save_prices_bulk(data_dict)
            else:
                if not data.empty and len(chunk) == 1:
                    db.save_prices_bulk({chunk[0]: data})

        except Exception as e:
            print(f"Error fetching chunk {i}: {e}")

        if progress_callback:
            progress_callback(min(i + chunk_size, total), total)

        time.sleep(APIConfig.WAIT_SEC)


def get_stock_data_cached(tickers: list, force_update: bool = False) -> dict:
    """
    DB優先で株価データを取得する。キャッシュミスのみAPI取得。

    Returns:
        dict: {ticker: DataFrame}
    """
    if not tickers:
        return {}

    data_map = db.load_prices(tickers)
    missing = [t for t in tickers if t not in data_map or data_map[t].empty]

    if missing:
        print(f"Fetching {len(missing)} missing tickers from API...")
        update_stock_data(missing)
        new_data = db.load_prices(missing)
        data_map.update(new_data)

    return data_map
