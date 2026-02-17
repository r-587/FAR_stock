"""
FAR_stock データベース操作モジュール
SQLiteを使用した株価データ・予測結果・ウォッチリストの永続化。
"""

import sqlite3
import datetime
import pandas as pd
from config import APIConfig


def get_connection():
    """DB接続を取得する。"""
    return sqlite3.connect(APIConfig.DB_PATH)


def init_db():
    """データベーステーブルを初期化する。"""
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY,
        name TEXT,
        market TEXT,
        sector TEXT,
        last_updated DATETIME
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT,
        date DATETIME,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (ticker, date)
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        predicted_at DATETIME,
        surge_probability REAL,
        rule_score REAL,
        ensemble_score REAL,
        signals_json TEXT
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY,
        note TEXT,
        added_at DATETIME
    )
    ''')

    conn.commit()
    conn.close()


def save_tickers(df: pd.DataFrame):
    """銘柄情報をupsertする。dfは Ticker, Name, Market, Sector カラムを持つ。"""
    conn = get_connection()
    data = []
    now = datetime.datetime.now()
    for _, row in df.iterrows():
        data.append((
            row['Ticker'], row['Name'], row['Market'],
            row['Sector'], now
        ))
    c = conn.cursor()
    c.executemany('''
    INSERT OR REPLACE INTO tickers (ticker, name, market, sector, last_updated)
    VALUES (?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()


def save_prices_bulk(data_dict: dict):
    """複数銘柄の株価データを一括保存する。data_dict: {ticker: DataFrame}"""
    conn = get_connection()
    c = conn.cursor()
    records = []

    for ticker, df in data_dict.items():
        if df is None or df.empty:
            continue
        df_reset = df.reset_index()
        for _, row in df_reset.iterrows():
            date_val = row.get('Date', row.name)
            if hasattr(date_val, 'to_pydatetime'):
                date_val = date_val.to_pydatetime()
            records.append((
                ticker, date_val,
                row.get('Open'), row.get('High'),
                row.get('Low'), row.get('Close'),
                row.get('Volume')
            ))

    if records:
        c.executemany('''
        INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)

    conn.commit()
    conn.close()


def load_prices(tickers: list, days: int = 400) -> dict:
    """DBから株価データを読み込む。{ticker: DataFrame}を返す。"""
    conn = get_connection()
    if not tickers:
        conn.close()
        return {}

    placeholders = ','.join(['?'] * len(tickers))
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)

    query = f'''
    SELECT ticker, date, open, high, low, close, volume
    FROM prices
    WHERE ticker IN ({placeholders}) AND date >= ?
    ORDER BY date ASC
    '''
    params = list(tickers) + [cutoff]
    df_all = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    conn.close()

    if df_all.empty:
        return {}

    result = {}
    for ticker, group in df_all.groupby('ticker'):
        group = group.set_index('date').sort_index()
        group = group.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }).drop(columns=['ticker'])
        result[ticker] = group
    return result


def save_prediction(ticker: str, surge_prob: float, rule_score: float,
                    ensemble_score: float, signals_json: str):
    """予測結果を保存する。"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
    INSERT INTO predictions (ticker, predicted_at, surge_probability,
                             rule_score, ensemble_score, signals_json)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, datetime.datetime.now(), surge_prob,
          rule_score, ensemble_score, signals_json))
    conn.commit()
    conn.close()


# ウォッチリスト操作
def add_to_watchlist(ticker: str, note: str = ""):
    conn = get_connection()
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO watchlist (ticker, note, added_at) VALUES (?, ?, ?)',
              (ticker, note, datetime.datetime.now()))
    conn.commit()
    conn.close()


def remove_from_watchlist(ticker: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM watchlist WHERE ticker = ?', (ticker,))
    conn.commit()
    conn.close()


def get_watchlist() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM watchlist ORDER BY added_at DESC', conn)
    conn.close()
    return df


def is_in_watchlist(ticker: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT 1 FROM watchlist WHERE ticker = ?', (ticker,))
    result = c.fetchone()
    conn.close()
    return result is not None


# 初期化を実行
init_db()
