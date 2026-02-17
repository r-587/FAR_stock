import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

def create_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    batch_size: int = 64
):
    """
    DataFrameからTFT学習用のTimeSeriesDataSetを作成する。
    
    Args:
        df: 'ticker', 'Date', 'Close', 'Volume', ... を含むDataFrame
        max_encoder_length: 入力系列長 (過去何日分を見るか)
        max_prediction_length: 予測期間 (未来何日分を予測するか)
        batch_size: バッチサイズ
        
    Returns:
        training_dataset, validation_dataset, train_dataloader, val_dataloader
    """
    
    # 日付を整数インデックスに変換 (time_idx)
    # 各tickerごとに日付順にソートし、連番を振る必要あり
    # ここでは簡易的に、全データの日付を一意なIDに変換するアプローチをとる
    # ただし、tickerごとに欠損日があるとずれるため、カレンダー日付を元にした通し番号が良い
    
    df = df.copy()
    if 'Date' not in df.columns:
        raise ValueError("DataFrame must have 'Date' column.")
        
    # Date型変換
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
         df['Date'] = pd.to_datetime(df['Date'])
         
    # time_idx作成 (最古の日付からの経過日数)
    min_date = df['Date'].min()
    df['time_idx'] = (df['Date'] - min_date).dt.days
    
    # Tickerはカテゴリ変数として扱う
    df['ticker'] = df['ticker'].astype(str)
    
    # 特徴量定義
    # 既知の未来 (Known Reals): 日付要素 (月, 日, 曜日)
    df['month'] = df['Date'].dt.month.astype(str).astype('category')
    df['day'] = df['Date'].dt.day.astype(str).astype('category')
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype('category')
    
    # データセット定義
    training_cutoff = df['time_idx'].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Close",  # 予測対象 (Close価格) ※リターン予測にする場合は変更検討
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,  # allow less history
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        time_varying_known_categoricals=["month", "day", "day_of_week"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "Close", "Volume", "Open", "High", "Low", 
            # テクニカル指標があればここに追加 (RSI, SMAなど)
        ],
        target_normalizer=GroupNormalizer(
            groups=["ticker"], transformation="softplus"
        ),  # Tickerごとに正規化
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True # 株式市場は休日があるのでTrue
    )
    
    # Validation Dataset (直近のデータを検証用にする)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    
    # Dataloader作成
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )
    
    return training, validation, train_dataloader, val_dataloader
