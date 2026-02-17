import torch
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet

def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    learning_rate: float = 0.03,
    hidden_size: int = 16,
    attention_head_size: int = 1,
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,
    lstm_layers: int = 1,
    output_size: int = 7,  # QuantileLoss: 7 quantiles by default
) -> TemporalFusionTransformer:
    """
    TFTモデルを初期化して返すファクトリ関数。
    TemporalFusionTransformerはpl.LightningModuleを継承しているため、
    そのままTrainerに渡すことができる。
    """
    
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        lstm_layers=lstm_layers,
        output_size=output_size, 
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return tft
