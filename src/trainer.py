import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from src.dataset import create_tft_dataset
from src.tft_model import create_tft_model
from config import GPU_AVAILABLE

class TFTTrainer:
    def __init__(self, data_dir: str = "models/tft_checkpoints"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def train(
        self,
        df,
        max_epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 1,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        lstm_layers: int = 1,
    ):
        # 1. データセット作成
        training, validation, train_dataloader, val_dataloader = create_tft_dataset(
            df, batch_size=batch_size
        )
        
        # 2. モデル作成
        tft = create_tft_model(
            training_dataset=training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            lstm_layers=lstm_layers,
        )
        
        # 3. Callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=5,
            verbose=False,
            mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.data_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        )
        
        # 4. Trainer設定
        # 4. Trainer設定
        # PyTorchのGPUが使えるなら使う (LightGBMのGPU判定とは独立させる)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if accelerator == "gpu" else "auto"
        
        logger = TensorBoardLogger("lightning_logs", name="tft_model")
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
        )
        
        # 5. 学習実行
        print(f"Starting TFT training on {accelerator}...")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # 6. ベストモデルのロード
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        return best_tft, trainer
