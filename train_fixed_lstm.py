import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split

from thesisproject.models.mpu import UNet, LitMPU
from thesisproject.models.encoder import Encoder, LitEncoder
from thesisproject.models.lstm import LSTMDataModule, LSTM, LitFixedLSTM

# Data
image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/image_samples.csv"

train_indices = np.load("/home/blg515/train_ids.npy", allow_pickle=True).astype(str)
train_indices, val_indices = train_test_split(train_indices, test_size=0.5)

lstm_data = LSTMDataModule(
    image_path,
    subjects_csv,
    batch_size=8,
    train_slices_per_epoch=1000,
    val_slices_per_epoch=500,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=val_indices
)

## Models
mpu_checkpoint = "/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_9494/checkpoints/epoch=36-step=4625.ckpt"
encoder_checkpoint = "/home/blg515/masters-thesis/model_saves/encoder/lightning_logs/version_3040/checkpoints/epoch=53-step=3402.ckpt"

unet = UNet(1, 9, 384)
lit_mpu = LitMPU(unet).load_from_checkpoint(mpu_checkpoint, unet=unet)

encoder = Encoder(1448, unet.fc_in, 500)
lit_encoder = LitEncoder(unet, encoder).load_from_checkpoint(encoder_checkpoint, unet=unet, encoder=encoder)

lstm = LSTM(encoder.vector_size + 1, 1000, 2) #input size is vector size + 1 if adding dt
model = LitFixedLSTM(unet, encoder, lstm)

# Callbacks
early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=15)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Training
checkpoint_path = None

num_gpus = torch.cuda.device_count()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/fixed-lstm/",
    profiler="simple",
    auto_lr_find=True,
    auto_scale_batch_size=False,
    enable_progress_bar=True,
    accumulate_grad_batches=2,
    max_epochs=100
)

#trainer.tune(model, datamodule=lstm_data)

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=lstm_data
)