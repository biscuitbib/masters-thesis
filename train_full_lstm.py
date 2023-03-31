import os
import sys

import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from thesisproject.models.mpu import UNet, LitMPU
from thesisproject.models.encoder import Encoder, LitEncoder
from thesisproject.models.lstm import LSTMDataModule, LSTM, LitFixedLSTM, LitFullLSTM

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]
index = int(args[0])

encoding_size_index = index % 3
hidden_size_index = index // 3

assert torch.cuda.is_available(), "No GPU available."

# Data
image_path = "/home/blg515/masters-thesis/oai_images" #"/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/masters-thesis/image_samples.csv"

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

lstm_data = LSTMDataModule(
    image_path,
    subjects_csv,
    batch_size=8,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
)

## Models
mpu_checkpoint = hparams["mpu_path"]

unet = UNet(1, 9, 384)
lit_mpu = LitMPU(unet).load_from_checkpoint(mpu_checkpoint, unet=unet)

encoding_size = hparams["encoder"]["encoding_size"][encoding_size_index]
encoder_checkpoint = hparams["encoder"]["encoding_path"][encoding_size_index]

encoder = Encoder(1448, unet.fc_in, encoding_size)
lit_encoder = LitEncoder(unet, encoder).load_from_checkpoint(encoder_checkpoint, unet=unet, encoder=encoder)

hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
lstm_checkpoint = hparams["lstm"]["fixed_lstm_path"][encoding_size_index * 3 + hidden_size_index]

lstm = LSTM(encoder.vector_size + 1, hidden_size, 2)
lit_lstm = LitFixedLSTM(unet, encoder, lstm).load_from_checkpoint(lstm_checkpoint, unet=unet, encoder=encoder, lstm=lstm)

print(f"Training end-to-end LSTM with encoding_size={encoding_size}, hidden_size={hidden_size}")

model = LitFullLSTM(unet, encoder, lstm)

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
    default_root_dir="model_saves/full-lstm/",
    profiler="simple",
    enable_progress_bar=True,
    accumulate_grad_batches=2,
    max_epochs=200
)

print(f"saving model to {trainer.logger.log_dir}")

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=lstm_data
)
