import os
import sys

import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split

from thesisproject.models.mpu import UNet, LitMPU
from thesisproject.models.encoder import Encoder, LitEncoder
from thesisproject.models.linear_model import LinearDataModule, LinearModel, LitFixedLinearModel, LitFullLinearModel

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]
index = int(args[0])

encoding_size_index = index % 3
n_visits_index = index // 3

assert torch.cuda.is_available(), "No GPU available."

# Data
image_path = "/home/blg515/masters-thesis/oai_images" #"/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/masters-thesis/image_samples.csv"

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]

linear_data = LinearDataModule(
    image_path,
    subjects_csv,
    batch_size=16,
    n_visits=n_visits,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
)

## Models
mpu_checkpoint = mpu_checkpoint = hparams["mpu_path"]

unet = UNet(1, 9, 384)
lit_mpu = LitMPU(unet).load_from_checkpoint(mpu_checkpoint, unet=unet)

encoding_size = hparams["encoder"]["encoding_size"][encoding_size_index]
encoder_checkpoint = hparams["encoder"]["encoding_path"][encoding_size_index]

encoder = Encoder(1448, unet.fc_in, encoding_size)
lit_encoder = LitEncoder(unet, encoder).load_from_checkpoint(encoder_checkpoint, unet=unet, encoder=encoder)

linear_checkpoint = hparams["linear_regression"]["fixed_linear_path"][encoding_size_index * 3 + n_visits_index]

linear = LinearModel(encoding_size * n_visits)
lit_linear = LitFixedLinearModel(unet, encoder, linear, n_visits=n_visits).load_from_checkpoint(linear_checkpoint, unet=unet, encoder=encoder, linear=linear, n_visits=n_visits)

model = LitFullLinearModel(unet, encoder, linear, n_visits=n_visits, lr=1e-4, weight_decay=0.01)
print(f"Training full Linear model with encoding_size={encoding_size} and n_visits={n_visits}")

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
    default_root_dir="model_saves/full-linear/",
    profiler="simple",
    enable_progress_bar=False,
    accumulate_grad_batches=2,
    max_epochs=200
)

print(f"saving model to {trainer.logger.log_dir}")

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=linear_data
)