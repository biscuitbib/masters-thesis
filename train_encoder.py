import os
import sys

import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split

from thesisproject.models.mpu import LitMPU, UNet
from thesisproject.models.encoder import Encoder, LitEncoder, EncoderDataModule

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]
encoding_size_index = int(args[0])

assert torch.cuda.is_available(), "No GPU available."

# Data
image_path = "/home/blg515/masters-thesis/oai_images" #"/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/masters-thesis/image_samples.csv"

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
train_indices, val_indices = train_test_split(train_indices, test_size=0.6, shuffle=True)

encoder_data = EncoderDataModule(
    image_path,
    subjects_csv,
    batch_size=16,
    train_slices_per_epoch=1000,
    val_slices_per_epoch=500,
    train_indices=train_indices,
    val_indices=val_indices
)

## Model
mpu_checkpoint = hparams["mpu_path"]

label_keys = ["Lateral femoral cart.",
                "Lateral meniscus",
                "Lateral tibial cart.",
                "Medial femoral cartilage",
                "Medial meniscus",
                "Medial tibial cart.",
                "Patellar cart.",
                "Tibia"]

unet = UNet(1, 9, 384, class_names=label_keys)

mpu = LitMPU.load_from_checkpoint(mpu_checkpoint, unet=unet)

unet: UNet = mpu.unet

encoding_size = hparams["encoder"]["encoding_size"][encoding_size_index]

print(f"Training encoder with encoding_size={encoding_size}")

encoder = Encoder(1448, unet.fc_in, encoding_size)

model = LitEncoder(unet, encoder)

# Training
checkpoint_path = None

# Callbacks
early_stopping = EarlyStopping("loss/train", mode="min", min_delta=0.0, patience=15)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

num_gpus = torch.cuda.device_count()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/encoder/",
    profiler="simple",
    enable_progress_bar=True,
    accumulate_grad_batches=2,
    max_epochs=100
)

print(f"saving model to {trainer.logger.log_dir}")

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=encoder_data
)