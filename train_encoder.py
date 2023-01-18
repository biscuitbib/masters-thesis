import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from thesisproject.models.mpu import LitMPU, UNet
from thesisproject.models.encoder import Encoder, LitEncoder, EncoderDataModule

# Data
image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/image_samples_edit.csv"

encoder_data = EncoderDataModule(
    image_path,
    subjects_csv,
    batch_size=12,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000,
    train_val_ratio=[0.8, 0.2]
)

## Model
mpu_checkpoint = "/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_9098/checkpoints/"
mpu_checkpoint += os.listdir(mpu_checkpoint)[0]#"/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_6692/checkpoints/unet.ckpt"

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

encoder = Encoder(1448, unet.fc_in, 200)

model = LitEncoder(unet, encoder)

# Training
checkpoint_path = None#"/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_6438/checkpoints/epoch=33-step=5678.ckpt"

# Callbacks
early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=9)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

num_gpus = torch.cuda.device_count()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/encoder/",
    profiler="simple",
    auto_lr_find=True,
    auto_scale_batch_size=False,
    enable_progress_bar=True,
    max_epochs=100
)

trainer.tune(model, datamodule=encoder_data)

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=encoder_data
)
