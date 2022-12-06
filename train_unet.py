import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from thesisproject.models.mpu import LitMPU, UNet, SegmentationDataModule

# Data
path = "../knee_data/"

segmentation_data = SegmentationDataModule(
    path,
    batch_size=8,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000
)

## Model
label_keys = ["Lateral femoral cart.",
                "Lateral meniscus",
                "Lateral tibial cart.",
                "Medial femoral cartilage",
                "Medial meniscus",
                "Medial tibial cart.",
                "Patellar cart.",
                "Tibia"]

unet = UNet(1, 9, 384, class_names=label_keys)

model = LitMPU(unet)

# Training
checkpoint_path = None#"/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_6438/checkpoints/epoch=33-step=5678.ckpt"

# Callbacks
early_stopping = EarlyStopping("val/dice", mode="max", min_delta=0.0, patience=6)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

num_gpus = torch.cuda.device_count()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/unet/",
    profiler="simple",
    auto_lr_find=False,
    auto_scale_batch_size=False,
    enable_progress_bar=True,
    max_epochs=50
)

#trainer.tune(model, datamodule=segmentation_data)

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=segmentation_data
)
