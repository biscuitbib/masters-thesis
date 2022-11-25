import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from thesisproject.data import ImagePairDataset, SegmentationDataModule
from thesisproject.models import LitMPU, UNet

# Data
#path = "../toy-data/"
path = "../knee_data/"

segmentation_data = SegmentationDataModule(
    path, batch_size=8,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000
)


# Model
checkpoint_path = "/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_4476/checkpoints/" + os.listdir("/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_4476/checkpoints/")[0]

label_keys = [
    "Lateral femoral cart.",
    "Lateral meniscus",
    "Lateral tibial cart.",
    "Medial femoral cartilage",
    "Medial meniscus",
    "Medial tibial cart.",
    "Patellar cart.",
    "Tibia"]

unet = UNet(1, 9, 384, class_names=label_keys)
litunet = LitMPU.load_from_checkpoint(checkpoint_path, unet=unet)
litunet.eval()

# initialize the Trainer
num_gpus = torch.cuda.device_count()
trainer = pl.Trainer(
    accelerator="gpu",
    devices=num_gpus,
    default_root_dir="model_saves/unet/"
)

# test the model
trainer.test(litunet, datamodule=segmentation_data)