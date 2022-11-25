"""
U-Net model to be used in Multiplaner U-Net.
Reworked from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from collections import OrderedDict
from thesisproject.models.unet import UNet
from thesisproject.utils import (create_overlay_figure, get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitEncoder(pl.LightningModule):
    """
    Pytorch Lightning module representing a multiplanar U-net encoder:
    Takes a pretrained U-net and.
    Training and validation is done on slices of image volumes and predictions are made on entire image volumes.
    """
    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet
        self.unet.encode = True

        self.fc = nn.Linear(self.unet.encoding_size, 2) #Linear layer for pretraining

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        encodings = self.unet(images)
        outputs = self.fc(encodings)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]

        encodings = self.unet(images)
        outputs = self.fc(encodings)

        loss = self.criterion(outputs, labels)

        # log metrics
        metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

        log_values = {
            "loss/val_loss": loss.detach(),
            "val/accuracy": np.mean(metrics["accuracy"]),
            "val/precision": np.mean(metrics["precision"]),
            "val/recall": np.mean(metrics["recall"]),
            "val/specificity": np.mean(metrics["specificity"]),
        }
        self.log_dict(log_values, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        encodings = self.unet(images)
        outputs = self.fc(encodings)

        loss = self.criterion(outputs, labels)

        # log metrics
        metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

        log_values = {
            "loss/val_loss": loss.detach(),
            "val/accuracy": np.mean(metrics["accuracy"]),
            "val/precision": np.mean(metrics["precision"]),
            "val/recall": np.mean(metrics["recall"]),
            "val/specificity": np.mean(metrics["specificity"]),
        }
        self.log_dict(log_values, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.unet.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val_loss",
                "frequency": 1
            },
        }