import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from collections import OrderedDict
from thesisproject.models.mpu import UNet
from thesisproject.utils import (create_overlay_figure, get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitFixedLSTM(pl.LightningModule):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4

    def forward(self, x):
        return self.lstm(x)

    def training_step(self, batch, _batch_idx):
        inputs, labels = batch[0], batch[1]

        outputs = self.lstm(inputs)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels = batch[0], batch[1]

        outputs = self.lstm(inputs)

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
        optimizer = optim.Adam(self.lstm.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val_loss",
                "frequency": 1
            },
        }