import numpy as np
import pytorch_lightning as pl
import torch
from .encoder import Encoder
from thesisproject.models.mpu import UNet
from thesisproject.utils import get_multiclass_metrics, save_metrics_csv
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitEncoder(pl.LightningModule):
    """
    Pytorch Lightning module representing a multiplanar U-net encoder:
    Takes a pretrained U-net and.
    Training and validation is done on slices of image volumes and predictions are made on entire image volumes.
    """
    def __init__(self, unet: UNet, encoder: Encoder):
        super().__init__()
        self.unet = unet
        self.unet.encode = True

        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.encoder.vector_size, 2)
        ) #Linear layer for pretraining

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4
        self.weight_decay = 0 #1e-6

    def forward(self, x):
        x = self.unet(x)
        x = self.encoder(x)
        return x

    def training_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        self.unet.eval()
        with torch.no_grad():
            unet_bottleneck = self.unet(images)

        encodings = self.encoder(unet_bottleneck)
        outputs = self.fc(encodings)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]

        self.unet.eval()
        with torch.no_grad():
            unet_bottleneck = self.unet(images)

            encodings = self.encoder(unet_bottleneck)
            outputs = self.fc(encodings)

        print(f"pred vs label: {torch.argmax(outputs, dim=1)} {labels}")

        loss = self.criterion(outputs, labels)

        # log metrics
        metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

        log_values = {
            "val/class_ratio": np.mean(labels.detach().cpu().numpy()),
            "loss/val_loss": loss.detach(),
            "val/accuracy": np.mean(metrics["accuracy"]),
            "val/precision": np.mean(metrics["precision"]),
            "val/recall": np.mean(metrics["recall"]),
            "val/specificity": np.mean(metrics["specificity"]),
        }
        self.log_dict(log_values, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        self.unet.eval()
        with torch.no_grad():
            unet_bottleneck = self.unet(images)

        encodings = self.encoder(unet_bottleneck)
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
        params = list(self.encoder.parameters()) + list(self.fc.parameters())
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val_loss",
                "frequency": 1
            },
        }