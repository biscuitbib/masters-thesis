"""
U-Net model to be used in Multiplaner U-Net.
Reworked from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .unet_blocks import *
from thesisproject.utils import create_overlay_figure, get_multiclass_metrics

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, image_size, encoding_size=1000, class_names=None, encode=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.class_names = class_names
        self.encoding_size = encoding_size
        self.image_size = image_size # Assumes square images
        self.encode = encode

        # Hard coded increase of filters by a factor of sqrt(2) for each layer
        self.inc = Double_Conv(n_channels, 90)
        self.down1 = Down(90, 181)
        self.down2 = Down(181, 362)
        self.down3 = Down(362, 724)
        self.down4 = Down(724, 1448)

        self.encoder = Encoder(
            1448,
            self._calculate_fc_in(),
            encoding_size
        )

        self.up1 = Up(1448, 724)
        self.up2 = Up(724, 362)
        self.up3 = Up(362, 181)
        self.up4 = Up(181, 90)
        self.outc = Out_Conv(90, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.encode:
            logits = self.encoder(x5)
            return logits

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _calculate_fc_in(self):
        bottleneck_imsize = self.image_size // 16 # 4 max_pool
        fc_imsize = bottleneck_imsize // 4 # 2 max_pool
        fc_feature_channels = 1448 // 4
        fc_in = fc_imsize * fc_feature_channels
        return fc_in

class LitUNet(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset=None, test_dataset=None, *unet_args, **unet_kwargs):
        super().__init__()
        self.unet = UNet(*unet_args, **unet_kwargs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 8
        self.learning_rate = 5e-5

    def training_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        outputs = self.unet(images)
        loss = self.criterion(outputs, labels)

        self.log("train loss", loss.item())
        return loss

    def validation_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        outputs = self.unet(images)
        loss = self.criterion(outputs, labels)

        # log metrics
        metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

        accuracy += np.mean(metrics["accuracy"])
        precision += np.mean(metrics["precision"])
        recall += np.mean(metrics["recall"])
        specificity += np.mean(metrics["specificity"])
        mean_dice += np.mean(metrics["dice"])
        per_class_dice += metrics["dice"]

        log_values = {
            "val loss": loss.item,
            "accuracy": np.mean(metrics["accuracy"]),
            "precision": np.mean(metrics["precision"]),
            "recall": np.mean(metrics["recall"]),
            "specificity": np.mean(metrics["specificity"]),
            "dice": np.mean(metrics["dice"])
        }
        self.log_dict(log_values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=2)
        return [optimizer], [lr_scheduler]