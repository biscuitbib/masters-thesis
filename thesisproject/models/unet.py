"""
U-Net model to be used in Multiplaner U-Net.
Reworked from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import torch
from torch import nn, optim
import pytorch_lightning as pl

from .unet_blocks import *
from thesisproject.utils import get_metrics, mask_to_rgb, segmentation_to_rgb, grayscale_to_rgb

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Hard coded increase of filters by a factor of sqrt(2) for each layer
        self.inc = Double_Conv(n_channels, 90)
        self.down1 = Down(90, 181)
        self.down2 = Down(181, 362)
        self.down3 = Down(362, 724)
        self.down4 = Down(724, 1448)

        self.up1 = Up(1448, 724)
        self.up2 = Up(724, 362)
        self.up3 = Up(362, 181)
        self.up4 = Up(181, 90)
        self.outc = Out_Conv(90, n_classes)

    def forward(self, x, encode=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if encode:
            return x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class LitUNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.unet = UNet(n_channels, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        inputs, labels = batch[0],batch[1]

        outputs = self.unet(inputs)
        loss = self.criterion(outputs, labels)

        # Average batch loss
        batch_samples = inputs.shape[0]
        current_loss = loss.item() / batch_samples
        self.log("loss/training", current_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]

        outputs = self.unet(inputs)

        loss = self.criterion(outputs, labels)

        # save statistics
        batch_samples = inputs.shape[0]
        current_loss = loss.item() / batch_samples

        metrics = get_metrics(outputs, labels, remove_bg=True)

        # Create image for first batch
        if batch_idx == 0:
            input_imgs = inputs.cpu() * 255
            input_imgs /= torch.max(input_imgs)
            input_imgs = grayscale_to_rgb(input_imgs)
            pred_imgs = segmentation_to_rgb(outputs.cpu())
            target_imgs = mask_to_rgb(labels.cpu())
            imgs = torch.cat((input_imgs, target_imgs, pred_imgs), dim=2)
            self.log("sample validation images", imgs[:4, ...])

        self.log_dict({"loss/validation": current_loss, **metrics})

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.unet.parameters(), lr=5e-5)
        return optimizer
