import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from thesisproject.utils import (get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitFullLSTM(pl.LightningModule):
    def __init__(self, unet, encoder, lstm):
        super().__init__()
        self.unet = unet
        self.unet.encode = True
        self.encoder = encoder
        self.lstm = lstm

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4

    def forward(self, x):
        return self.lstm(x)

    def training_step(self, batch, _batch_idx):
        """
        Batches are B x K x 1 x H x W
        """
        inputs, labels = batch[0], batch[1]

        encoded_inputs = []
        for series in inputs:
            unet_series = self.unet(series)
            encoding_series = self.encoder(unet_series)
            encoded_inputs.append(encoding_series)

        encoded_inputs = torch.stack(encoded_inputs, dim=0).squeeze()

        if encoded_inputs.ndim == 2:
            encoded_inputs = encoded_inputs.unsqueeze(1)

        outputs = self.lstm(encoded_inputs)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels = batch[0], batch[1]

        encoded_inputs = []
        for series in inputs:
            unet_series = self.unet(series)
            encoding_series = self.encoder(unet_series)
            encoded_inputs.append(encoding_series)

        encoded_inputs = torch.stack(encoded_inputs, dim=0).squeeze()

        if encoded_inputs.ndim == 2:
            encoded_inputs = encoded_inputs.unsqueeze(1)

        outputs = self.lstm(encoded_inputs)

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
        params = list(self.unet.parameters()) + list(self.encoder.parameters()) + list(self.lstm.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val_loss",
                "frequency": 1
            },
        }