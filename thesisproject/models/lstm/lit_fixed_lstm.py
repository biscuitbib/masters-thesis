import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from thesisproject.utils import (get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitFixedLSTM(pl.LightningModule):
    def __init__(self, unet, encoder, lstm, lr=1e-5, weight_decay=1e-4):
        super().__init__()
        self.unet = unet
        self.unet.encode = True
        self.encoder = encoder
        self.lstm = lstm

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.lstm(x)

    def training_step(self, batch, _batch_idx):
        """
        Batches are B x K x 1 x H x W
        """
        inputs, labels, timedeltas = batch[0], batch[1], batch[2]

        self.unet.eval()
        self.encoder.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series)

        encoded_inputs = torch.stack(encoded_inputs, dim=0)
        encoded_inputs_dt = torch.cat([timedeltas.unsqueeze(-1), encoded_inputs], dim=2)

        outputs = self.lstm(encoded_inputs_dt)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels, timedeltas = batch[0], batch[1], batch[2]

        self.unet.eval()
        self.encoder.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series)

        encoded_inputs = torch.stack(encoded_inputs, dim=0)
        encoded_inputs_dt = torch.cat([timedeltas.unsqueeze(-1), encoded_inputs], dim=2)

        outputs = self.lstm(encoded_inputs_dt)

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

    def predict(self, data):
        inputs, timedeltas = data[0], data[2]
        self.unet.eval()
        self.encoder.eval()
        self.lstm.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series)

            encoded_inputs = torch.stack(encoded_inputs, dim=0)
            encoded_inputs_dt = torch.cat([timedeltas.unsqueeze(-1), encoded_inputs], dim=2)

            outputs = self.lstm(encoded_inputs_dt)
            return softmax(outputs, dim=1)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.lstm.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/train",
                "frequency": 1
            },
        }