import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from thesisproject.utils import (get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitFixedLinearModel(pl.LightningModule):
    def __init__(self, unet, encoder, linear, n_visits=1, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.unet = unet
        self.unet.encode = True
        self.encoder = encoder
        self.linear = linear

        self.n_visits = n_visits

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, _batch_idx):
        """
        Batches are B x K x 1 x H x W
        Same as LSTM
        """
        inputs, labels, _timedeltas = batch[0], batch[1], batch[2]

        self.unet.eval()
        self.encoder.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series.squeeze())

        encoded_stacks = torch.stack([series.reshape(-1) for series in encoded_inputs])

        outputs = self.linear(encoded_stacks)

        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs, labels, _timedeltas = batch[0], batch[1], batch[2]

        self.unet.eval()
        self.encoder.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series.squeeze())

        encoded_stacks = torch.stack([series.reshape(-1) for series in encoded_inputs])

        outputs = self.linear(encoded_stacks)

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

    def predict(self, inputs):
        inputs = inputs[0]
        self.unet.eval()
        self.encoder.eval()
        self.linear.eval()
        encoded_inputs = []
        with torch.no_grad():
            for series in inputs:
                unet_series = self.unet(series)
                encoding_series = self.encoder(unet_series)
                encoded_inputs.append(encoding_series)

            encoded_stacks = torch.stack([series.reshape(-1) for series in encoded_inputs])

            outputs = self.linear(encoded_stacks)
            return softmax(outputs, dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.linear.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/train",
                "frequency": 1
            },
        }