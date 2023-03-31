import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

from thesisproject.utils import (get_multiclass_metrics,
                                 save_metrics_csv)

class LitFullLinearModel(pl.LightningModule):
    def __init__(self, unet, encoder, linear, n_visits=1, lr=5e-5, weight_decay=1e-5):
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

        encoded_inputs = []
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

        encoded_inputs = []
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

    def _predict_view(self, image_volumes):
        with torch.no_grad():
            view_predictions = []
            for image_slice_batch in torch.split(image_volumes, 8, dim=1):
                self.unet.eval()
                self.encoder.eval()
                self.linear.eval()
                encoded_inputs = None
                encoded_series = []
                for image_series in image_slice_batch:
                     # Standardize slice
                    image_series = torch.stack([(image_slice - torch.mean(image_slice)) / torch.std(image_slice) if torch.std(image_slice) > 0 else image_slice for image_slice in image_series], dim=0)

                    unet_series = self.unet(image_series.unsqueeze(1))
                    encoding_series = self.encoder(unet_series)
                    encoded_series.append(encoding_series.squeeze())

                encoded_series = torch.stack(encoded_series).permute(1, 0, 2)
                encoded_series = encoded_series.reshape((encoded_series.shape[0], -1))

                encoded_batch = encoded_series.unsqueeze(0)

                outputs = self.linear(encoded_batch)
                prediction = softmax(outputs, dim=2).squeeze()

                view_predictions.append(prediction[:,1].detach().cpu())
            return torch.cat(view_predictions)

    def _predict_volumes(self, image_volumes):
        # First view
        prediction_view = []
        prediction_view.append(self._predict_view(image_volumes))

        # Second view
        rot_img = torch.stack([image_volume.permute(1, 0, 2) for image_volume in image_volumes])
        prediction_view.append(self._predict_view(rot_img))

        # Third view
        rot_img = torch.stack([image_volume.permute(2, 0, 1) for image_volume in image_volumes])
        prediction_view.append(self._predict_view(rot_img))

        return prediction_view

    def predict(self, image_volumes):
        """
        image_volumes is series of image volumes: K x H x W x D
        """
        volume_prediction = self._predict_volumes(image_volumes.squeeze(1))
        volume_prediction = torch.stack([torch.mean(view_prediction) for view_prediction in volume_prediction], 0)
        return torch.mean(volume_prediction)

    def on_test_start(self):
        self.prediction_list = []
        self.label_list = []

    def on_test_end(self):
        return self.prediction_list, self.label_list

    def test_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        prediction = self.predict(images)

        self.prediction_list.append(prediction)
        self.label_list.append(labels)


    def configure_optimizers(self):
        params = list(self.unet.parameters()) + list(self.encoder.parameters()) + list(self.linear.parameters())
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val_loss",
                "frequency": 1
            },
        }