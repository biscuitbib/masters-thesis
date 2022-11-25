import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from collections import OrderedDict
from thesisproject.models import UNet
from thesisproject.utils import (create_overlay_figure, get_multiclass_metrics,
                                 save_metrics_csv)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitMPU(pl.LightningModule):
    """
    Pytorch Lightning module representing a multiplanar U-net:
    Training and validation is done on slices of image volumes and predictions are made on entire image volumes.
    """
    def __init__(self, unet: UNet, save_test_preds=False, ):
        super().__init__()
        self.unet = unet
        self.save_test_preds = save_test_preds

        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 8
        self.lr = 5e-5

    def training_step(self, batch, _batch_idx):
        images, labels = batch[0], batch[1]

        outputs = self.unet(images)
        loss = self.criterion(outputs, labels)

        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]

        outputs = self.unet(images)
        loss = self.criterion(outputs, labels)

        if batch_idx == 0:
            overlay_fig, _ = create_overlay_figure(images, labels, outputs, images_per_batch=4)
            self.logger.experiment.add_figure("images/val", overlay_fig, self.current_epoch)

        # log metrics
        metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

        log_dice = {f"dice/{class_name}": dice for class_name, dice in zip(self.unet.class_names, metrics["dice"])}
        log_values = {
            "loss/val_loss": loss.detach(),
            "val/accuracy": np.mean(metrics["accuracy"]),
            "val/precision": np.mean(metrics["precision"]),
            "val/recall": np.mean(metrics["recall"]),
            "val/specificity": np.mean(metrics["specificity"]),
            "val/dice": np.mean(metrics["dice"]),
            **log_dice
        }
        self.log_dict(log_values, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, _batch_idx):
        """
        Testing expects batch of single ImagePair object
        """
        imagepair = batch[0]
        with imagepair.loaded_in_context():
            image, label = imagepair.image, imagepair.label

            image -= torch.min(image)
            image /= torch.max(image)

            # Softmax prediction
            prediction = self._predict_volume(image, class_index=False)

            if self.save_test_preds:
                prediction_nii = nib.Nifti1Image(prediction.squeeze().detach().cpu().numpy(), affine=np.eye(4), dtype=np.uint8)
                if not os.path.exists("predictions"):
                    os.mkdir("predictions")

                nib.save(prediction_nii, os.path.join("predictions", f"{imagepair.identifier}.nii.gz"))

            metrics = get_multiclass_metrics(
                prediction.unsqueeze(0).detach().cpu(), # adding batch dim.
                label.unsqueeze(0).unsqueeze(0).detach().cpu(), # adding batch and channel dim.
                remove_bg=True
            )

            for metric_key, metric_values in metrics.items():
                self.per_class_metrics[metric_key] += metric_values

    def on_test_start(self):
        self.per_class_metrics = {
            "accuracy": np.zeros(self.unet.n_classes - 1),
            "precision": np.zeros(self.unet.n_classes - 1),
            "recall": np.zeros(self.unet.n_classes - 1),
            "specificity": np.zeros(self.unet.n_classes - 1),
            "dice": np.zeros(self.unet.n_classes - 1)
        }

    def on_test_end(self):
        save_metrics_csv(self.per_class_metrics, self.unet.class_names)

    def _predict_view(self, image_volume):
        tmp_vol = None
        i = 0
        for image_batch in torch.split(image_volume, self.batch_size, dim=0):
            image_batch = image_batch.unsqueeze(1).to(self._device)

            prediction = self.unet(image_batch).squeeze(1).detach().cpu()

            """
            ########## SAVE PER VIEW IMAGES
            input_imgs = image_batch.detach().cpu()
            input_imgs /= torch.max(input_imgs)
            input_imgs = grayscale_to_rgb(input_imgs)

            b, h, w, c = input_imgs.shape

            pred_imgs = segmentation_to_rgb(prediction.detach().cpu())
            pred_overlay = (input_imgs / 2) + (pred_imgs / 2)
            pred_overlay = torch.cat([pred_overlay[i, ...] for i in range(pred_overlay.shape[0])], dim=1).numpy()

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(pred_overlay)
            ax.set_title(f"Prediction batch along {view} view")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            fig.savefig(f"predictions/per_slice/{view}_{i}.jpg")

            plt.close()
            i += 1
            ###################
            """

            if tmp_vol is None:
                tmp_vol = prediction
            else:
                tmp_vol = torch.cat([tmp_vol, prediction], dim=0)

        tmp_vol = tmp_vol.permute(1, 0, 2, 3)
        return F.softmax(tmp_vol, dim=0)

    def _predict_volume(self, image_volume, class_index=True):
        """
        Create single prediction volume using prediction volumes along the three standard axes
        """
        prediction_image = torch.zeros((self.unet.n_classes, *image_volume.shape), dtype=image_volume.dtype)

        # First view
        prediction_image += self._predict_view(image_volume)

        # Second view
        rot_img = image_volume.permute(1, 0, 2)
        prediction_image += self._predict_view(rot_img).permute(0, 2, 1, 3)

        # Third view
        rot_img = image_volume.permute(2, 0, 1)
        prediction_image += self._predict_view(rot_img).permute(0, 2, 3, 1)

        if class_index:
            return torch.argmax(prediction_image, dim=0)
        else:
            return prediction_image / 3.

    def predict_step(self, batch, _batch_idx):
        """
        Create prediction volume for the image in batch.
        Expects a batch of a single image_volume
        """
        image_volume = batch[0]
        return self._predict_volume(image_volume)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.unet.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/dice",
                "frequency": 1
            },
        }

    def model_from_checkpoint(self, checkpoint):
        modified_key_vals = [(key.split(".", 1)[1], val) for key, val in checkpoint["state_dict"].items()]
        state_dict = OrderedDict(modified_key_vals)
        unet = self.unet.clone()
        unet.load_state_dict(state_dict)
        return unet
