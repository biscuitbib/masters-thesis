import os
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from tqdm import tqdm

from thesisproject.predict import Predict
from thesisproject.utils import get_multiclass_metrics

def save_metrics_csv(metrics, class_names):
    with open("eval.csv", "w") as f:
        headers = ["metric", *[f"{class_name}" for class_name in class_names], "mean"]
        f.write(",".join(headers) + "\n")
        for metric, values in metrics.items():
            row = [metric, *values, np.mean(values)]
            f.write(",".join([str(v) for v in row]) + "\n")


class Test:
    def __init__(self, net, test_loader, save_preds=False):
        self.net = net
        self.n_classes = self.net.n_classes - 1
        self.test_loader = test_loader
        self.pbar = tqdm(total=len(self.test_loader), unit="images")
        self.save_preds = save_preds
        self.predict = Predict(self.net, batch_size=1, show_progress=False)
        self.n_samples = 0
        self.per_class_metrics = {
            "accuracy": np.zeros(self.n_classes),
            "precision": np.zeros(self.n_classes),
            "recall": np.zeros(self.n_classes),
            "specificity": np.zeros(self.n_classes),
            "dice": np.zeros(self.n_classes)
        }

    def _test_volume(self, imagepair):
        self.pbar.set_description(f"Testing image {imagepair.identifier}")
        with imagepair.loaded_in_context():
            image, label = imagepair.image, imagepair.label

            image -= torch.min(image)
            image /= torch.max(image)

            # Softmax prediction
            prediction = self.predict(image, class_index=False)

            if self.save_preds:
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

            self.n_samples += 1
            self.pbar.update(1)
            return

    def __call__(self):
        with torch.no_grad():
            for i, [imagepair] in enumerate(self.test_loader, 0):
                self._test_volume(imagepair)


            self.pbar.close()
            print("Calculating per class metrics.")
            for key in self.per_class_metrics.keys():
                self.per_class_metrics[key] /= self.n_samples

            save_metrics_csv(self.per_class_metrics, self.net.class_names)