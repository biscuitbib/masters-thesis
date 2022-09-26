import os
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from tqdm import tqdm

from thesisproject.predict import predict_volume
from thesisproject.utils import get_multiclass_metrics

def test_loop(net, test_loader, n_classes, save_preds=False):
    
    per_class_metrics = {
        "accuracy": np.zeros(n_classes),
        "precision": np.zeros(n_classes),
        "recall": np.zeros(n_classes),
        "specificity": np.zeros(n_classes),
        "dice": np.zeros(n_classes)
    }
    
    n_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for i, [imagepair] in enumerate(test_loader, 0):
            pbar.set_description(f"Testing image {imagepair.identifier}")
            with imagepair.loaded_in_context():
                image, label = imagepair.image, imagepair.label

                prediction = predict_volume(net, image)

                if save_preds:
                    prediction_nii = nib.Nifti1Image(prediction.squeeze().detach().cpu().numpy(), affine=np.eye(4), dtype=np.float32)
                    if not os.path.exists("predictions"):
                        os.mkdir("predictions")
                        
                    nib.save(prediction_nii, os.path.join("predictions", f"{imagepair.identifier}.nii.gz"))

                metrics = get_multiclass_metrics(
                    prediction.unsqueeze(0).detach().cpu(), 
                    label.unsqueeze(0).detach().cpu(), 
                    net.n_classes, 
                    remove_bg=True
                )

                for i, class_metric in enumerate(metrics):
                    for metric_key, metric_value in class_metric.items():
                        per_class_metrics[metric_key][i] += metric_value
                n_samples += 1
                pbar.update(1)
            
        pbar.close()
            
        for key in per_class_metrics.keys():
            per_class_metrics[key] /= n_samples
            
        print(per_class_metrics)