import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from thesisproject.predict import predict_volume
from thesisproject.utils import get_multiclass_metrics

def test_loop(net, test_loader, n_classes):
    
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
        for data in test_loader:
            image, label = data[0], data[1]

            prediction = predict_volume(net, image)

            metrics = get_multiclass_metrics(
                prediction.unsqueeze(0).detach().cpu(), 
                label.detach().cpu(), 
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