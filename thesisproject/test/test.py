import torch
import pandas as pd
from thesisproject.predict import predict_volume
from thesisproject.utils import get_multiclass_metrics

def test_loop(net, test_loader):
    
    metrics = pd.DataFrame()
    with torch.no_grad():
        for data in test_loader:
            image, label = data[0].squeeze(), data[1]

            prediction = predict_volume(net, image)

            metrics = get_multiclass_metrics(
                prediction.unsqueeze(0).detach().cpu(), 
                label.detach().cpu(), 
                net.n_classes, 
                remove_bg=True
            )
            print(metrics)
            exit()