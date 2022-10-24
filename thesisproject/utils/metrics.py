import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

def get_multiclass_metrics(pred, target, remove_bg=False):
    """
    Takes pred = (batch X n classes X height X width X depth) softmax prediction, 
    and target = (batch X 1 X height X width X depth target), 
    and calculates metrics from confusion matrics
    """
    n_classes = pred.shape[1]
    offset = int(remove_bg)

    pred_class_indices = torch.argmax(pred, dim=1)
    
    pred_flat = torch.flatten(pred_class_indices).numpy()
    target_flat = torch.flatten(target).numpy()
    
    cm = confusion_matrix(pred_flat, target_flat, labels=np.arange(n_classes))
    if remove_bg:
        cm = cm[1:, 1:]
          
    metrics = {
        "accuracy": np.zeros(n_classes - offset),
        "precision": np.zeros(n_classes - offset),
        "recall": np.zeros(n_classes - offset),
        "specificity": np.zeros(n_classes - offset),
        "dice": np.zeros(n_classes - offset),
    }
    
    sum_cm = np.sum(cm)
    eps = 1e-6
    for c in range(n_classes - offset):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = sum_cm - tp - fp - fn
        
        accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        
        metrics["accuracy"][c] = accuracy
        metrics["precision"][c] = precision
        metrics["recall"][c] = recall
        metrics["specificity"][c] = specificity
        metrics["dice"][c] = dice
        
    return metrics
    
if __name__ == "__main__":
    test_pred = torch.tensor([
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]])
    
    test_target = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 2],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 2]])
        
    pred_flat = torch.flatten(test_pred).numpy()
    target_flat = torch.flatten(test_target).numpy()
    
    cm = confusion_matrix(pred_flat, target_flat, labels=np.arange(3))
    
    print(cm)
    
    sum_cm = np.sum(cm)
    eps = 1e-6
    for c in range(3):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = sum_cm - tp - fp - fn
        
        accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
        print(f"""
class {c}
accuracy = {accuracy}
precision = {precision}
recall = {recall}
specificity = {specificity}
dice = {dice}
        """)