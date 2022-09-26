import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import confusion_matrix

def one_hot_pred_and_target(pred, target, num_classes):
    """
    Takes pred = batch X num classes X height X width prediction, 
    and target = batch X height X width target, and makes them same size
    """
    labels_onehot = F.one_hot(target, num_classes=num_classes)
    labels_onehot = labels_onehot.permute(0, 3, 1, 2)
    
    max_idx = torch.argmax(pred, 1, keepdim=True).squeeze()
    pred_onehot = F.one_hot(max_idx, num_classes=num_classes).permute(0, 3, 1, 2)
    
    return pred_onehot, labels_onehot

def one_hot_volumes(pred, target, num_classes):
    """
    Takes pred = (batch X 1 X height X width X depth) prediction, 
    and target = (batch X 1 X height X width X depth target), and makes them same size
    """
    #max_idx = torch.argmax(pred, 1, keepdim=True).squeeze(1)
    #print(np.max(max_idx))
    pred_onehot = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3)

    labels_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3)
    
    return pred_onehot, labels_onehot
    
def calculate_metrics(pred, target):
    tn, fp, fn, tp = confusion_matrix(pred.flatten(), target.flatten()).ravel()
    
    eps = 1e-6
    accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice = (2 * tp) / (2*tp + fp + fn + eps)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice
    }


def get_metrics(pred, target, num_classes, remove_bg=False):
    pred, target = one_hot_pred_and_target(pred, target, num_classes)
    
    if remove_bg:
        pred = pred[:, 1:, ...]
        target = target[:, 1:, ...]
        
    metrics = calculate_metrics(pred, target)
    
    return metrics


def get_multiclass_metrics(pred, target, num_classes, remove_bg=False):
    pred, target = one_hot_volumes(pred, target, num_classes)
    
    if remove_bg:
        pred = pred[:, 1:, ...]
        target = target[:, 1:, ...]
        
    class_metrics = []
    
    offset = 1 if remove_bg else 0
    for c in range(pred.shape[1]):
        metrics = calculate_metrics(pred[:, c, ...], target[:, c, ...])
        class_metrics.append(metrics)
        
    return class_metrics
    