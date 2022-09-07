import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

def prepare_input_and_target(pred, target):
    """
    Takes pred = batch X num classes X height X width prediction, 
    and target = batch X height X width target, and makes them same size
    """
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    
    num_classes = pred.shape[1]
    labels_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    
    max_idx = torch.argmax(pred, 1, keepdim=True).squeeze()
    pred_onehot = F.one_hot(max_idx, num_classes=num_classes).permute(0, 3, 1, 2)
    
    return pred_onehot, labels_onehot
    


def get_metrics(pred, target):
    pred, target = prepare_input_and_target(pred, target)
    
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