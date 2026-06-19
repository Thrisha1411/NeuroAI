from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculates Accuracy, Precision, Recall, F1, and Confusion Matrix.
    
    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        average (str): Averaging method for multiclass ('micro', 'macro', 'weighted').
        
    Returns:
        metrics (dict): Dictionary containing computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }
