import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'f1': f1, 'confusion_matrix': cm.tolist()}
