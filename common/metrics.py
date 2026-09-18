import numpy as np
from sklearn.metrics import f1_score

def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))

def prediction_consistency(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    return float((preds_a == preds_b).mean())
