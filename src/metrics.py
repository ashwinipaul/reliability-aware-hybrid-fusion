import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    binary = probs.shape[1] == 2
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary" if binary else "macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary" if binary else "macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary" if binary else "macro", zero_division=0),
    }
    try:
        result["auc"] = roc_auc_score(y_true, probs[:, 1]) if binary else roc_auc_score(y_true, probs, multi_class="ovr")
    except ValueError:
        result["auc"] = float("nan")
    return result
