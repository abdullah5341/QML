"""Evaluation metrics. The positive class is malignant throughout."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]


def malignant_score(model, X):
    """Continuous score, increasing with P(malignant), used for the ROC curve."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate(y_true, y_pred, y_score):
    """Hold-out metrics for one model. `y_score` is P(malignant) or any monotone score."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
