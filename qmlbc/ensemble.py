"""VQC ensemble and sensitivity-targeted threshold calibration."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from . import config
from .vqc import VQC


class VQCEnsemble(ClassifierMixin, BaseEstimator):
    """Several VQCs that differ only in their random initialisation, averaged.

    The members' logits are averaged, so the ensemble is itself a logistic model
    and attributions of the ensemble are the mean of the members' attributions.
    `threshold` is the probability at or above which a patient is called
    malignant; 0.5 is the uncalibrated default.
    """

    def __init__(self, seeds=tuple(config.VQC_SEEDS), n_qubits=config.N_QUBITS,
                 n_layers=config.N_LAYERS, epochs=config.VQC_EPOCHS, threshold=0.5):
        self.seeds = seeds
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.members_ = [
            VQC(n_qubits=self.n_qubits, n_layers=self.n_layers, epochs=self.epochs,
                seed=seed).fit(X, y)
            for seed in self.seeds
        ]
        return self

    def decision_function(self, X):
        return np.mean([m.decision_function(X) for m in self.members_], axis=0)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def threshold_for_sensitivity(y_true, p_malignant, target):
    """Largest threshold whose sensitivity on (y_true, p_malignant) reaches `target`.

    Must be called on predictions for patients the model was not trained on
    (out-of-fold or validation), never on the test set.
    """
    positives = np.sort(np.asarray(p_malignant)[np.asarray(y_true) == 1])
    allowed_misses = int(np.floor((1 - target) * len(positives)))
    return float(positives[allowed_misses])


def sensitivity_specificity(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {
        "Sensitivity": tp / np.sum(y_true == 1),
        "Specificity": tn / np.sum(y_true == 0),
        "Accuracy": (tp + tn) / len(y_true),
        "Missed cancers": int(np.sum((y_true == 1) & (y_pred == 0))),
        "False alarms": int(np.sum((y_true == 0) & (y_pred == 1))),
    }
