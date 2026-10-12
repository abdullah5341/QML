"""Dataset loading, splitting and the classical-to-quantum preprocessing chain."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config

CLASS_NAMES = ["Benign", "Malignant"]


def load_dataset():
    """Return (X, y) with y = 1 for malignant and y = 0 for benign.

    scikit-learn ships the labels the other way round (0 = malignant). They are
    flipped here so that the positive class is the disease: recall is then the
    clinical sensitivity, i.e. the fraction of cancers that are caught.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = (1 - raw.target).rename("malignant")
    return X, y


def split(X, y):
    """Stratified hold-out split; the test set is touched once, at the very end."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )


def reduction_steps(n_components):
    """StandardScaler -> PCA. Shared by the classical and quantum reduced models."""
    return [
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=config.SEED)),
    ]


def quantum_steps(n_qubits):
    """StandardScaler -> PCA -> rescale each component to a rotation angle in [0, pi].

    Used as pipeline steps so every transform is fitted on training folds only.
    Test points falling outside the training range are clipped, which keeps the
    angle encoding injective (RY is 2*pi periodic in general, but one-to-one on
    [0, pi]).
    """
    return reduction_steps(n_qubits) + [
        ("angle", MinMaxScaler(feature_range=(0.0, np.pi), clip=True)),
    ]


def target_correlations(X, y):
    """Point-biserial correlation of each feature with the malignant label."""
    return X.apply(lambda col: np.corrcoef(col, y)[0, 1]).sort_values(
        key=np.abs, ascending=False
    )


def class_counts(y):
    counts = pd.Series(y).value_counts().sort_index()
    return {CLASS_NAMES[i]: int(n) for i, n in counts.items()}
