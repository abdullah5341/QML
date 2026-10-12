"""Quantum-kernel support vector machine (QSVM), scikit-learn compatible."""
from itertools import combinations

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from . import config


def build_feature_map(n_qubits, reps):
    """IQP / ZZ feature map (Havlicek et al., 2019) returning the state vector.

    Each repetition applies H on every qubit, RZ(x_i) on qubit i and
    exp(-i x_i x_j Z_i Z_j / 2) on every pair, so the embedding entangles the
    features and is conjectured hard to simulate classically at scale.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=None, diff_method=None)
    def feature_map(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[..., i], wires=i)
            for i, j in combinations(range(n_qubits), 2):
                qml.IsingZZ(x[..., i] * x[..., j], wires=[i, j])
        return qml.state()

    return feature_map


class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVM on the fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2.

    On hardware each kernel entry is estimated by running phi(x')^dagger phi(x)
    and counting all-zero outcomes. On a noiseless simulator the same number is
    obtained exactly from the state vectors, which is what is done here.
    `bandwidth` rescales the inputs before embedding; it controls how fast the
    kernel decays and guards against the kernel concentrating towards identity.
    """

    def __init__(self, n_qubits=config.N_QUBITS, reps=2, bandwidth=1.0, C=1.0,
                 seed=config.SEED):
        self.n_qubits = n_qubits
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C
        self.seed = seed

    def __getstate__(self):
        # The QNode is not picklable; it is rebuilt on first use after loading.
        state = super().__getstate__().copy()
        state.pop("feature_map_", None)
        return state

    def _states(self, X):
        if not hasattr(self, "feature_map_"):
            self.feature_map_ = build_feature_map(self.n_qubits, self.reps)
        X = np.atleast_2d(np.asarray(X, dtype=float)) * self.bandwidth
        return np.asarray(self.feature_map_(X))

    def _kernel(self, states):
        return np.abs(states @ self.train_states_.conj().T) ** 2

    def fit(self, X, y):
        self.train_states_ = self._states(X)
        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self.svc_.fit(self._kernel(self.train_states_), y)
        self.classes_ = self.svc_.classes_
        return self

    def kernel_matrix(self, X):
        return self._kernel(self._states(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self.kernel_matrix(X))

    def predict(self, X):
        return self.svc_.predict(self.kernel_matrix(X))
