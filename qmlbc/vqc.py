"""Variational Quantum Classifier (hybrid quantum-classical), scikit-learn compatible."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin

from . import config


def encode(x, n_qubits):
    """Angle encoding: feature i becomes an RY rotation on qubit i.

    |psi(x)> = RY(x_1)|0> (x) ... (x) RY(x_n)|0>, a product state.
    `x` may carry a leading batch dimension (parameter broadcasting).
    """
    for i in range(n_qubits):
        qml.RY(x[..., i], wires=i)


def variational_layer(layer_weights, n_qubits):
    """RX-RY-RZ on every qubit, then a ring of CNOTs to entangle neighbours."""
    for i in range(n_qubits):
        qml.RX(layer_weights[i, 0], wires=i)
        qml.RY(layer_weights[i, 1], wires=i)
        qml.RZ(layer_weights[i, 2], wires=i)
    if n_qubits > 1:
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


def build_circuit(n_qubits, reupload=False):
    """QNode returning <Z_0>, the expectation value read out as the class score."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, x):
        if not reupload:
            encode(x, n_qubits)
        for layer_weights in weights:
            if reupload:
                encode(x, n_qubits)
            variational_layer(layer_weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


class VQC(ClassifierMixin, BaseEstimator):
    """Variational quantum classifier with a classical logistic read-out.

    The quantum circuit produces f(x) = <Z_0> in [-1, 1]. A classical affine map
    and sigmoid turn it into P(malignant | x) = sigmoid(a * f(x) + b), and the
    circuit weights together with (a, b) are trained by Adam on the binary
    cross-entropy. Inputs must already be rotation angles (see data.quantum_steps).
    """

    def __init__(self, n_qubits=config.N_QUBITS, n_layers=config.N_LAYERS,
                 reupload=False, epochs=config.VQC_EPOCHS,
                 batch_size=config.VQC_BATCH_SIZE, lr=config.VQC_LR, seed=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def _circuit(self):
        if not hasattr(self, "circuit_"):
            self.circuit_ = build_circuit(self.n_qubits, self.reupload)
        return self.circuit_

    def __getstate__(self):
        # The QNode is not picklable; it is rebuilt on first use after loading.
        state = super().__getstate__().copy()
        state.pop("circuit_", None)
        return state

    def _logits(self, weights, scale, bias, X):
        return scale * self._circuit()(weights, X) + bias

    def _loss(self, weights, scale, bias, X, y):
        p = 1.0 / (1.0 + pnp.exp(-self._logits(weights, scale, bias, X)))
        p = pnp.clip(p, 1e-7, 1 - 1e-7)
        return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))

    def fit(self, X, y):
        X = pnp.array(np.asarray(X, dtype=float), requires_grad=False)
        y = pnp.array(np.asarray(y, dtype=float), requires_grad=False)
        self.classes_ = np.array([0, 1])

        rng = np.random.default_rng(self.seed)
        weights = pnp.array(
            rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )
        scale = pnp.array(1.0, requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.loss_history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = order[start:start + self.batch_size]
                weights, scale, bias, _, _ = opt.step(
                    self._loss, weights, scale, bias, X[idx], y[idx]
                )
            self.loss_history_.append(float(self._loss(weights, scale, bias, X, y)))

        self.weights_ = np.array(weights)
        self.scale_, self.bias_ = float(scale), float(bias)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(self._logits(self.weights_, self.scale_, self.bias_, X))

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    @property
    def n_parameters_(self):
        return self.n_layers * self.n_qubits * 3 + 2
