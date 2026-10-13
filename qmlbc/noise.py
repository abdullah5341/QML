"""Hardware-noise study: gate errors (depolarising) and finite measurement shots."""
from functools import lru_cache

import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression

# Single-qubit gates on current devices are roughly ten times more accurate
# than two-qubit gates, so one knob (the CNOT error) sets both.
SINGLE_TO_TWO_QUBIT_ERROR = 0.1


@lru_cache(maxsize=None)
def build_noisy_circuit(n_qubits):
    """The VQC of vqc.build_circuit with a depolarising channel after every gate.

    A depolarising channel of strength p replaces the qubit's state by the
    maximally mixed state with probability p. Simulated exactly with density
    matrices, so the returned <Z_0> is the infinite-shot value on a noisy device.
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit(weights, x, p_two):
        p_one = SINGLE_TO_TWO_QUBIT_ERROR * p_two

        def one_qubit(gate, angle, wire):
            gate(angle, wires=wire)
            qml.DepolarizingChannel(p_one, wires=wire)

        for i in range(n_qubits):
            one_qubit(qml.RY, x[..., i], i)
        for layer_weights in weights:
            for i in range(n_qubits):
                one_qubit(qml.RX, layer_weights[i, 0], i)
                one_qubit(qml.RY, layer_weights[i, 1], i)
                one_qubit(qml.RZ, layer_weights[i, 2], i)
            for i in range(n_qubits):
                j = (i + 1) % n_qubits
                qml.CNOT(wires=[i, j])
                qml.DepolarizingChannel(p_two, wires=i)
                qml.DepolarizingChannel(p_two, wires=j)
        return qml.expval(qml.PauliZ(0))

    return circuit


def noisy_expectation(vqc, angles, p_two):
    """<Z_0> of a fitted VQC for each row of `angles` at CNOT error rate `p_two`."""
    circuit = build_noisy_circuit(vqc.n_qubits)
    return np.asarray(circuit(vqc.weights_, np.atleast_2d(angles), p_two), dtype=float)


def sample_shots(expectation, shots, rng):
    """Estimate <Z_0> from `shots` measurements: each returns +1 with prob (1 + <Z_0>) / 2."""
    p_plus = np.clip((1 + np.asarray(expectation)) / 2, 0, 1)
    return 2 * rng.binomial(shots, p_plus) / shots - 1


def recalibrate_readout(z_train, y_train):
    """Refit the two classical read-out parameters (a, b) on noisy training outputs.

    Costs no extra quantum training: the circuit weights stay as they are and
    only P(malignant) = sigmoid(a * <Z_0> + b) is re-estimated.
    """
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.reshape(z_train, (-1, 1)), y_train)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def classify(z, scale, bias):
    return (scale * np.asarray(z) + bias > 0).astype(int)


ERROR_MODELS = ("depolarizing", "amplitude damping", "readout", "coherent over-rotation")


@lru_cache(maxsize=None)
def build_error_model_circuit(n_qubits, kind):
    """The VQC under one of four error models, all controlled by one strength s.

    depolarizing             unital gate noise, as build_noisy_circuit
    amplitude damping        non-unital gate noise (energy relaxation towards |0>),
                             strength s after each CNOT qubit and s/10 after 1-qubit gates
    readout                  the measured qubit is flipped with probability s
    coherent over-rotation   every trained rotation angle is applied as (1 + s) x angle,
                             a systematic calibration error that leaves the state pure

    The first three are channels that act on <Z_0> close to an affine map, the
    last changes the function the circuit computes.
    """
    if kind not in ERROR_MODELS:
        raise ValueError(kind)
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit(weights, x, s):
        def gate_noise(wires, strength):
            for w in wires:
                if kind == "depolarizing":
                    qml.DepolarizingChannel(strength, wires=w)
                elif kind == "amplitude damping":
                    qml.AmplitudeDamping(strength, wires=w)

        over = 1 + s if kind == "coherent over-rotation" else 1.0
        p_one = SINGLE_TO_TWO_QUBIT_ERROR * s
        for i in range(n_qubits):
            qml.RY(x[..., i], wires=i)
            gate_noise([i], p_one)
        for layer_weights in weights:
            for i in range(n_qubits):
                for gate, k in ((qml.RX, 0), (qml.RY, 1), (qml.RZ, 2)):
                    gate(over * layer_weights[i, k], wires=i)
                    gate_noise([i], p_one)
            for i in range(n_qubits):
                j = (i + 1) % n_qubits
                qml.CNOT(wires=[i, j])
                gate_noise([i, j], s)
        if kind == "readout":
            qml.BitFlip(s, wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


def error_model_expectation(vqc, angles, kind, s):
    circuit = build_error_model_circuit(vqc.n_qubits, kind)
    return np.asarray(circuit(vqc.weights_, np.atleast_2d(angles), s), dtype=float)
