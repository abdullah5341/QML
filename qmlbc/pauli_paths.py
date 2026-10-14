"""Pauli-path decomposition of the VQC output under local depolarising noise.

The measured observable Z_0 is propagated backwards (Heisenberg picture)
through the trained circuit, keeping track of how many noisy locations each
Pauli path passes with a non-identity Pauli on the affected qubit. A
depolarising channel of strength p multiplies every non-identity Pauli on its
qubit by q = 1 - 4p/3 and leaves the identity alone, so the noisy output is
exactly

    f_p(x) = sum_{k, m} q2(p)^k q1(p)^m g_{k,m}(x),

where k counts the two-qubit (CNOT) noise locations and m the single-qubit
ones a path is damped at, and g_{k,m} is the part of the noiseless output
carried by such paths. The "hit-count spectrum" g_{k,m} is computed once from
the noiseless circuit; every noise level then follows without simulation.
"""
import itertools
from functools import lru_cache

import numpy as np
import pennylane as qml

from .noise import SINGLE_TO_TWO_QUBIT_ERROR

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@lru_cache(maxsize=None)
def pauli_basis(n):
    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=n)]
    mats = []
    for label in labels:
        m = np.array([[1.0 + 0j]])
        for c in label:
            m = np.kron(m, _PAULI[c])
        mats.append(m)
    return labels, np.array(mats)


def heisenberg_matrix(U, n):
    """M with M[i, j] = Tr(P_i U^dag P_j U) / 2^n: coefficients of U^dag O U are M @ o."""
    _, P = pauli_basis(n)
    conj = np.einsum("ab,jbc,cd->jad", U.conj().T, P, U)
    return np.einsum("iab,jba->ij", P, conj).real / 2 ** n


def _forward_operations(weights, n):
    """The trained part of the circuit, gate by gate, with its noise locations."""
    ops = [("n1", i) for i in range(n)]               # noise after each encoding RY
    for layer in weights:
        for i in range(n):
            for gate, k in ((qml.RX, 0), (qml.RY, 1), (qml.RZ, 2)):
                ops.append(("u", qml.matrix(gate(layer[i, k], wires=i), wire_order=range(n))))
                ops.append(("n1", i))
        for i in range(n):
            j = (i + 1) % n
            ops.append(("u", qml.matrix(qml.CNOT(wires=[i, j]), wire_order=range(n))))
            ops.append(("n2", i))
            ops.append(("n2", j))
    return ops


def hit_coefficients(weights, n):
    """o[P, k, m]: coefficient of Pauli P on the encoded state from paths hit k and m times."""
    labels, _ = pauli_basis(n)
    ops = _forward_operations(np.asarray(weights), n)
    K = sum(op == "n2" for op, _ in ops)
    M = sum(op == "n1" for op, _ in ops)
    non_identity = np.array([[c != "I" for c in label] for label in labels])
    o = np.zeros((len(labels), K + 1, M + 1))
    o[labels.index("Z" + "I" * (n - 1)), 0, 0] = 1.0
    for kind, arg in reversed(ops):
        if kind == "u":
            o = np.einsum("ij,jkm->ikm", heisenberg_matrix(arg, n), o)
        else:
            hit = non_identity[:, arg]
            shifted = np.zeros_like(o[hit])
            if kind == "n2":
                shifted[:, 1:, :] = o[hit][:, :-1, :]
            else:
                shifted[:, :, 1:] = o[hit][:, :, :-1]
            o[hit] = shifted
    return o


def encoded_expectations(angles, n):
    """<P> on RY(angle)|0> product states: I -> 1, X -> sin, Y -> 0, Z -> cos."""
    labels, _ = pauli_basis(n)
    angles = np.atleast_2d(angles)
    single = {"I": np.ones_like(angles), "X": np.sin(angles),
              "Y": np.zeros_like(angles), "Z": np.cos(angles)}
    return np.stack([np.prod([single[c][:, w] for w, c in enumerate(label)], axis=0)
                     for label in labels], axis=1)


def hit_spectrum(weights, angles, n):
    """g[x, k, m]: the noiseless output of each patient split by hit counts."""
    o = hit_coefficients(weights, n)
    E = encoded_expectations(angles, n)
    return (E @ o.reshape(o.shape[0], -1)).reshape(len(E), *o.shape[1:])


def damping(p):
    """(q2, q1): Pauli damping at CNOT and single-qubit noise locations for CNOT error p."""
    return 1 - 4 * p / 3, 1 - 4 * SINGLE_TO_TWO_QUBIT_ERROR * p / 3


def noisy_output(g, p):
    q2, q1 = damping(p)
    k = np.arange(g.shape[1])[:, None]
    m = np.arange(g.shape[2])[None, :]
    return np.tensordot(g, q2 ** k * q1 ** m, axes=([1, 2], [0, 1]))


def structural_summary(g):
    """p-independent quantities that predict the noisy behaviour.

    w[k, m]  projection of each sector onto the clean output (sums to 1)
    k_eff    mean number of damping events, sum w (k + m/10): lambda ~ (1 - 4p/3)^k_eff
    D        dispersion of the part of the first-order damping that is not
             proportional to f: 1 - R^2 ~ (4p/3)^2 D for small p
    """
    f = g.sum(axis=(1, 2))
    fc = f - f.mean()
    var = fc @ fc / len(f)
    gc = g - g.mean(axis=0)
    w = np.einsum("x,xkm->km", fc, gc) / len(f) / var
    k = np.arange(g.shape[1])[:, None]
    m = np.arange(g.shape[2])[None, :]
    events = k + SINGLE_TO_TWO_QUBIT_ERROR * m
    k_eff = float((w * events).sum())
    # First-order damped part, minus its projection on f.
    h = np.tensordot(gc, events, axes=([1, 2], [0, 1]))
    residual = h - (h @ fc / len(f) / var) * fc
    D = float(residual @ residual / len(f) / var)
    return {"w": w, "k_eff": k_eff, "D": D,
            "negative_weight": float(-w[w < 0].sum())}


def affine_fit(f_clean, f_noisy):
    lam, c = np.polyfit(f_clean, f_noisy, 1)
    resid = f_noisy - (lam * f_clean + c)
    return lam, c, 1 - resid.var() / f_noisy.var()


def critical_noise(g, r2_target=0.99, grid=np.linspace(1e-4, 0.3, 600)):
    """Smallest CNOT error at which the exact R^2 falls below the target."""
    f = g.sum(axis=(1, 2))
    for p in grid:
        if affine_fit(f, noisy_output(g, p))[2] < r2_target:
            return float(p)
    return float("nan")
