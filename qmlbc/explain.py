"""Per-patient explanations: which of the 30 measurements drove a prediction."""
import numpy as np
import pandas as pd

BASE_QUANTITIES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                   "concavity", "concave points", "symmetry", "fractal dimension"]


def _angle_gradient(member, angles):
    """d(logit)/d(angle) of one VQC by the parameter-shift rule, for a batch of angles.

    Each angle enters through a single RY gate, so the derivative of <Z_0> is
    exactly [f(angle + pi/2) - f(angle - pi/2)] / 2: two circuit runs per qubit,
    the same recipe that works on real hardware.
    """
    circuit = member._circuit()
    grad = np.empty_like(angles)
    for k in range(angles.shape[1]):
        shift = np.zeros(angles.shape[1])
        shift[k] = np.pi / 2
        plus = np.asarray(circuit(member.weights_, angles + shift))
        minus = np.asarray(circuit(member.weights_, angles - shift))
        grad[:, k] = member.scale_ * (plus - minus) / 2
    return grad


def integrated_gradients(pipeline, X, baseline, steps=64):
    """Attribute the ensemble's logit to the original features (Sundararajan et al., 2017).

    The gradient of the logit is integrated along the straight line from
    `baseline` (the average training patient) to each patient x. The attributions
    are in logit units and add up to logit(x) - logit(baseline), so every unit
    of the prediction is accounted for. Positive = pushes towards malignant.

    `pipeline` is StandardScaler -> PCA -> MinMaxScaler(clip) -> VQCEnsemble.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    baseline = np.asarray(baseline, dtype=float)
    scaler, pca, angle = pipeline["scale"], pipeline["pca"], pipeline["angle"]
    members = pipeline[-1].members_
    preprocess = pipeline[:-1]

    # d(angle_k)/d(x_j) for the linear part of the preprocessing chain.
    jacobian = (pca.components_ / scaler.scale_) * angle.scale_[:, None]

    ts = (np.arange(steps) + 0.5) / steps
    delta = X - baseline
    path = baseline + ts[:, None, None] * delta                      # (steps, n, 30)
    flat = pd.DataFrame(path.reshape(-1, X.shape[1]), columns=scaler.feature_names_in_)
    angles = preprocess.transform(flat)
    lo, hi = angle.feature_range
    inside = (angles > lo) & (angles < hi)       # clipped angles do not respond to x

    grad_angle = np.mean([_angle_gradient(m, angles) for m in members], axis=0)
    grad_x = (grad_angle * inside) @ jacobian
    return delta * grad_x.reshape(steps, *X.shape).mean(axis=0)


def linear_attributions(pipeline, X, baseline):
    """Exact attributions of a StandardScaler -> LogisticRegression pipeline.

    For a linear logit they are simply coefficient x (deviation from baseline),
    which is also what integrated gradients returns for a linear model.
    """
    scaler, clf = pipeline["scale"], pipeline["clf"]
    return (np.asarray(X, dtype=float) - np.asarray(baseline)) / scaler.scale_ * clf.coef_[0]


def by_quantity(values, feature_names):
    """Sum per-feature values over the mean / error / worst variants of each quantity."""
    def quantity(feature):
        return feature.replace("mean ", "").replace("worst ", "").replace(" error", "")

    values = pd.Series(np.asarray(values), index=[quantity(f) for f in feature_names])
    return values.groupby(level=0).sum()[BASE_QUANTITIES]
