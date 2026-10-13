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


def integrated_gradients(pipeline, X, baseline, steps=64, members=None):
    """Attribute the ensemble's logit to the original features (Sundararajan et al., 2017).

    The gradient of the logit is integrated along the straight line from
    `baseline` (the average training patient) to each patient x. The attributions
    are in logit units and add up to logit(x) - logit(baseline), so every unit
    of the prediction is accounted for. Positive = pushes towards malignant.

    Because the features reach the circuit only through a linear map to four
    angles, the result has the form delta_x * (g @ jacobian): for each patient
    the 30 attributions are fixed by four numbers g, one per qubit.

    `pipeline` is StandardScaler -> PCA -> MinMaxScaler(clip) -> VQCEnsemble.
    `members` restricts the explanation to a subset of the ensemble.
    """
    delta, g, jacobian = _path_gradients(pipeline, X, baseline, steps, members)
    return delta * (g @ jacobian)


def angle_attributions(pipeline, X, baseline, steps=64, members=None):
    """Integrated gradients at the level of the four qubit angles (principal components).

    The same explanation grouped by the component each feature acts through:
    sum_k (delta_x @ jacobian.T)_k * g_k equals the sum of the feature attributions.
    """
    delta, g, jacobian = _path_gradients(pipeline, X, baseline, steps, members)
    return (delta @ jacobian.T) * g


def _path_gradients(pipeline, X, baseline, steps, members):
    """Path-averaged d(logit)/d(angle) on the straight line from baseline to each patient.

    Returns delta_x = X - baseline, g = the average angle gradient along the path
    with clipped angles set to zero (n x qubits), and jacobian = d(angle)/d(x) of
    the linear preprocessing (qubits x features).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    baseline = np.asarray(baseline, dtype=float)
    scaler, pca, angle = pipeline["scale"], pipeline["pca"], pipeline["angle"]
    members = pipeline[-1].members_ if members is None else members
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
    g = (grad_angle * inside).reshape(steps, X.shape[0], -1).mean(axis=0)
    return delta, g, jacobian


def linear_attributions(pipeline, X, baseline):
    """Exact attributions of a StandardScaler -> LogisticRegression pipeline.

    For a linear logit they are simply coefficient x (deviation from baseline),
    which is also what integrated gradients returns for a linear model.
    """
    scaler, clf = pipeline["scale"], pipeline["clf"]
    return (np.asarray(X, dtype=float) - np.asarray(baseline)) / scaler.scale_ * clf.coef_[0]


def quantity_of(feature):
    """'worst concave points' -> 'concave points'."""
    return feature.replace("mean ", "").replace("worst ", "").replace(" error", "")


def by_quantity(values, feature_names):
    """Sum per-feature values over the mean / error / worst variants of each quantity."""
    values = pd.Series(np.asarray(values), index=[quantity_of(f) for f in feature_names])
    return values.groupby(level=0).sum()[BASE_QUANTITIES]
