"""Statistical helpers: intervals and tests for comparing classifiers."""
import numpy as np
from scipy import stats as st
from sklearn.metrics import roc_curve


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k / n."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def corrected_ttest(diff, test_train_ratio):
    """Corrected resampled t-test for repeated k-fold CV (Nadeau and Bengio, 2003).

    Folds share training data, so the naive variance of the per-fold differences
    is too small; it is inflated by test_train_ratio (1 / (k - 1) for k folds).
    Returns the mean difference, its 95% interval and the two-sided p-value.
    """
    diff = np.asarray(diff, dtype=float)
    J, mean = len(diff), diff.mean()
    se = np.sqrt((1 / J + test_train_ratio) * diff.var(ddof=1))
    if se == 0:
        return mean, mean, mean, 1.0 if mean == 0 else 0.0
    half = st.t.ppf(0.975, J - 1) * se
    p = 2 * st.t.sf(abs(mean / se), J - 1)
    return mean, mean - half, mean + half, p


def holm(pvalues):
    """Holm-Bonferroni adjustment for a family of tests."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired predictions. Returns (only A right, only B right, p)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = st.binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def bootstrap_ci(statistic, y, *arrays, n_boot=2000, seed=0):
    """Percentile interval of statistic(y, *arrays) over stratified bootstrap resamples."""
    y = np.asarray(y)
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(statistic(y[idx], *[a[idx] for a in arrays]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def expected_calibration_error(y, p, bins=10):
    """Mean |observed rate - predicted probability| over equal-width bins, weighted by size."""
    y, p = np.asarray(y), np.asarray(p)
    which = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(np.mean(which == b) * abs(y[which == b].mean() - p[which == b].mean())
                     for b in range(bins) if np.any(which == b)))


def sensitivity_at_specificity(y, score, specificity):
    """Highest sensitivity reachable while keeping specificity at or above the target."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= 1 - specificity + 1e-12].max())
