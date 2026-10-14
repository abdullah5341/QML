"""A classical baseline with exactly the VQC's function class.

    python run_trig_baseline.py      # about 2 minutes on a laptop CPU

With single angle encoding, every expectation value of the circuit is a
linear combination of the 81 products prod_k {1, cos x_k, sin x_k} over the
four angles: these are the Pauli expectations <P> of the encoded product
state for the 81 Pauli strings without Y (qmlbc/pauli_paths.py). The VQC is
therefore a logistic regression on these 81 features whose coefficients are
restricted to what a 3-layer circuit can reach. Logistic regression on the
same 81 features, unrestricted, isolates what the circuit's restriction adds
or costs.

Uses the same 15 cross-validation partitions as run_fair_comparison.py and
compares fold by fold with the VQC results saved there.
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from qmlbc import config, data, stats

N_SPLITS, N_REPEATS, INNER_FOLDS = 5, 3, 3
NAME = "Logistic regression, 81 trigonometric features"
VQC_NAME = "VQC (4 qubits, 38 parameters)"


def trigonometric_features(angles):
    """All 81 products of {1, cos, sin} across the four angles."""
    angles = np.asarray(angles)
    basis = [np.stack([np.ones_like(a), np.cos(a), np.sin(a)], axis=1) for a in angles.T]
    return np.stack([np.prod([basis[q][:, c] for q, c in enumerate(combo)], axis=0)
                     for combo in itertools.product(range(3), repeat=angles.shape[1])], axis=1)


def model(seed):
    pipe = Pipeline(data.quantum_steps(config.N_QUBITS) + [
        ("trig", FunctionTransformer(trigonometric_features)),
        ("std", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10000)),
    ])
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=inner, scoring="roc_auc")


def main():
    X, y = data.load_dataset()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=config.SEED)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        m = model(config.SEED + fold).fit(X.iloc[tr], y.iloc[tr])
        yt = np.asarray(y.iloc[te])
        p = m.predict_proba(X.iloc[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"fold": fold, "model": NAME, "AUC": roc_auc_score(yt, p),
                     "Accuracy": accuracy_score(yt, pred),
                     "Sensitivity": np.mean(pred[yt == 1] == 1),
                     "Specificity": np.mean(pred[yt == 0] == 0),
                     "Sensitivity at 95% specificity": stats.sensitivity_at_specificity(yt, p, 0.95),
                     "Brier": brier_score_loss(yt, p), "C": m.best_params_["clf__C"]})
    trig = pd.DataFrame(rows)

    folds = pd.read_csv(config.RESULTS_DIR / "fair_comparison_folds.csv")
    vqc = folds[folds["model"] == VQC_NAME].set_index("fold")
    lr4 = folds[folds["model"] == "Logistic regression"].set_index("fold")
    t = trig.set_index("fold")
    out = {}
    for metric in ["AUC", "Accuracy", "Sensitivity at 95% specificity"]:
        for other, label in [(t, "VQC minus trigonometric LR"), (lr4, "VQC minus linear LR")]:
            mean, lo, hi, p = stats.corrected_ttest(vqc[metric] - other[metric], 1 / (N_SPLITS - 1))
            out[f"{metric}: {label}"] = {"mean": mean, "ci": [lo, hi], "p": p}
    metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity",
               "Sensitivity at 95% specificity", "Brier"]
    summary = pd.concat([trig, folds[folds["model"].isin([VQC_NAME, "Logistic regression"])]]) \
        .groupby("model")[metrics].agg(["mean", "std"])
    trig.to_csv(config.RESULTS_DIR / "trig_baseline_folds.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "trig_baseline_summary.csv")
    (config.RESULTS_DIR / "trig_baseline.json").write_text(json.dumps(
        {"tests": out, "chosen_C": trig["C"].value_counts().to_dict()}, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string())
    print(json.dumps(out, indent=2, default=float))
    print("chosen C:", trig["C"].value_counts().to_dict())


if __name__ == "__main__":
    main()
