"""Fair comparison of quantum and classical models, with uncertainty and tests.

    python run_fair_comparison.py      # about 15 minutes on a laptop CPU

Addresses three weaknesses of the single hold-out comparison in run_experiment.py:

1. One split of 114 patients. Here: 5-fold cross-validation repeated 3 times
   over all 569 patients (15 train/test partitions).
2. Unequal treatment. Here every model sees the same 4 principal components.
   The classical models are tuned by inner cross-validation, the QSVM as before,
   and a small neural network with the same number of parameters as the VQC
   (37 vs 38) is added, trained without tuning like the VQC. The VQC
   architecture was fixed before this experiment, so tuning favours the
   classical side: any gap is an upper bound on the VQC's disadvantage.
3. No uncertainty. Here: corrected resampled t-tests across folds with Holm
   correction, exact McNemar tests on pooled out-of-fold predictions, and
   calibration (Brier score, expected calibration error) next to discrimination.

Part B adds bootstrap and Wilson intervals to the original test-set results.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from qmlbc import config, data, plots, stats
from qmlbc.metrics import malignant_score
from qmlbc.qkernel import QuantumKernelSVC
from qmlbc.vqc import VQC

N_SPLITS, N_REPEATS, INNER_FOLDS = 5, 3, 3
REFERENCE = "Logistic regression, 30 features"
VQC_NAME = "VQC (4 qubits, 38 parameters)"
MLP_NAME = "Neural network (37 parameters)"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidates(seed):
    """Every model, keyed by name. All except the reference see the same 4 components."""
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)

    def pca(estimator):
        return Pipeline(data.reduction_steps(config.N_QUBITS) + [("clf", estimator)])

    def tuned(estimator, grid):
        return GridSearchCV(pca(estimator), {f"clf__{k}": v for k, v in grid.items()},
                            cv=inner, scoring="roc_auc")

    quantum = data.quantum_steps(config.N_QUBITS)
    return {
        "Logistic regression": tuned(LogisticRegression(max_iter=5000),
                                     {"C": [0.01, 0.1, 1, 10, 100]}),
        "SVM (RBF)": tuned(SVC(random_state=seed),
                           {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}),
        "Random forest": tuned(RandomForestClassifier(n_estimators=200, random_state=seed),
                               {"max_depth": [2, 4, None], "min_samples_leaf": [1, 5]}),
        "XGBoost": tuned(XGBClassifier(n_estimators=200, eval_metric="logloss",
                                       random_state=seed),
                         {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1, 0.3]}),
        # 4 inputs -> 6 tanh units -> 1 output: 4*6 + 6 + 6 + 1 = 37 parameters.
        MLP_NAME: pca(MLPClassifier(hidden_layer_sizes=(6,), activation="tanh",
                                    solver="lbfgs", max_iter=5000, random_state=seed)),
        "QSVM (quantum kernel)": GridSearchCV(
            Pipeline(quantum + [("qsvm", QuantumKernelSVC())]), config.QSVM_GRID,
            cv=inner, scoring="roc_auc"),
        VQC_NAME: Pipeline(quantum + [("vqc", VQC(seed=seed))]),
        REFERENCE: Pipeline([("scale", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000))]),
    }


def fold_metrics(model, X, y):
    score = malignant_score(model, X)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    y = np.asarray(y)
    return {
        "AUC": roc_auc_score(y, score),
        "Accuracy": accuracy_score(y, pred),
        "Sensitivity": np.mean(pred[y == 1] == 1),
        "Specificity": np.mean(pred[y == 0] == 0),
        "Sensitivity at 95% specificity": stats.sensitivity_at_specificity(y, score, 0.95),
        "Brier": brier_score_loss(y, proba) if proba is not None else np.nan,
    }, pred, proba


def cross_validate(X, y):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=config.SEED)
    rows, oof = [], {}
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        repeat = fold // N_SPLITS
        for name, model in candidates(config.SEED + fold).items():
            model.fit(X.iloc[tr], y.iloc[tr])
            metrics, pred, proba = fold_metrics(model, X.iloc[te], y.iloc[te])
            rows.append({"fold": fold, "repeat": repeat, "model": name, **metrics})
            if repeat == 0:
                store = oof.setdefault(name, {"pred": np.full(len(y), -1),
                                              "proba": np.full(len(y), np.nan)})
                store["pred"][te] = pred
                if proba is not None:
                    store["proba"][te] = proba
        log(f"  fold {fold + 1}/{N_SPLITS * N_REPEATS}: VQC AUC "
            f"{rows[-2]['AUC']:.3f}, accuracy {rows[-2]['Accuracy']:.3f}")
    return pd.DataFrame(rows), oof


def compare_to_vqc(folds, oof, y):
    """Paired tests of every model against the VQC, Holm-corrected per metric."""
    wide = {m: folds.pivot(index="fold", columns="model", values=m)
            for m in ["AUC", "Accuracy", "Sensitivity at 95% specificity"]}
    y = np.asarray(y)
    correct_vqc = oof[VQC_NAME]["pred"] == y
    rows = []
    for name in folds["model"].unique():
        if name == VQC_NAME:
            continue
        row = {"model": name}
        for metric, table in wide.items():
            mean, lo, hi, p = stats.corrected_ttest(table[VQC_NAME] - table[name],
                                                    1 / (N_SPLITS - 1))
            row.update({f"{metric}: VQC minus model": mean, f"{metric}: CI low": lo,
                        f"{metric}: CI high": hi, f"{metric}: p": p})
        only_vqc, only_model, p = stats.mcnemar_exact(correct_vqc, oof[name]["pred"] == y)
        row.update({"McNemar: only VQC right": only_vqc,
                    "McNemar: only model right": only_model, "McNemar: p": p})
        rows.append(row)
    tests = pd.DataFrame(rows).set_index("model")
    for metric in list(wide) + ["McNemar"]:
        tests[f"{metric}: p (Holm)"] = stats.holm(tests[f"{metric}: p"])
    return tests


def calibration(oof, y):
    y = np.asarray(y)
    out = {}
    for name, store in oof.items():
        p = store["proba"]
        if np.isnan(p).any():
            continue
        out[name] = {"Brier": brier_score_loss(y, p),
                     "ECE": stats.expected_calibration_error(y, p),
                     "proba": p}
    return out


def test_set_intervals():
    """Bootstrap AUC and Wilson sensitivity/specificity intervals on the original test set."""
    X, y = data.load_dataset()
    _, X_test, _, y_test = data.split(X, y)
    y_test = np.asarray(y_test)
    models = {
        "Logistic regression, 30 features": joblib.load(
            config.MODEL_DIR / "logistic_regression.joblib"),
        "QSVM (quantum kernel)": joblib.load(config.MODEL_DIR / "qsvm.joblib"),
        "VQC ensemble": joblib.load(config.MODEL_DIR / "vqc_ensemble.joblib"),
    }
    rows = {}
    for name, model in models.items():
        score = malignant_score(model, X_test)
        thresholds = [("0.5", 0.5)]
        if name == "VQC ensemble":
            thresholds.append(("calibrated", model["vqc"].threshold))
        for label, thr in thresholds:
            # The saved ensemble carries its calibrated threshold, so apply each
            # threshold to the probabilities explicitly; the QSVM has only a margin.
            if hasattr(model, "predict_proba"):
                pred = (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
            else:
                pred = np.asarray(model.predict(X_test))
            tp = int(np.sum((pred == 1) & (y_test == 1)))
            tn = int(np.sum((pred == 0) & (y_test == 0)))
            n_pos, n_neg = int(np.sum(y_test == 1)), int(np.sum(y_test == 0))
            auc_lo, auc_hi = stats.bootstrap_ci(roc_auc_score, y_test, score)
            rows[f"{name}, threshold {label}"] = {
                "AUC": roc_auc_score(y_test, score), "AUC CI low": auc_lo, "AUC CI high": auc_hi,
                "Sensitivity": tp / n_pos,
                "Sensitivity CI low": stats.wilson(tp, n_pos)[0],
                "Sensitivity CI high": stats.wilson(tp, n_pos)[1],
                "Specificity": tn / n_neg,
                "Specificity CI low": stats.wilson(tn, n_neg)[0],
                "Specificity CI high": stats.wilson(tn, n_neg)[1],
            }
    lr = malignant_score(models["Logistic regression, 30 features"], X_test)
    ens = malignant_score(models["VQC ensemble"], X_test)
    diff_lo, diff_hi = stats.bootstrap_ci(
        lambda yy, a, b: roc_auc_score(yy, a) - roc_auc_score(yy, b), y_test, ens, lr)
    return pd.DataFrame(rows).T, {"auc_difference_vqc_minus_logreg":
                                  roc_auc_score(y_test, ens) - roc_auc_score(y_test, lr),
                                  "ci": [diff_lo, diff_hi]}


def main():
    X, y = data.load_dataset()
    log(f"Repeated {N_SPLITS}-fold cross-validation x {N_REPEATS} on all {len(X)} patients")
    folds, oof = cross_validate(X, y)
    folds.to_csv(config.RESULTS_DIR / "fair_comparison_folds.csv", index=False)

    metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity",
               "Sensitivity at 95% specificity", "Brier"]
    order = list(candidates(0))
    summary = folds.groupby("model")[metrics].agg(["mean", "std"]).reindex(order)
    summary.to_csv(config.RESULTS_DIR / "fair_comparison_summary.csv")
    tests = compare_to_vqc(folds, oof, y).reindex([m for m in order if m != VQC_NAME])
    tests.to_csv(config.RESULTS_DIR / "fair_comparison_tests.csv")
    calib = calibration(oof, y)
    pd.DataFrame({k: {"Brier": v["Brier"], "ECE": v["ECE"]} for k, v in calib.items()}).T \
        .to_csv(config.RESULTS_DIR / "fair_comparison_calibration.csv")

    plots.fair_comparison(folds, order, VQC_NAME, REFERENCE)
    plots.reliability({k: v["proba"] for k, v in calib.items()
                       if k in ("Logistic regression", MLP_NAME, VQC_NAME)}, np.asarray(y))

    log("Test-set intervals")
    intervals, auc_diff = test_set_intervals()
    intervals.to_csv(config.RESULTS_DIR / "test_set_intervals.csv")
    (config.RESULTS_DIR / "fair_comparison.json").write_text(json.dumps({
        "design": {"splits": N_SPLITS, "repeats": N_REPEATS, "inner_folds": INNER_FOLDS},
        "test_set_auc_difference": auc_diff}, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nMean (s.d.) over 15 folds:\n" + summary.round(4).to_string())
    print("\nVQC minus each model (corrected t-test) and McNemar:\n" + tests.round(4).T.to_string())
    print("\nCalibration of pooled out-of-fold probabilities (repeat 1):\n" + pd.DataFrame(
        {k: {"Brier": v["Brier"], "ECE": v["ECE"]} for k, v in calib.items()}).T.round(4).to_string())
    print("\nTest-set intervals:\n" + intervals.round(3).to_string())
    print("\n" + json.dumps(auc_diff, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
