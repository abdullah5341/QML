"""Confirm the calibrated threshold and the noise recalibration without the test set.

    python run_validation_check.py      # about 8 minutes on a laptop CPU

The main study reports both results on one held-out test set that several
analyses have looked at. Here the test set is never touched. The 455 training
patients are split three ways, five times over with different random splits:

    fit (60%)          train the five circuits
    calibration (20%)  choose the decision threshold, refit the noisy read-out
    validation (20%)   report

so every reported number comes from patients used for neither training nor selection.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from qmlbc import classical, config, data, noise
from qmlbc.ensemble import (VQCEnsemble, sensitivity_specificity,
                            threshold_for_sensitivity)

REPEATS = 5
NOISE_LEVELS = [0.0, 0.05, 0.1]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def three_way_split(X, y, seed):
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return X_fit, y_fit, X_cal, y_cal, X_val, y_val


def operating_rows(repeat, name, model, X_cal, y_cal, X_val, y_val):
    """Validation results at 0.5 and at the threshold chosen on the calibration split."""
    threshold = threshold_for_sensitivity(
        y_cal, model.predict_proba(X_cal)[:, 1], config.TARGET_SENSITIVITY)
    p_val = model.predict_proba(X_val)[:, 1]
    return [
        {"repeat": repeat, "model": name, "threshold": label, "value": value,
         "cancers": int(np.sum(y_val == 1)),
         **sensitivity_specificity(y_val, (p_val >= value).astype(int))}
        for label, value in [("0.5", 0.5), ("calibrated", threshold)]
    ]


def noise_rows(repeat, ensemble, X_cal, y_cal, X_val, y_val):
    angles_cal = ensemble[:-1].transform(X_cal)
    angles_val = ensemble[:-1].transform(X_val)
    y_cal, y_val = np.asarray(y_cal), np.asarray(y_val)
    rows = []
    for p in NOISE_LEVELS:
        raw, recal, auc = [], [], []
        for member in ensemble["vqc"].members_:
            z_cal = noise.noisy_expectation(member, angles_cal, p)
            z_val = noise.noisy_expectation(member, angles_val, p)
            a, b = noise.recalibrate_readout(z_cal, y_cal)
            raw.append(np.mean(noise.classify(z_val, member.scale_, member.bias_) == y_val))
            recal.append(np.mean(noise.classify(z_val, a, b) == y_val))
            auc.append(roc_auc_score(y_val, member.scale_ * z_val))
        rows.append({"repeat": repeat, "cnot_error": p, "acc_as_trained": np.mean(raw),
                     "acc_recalibrated": np.mean(recal), "auc": np.mean(auc)})
    return rows


def main():
    X, y = data.load_dataset()
    X_train, _, y_train, _ = data.split(X, y)   # the test set is discarded here

    operating, noisy = [], []
    for repeat in range(REPEATS):
        X_fit, y_fit, X_cal, y_cal, X_val, y_val = three_way_split(
            X_train, y_train, config.SEED + repeat)
        ensemble = Pipeline(data.quantum_steps(config.N_QUBITS) + [("vqc", VQCEnsemble())])
        ensemble.fit(X_fit, y_fit)
        logreg = classical.full_feature_models()["Logistic Regression"].fit(X_fit, y_fit)
        operating += operating_rows(repeat, "VQC ensemble", ensemble, X_cal, y_cal, X_val, y_val)
        operating += operating_rows(repeat, "Logistic regression", logreg, X_cal, y_cal, X_val, y_val)
        noisy += noise_rows(repeat, ensemble, X_cal, y_cal, X_val, y_val)
        log(f"repeat {repeat + 1}/{REPEATS} done "
            f"(fit {len(X_fit)}, calibration {len(X_cal)}, validation {len(X_val)})")

    operating, noisy = pd.DataFrame(operating), pd.DataFrame(noisy)
    operating.to_csv(config.RESULTS_DIR / "validation_operating_points.csv", index=False)
    noisy.to_csv(config.RESULTS_DIR / "validation_noise.csv", index=False)

    pd.set_option("display.width", 200)
    grouped = operating.groupby(["model", "threshold"], sort=False)
    summary = grouped[["value", "Sensitivity", "Specificity", "Accuracy"]].agg(["mean", "std"])
    pooled = grouped[["cancers", "Missed cancers", "False alarms"]].sum()
    print("\nPer repeat:\n" + operating.round(3).to_string(index=False))
    print("\nMean and s.d. over repeats:\n" + summary.round(3).to_string())
    print("\nPooled over repeats:\n" + pooled.to_string())
    print("\nNoise, mean and s.d. over repeats:\n" + noisy.groupby("cnot_error")[
        ["acc_as_trained", "acc_recalibrated", "auc"]].agg(["mean", "std"]).round(3).to_string())
    log("Done.")


if __name__ == "__main__":
    main()
