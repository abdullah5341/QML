"""Extensions to the base study: is the quantum model reliable, robust and explainable?

    python run_extensions.py        # about 20 minutes on a laptop CPU

1. Ensemble of VQCs with a decision threshold calibrated to a target sensitivity.
2. Hardware-noise study: gate errors and finite measurement shots.
3. Explainability: integrated-gradient attributions on the 30 original features.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from qmlbc import classical, config, data, explain, noise, plots
from qmlbc.ensemble import (VQCEnsemble, sensitivity_specificity,
                            threshold_for_sensitivity)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calibration_study(ensemble, logreg, X_train, y_train, X_test, y_test):
    """Pick thresholds on out-of-fold training predictions, then score on the test set."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.empty(len(X_train))
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        model = clone(ensemble).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        log(f"  out-of-fold predictions, fold {fold + 1}/{config.CV_FOLDS}")
    oof_lr = cross_val_predict(logreg, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = {
        "vqc": threshold_for_sensitivity(y_train, oof, config.TARGET_SENSITIVITY),
        "logreg": threshold_for_sensitivity(y_train, oof_lr, config.TARGET_SENSITIVITY),
    }
    ensemble["vqc"].threshold = thresholds["vqc"]

    single, ens, lr = "Single VQC run", "VQC ensemble (5 runs)", "Logistic regression, 30 features"
    p_ens = ensemble.predict_proba(X_test)[:, 1]
    p_lr = logreg.predict_proba(X_test)[:, 1]
    angles = ensemble[:-1].transform(X_test)
    rows = {}
    for seed, member in zip(ensemble["vqc"].seeds, ensemble["vqc"].members_):
        rows[f"VQC seed {seed}, threshold 0.5"] = (single, member.predict(angles))
    rows["VQC ensemble, threshold 0.5"] = (ens, p_ens >= 0.5)
    rows[f"VQC ensemble, threshold {thresholds['vqc']:.2f} (calibrated)"] = (
        ens, p_ens >= thresholds["vqc"])
    rows["Logistic regression, threshold 0.5"] = (lr, p_lr >= 0.5)
    rows[f"Logistic regression, threshold {thresholds['logreg']:.2f} (calibrated)"] = (
        lr, p_lr >= thresholds["logreg"])

    table = pd.DataFrame({
        name: {"group": group, **sensitivity_specificity(y_test, pred.astype(int))}
        for name, (group, pred) in rows.items()
    }).T
    table.index.name = "Model"
    table["ROC AUC"] = np.nan
    table.loc[table["group"] == ens, "ROC AUC"] = roc_auc_score(y_test, p_ens)
    table.loc[table["group"] == lr, "ROC AUC"] = roc_auc_score(y_test, p_lr)
    return table, thresholds


def noise_study(ensemble, X_train, y_train, X_test, y_test):
    members = ensemble["vqc"].members_
    angles_train = ensemble[:-1].transform(X_train)
    angles_test = ensemble[:-1].transform(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    rng = np.random.default_rng(config.SEED)
    levels = sorted(set(config.NOISE_CNOT_ERRORS) | set(config.NOISE_SHOT_LEVELS))

    gate_rows, shot_rows = [], []
    for p in levels:
        raw, recal, auc, shrink = [], [], [], []
        shot_acc = {shots: [] for shots in config.NOISE_SHOTS}
        for member in members:
            z_train = noise.noisy_expectation(member, angles_train, p)
            z_test = noise.noisy_expectation(member, angles_test, p)
            if p == 0:
                exact = member.decision_function(angles_test)
                assert np.allclose(member.scale_ * z_test + member.bias_, exact, atol=1e-6)
                clean_spread = z_test.std()
            a, b = noise.recalibrate_readout(z_train, y_train)
            raw.append(np.mean(noise.classify(z_test, member.scale_, member.bias_) == y_test))
            recal.append(np.mean(noise.classify(z_test, a, b) == y_test))
            auc.append(roc_auc_score(y_test, member.scale_ * z_test))
            shrink.append(z_test.std())
            if p in config.NOISE_SHOT_LEVELS:
                for shots in config.NOISE_SHOTS:
                    for _ in range(config.NOISE_SHOT_REPEATS):
                        z_hat = noise.sample_shots(z_test, shots, rng)
                        shot_acc[shots].append(np.mean(noise.classify(z_hat, a, b) == y_test))
        if p in config.NOISE_CNOT_ERRORS:
            gate_rows.append({
                "cnot_error": p,
                "acc_raw_mean": np.mean(raw), "acc_raw_std": np.std(raw),
                "acc_recal_mean": np.mean(recal), "acc_recal_std": np.std(recal),
                "auc_mean": np.mean(auc), "auc_std": np.std(auc),
                "signal_spread": np.mean(shrink),
            })
            log(f"  CNOT error {p:g}: accuracy {np.mean(raw):.3f} as trained, "
                f"{np.mean(recal):.3f} recalibrated, AUC {np.mean(auc):.3f}")
        if p in config.NOISE_SHOT_LEVELS:
            shot_rows += [{"cnot_error": p, "shots": shots, "acc_mean": np.mean(accs),
                           "acc_std": np.std(accs)} for shots, accs in shot_acc.items()]
    gate = pd.DataFrame(gate_rows)
    gate["signal_retained"] = gate["signal_spread"] / gate["signal_spread"].iloc[0]
    return gate, pd.DataFrame(shot_rows)


def explain_study(ensemble, logreg, X_train, X_test, y_test):
    baseline = X_train.mean().values
    features = list(X_train.columns)
    attr_vqc = explain.integrated_gradients(ensemble, X_test.values, baseline, config.IG_STEPS)
    attr_lr = explain.linear_attributions(logreg, X_test.values, baseline)

    # Completeness: attributions must add up to logit(patient) - logit(average patient).
    base_logit = ensemble.decision_function(pd.DataFrame([baseline], columns=features))[0]
    gap = attr_vqc.sum(axis=1) - (ensemble.decision_function(X_test) - base_logit)

    per_feature = pd.DataFrame({
        "VQC ensemble (4 qubits)": np.abs(attr_vqc).mean(axis=0),
        "Logistic regression (30 features)": np.abs(attr_lr).mean(axis=0),
    }, index=pd.Index(features, name="feature"))
    shares = per_feature / per_feature.sum()
    by_quantity = shares.apply(lambda col: explain.by_quantity(col.values, features))
    plots.global_attribution(by_quantity)

    # Example: the detected cancer the model was least sure about, where the
    # evidence points both ways and an explanation is most useful.
    p_test = ensemble.predict_proba(X_test)[:, 1]
    detected = (np.asarray(y_test) == 1) & (p_test >= ensemble["vqc"].threshold)
    row = int(np.flatnonzero(detected)[np.argmin(p_test[detected])])
    case, p_case = X_test.index[row], p_test[row]
    plots.patient_attribution(
        pd.Series(attr_vqc[row], index=features),
        f"A borderline cancer: why the VQC ensemble flags case {case} (P = {p_case:.2f})",
    )
    rho = spearmanr(per_feature.iloc[:, 0], per_feature.iloc[:, 1])
    summary = {
        "completeness_max_abs_error": float(np.abs(gap).max()),
        "spearman_vqc_vs_logreg": float(rho.statistic),
        "spearman_p_value": float(rho.pvalue),
        "top_features_vqc": shares.iloc[:, 0].sort_values(ascending=False).head(5).round(4).to_dict(),
        "top_features_logreg": shares.iloc[:, 1].sort_values(ascending=False).head(5).round(4).to_dict(),
        "example_case": int(case),
    }
    return shares, by_quantity, summary


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    X, y = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split(X, y)

    log("Training the VQC ensemble")
    ensemble = Pipeline(data.quantum_steps(config.N_QUBITS) + [("vqc", VQCEnsemble())])
    ensemble.fit(X_train, y_train)
    logreg = classical.full_feature_models()["Logistic Regression"].fit(X_train, y_train)

    log("1. Ensemble and threshold calibration")
    operating, thresholds = calibration_study(ensemble, logreg, X_train, y_train, X_test, y_test)
    operating.to_csv(config.RESULTS_DIR / "operating_points.csv")
    plots.operating_points(operating)
    joblib.dump(ensemble, config.MODEL_DIR / "vqc_ensemble.joblib")

    log("2. Hardware-noise study")
    gate, shots = noise_study(ensemble, X_train, y_train, X_test, y_test)
    gate.to_csv(config.RESULTS_DIR / "noise_gate.csv", index=False)
    shots.to_csv(config.RESULTS_DIR / "noise_shots.csv", index=False)
    plots.gate_noise(gate)
    plots.shot_noise(shots)

    log("3. Explainability")
    shares, by_quantity, explain_summary = explain_study(ensemble, logreg, X_train, X_test, y_test)
    shares.to_csv(config.RESULTS_DIR / "attribution_by_feature.csv")
    by_quantity.to_csv(config.RESULTS_DIR / "attribution_by_quantity.csv")

    summary = {"target_sensitivity": config.TARGET_SENSITIVITY, "thresholds": thresholds,
               "explainability": explain_summary}
    (config.RESULTS_DIR / "extensions.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    print("\nOperating points:\n" + operating.drop(columns="group").to_string())
    print("\nGate noise:\n" + gate.round(4).to_string(index=False))
    print("\nShots:\n" + shots.pivot(index="shots", columns="cnot_error", values="acc_mean")
          .round(4).to_string())
    print("\nAttribution share by quantity:\n" + by_quantity.round(3).to_string())
    print("\n" + json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
