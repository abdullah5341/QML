"""Why does refitting two classical numbers undo gate noise, and when does it not?

    python run_noise_mechanism.py      # about 15 minutes on a laptop CPU

Hypothesis. Noise that is close to an affine map of the circuit output,
<Z_0>_noisy = lambda * <Z_0>_clean + c, keeps the ranking of patients (AUC)
and is undone exactly by refitting the read-out sigmoid(a z + b): the refit
must find a' = a / lambda. Noise that changes the function the circuit
computes cannot be undone this way.

Tests, on the trained five-circuit ensemble and without retraining it:

1. Affinity. Regress noisy on clean outputs for all 569 patients: lambda, c, R².
2. Prediction of the refit. Compare the refitted scale a' with a0 / lambda,
   where a0 is the scale refitted on noiseless outputs.
3. Four error models, three expected to be near-affine (depolarizing, amplitude
   damping, read-out flips) and one that is not (coherent over-rotation of the
   trained angles). Recalibration should work for the first three only.
4. Shot cost. If noise only shrinks the signal by lambda, the shots needed
   scale as 1 / lambda², so accuracy curves for different noise levels should
   coincide when plotted against N * lambda².

The read-out is refitted on training patients and evaluated on test patients.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from qmlbc import config, data, noise, plots

STRENGTHS = {
    "depolarizing": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "amplitude damping": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "readout": [0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
    "coherent over-rotation": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
}
SHOT_LEVELS = [0.0, 0.02, 0.05, 0.1]
SHOTS = [10, 30, 100, 300, 1000, 3000, 10000]
SHOT_REPEATS = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def affine_fit(z_clean, z_noisy):
    lam, c = np.polyfit(z_clean, z_noisy, 1)
    resid = z_noisy - (lam * z_clean + c)
    r2 = 1 - resid.var() / z_noisy.var() if z_noisy.var() > 0 else np.nan
    return lam, c, r2


def main():
    X, y = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split(X, y)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    ensemble = joblib.load(config.MODEL_DIR / "vqc_ensemble.joblib")
    members = ensemble["vqc"].members_
    a_train = ensemble[:-1].transform(X_train)
    a_test = ensemble[:-1].transform(X_test)
    n_train = len(a_train)
    angles = np.vstack([a_train, a_test])

    clean = [np.asarray(m._circuit()(m.weights_, angles)) for m in members]
    # Read-out refitted on noiseless outputs: the reference the noisy refit is compared to.
    a_clean = [noise.recalibrate_readout(z0[:n_train], y_train)[0] for z0 in clean]
    check = noise.error_model_expectation(members[0], a_test[:10], "depolarizing", 0.0)
    assert np.allclose(check, clean[0][n_train:n_train + 10], atol=1e-6)

    rows = []
    for kind, levels in STRENGTHS.items():
        for s in levels:
            per = []
            for m, z0, a0 in zip(members, clean, a_clean):
                z = noise.error_model_expectation(m, angles, kind, s)
                lam, c, r2 = affine_fit(z0, z)
                a_fit, b_fit = noise.recalibrate_readout(z[:n_train], y_train)
                zt = z[n_train:]
                per.append({
                    "lambda": lam, "offset": c, "r2": r2,
                    "spearman": spearmanr(z0[n_train:], zt).statistic,
                    "auc": roc_auc_score(y_test, m.scale_ * zt),
                    "acc_as_trained": np.mean(noise.classify(zt, m.scale_, m.bias_) == y_test),
                    "acc_recalibrated": np.mean(noise.classify(zt, a_fit, b_fit) == y_test),
                    # 1 if the refit does exactly what the affine model predicts (a0 / lambda).
                    "refit_vs_prediction": a_fit * lam / a0,
                })
            row = {"kind": kind, "strength": s, **pd.DataFrame(per).mean().to_dict(),
                   "acc_recalibrated_std": pd.DataFrame(per)["acc_recalibrated"].std(ddof=0)}
            rows.append(row)
            log(f"  {kind:<23} {s:<6g} lambda {row['lambda']:.3f}  R² {row['r2']:.4f}  "
                f"AUC {row['auc']:.3f}  acc {row['acc_as_trained']:.3f} -> "
                f"{row['acc_recalibrated']:.3f}  refit/prediction {row['refit_vs_prediction']:.3f}")
    mech = pd.DataFrame(rows)
    mech.to_csv(config.RESULTS_DIR / "noise_mechanism.csv", index=False)
    plots.noise_linearity(mech)

    # Effective number of fully damping CNOT channels: lambda ~ (1 - 4p/3)^k.
    dep = mech[mech["kind"] == "depolarizing"]
    k_eff = np.polyfit(np.log(1 - 4 * dep["strength"] / 3), np.log(dep["lambda"]), 1)[0]

    log("Shot cost against effective shots")
    rng = np.random.default_rng(config.SEED)
    shot_rows = []
    for p in SHOT_LEVELS:
        accs = {n: [] for n in SHOTS}
        lams = []
        for m, z0 in zip(members, clean):
            z = noise.error_model_expectation(m, angles, "depolarizing", p)
            lams.append(affine_fit(z0, z)[0])
            a_fit, b_fit = noise.recalibrate_readout(z[:n_train], y_train)
            for n in SHOTS:
                for _ in range(SHOT_REPEATS):
                    z_hat = noise.sample_shots(z[n_train:], n, rng)
                    accs[n].append(np.mean(noise.classify(z_hat, a_fit, b_fit) == y_test))
        shot_rows += [{"cnot_error": p, "lambda": np.mean(lams), "shots": n,
                       "effective_shots": n * np.mean(lams) ** 2,
                       "acc_mean": np.mean(v), "acc_std": np.std(v)} for n, v in accs.items()]
    shots = pd.DataFrame(shot_rows)
    shots.to_csv(config.RESULTS_DIR / "noise_shot_collapse.csv", index=False)
    plots.shot_collapse(shots)

    # How well does N * lambda² predict accuracy? Interpolate each noisy curve
    # onto the noiseless one at equal effective shots.
    ref = shots[shots["cnot_error"] == 0]
    gaps = []
    for p, grp in shots[shots["cnot_error"] > 0].groupby("cnot_error"):
        inside = grp[(grp["effective_shots"] >= ref["shots"].min()) &
                     (grp["effective_shots"] <= ref["shots"].max())]
        predicted = np.interp(np.log(inside["effective_shots"]), np.log(ref["shots"]),
                              ref["acc_mean"])
        gaps.append(float(np.max(np.abs(predicted - inside["acc_mean"]))))

    summary = {"k_eff": float(k_eff), "max_accuracy_gap_after_collapse": max(gaps),
               "noisy_cnot_channels_in_circuit": 2 * config.N_LAYERS * config.N_QUBITS}
    (config.RESULTS_DIR / "noise_mechanism.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    print("\nMechanism:\n" + mech.round(4).to_string(index=False))
    print("\nShots:\n" + shots.pivot(index="shots", columns="cnot_error", values="acc_mean")
          .round(4).to_string())
    print("\n" + json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
