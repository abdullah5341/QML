"""From circuit structure to noise behaviour: the Pauli-path theory of the affine regularity.

    python run_pauli_paths.py      # about 15 minutes on a laptop CPU

Section 7.3 found empirically that the noisy VQC output is close to an affine
copy of the noiseless one. Here the noisy output is decomposed exactly into
Pauli paths grouped by how many noisy locations they pass (qmlbc/pauli_paths.py),
which turns the observation into a statement about circuit structure:

    lambda(p) = sum_{k,m} w_km q2^k q1^m       (exact, no noise simulation)
    lambda(p) ~ (1 - 4p/3)^k_eff               (closed form)
    1 - R^2   ~ (4p/3)^2 D                     (first order)

where w, k_eff and D are computed once from the noiseless circuit. The
regularity holds while paths of different length carry the same function of
x (small D) and fails when they do not.

Tests:
A. The decomposition reproduces density-matrix simulation exactly.
B. How well the closed forms predict lambda and the breakdown point p*
   (the CNOT error at which R^2 falls below 0.99).
C. Proposal hypothesis RQ1: affinity breaks earlier in deeper circuits.
   Circuits with 1 to 5 layers (3 seeds each) are trained and p* is compared
   across depths; D should rank circuits by p* if it captures the mechanism.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from qmlbc import config, data, noise, pauli_paths, plots
from qmlbc.vqc import VQC

DEPTHS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
P_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def analyse(member, angles, check_p=0.05):
    g = pauli_paths.hit_spectrum(member.weights_, angles, member.n_qubits)
    summary = pauli_paths.structural_summary(g)
    sim = noise.noisy_expectation(member, angles, check_p)
    f = g.sum(axis=(1, 2))
    return g, f, {
        "k_eff": summary["k_eff"], "D": summary["D"],
        "negative_weight": summary["negative_weight"],
        "p_star_exact": pauli_paths.critical_noise(g),
        "p_star_first_order": 0.75 * 0.1 / np.sqrt(summary["D"]),
        "decomposition_error": float(np.abs(pauli_paths.noisy_output(g, check_p) - sim).max()),
        "output_spread": float(f.std()),
    }


def summarise(main_circuits, depth):
    # Circuits whose R^2 never falls below 0.99 on the grid (one-layer circuits,
    # D = 0) are censored at the top of the grid for the rank statistics.
    censored = depth.assign(p_star=depth["p_star_exact"].fillna(0.3))
    rho_D = spearmanr(censored["D"], censored["p_star"])
    rho_L = spearmanr(censored["layers"], censored["p_star"])
    finite = depth.dropna(subset=["p_star_exact"])
    ratio = finite["p_star_first_order"] / finite["p_star_exact"]
    return {
        "max_decomposition_error": float(max(main_circuits["decomposition_error"].max(),
                                             depth["decomposition_error"].max())),
        "main_k_eff_mean": float(main_circuits["k_eff"].mean()),
        "main_p_star_exact_mean": float(main_circuits["p_star_exact"].mean()),
        "main_p_star_first_order_mean": float(main_circuits["p_star_first_order"].mean()),
        "never_break_on_grid": int(depth["p_star_exact"].isna().sum()),
        "spearman_D_vs_p_star": [float(rho_D.statistic), float(rho_D.pvalue)],
        "spearman_depth_vs_p_star": [float(rho_L.statistic), float(rho_L.pvalue)],
        "first_order_over_exact_p_star": {"median": float(ratio.median()),
                                          "min": float(ratio.min()), "max": float(ratio.max())},
    }


def main():
    X, y = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split(X, y)

    log("A/B. The five trained circuits of the main study (3 layers)")
    ensemble = joblib.load(config.MODEL_DIR / "vqc_ensemble.joblib")
    angles = ensemble[:-1].transform(X)
    rows, lam_rows = [], []
    for seed, member in zip(ensemble["vqc"].seeds, ensemble["vqc"].members_):
        g, f, row = analyse(member, angles)
        rows.append({"seed": seed, **row})
        for p in P_GRID:
            lam, _, r2 = pauli_paths.affine_fit(f, pauli_paths.noisy_output(g, p))
            lam_rows.append({
                "seed": seed, "p": p, "lambda_exact": lam, "r2_exact": r2,
                "lambda_closed_form": (1 - 4 * p / 3) ** row["k_eff"],
                "r2_first_order": 1 - (4 * p / 3) ** 2 * row["D"],
            })
        log(f"  seed {seed}: k_eff {row['k_eff']:.2f}, D {row['D']:.3f}, p* exact "
            f"{row['p_star_exact']:.3f}, first order {row['p_star_first_order']:.3f}, "
            f"decomposition error {row['decomposition_error']:.1e}")
    main_circuits = pd.DataFrame(rows)
    lam = pd.DataFrame(lam_rows).groupby("p").mean().drop(columns="seed")

    log("C. Depth study")
    depth_rows = []
    for L in DEPTHS:
        for seed in SEEDS:
            model = Pipeline(data.quantum_steps(config.N_QUBITS) +
                             [("vqc", VQC(n_layers=L, seed=seed))]).fit(X_train, y_train)
            _, _, row = analyse(model["vqc"], model[:-1].transform(X))
            row.update({"layers": L, "seed": seed, "test_accuracy": model.score(X_test, y_test)})
            depth_rows.append(row)
            log(f"  L={L} seed {seed}: accuracy {row['test_accuracy']:.3f}, k_eff "
                f"{row['k_eff']:.1f}, D {row['D']:.3f}, p* {row['p_star_exact']:.3f}")
    depth = pd.DataFrame(depth_rows)

    summary = summarise(main_circuits, depth)
    main_circuits.to_csv(config.RESULTS_DIR / "pauli_paths_main.csv", index=False)
    lam.to_csv(config.RESULTS_DIR / "pauli_paths_lambda.csv")
    depth.to_csv(config.RESULTS_DIR / "pauli_paths_depth.csv", index=False)
    (config.RESULTS_DIR / "pauli_paths.json").write_text(json.dumps(summary, indent=2))
    plots.pauli_paths(lam, depth)

    pd.set_option("display.width", 200)
    print("\nMain circuits:\n" + main_circuits.round(4).to_string(index=False))
    print("\nLambda and R^2 (mean of 5 circuits):\n" + lam.round(4).to_string())
    print("\nDepth study:\n" + depth.round(4).to_string(index=False))
    print("\nBy depth:\n" + depth.groupby("layers")[["test_accuracy", "k_eff", "D", "p_star_exact",
                                                     "p_star_first_order"]].mean().round(4).to_string())
    print("\n" + json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
