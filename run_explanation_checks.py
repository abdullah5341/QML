"""Do the VQC's explanations deserve trust? Four checks on integrated gradients.

    python run_explanation_checks.py      # about 5 minutes on a laptop CPU

1. Agreement with an independent method. Exact Shapley values over the ten
   nucleus properties (2^10 = 1024 coalitions per patient, missing properties
   set to the average patient). Shapley values and integrated gradients make
   different assumptions; if they agree, the explanation is not an artefact of
   one method.
2. Faithfulness (deletion test). Reset features to the average patient in order
   of importance and track how fast the prediction moves. A faithful ranking
   moves it faster than a random order or than an order borrowed from another
   model (logistic regression).
3. Sanity check (Adebayo et al., 2018). Recompute the explanations with the
   trained circuit weights replaced by random ones. Explanations that hardly
   change describe the input pipeline, not what the circuit learned.
4. Stability. Agreement of the explanations of the five independently trained
   circuits.

Also reported: attribution per qubit (principal component), which is what the
circuit itself decides; the 30 feature attributions follow from these four
numbers and the fixed PCA map.
"""
import itertools
import math
import json
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qmlbc import classical, config, data, explain, plots

N_RANDOM_ORDERS = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def logits(model, rows, columns, chunk=16384):
    return np.concatenate([
        model.decision_function(pd.DataFrame(rows[i:i + chunk], columns=columns))
        for i in range(0, len(rows), chunk)
    ])


def grouped_shapley(model, X, baseline, columns):
    """Exact Shapley values of the logit over the ten nucleus-property groups."""
    groups = [[j for j, f in enumerate(columns) if explain.quantity_of(f) == q]
              for q in explain.BASE_QUANTITIES]
    G = len(groups)
    masks = np.array(list(itertools.product([0, 1], repeat=G)))          # (1024, G)
    feature_mask = np.zeros((len(masks), len(columns)))
    for g, cols in enumerate(groups):
        feature_mask[:, cols] = masks[:, [g]]
    rows = (baseline + feature_mask[None] * (X - baseline)[:, None]).reshape(-1, len(columns))
    v = logits(model, rows, columns).reshape(len(X), len(masks))

    code = masks @ (1 << np.arange(G))
    position = np.empty(len(masks), int)
    position[code] = np.arange(len(masks))
    size = masks.sum(axis=1)
    weight = np.array([math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G)
                       if k < G else 0.0 for k in size])
    phi = np.zeros((len(X), G))
    for g in range(G):
        without = np.flatnonzero(masks[:, g] == 0)
        with_g = position[code[without] | (1 << g)]
        phi[:, g] = (v[:, with_g] - v[:, without]) @ weight[without]
    return phi


def deletion_steps(x, baseline, order):
    """Copies of x with the features in `order` reset to the baseline one at a time."""
    steps = np.repeat(x[None], len(order) + 1, axis=0)
    for k, j in enumerate(order):
        steps[k + 1:, j] = baseline[j]
    return steps


def mean_spearman(a, b):
    return float(np.nanmean([spearmanr(u, v).statistic for u, v in zip(a, b)]))


def main():
    X, y = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split(X, y)
    columns = list(X.columns)
    baseline = X_train.mean().values
    Xt = X_test.values
    ensemble = joblib.load(config.MODEL_DIR / "vqc_ensemble.joblib")
    logreg = classical.full_feature_models()["Logistic Regression"].fit(X_train, y_train)
    F = ensemble.decision_function(X_test)
    F0 = ensemble.decision_function(pd.DataFrame([baseline], columns=columns))[0]

    log("Integrated gradients")
    ig = explain.integrated_gradients(ensemble, Xt, baseline, config.IG_STEPS)
    ig_groups = np.array([explain.by_quantity(r, columns).values for r in ig])
    lr = explain.linear_attributions(logreg, Xt, baseline)
    lr_groups = np.array([explain.by_quantity(r, columns).values for r in lr])

    log("1. Exact grouped Shapley values")
    shap = grouped_shapley(ensemble, Xt, baseline, columns)
    shap_gap = np.abs(shap.sum(axis=1) - (F - F0)).max()
    top_agree = np.mean(np.argmax(np.abs(ig_groups), 1) == np.argmax(np.abs(shap), 1))
    shares = pd.DataFrame({
        "VQC, integrated gradients": np.abs(ig_groups).mean(0),
        "VQC, Shapley values": np.abs(shap).mean(0),
        "Logistic regression (exact)": np.abs(lr_groups).mean(0),
    }, index=explain.BASE_QUANTITIES)
    shares = shares / shares.sum()
    plots.explanation_agreement(shares)

    log("2. Deletion test")
    rng = np.random.default_rng(config.SEED)
    orders = {
        "VQC integrated gradients": [np.argsort(-np.abs(r)) for r in ig],
        "Logistic regression attributions": [np.argsort(-np.abs(r)) for r in lr],
    }
    random_orders = [[rng.permutation(len(columns)) for _ in Xt] for _ in range(N_RANDOM_ORDERS)]
    keep = np.abs(F - F0) > 1.0          # patients whose prediction differs from the baseline
    curves, areas = {}, {}
    for name, per_patient in list(orders.items()) + [("Random order", None)]:
        sets = random_orders if per_patient is None else [per_patient]
        runs = []
        for order_set in sets:
            rows = np.concatenate([deletion_steps(Xt[i], baseline, order_set[i])
                                   for i in range(len(Xt))])
            v = logits(ensemble, rows, columns).reshape(len(Xt), len(columns) + 1)
            runs.append((v - F0) / (F - F0)[:, None])
        r = np.mean(runs, axis=0)[keep]
        curves[name] = np.median(r, axis=0)
        areas[name] = float(np.mean(np.trapezoid(np.abs(r), dx=1, axis=1) / len(columns)))
    plots.deletion_curves(curves)

    log("3. Sanity check with randomised circuit weights")
    random_ens = deepcopy(ensemble)
    for m in random_ens["vqc"].members_:
        m.weights_ = rng.uniform(0, 2 * np.pi, size=m.weights_.shape)
    ig_random = explain.integrated_gradients(random_ens, Xt, baseline, config.IG_STEPS)
    sanity = mean_spearman(np.abs(ig), np.abs(ig_random))
    sanity_angles = mean_spearman(
        np.abs(explain.angle_attributions(ensemble, Xt, baseline, config.IG_STEPS)),
        np.abs(explain.angle_attributions(random_ens, Xt, baseline, config.IG_STEPS)))

    log("4. Stability across the five circuits")
    members = ensemble["vqc"].members_
    per_member = [explain.integrated_gradients(ensemble, Xt, baseline, config.IG_STEPS, [m])
                  for m in members]
    pairs = list(itertools.combinations(range(len(members)), 2))
    stability_patient = np.mean([mean_spearman(np.abs(per_member[a]), np.abs(per_member[b]))
                                 for a, b in pairs])
    stability_global = np.mean([spearmanr(np.abs(per_member[a]).mean(0),
                                          np.abs(per_member[b]).mean(0)).statistic
                                for a, b in pairs])

    angle = explain.angle_attributions(ensemble, Xt, baseline, config.IG_STEPS)
    qubit_share = np.abs(angle).mean(0) / np.abs(angle).mean(0).sum()

    summary = {
        "shapley_completeness_max_error": float(shap_gap),
        "ig_vs_shapley_mean_spearman_per_patient": mean_spearman(ig_groups, shap),
        "ig_vs_shapley_top_property_agreement": float(top_agree),
        "ig_vs_shapley_global_spearman": float(spearmanr(shares.iloc[:, 0], shares.iloc[:, 1]).statistic),
        "vqc_vs_logreg_global_spearman": float(spearmanr(shares.iloc[:, 0], shares.iloc[:, 2]).statistic),
        "deletion_patients": int(keep.sum()),
        "deletion_area": areas,
        "sanity_spearman_trained_vs_random_features": sanity,
        "sanity_spearman_trained_vs_random_qubits": sanity_angles,
        "stability_mean_spearman_per_patient": float(stability_patient),
        "stability_mean_spearman_global": float(stability_global),
        "qubit_share": dict(zip([f"qubit {k} (PC{k + 1})" for k in range(len(qubit_share))],
                                map(float, qubit_share))),
    }
    shares.to_csv(config.RESULTS_DIR / "explanation_methods_by_quantity.csv")
    pd.DataFrame(curves).to_csv(config.RESULTS_DIR / "deletion_curves.csv", index_label="deleted")
    (config.RESULTS_DIR / "explanation_checks.json").write_text(json.dumps(summary, indent=2))
    print("\nShares by property:\n" + shares.round(3).to_string())
    print("\n" + json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
