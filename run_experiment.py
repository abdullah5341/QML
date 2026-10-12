"""Run the full study: EDA, classical baselines, QSVM, VQC, ablation, figures, tables.

    python run_experiment.py            # everything (about 8 minutes on a laptop CPU)
    python run_experiment.py --quick    # skip cross-validation of the VQC and the ablation
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

from qmlbc import classical, config, data, plots
from qmlbc.metrics import METRICS, evaluate, malignant_score
from qmlbc.qkernel import QuantumKernelSVC
from qmlbc.vqc import VQC

FULL, REDUCED, QUANTUM = (
    "Classical, 30 features",
    f"Classical, {config.N_QUBITS} PCA components",
    f"Quantum, {config.N_QUBITS} qubits",
)
PCA_SUFFIX = f" (PCA-{config.N_QUBITS})"
VQC_ONE = f"VQC (seed {config.VQC_SEEDS[0]})"
VQC_MEAN = f"VQC (mean of {len(config.VQC_SEEDS)} seeds)"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eda(X, y, X_train, y_train):
    plots.class_distribution(data.class_counts(y))
    plots.target_correlation(data.target_correlations(X, y))
    plots.correlation_heatmap(X)

    # PCA is fitted on the training set only, as it is inside every model pipeline.
    full_pca = Pipeline(data.reduction_steps(None)).fit(X_train)
    explained = full_pca["pca"].explained_variance_ratio_
    plots.pca_variance(explained, config.N_QUBITS)
    plots.pca_scatter(full_pca.transform(X_train), y_train)
    return explained


def vqc_pipeline(**kwargs):
    return Pipeline(data.quantum_steps(config.N_QUBITS) + [("vqc", VQC(**kwargs))])


def run_ablation(X_train, y_train):
    """Depth x encoding sweep, scored on a validation split so the test set stays unseen."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=config.ABLATION_VAL_SIZE, stratify=y_train,
        random_state=config.SEED,
    )
    rows = []
    for reupload in (False, True):
        for layers in config.ABLATION_LAYERS:
            accs = [
                vqc_pipeline(n_layers=layers, reupload=reupload, seed=seed,
                             epochs=config.ABLATION_EPOCHS).fit(X_fit, y_fit).score(X_val, y_val)
                for seed in config.ABLATION_SEEDS
            ]
            rows.append({
                "encoding": "Data re-uploading" if reupload else "Single encoding",
                "layers": layers, "parameters": layers * config.N_QUBITS * 3 + 2,
                "mean": np.mean(accs), "std": np.std(accs),
            })
            log(f"  ablation {rows[-1]['encoding']}, L={layers}: {np.mean(accs):.3f}")
    return pd.DataFrame(rows)


def main(quick=False):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = data.load_dataset()
    X_train, X_test, y_train, y_test = data.split(X, y)
    log(f"Loaded {len(X)} samples, {X.shape[1]} features; "
        f"train {len(X_train)}, test {len(X_test)}")

    explained = eda(X, y, X_train, y_train)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    rows, families, scores, matrices = {}, {}, {}, {}

    def record(name, family, model, cv_scores=None):
        pred = model.predict(X_test)
        score = malignant_score(model, X_test)
        rows[name] = evaluate(y_test, pred, score)
        if cv_scores is not None:
            rows[name]["CV accuracy"] = np.mean(cv_scores)
            rows[name]["CV std"] = np.std(cv_scores)
        families[name] = family
        scores[name] = (score, rows[name]["ROC AUC"])
        matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        log(f"  {name}: accuracy {rows[name]['Accuracy']:.3f}, recall {rows[name]['Recall']:.3f}")

    log("Classical baselines")
    for family, models, suffix in [
        (FULL, classical.full_feature_models(), ""),
        (REDUCED, classical.reduced_feature_models(config.N_QUBITS), PCA_SUFFIX),
    ]:
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
            record(name + suffix, family, model.fit(X_train, y_train), cv_scores)
    logreg = classical.full_feature_models()["Logistic Regression"].fit(X_train, y_train)

    log("Quantum-kernel SVM")
    qsvm = Pipeline(data.quantum_steps(config.N_QUBITS) + [("qsvm", QuantumKernelSVC())])
    search = GridSearchCV(qsvm, config.QSVM_GRID, cv=cv).fit(X_train, y_train)
    best = search.cv_results_["params"][search.best_index_]
    qsvm_cv = [search.cv_results_[f"split{i}_test_score"][search.best_index_]
               for i in range(config.CV_FOLDS)]
    log(f"  selected {best}")
    record("QSVM (quantum kernel)", QUANTUM, search.best_estimator_, qsvm_cv)

    log("Variational quantum classifier")
    vqc_runs, histories = [], []
    for seed in config.VQC_SEEDS:
        model = vqc_pipeline(seed=seed).fit(X_train, y_train)
        vqc_runs.append(evaluate(y_test, model.predict(X_test), malignant_score(model, X_test)))
        histories.append(model["vqc"].loss_history_)
        log(f"  seed {seed}: accuracy {vqc_runs[-1]['Accuracy']:.3f}")
        if seed == config.VQC_SEEDS[0]:
            first = model
    vqc_cv = None if quick else cross_val_score(vqc_pipeline(seed=0), X_train, y_train, cv=cv)
    # A single initialisation can be lucky or unlucky, so the VQC row of the
    # table is the mean over all seeds. The confusion matrix and ROC curve need
    # one concrete model and use the first seed, fixed in advance.
    record(VQC_ONE, QUANTUM, first, vqc_cv)
    seeds_df = pd.DataFrame(vqc_runs, index=pd.Index(config.VQC_SEEDS, name="seed"))
    rows[VQC_MEAN] = {**rows.pop(VQC_ONE), **seeds_df.mean().to_dict()}
    rows[VQC_MEAN].update({f"{m} std": seeds_df[m].std(ddof=0) for m in METRICS})
    families[VQC_MEAN] = families.pop(VQC_ONE)

    # Fitted pipelines for the demo app (raw 30 features in, prediction out).
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(logreg, config.MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(search.best_estimator_, config.MODEL_DIR / "qsvm.joblib")
    joblib.dump(first, config.MODEL_DIR / "vqc.joblib")

    table = pd.DataFrame(rows).T
    table.index.name = "Model"
    table.insert(0, "Family", pd.Series(families))
    table.to_csv(config.RESULTS_DIR / "metrics.csv")
    seeds_df.to_csv(config.RESULTS_DIR / "vqc_seeds.csv")

    plots.circuit_diagram(config.N_QUBITS, config.N_LAYERS)
    plots.training_curves(histories)
    shown = [n for n in matrices if not n.endswith(PCA_SUFFIX)]
    plots.confusion_matrices({n: matrices[n] for n in shown})
    plots.roc_curves(y_test, {n: scores[n] for n in shown},
                     "ROC curves: 30-feature baselines vs quantum models", "09_roc_curves.png")
    like_for_like = [n for n in matrices if n not in classical.full_feature_models()]
    plots.roc_curves(y_test, {n: scores[n] for n in like_for_like},
                     f"ROC curves: all models on the same {config.N_QUBITS} PCA components",
                     "10_roc_curves_pca.png")
    plots.model_comparison(table[METRICS].astype(float), families)

    ablation = None
    if not quick:
        log("VQC ablation")
        ablation = run_ablation(X_train, y_train)
        ablation.to_csv(config.RESULTS_DIR / "vqc_ablation.csv", index=False)
        plots.ablation(ablation)

    summary = {
        "n_samples": len(X), "n_train": len(X_train), "n_test": len(X_test),
        "class_counts": data.class_counts(y),
        "pca_explained_variance": explained[:config.N_QUBITS].tolist(),
        "pca_cumulative_variance": float(explained[:config.N_QUBITS].sum()),
        "qsvm_selected": best,
        "vqc_parameters": first["vqc"].n_parameters_,
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("\n" + table[METRICS + ["CV accuracy"]].astype(float).round(4).to_string())
    print("\nVQC over seeds:\n" + seeds_df[METRICS].round(4).to_string())
    if ablation is not None:
        print("\nAblation:\n" + ablation.round(4).to_string(index=False))
    log(f"Done. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="skip VQC cross-validation and the ablation study")
    main(parser.parse_args().quick)
