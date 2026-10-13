"""Figures. One function per figure; each saves a PNG into results/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pennylane as qml
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

from . import config
from .data import CLASS_NAMES
from .vqc import encode, variational_layer

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots, always assigned in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASS_COLORS = {"Benign": SERIES[0], "Malignant": SERIES[1]}

SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef5fd", "#86b6ef", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name)
    plt.close(fig)


def class_distribution(counts):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    total = sum(counts.values())
    bars = ax.bar(list(counts), list(counts.values()), width=0.5,
                  color=[CLASS_COLORS[c] for c in counts])
    for bar, n in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 6, f"{n} ({n / total:.1%})",
                ha="center", color=INK)
    ax.set(title="Class distribution (N = 569)", ylabel="Samples",
           ylim=(0, max(counts.values()) * 1.15))
    ax.grid(axis="x", visible=False)
    _save(fig, "01_class_distribution.png")


def target_correlation(corr, top=12):
    corr = corr.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(corr.index, corr.values, height=0.6, color=SERIES[0])
    for i, v in enumerate(corr.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set(title=f"Top {top} features by correlation with malignancy",
           xlabel="Point-biserial correlation r", xlim=(0, 1))
    ax.grid(axis="y", visible=False)
    _save(fig, "02_target_correlation.png")


def correlation_heatmap(X):
    cols = [c for c in X.columns if c.startswith("mean ")]
    corr = X[cols].corr().values
    labels = [c.replace("mean ", "") for c in cols]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Pearson correlation between the ten mean-value features")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    _save(fig, "03_feature_correlation.png")


def pca_variance(explained, n_used):
    cum = np.cumsum(explained)
    k = np.arange(1, len(explained) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(k, cum, color=SERIES[0], marker="o", markersize=4)
    ax.axvline(n_used, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"{n_used} components kept\n{cum[n_used - 1]:.1%} of variance",
                xy=(n_used, cum[n_used - 1]), xytext=(n_used + 1.5, cum[n_used - 1] - 0.22),
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set(title="Cumulative variance explained by principal components",
           xlabel="Number of principal components", ylabel="Cumulative explained variance",
           ylim=(0.3, 1.02), xlim=(0.5, 15.5))
    _save(fig, "04_pca_variance.png")


def pca_scatter(Z, y):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for label, name in enumerate(CLASS_NAMES):
        pts = Z[np.asarray(y) == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=CLASS_COLORS[name], label=name,
                   alpha=0.8, edgecolor=SURFACE, linewidth=0.6)
    ax.set(title="Training set in the first two principal components",
           xlabel="PC 1", ylabel="PC 2")
    ax.legend()
    _save(fig, "05_pca_scatter.png")


def circuit_diagram(n_qubits, n_layers):
    def circuit(weights, x):
        encode(x, n_qubits)
        for layer_weights in weights:
            variational_layer(layer_weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    fig, _ = qml.draw_mpl(circuit)(np.zeros((n_layers, n_qubits, 3)), np.zeros(n_qubits))
    fig.set_facecolor("white")
    fig.suptitle(f"VQC: RY angle encoding, {n_layers} variational layers, <Z0> read-out",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    _save(fig, "06_vqc_circuit.png")


def training_curves(histories):
    H = np.array(histories)
    epochs = np.arange(1, H.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for h in H:
        ax.plot(epochs, h, color=SERIES[0], linewidth=1, alpha=0.35)
    ax.plot(epochs, H.mean(axis=0), color=SERIES[0], label=f"Mean of {len(H)} seeds")
    ax.plot([], [], color=SERIES[0], linewidth=1, alpha=0.35, label="Individual seeds")
    ax.set(title="VQC training loss", xlabel="Epoch",
           ylabel="Binary cross-entropy (training set)", ylim=(0, None))
    ax.legend()
    _save(fig, "07_vqc_training.png")


def confusion_matrices(matrices):
    """`matrices` maps a model name to its 2x2 confusion matrix (rows = truth)."""
    n = len(matrices)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows))
    vmax = max(m.max() for m in matrices.values())
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > vmax / 2 else INK)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES, rotation=90, va="center")
        ax.set_title(name, fontsize=10)
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("Predicted", color=INK_2, fontsize=10)
    fig.supylabel("True", color=INK_2, fontsize=10)
    fig.tight_layout()
    _save(fig, "08_confusion_matrices.png")


def roc_curves(y_true, scores, title, filename):
    """`scores` maps a model name to its continuous malignancy score on the test set."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    for color, (name, (score, auc)) in zip(SERIES, scores.items()):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {auc:.3f})")
    ax.set(title=title, xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)", xlim=(-0.01, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, filename)


def model_comparison(table, families):
    """Dot plot of the four headline metrics, one panel each, coloured by model family."""
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    family_color = dict(zip(dict.fromkeys(families.values()), SERIES))
    names = list(table.index)[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 0.36 * len(names) + 1.6), sharey=True)
    lo = np.floor(table[metrics].values.min() * 20) / 20 - 0.01
    for ax, metric in zip(axes, metrics):
        for i, name in enumerate(names):
            v = table.loc[name, metric]
            ax.plot([lo, v], [i, i], color=GRID, linewidth=1.5, zorder=1)
            ax.scatter(v, i, s=55, color=family_color[families[name]], zorder=2,
                       edgecolor=SURFACE, linewidth=1)
            ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set(title=metric, xlim=(lo, 1.035))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f)
               for f, c in family_color.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Hold-out test performance by model", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_model_comparison.png")


def ablation(df):
    """Validation accuracy against circuit depth, with and without data re-uploading."""
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for color, (label, grp) in zip(SERIES, df.groupby("encoding", sort=False)):
        ax.errorbar(grp["layers"], grp["mean"], yerr=grp["std"], color=color, marker="o",
                    markersize=6, capsize=3, label=label)
    ax.set(title="VQC ablation: depth and encoding strategy",
           xlabel="Variational layers L", ylabel="Validation accuracy (mean ± s.d., 3 seeds)",
           xticks=sorted(df["layers"].unique()))
    ax.legend(loc="lower right")
    _save(fig, "12_vqc_ablation.png")


def operating_points(df):
    """Sensitivity and specificity per model/threshold. `df` has a 'group' column."""
    groups = list(dict.fromkeys(df["group"]))
    color = dict(zip(groups, SERIES))
    names = list(df.index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.42 * len(names) + 1.7), sharey=True)
    for ax, metric in zip(axes, ["Sensitivity", "Specificity"]):
        lo = min(0.75, df[metric].min() - 0.03)
        for i, name in enumerate(names):
            v = df.loc[name, metric]
            ax.plot([lo, v], [i, i], color=GRID, linewidth=1.5, zorder=1)
            ax.scatter(v, i, s=55, color=color[df.loc[name, "group"]], zorder=2,
                       edgecolor=SURFACE, linewidth=1)
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set(title=metric, xlim=(lo, 1.05))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g)
               for g, c in color.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Test-set operating points: single runs, ensemble and calibrated threshold",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "13_operating_points.png")


def gate_noise(df):
    """Accuracy (as trained vs read-out recalibrated) and AUC against CNOT error rate."""
    x = np.arange(len(df))
    labels = [f"{p:g}" for p in df["cnot_error"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, (col, label) in zip(SERIES, [("acc_raw", "Read-out as trained"),
                                           ("acc_recal", "Read-out recalibrated")]):
        axes[0].errorbar(x, df[f"{col}_mean"], yerr=df[f"{col}_std"], color=color,
                         marker="o", markersize=6, capsize=3, label=label)
    axes[0].set(title="Test accuracy under gate noise", ylabel="Accuracy (mean ± s.d., 5 seeds)")
    axes[0].legend(loc="lower left")
    axes[1].errorbar(x, df["auc_mean"], yerr=df["auc_std"], color=SERIES[2], marker="o",
                     markersize=6, capsize=3)
    axes[1].set(title="Ranking quality under gate noise", ylabel="ROC AUC (mean ± s.d., 5 seeds)",
                ylim=(min(0.9, df["auc_mean"].min() - 0.02), 1.005))
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("CNOT depolarising error rate (single-qubit gates: one tenth)")
    fig.tight_layout()
    _save(fig, "14_gate_noise.png")


def shot_noise(df):
    """Accuracy against the number of measurement shots, one line per gate-noise level."""
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for color, (level, grp) in zip(SERIES, df.groupby("cnot_error", sort=True)):
        ax.plot(grp["shots"], grp["acc_mean"], color=color, marker="o", markersize=6,
                label=f"CNOT error {level:g}")
        ax.fill_between(grp["shots"], grp["acc_mean"] - grp["acc_std"],
                        grp["acc_mean"] + grp["acc_std"], color=color, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set(title="Test accuracy against measurement shots per patient",
           xlabel="Shots (circuit repetitions per patient)",
           ylabel="Accuracy (mean ± s.d. over repetitions)")
    ax.legend(loc="lower right")
    _save(fig, "15_shot_noise.png")


def global_attribution(df):
    """Share of total attribution per cell-nucleus quantity, one bar per model."""
    df = df.sort_values(df.columns[0])
    y = np.arange(len(df))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for k, (color, col) in enumerate(zip(SERIES, df.columns)):
        pos = y + (0.5 - k) * (h + 0.04)
        ax.barh(pos, df[col], height=h, color=color, label=col)
        for yy, v in zip(pos, df[col]):
            ax.text(v + 0.002, yy, f"{v:.1%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, df.index)
    ax.set(title="What the models rely on: share of total attribution",
           xlabel="Share of mean |attribution| over test patients",
           xlim=(0, df.values.max() * 1.12))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, "16_global_attribution.png")


def patient_attribution(attr, title, top=10):
    """Signed attributions for one patient. `attr` is a Series indexed by feature."""
    attr = attr.reindex(attr.abs().sort_values(ascending=False).index).head(top)[::-1]
    towards = {"Towards malignant": "#e34948", "Towards benign": SERIES[0]}
    colors = [towards["Towards malignant" if v > 0 else "Towards benign"] for v in attr]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(attr.index, attr.values, height=0.6, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set(title=title, xlabel="Contribution to the logit (integrated gradients)")
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l)
                        for l, c in towards.items()], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08))
    _save(fig, "17_patient_attribution.png")


def fair_comparison(folds, order, highlight, reference):
    """Per-fold AUC and accuracy for every model (dots) with the mean (bar)."""
    names = order[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.45 * len(names) + 1.6), sharey=True)
    metrics = ["AUC", "Accuracy", "Sensitivity at 95% specificity"]
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, metrics):
        for i, name in enumerate(names):
            v = folds.loc[folds["model"] == name, metric].values
            color = SERIES[2] if name == highlight else (MUTED if name == reference else SERIES[0])
            ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=12, color=color, alpha=0.55,
                       linewidth=0)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=color, linewidth=2.5)
            ax.text(1.002, i, f"{v.mean():.3f}", va="center", fontsize=8, color=INK,
                    transform=ax.get_yaxis_transform())
        ax.set_title(metric)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle("Repeated cross-validation: 15 folds per model, same 4 principal components",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "18_fair_comparison.png")


def reliability(probas, y, bins=10):
    """Reliability diagram: observed malignancy rate against predicted probability."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    edges = np.linspace(0, 1, bins + 1)
    for color, (name, p) in zip(SERIES, probas.items()):
        which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        xs = [p[which == b].mean() for b in range(bins) if np.any(which == b)]
        ys = [y[which == b].mean() for b in range(bins) if np.any(which == b)]
        ax.plot(xs, ys, color=color, marker="o", markersize=5, label=name)
    ax.set(title="Calibration of out-of-fold probabilities", xlabel="Predicted P(malignant)",
           ylabel="Observed fraction malignant", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "19_reliability.png")


def noise_linearity(df):
    """Left: contraction factor against error strength. Right: rank preservation (AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for color, (kind, grp) in zip(SERIES, df.groupby("kind", sort=False)):
        axes[0].plot(grp["strength"], grp["r2"], color=color, marker="o", markersize=5,
                     label=kind)
        axes[1].plot(grp["strength"], grp["auc"], color=color, marker="o", markersize=5,
                     label=kind)
    axes[0].set(title="Is the noisy output an affine copy of the clean one?",
                xlabel="Error strength", ylabel="R² of noisy on clean ⟨Z₀⟩")
    axes[1].set(title="Ranking of test patients", xlabel="Error strength", ylabel="ROC AUC")
    axes[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, "20_noise_mechanism.png")


def shot_collapse(df):
    """Accuracy against shots and against effective shots N·λ²."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for color, (p, grp) in zip(SERIES, df.groupby("cnot_error", sort=True)):
        lam = grp["lambda"].iloc[0]
        for ax, x in zip(axes, [grp["shots"], grp["shots"] * lam ** 2]):
            ax.plot(x, grp["acc_mean"], color=color, marker="o", markersize=5,
                    label=f"CNOT error {p:g} (λ = {lam:.2f})")
    axes[0].set(title="Accuracy against shots", xlabel="Shots per patient", xscale="log",
                ylabel="Test accuracy (recalibrated)")
    axes[1].set(title="Against effective shots N·λ²", xlabel="N·λ²", xscale="log")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "21_shot_collapse.png")


def deletion_curves(curves):
    """Fraction of the prediction remaining as features are reset to the baseline."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for color, (name, c) in zip(SERIES, curves.items()):
        ax.plot(np.arange(len(c)), c, color=color, label=name)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set(title="Faithfulness: deleting the most important features first",
           xlabel="Features reset to the average patient",
           ylabel="Median fraction of logit change remaining", xlim=(0, len(c) - 1))
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "22_deletion_curves.png")


def explanation_agreement(shares):
    """Share of attribution per nucleus property under three explanation methods."""
    shares = shares.sort_values(shares.columns[0])
    y = np.arange(len(shares))
    h = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for k, (color, col) in enumerate(zip(SERIES, shares.columns)):
        ax.barh(y + (1 - k) * (h + 0.02), shares[col], height=h, color=color, label=col)
    ax.set_yticks(y, shares.index)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set(title="Two explanation methods for the VQC, and logistic regression",
           xlabel="Share of mean |attribution| over test patients")
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.12))
    _save(fig, "23_explanation_agreement.png")
