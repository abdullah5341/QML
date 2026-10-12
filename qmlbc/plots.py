"""Figures. One function per figure; each saves a PNG into results/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
