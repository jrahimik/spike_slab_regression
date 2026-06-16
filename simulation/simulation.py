
"""
simulation.py
=======

Here I am running the full simulation study: I define the four sweep scenarios,
run them all through the ExperimentRunner, then emit a summary table and a
multi-panel figure.

Run from the repo root (with the ``spikeslab`` package importable):
    python main.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from Model import SpikeSlabVI
import matplotlib.pyplot as plt

from trainer import train
import torch
from torch import tensor

# Define your matrix dimensions
n = 500  # Number of rows
p = 40  # Number of columns

# 1. Initialize the modern random number generator
rng = np.random.default_rng(seed=42)  # Optional: Seed ensures reproducibility

# 2. Generate an n x p matrix from a standard normal distribution (mean=0, std=1)
X = rng.normal(loc=0.0, scale=1.0, size=(n, p))

beta_sim=np.zeros(p)  
beta_sim[0]=2
beta_sim[1]=3
beta_sim[2]=4
beta_sim[9]=10

sigma_o= rng.normal(loc=0.0, scale=1, size=())

y =X@beta_sim+sigma_o


X_t=tensor(X,dtype=torch.float32)
y_t=tensor(y,dtype=torch.float32)

model=SpikeSlabVI(X_t,obs_std=2)
result =train(model,X_t,y_t)

plt.plot(result)
plt.xlabel("step")
plt.ylabel("loss")
plt.show()


pip = model.get_inclusion_prob()
print(pip)


from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap


ArrayLike = Union[Sequence[float], np.ndarray]


# Publication-oriented style
_TEXT = "#222222"
_GRID = "#FFFFFF"
_CMAP = LinearSegmentedColormap.from_list(
    "pub_teal",
    ["#F7FBFC", "#D6ECF1", "#9FD3DE", "#5FB3C3", "#247F96", "#0F4C5C"]
)

_PUB_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9.5,

    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": _TEXT,
    "axes.labelcolor": _TEXT,
    "axes.titleweight": "regular",

    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": _TEXT,
    "ytick.color": _TEXT,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "legend.fontsize": 9.0,
    "legend.frameon": False,

    # Keep text editable in vector export
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 160,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def _to_numpy_1d(x: ArrayLike) -> np.ndarray:
    """Convert NumPy / list-like / Torch-like input to a flat NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x).ravel()


def _validate_inputs(
    beta_true: ArrayLike,
    pip: ArrayLike,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate and prepare inputs."""
    beta_true = _to_numpy_1d(beta_true).astype(float)
    pip = _to_numpy_1d(pip).astype(float)
    threshold = float(threshold)

    if beta_true.size != pip.size:
        raise ValueError(
            "`beta_true` and `pip` must have the same length. "
            f"Got {beta_true.size} and {pip.size}."
        )

    if beta_true.size == 0:
        raise ValueError("Inputs must be non-empty.")

    if not np.all(np.isfinite(beta_true)):
        raise ValueError("`beta_true` contains non-finite values.")

    if not np.all(np.isfinite(pip)):
        raise ValueError("`pip` contains non-finite values.")

    if np.min(pip) < -1e-8 or np.max(pip) > 1 + 1e-8:
        raise ValueError("`pip` values must lie in [0, 1].")

    if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError("`threshold` must lie in [0, 1].")

    pip = np.clip(pip, 0.0, 1.0)

    return beta_true, pip, threshold


def plot_confusion_matrix(
    beta_true: ArrayLike,
    pip: ArrayLike,
    threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = "Feature-selection confusion matrix",
    show: bool = True,
) -> Tuple[np.ndarray, dict, Figure]:
    """
    Build and plot a publication-quality confusion matrix for feature selection.

    Parameters
    ----------
    beta_true
        True coefficient vector. Nonzero entries are treated as active features.
    pip
        Posterior inclusion probabilities.
    threshold
        PIP cutoff used to classify a feature as selected.
    save_path
        Optional output path. Prefer .pdf or .svg for publication use.
    title
        Optional figure title. Set to None for caption-only figures.
    show
        Whether to display the figure.

    Returns
    -------
    cm : np.ndarray
        2x2 confusion matrix:
            rows    = actual class   [Inactive, Active]
            columns = predicted      [Rejected, Selected]
    metrics : dict
        Precision, recall, F1, accuracy, specificity.
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    beta_true, pip, threshold = _validate_inputs(beta_true, pip, threshold)

    true_active = beta_true != 0
    pred_active = pip >= threshold

    TP = int(np.sum(true_active & pred_active))
    FN = int(np.sum(true_active & ~pred_active))
    FP = int(np.sum(~true_active & pred_active))
    TN = int(np.sum(~true_active & ~pred_active))

    cm = np.array([
        [TN, FP],   # actual inactive
        [FN, TP],   # actual active
    ])

    total = int(cm.sum())

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    accuracy = (TP + TN) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=False)

        im = ax.imshow(cm, cmap=_CMAP, aspect="equal")

        # Annotate cells with count and percent
        labels = np.array([
            ["True negative", "False positive"],
            ["False negative", "True positive"],
        ])

        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = 100.0 * count / total if total else 0.0
                shade = im.norm(count)
                txt_color = "white" if shade > 0.50 else "#0F4C5C"

                ax.text(
                    j, i - 0.08, f"{count}",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color=txt_color
                )
                ax.text(
                    j, i + 0.16, labels[i, j],
                    ha="center", va="center",
                    fontsize=8.7, color=txt_color
                )
                ax.text(
                    j, i + 0.33, f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=8.2, color=txt_color, alpha=0.85
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Rejected", "Selected"])
        ax.set_yticklabels(["Inactive", "Active"])

        ax.set_xlabel("Model prediction", labelpad=8)
        ax.set_ylabel("Ground truth", labelpad=8)

        if title is not None:
            ax.set_title(title, pad=10)

        # White separators between cells
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=_GRID, linewidth=2.5)
        ax.tick_params(which="both", length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Metrics strip
        metrics_text = (
            rf"Precision = {precision:.2f}   ·   "
            rf"Recall = {recall:.2f}   ·   "
            rf"Specificity = {specificity:.2f}   ·   "
            rf"F1 = {f1:.2f}   ·   "
            rf"Accuracy = {accuracy:.2f}"
        )
        fig.text(
            0.5, 0.02, metrics_text,
            ha="center", va="bottom",
            fontsize=9.2, color="#0F4C5C", fontweight="semibold"
        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
        cbar.set_label("Feature count", fontsize=9.5)
        cbar.outline.set_visible(False)

        fig.subplots_adjust(bottom=0.18, top=0.88)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return cm, metrics, fig
