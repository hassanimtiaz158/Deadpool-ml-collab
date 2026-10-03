"""Shared plotting helpers for notebooks and the evaluation report."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from fruit_classification import config  # noqa: E402

PALETTE = "husl"


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig: plt.Figure, name: str) -> str:
    """Save a figure into ``reports/figures`` and return the relative path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(config.PROJECT_ROOT))


def plot_class_distribution(counts: pd.Series, title: str, name: str, top: int | None = 30):
    counts = counts.sort_values(ascending=False)
    shown = counts if top is None else counts.head(top)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.22 * len(shown) + 1.5)))
    ax.barh(range(len(shown)), shown.values, color=sns.color_palette(PALETTE, len(shown)))
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_title(title)
    return fig, save_figure(fig, name)


def plot_confusion_matrix(labels_file="confusion_matrix_labels.json", matrix_file="confusion_matrix.npy", top: int = 30):
    cm = np.load(config.FIGURES_DIR / matrix_file)
    labels = json.loads((config.FIGURES_DIR / labels_file).read_text(encoding="utf-8"))
    totals = cm.sum(axis=1)
    order = np.argsort(totals)[::-1][:top]
    sub = cm[np.ix_(order, order)]
    sub_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(1 + 0.28 * top, 1 + 0.28 * top))
    sns.heatmap(
        np.log1p(sub),
        cmap="mako",
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        ax=ax,
        cbar_kws={"label": "log(1 + images)"},
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(f"Confusion matrix - {top} most frequent test classes")
    return fig, save_figure(fig, "confusion_matrix.png")


def plot_metrics_bars(metrics: dict, name: str = "metrics_overview.png"):
    rows = []
    for split in ("val", "test"):
        for key, value in metrics.get(split, {}).items():
            if isinstance(value, (int, float)):
                rows.append({"split": split, "metric": key, "value": value})
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, frame["split"].unique()):
        sub = frame[frame["split"] == split]
        ax.barh(sub["metric"], sub["value"], color=sns.color_palette(PALETTE, len(sub)))
        ax.set_title(f"{split} split")
        ax.set_xlim(0, 1.0)
    axes[0].invert_yaxis()
    fig.suptitle("Model performance")
    return fig, save_figure(fig, name)