"""Matplotlib plotting helpers (UI-agnostic).

Each function accepts already-aggregated data and returns a
:class:`matplotlib.figure.Figure` instance, so the caller decides where
to render (notebook display, ``fig.savefig(...)``, Streamlit
``st.pyplot``, etc.). Matplotlib is imported lazily so importing
:mod:`deap_pettingzoo.analysis` does not pull it in.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: WPS433
    return plt


def plot_learning_curves(
    curves_by_algo: Mapping[str, Any],
    *,
    x_col: str = "env_steps",
    y_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "score",
    figsize: tuple = (7.0, 4.5),
) -> Any:
    """Overlay multiple algorithms' learning curves with CI bands.

    ``curves_by_algo`` maps each algorithm to a DataFrame as produced by
    :func:`learning_curve_with_ci` (columns ``x_col``, ``mean`` or
    ``iqm``, ``lo``, ``hi``).
    """
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=figsize)
    for algo, df in curves_by_algo.items():
        if df is None or len(df) == 0:
            continue
        # Auto-detect y column
        candidate = y_col
        if candidate is None:
            for c in ("iqm", "mean"):
                if c in df.columns:
                    candidate = c
                    break
        if candidate is None or candidate not in df.columns:
            continue
        xs = df[x_col].to_numpy()
        ys = df[candidate].to_numpy()
        lo = df["lo"].to_numpy() if "lo" in df.columns else ys
        hi = df["hi"].to_numpy() if "hi" in df.columns else ys
        line = ax.plot(xs, ys, label=algo, linewidth=2.0)[0]
        ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_eval_distribution(
    score_groups: Mapping[str, Sequence[float]],
    *,
    title: Optional[str] = None,
    ylabel: str = "score",
    figsize: tuple = (6.0, 4.5),
    show_points: bool = True,
) -> Any:
    """Box + strip plot of per-seed final scores by algorithm."""
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=figsize)
    labels = sorted(score_groups.keys())
    data = [
        np.asarray(list(score_groups[k]), dtype=float)[
            np.isfinite(np.asarray(list(score_groups[k]), dtype=float))
        ]
        for k in labels
    ]
    ax.boxplot(data, labels=labels, showfliers=False, widths=0.5)
    if show_points:
        rng = np.random.default_rng(0)
        for i, sample in enumerate(data, start=1):
            if sample.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=sample.size)
            ax.scatter(
                np.full(sample.size, i) + jitter,
                sample,
                alpha=0.7,
                s=18,
                edgecolors="black",
                linewidths=0.4,
            )
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_performance_profile(
    taus: np.ndarray,
    point_estimates: Mapping[str, np.ndarray],
    intervals: Optional[Mapping[str, np.ndarray]] = None,
    *,
    title: Optional[str] = None,
    figsize: tuple = (7.0, 4.5),
) -> Any:
    """Plot performance profiles (fraction of runs ≥ τ) per algorithm.

    The arguments match the return signature of
    :func:`rliable_bridge.performance_profile`. ``intervals`` is
    optional; when omitted, only the point estimate is shown.
    """
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=figsize)
    for algo, point in point_estimates.items():
        line = ax.plot(taus, np.asarray(point).ravel(), label=algo, linewidth=2.0)[0]
        if intervals is not None and algo in intervals:
            iv = np.asarray(intervals[algo])
            if iv.shape[0] >= 2:
                ax.fill_between(
                    taus,
                    iv[0].ravel(),
                    iv[1].ravel(),
                    alpha=0.2,
                    color=line.get_color(),
                )
    ax.set_xlabel(r"score threshold $\tau$")
    ax.set_ylabel(r"$P(\mathrm{score} \geq \tau)$")
    ax.set_ylim(-0.02, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pi_matrix(
    pi_table: Any,
    *,
    title: Optional[str] = None,
    figsize: tuple = (5.5, 5.0),
) -> Any:
    """Heatmap of the probability-of-improvement matrix.

    Accepts the long-form DataFrame returned by
    :func:`rliable_bridge.probability_of_improvement`.
    """
    plt = _lazy_plt()
    import pandas as pd  # noqa: WPS433

    if pi_table is None or len(pi_table) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    pivot = pd.pivot_table(
        pi_table, index="algo_x", columns="algo_y", values="probability"
    )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=8,
                )
    ax.set_xlabel("y (compared against)")
    ax.set_ylabel("x (target)")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$P(x > y)$", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
