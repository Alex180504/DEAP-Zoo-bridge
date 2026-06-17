"""Pairwise comparison utilities.

All functions accept two 1-D samples and return a small dataclass-like
:class:`dict`. The non-parametric helpers (:func:`mann_whitney`,
:func:`cliffs_delta`) are pure-numpy implementations so they work
without ``scipy``; :func:`welch_t` lazy-imports ``scipy.stats`` and
falls back to a manual implementation if scipy is unavailable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from deap_pettingzoo.analysis.loaders import Experiment

# Effect-size helpers


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's *d* with pooled standard deviation.

    Returns ``nan`` when both samples have zero variance or fewer than
    two observations combined.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0.0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta (non-parametric effect size).

    Defined as ``(#(x>y) - #(x<y)) / (n_x * n_y)``. Returns ``nan`` for
    empty inputs. Result lies in ``[-1, 1]``: positive favours ``x``.
    Uses a vectorised O(n_x * n_y) computation; fine for the seed-count
    sizes typical in MARL experiments.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / float(a.size * b.size)


# Hypothesis tests


def mann_whitney(
    x: Sequence[float],
    y: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Mann-Whitney U test using :func:`scipy.stats.mannwhitneyu` if
    available; falls back to a normal-approximation implementation.

    Returns a dict ``{statistic, p_value, n_x, n_y, effect_cliffs_delta}``.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_x": int(a.size),
            "n_y": int(b.size),
            "effect_cliffs_delta": float("nan"),
        }
    try:
        from scipy import stats  # noqa: WPS433
        res = stats.mannwhitneyu(a, b, alternative=alternative)
        stat = float(res.statistic)
        p = float(res.pvalue)
    except ImportError:  # pragma: no cover - scipy missing
        stat, p = _mannwhitney_fallback(a, b, alternative=alternative)
    return {
        "statistic": stat,
        "p_value": p,
        "n_x": int(a.size),
        "n_y": int(b.size),
        "effect_cliffs_delta": cliffs_delta(a, b),
    }


def _mannwhitney_fallback(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> tuple:
    """Normal-approximation Mann-Whitney U for the no-scipy path."""
    n1, n2 = a.size, b.size
    combined = np.concatenate([a, b])
    ranks = _rankdata(combined)
    u1 = float(np.sum(ranks[:n1])) - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0.0:
        return float(u), float("nan")
    z = (u - mu) / sigma
    # Two-sided p-value via std normal CDF; one-sided variants halve.
    p_two = math.erfc(abs(z) / math.sqrt(2.0))
    if alternative == "two-sided":
        p = p_two
    else:
        p = p_two / 2.0
    return float(u1), float(p)


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-rank with tie handling, matching scipy.stats.rankdata."""
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(arr.size)
    sorted_arr = arr[sorter]
    obs = np.concatenate([[True], sorted_arr[1:] != sorted_arr[:-1]])
    dense = obs.cumsum()[inv]
    # Average ranks on ties
    count = np.concatenate([np.nonzero(obs)[0], [len(obs)]])
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def welch_t(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Welch's t-test (unequal variances).

    Returns ``{t, p_value, df, n_x, n_y, effect_cohens_d}``. Requires
    scipy for the exact p-value; without scipy, a normal-approximation
    p-value is returned.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {
            "t": float("nan"),
            "p_value": float("nan"),
            "df": float("nan"),
            "n_x": int(a.size),
            "n_y": int(b.size),
            "effect_cohens_d": float("nan"),
        }
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(va / a.size + vb / b.size)
    if se == 0.0:
        return {
            "t": float("nan"),
            "p_value": float("nan"),
            "df": float("nan"),
            "n_x": int(a.size),
            "n_y": int(b.size),
            "effect_cohens_d": cohens_d(a, b),
        }
    t = (ma - mb) / se
    df = (va / a.size + vb / b.size) ** 2 / (
        (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1)
    )
    try:
        from scipy import stats  # noqa: WPS433
        p = float(2.0 * stats.t.sf(abs(t), df))
    except ImportError:  # pragma: no cover
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {
        "t": float(t),
        "p_value": float(p),
        "df": float(df),
        "n_x": int(a.size),
        "n_y": int(b.size),
        "effect_cohens_d": cohens_d(a, b),
    }


# Pairwise tables


def pairwise_table(
    experiment: Experiment,
    *,
    metric: str = "final_fitness",
    test: str = "mann_whitney",
    algorithms: Optional[Sequence[str]] = None,
) -> Any:
    """Build a long-form pairwise comparison table.

    For every unordered pair of algorithms with at least two seeds each,
    runs the requested ``test`` and records the effect size.

    Returns a :class:`pandas.DataFrame` with columns: ``algo_x``,
    ``algo_y``, ``n_x``, ``n_y``, ``statistic``, ``p_value``, ``effect``,
    ``effect_name``.
    """
    import pandas as pd  # noqa: WPS433

    if test not in {"mann_whitney", "welch_t"}:
        raise ValueError(f"test must be 'mann_whitney' or 'welch_t', got {test!r}")

    groups = experiment.final_scores(metric=metric)
    if algorithms is not None:
        groups = {k: v for k, v in groups.items() if k in algorithms}
    keys = sorted(groups.keys())

    rows: list = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            xs, ys = groups[ka], groups[kb]
            if test == "mann_whitney":
                res = mann_whitney(xs, ys)
                effect = res["effect_cliffs_delta"]
                effect_name = "cliffs_delta"
            else:
                res = welch_t(xs, ys)
                effect = res["effect_cohens_d"]
                effect_name = "cohens_d"
            stat_key = "statistic" if test == "mann_whitney" else "t"
            rows.append(
                {
                    "algo_x": ka,
                    "algo_y": kb,
                    "n_x": res["n_x"],
                    "n_y": res["n_y"],
                    "statistic": res[stat_key],
                    "p_value": res["p_value"],
                    "effect": effect,
                    "effect_name": effect_name,
                }
            )
    return pd.DataFrame(rows)
