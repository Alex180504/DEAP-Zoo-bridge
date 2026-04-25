"""Thin wrappers around the ``rliable`` package.

Every function in this module raises :class:`ImportError` with a clear
install hint when ``rliable`` is missing, rather than failing later
inside the wrapped call. Numerical results are passed through
unchanged so the rliable API contract is preserved.

References
----------
Agarwal, R. *et al.* (2021). *Deep Reinforcement Learning at the Edge
of the Statistical Precipice.* NeurIPS 2021.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _require_rliable():
    try:
        import rliable  # noqa: F401, WPS433
        from rliable import library, metrics  # noqa: WPS433
        return library, metrics
    except ImportError as exc:  # pragma: no cover - dep missing
        raise ImportError(
            "rliable is required for this function. Install with "
            "`pip install deap-pettingzoo[analysis]` or `pip install rliable`."
        ) from exc


def _stack_scores(score_groups: Mapping[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    """Convert ``{algo: [seed_scores...]}`` to ``{algo: ndarray[N, 1]}``.

    rliable's API expects a 2-D ``(n_runs, n_tasks)`` matrix; for
    single-task analysis we treat each algo's seeds as the run axis.
    """
    out: Dict[str, np.ndarray] = {}
    for algo, vals in score_groups.items():
        arr = np.asarray(list(vals), dtype=float).reshape(-1, 1)
        if arr.size == 0:
            continue
        out[algo] = arr
    return out


def aggregate_metrics_table(
    score_groups: Mapping[str, Sequence[float]],
    *,
    reps: int = 2000,
    confidence_interval_size: float = 0.95,
) -> Any:
    """rliable aggregate scores (mean, median, IQM, optimality gap).

    Returns a :class:`pandas.DataFrame` with one row per algorithm and
    columns ``mean``, ``median``, ``iqm``, ``optimality_gap``, each
    accompanied by ``_lo`` / ``_hi`` for the bootstrap CI.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    matrices = _stack_scores(score_groups)
    if not matrices:
        return pd.DataFrame()

    def aggregate_func(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [
                metrics.aggregate_mean(scores),
                metrics.aggregate_median(scores),
                metrics.aggregate_iqm(scores),
                metrics.aggregate_optimality_gap(scores),
            ]
        )

    point, intervals = library.get_interval_estimates(
        matrices,
        aggregate_func,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )

    cols = ["mean", "median", "iqm", "optimality_gap"]
    rows = []
    for algo in matrices:
        row = {"algorithm": algo}
        for i, name in enumerate(cols):
            row[name] = float(point[algo][i])
            row[f"{name}_lo"] = float(intervals[algo][0, i])
            row[f"{name}_hi"] = float(intervals[algo][1, i])
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm")


def performance_profile(
    score_groups: Mapping[str, Sequence[float]],
    *,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    n_tau: int = 50,
    reps: int = 2000,
    confidence_interval_size: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute performance profiles ``P(score >= tau)`` per algorithm.

    Returns ``(taus, point_estimates, intervals)`` where
    ``point_estimates[algo]`` is a 1-D array of length ``n_tau`` and
    ``intervals[algo]`` is a ``(2, n_tau)`` array of lo/hi bounds.
    """
    library, _ = _require_rliable()
    matrices = _stack_scores(score_groups)
    if not matrices:
        return np.empty(0), {}, {}

    all_scores = np.concatenate([m.ravel() for m in matrices.values()])
    finite = all_scores[np.isfinite(all_scores)]
    if tau_min is None:
        tau_min = float(np.min(finite))
    if tau_max is None:
        tau_max = float(np.max(finite))
    taus = np.linspace(tau_min, tau_max, n_tau)

    point, intervals = library.create_performance_profile(
        matrices,
        taus,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )
    return taus, point, intervals


def probability_of_improvement(
    score_groups: Mapping[str, Sequence[float]],
    *,
    reps: int = 2000,
) -> Any:
    """Pairwise P(X > Y) matrix with bootstrap CIs.

    Returns a :class:`pandas.DataFrame` with one row per ordered pair
    ``(algo_x, algo_y)``: ``probability``, ``ci_lo``, ``ci_hi``.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    matrices = _stack_scores(score_groups)
    keys = sorted(matrices.keys())

    pairs: Dict[str, tuple] = {}
    for ka in keys:
        for kb in keys:
            if ka == kb:
                continue
            pairs[f"{ka}__vs__{kb}"] = (matrices[ka], matrices[kb])

    if not pairs:
        return pd.DataFrame()

    point, intervals = library.get_interval_estimates(
        pairs,
        metrics.probability_of_improvement,
        reps=reps,
    )

    rows = []
    for label, (ka_kb,) in zip(pairs.keys(), [(p,) for p in pairs.keys()]):  # noqa: B905
        ka, kb = label.split("__vs__")
        rows.append(
            {
                "algo_x": ka,
                "algo_y": kb,
                "probability": float(np.asarray(point[label]).ravel()[0]),
                "ci_lo": float(np.asarray(intervals[label][0]).ravel()[0]),
                "ci_hi": float(np.asarray(intervals[label][1]).ravel()[0]),
            }
        )
    return pd.DataFrame(rows)


def sample_efficiency_curve(
    runs_by_algo: Mapping[str, Sequence[Any]],
    *,
    x_col: str = "env_steps",
    y_col: str = "best",
    n_points: int = 25,
    reps: int = 1000,
    confidence_interval_size: float = 0.95,
) -> Any:
    """IQM-vs-budget curves with bootstrap CI bands.

    ``runs_by_algo`` maps an algorithm name to an iterable of
    :class:`~deap_pettingzoo.analysis.loaders.RunRecord`. Returns a
    long-form :class:`pandas.DataFrame` with columns ``algorithm``,
    ``x``, ``iqm``, ``lo``, ``hi``.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    # Determine a common x grid in env_steps spanning all runs.
    x_values: list = []
    series_by_algo: Dict[str, list] = {}
    for algo, runs in runs_by_algo.items():
        kept = []
        for r in runs:
            if r.metrics is None:
                continue
            cols = set(r.metrics.columns)
            if x_col not in cols or y_col not in cols:
                continue
            df = r.metrics[[x_col, y_col]].dropna().sort_values(x_col)
            if df.empty:
                continue
            kept.append(df)
            x_values.extend(df[x_col].tolist())
        if kept:
            series_by_algo[algo] = kept

    if not series_by_algo:
        return pd.DataFrame()

    x_grid = np.linspace(min(x_values), max(x_values), n_points)

    def _interp(df) -> np.ndarray:
        # Step interpolation: forward-fill at each grid point.
        xs = df[x_col].to_numpy(dtype=float)
        ys = df[y_col].to_numpy(dtype=float)
        out = np.empty(x_grid.size, dtype=float)
        for i, xv in enumerate(x_grid):
            idx = np.searchsorted(xs, xv, side="right") - 1
            out[i] = ys[idx] if idx >= 0 else float("nan")
        return out

    matrices: Dict[str, np.ndarray] = {}
    for algo, dfs in series_by_algo.items():
        matrices[algo] = np.stack([_interp(df) for df in dfs], axis=0)

    point, intervals = library.get_interval_estimates(
        matrices,
        metrics.aggregate_iqm,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )

    rows = []
    for algo in matrices:
        for i, xv in enumerate(x_grid):
            rows.append(
                {
                    "algorithm": algo,
                    "x": float(xv),
                    "iqm": float(np.asarray(point[algo]).ravel()[i]),
                    "lo": float(np.asarray(intervals[algo][0]).ravel()[i]),
                    "hi": float(np.asarray(intervals[algo][1]).ravel()[i]),
                }
            )
    return pd.DataFrame(rows)
