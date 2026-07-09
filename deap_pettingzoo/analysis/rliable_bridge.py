"""Thin wrappers around the ``rliable`` package.

Every function in this module raises :class:`ImportError` with a clear
install hint when ``rliable`` is missing, rather than failing later
inside the wrapped call. Numerical results are passed through
unchanged so that the rliable API contract is preserved.
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


def _stack_score_matrices(
    score_matrices: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Validate and pass through canonical ``(n_runs, n_tasks)`` matrices.

    Accepts the multi-task layout produced by
    :func:`deap_pettingzoo.analysis.collect_score_matrices`. Empty
    algorithms are dropped; 1-D inputs are reshaped to a single task
    column for symmetry with :func:`_stack_scores`.
    """
    out: Dict[str, np.ndarray] = {}
    for algo, mat in score_matrices.items():
        arr = np.asarray(mat, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"score matrix for {algo!r} must be 1-D or 2-D, got "
                f"shape {arr.shape}"
            )
        out[algo] = arr
    return out


def _resolve_matrices(
    score_groups: Optional[Mapping[str, Sequence[float]]],
    score_matrices: Optional[Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Pick the multi-task ``score_matrices`` path when given, else fall back.

    Exactly one of ``score_groups`` / ``score_matrices`` must be
    non-``None``.
    """
    if score_matrices is not None and score_groups is not None:
        raise ValueError(
            "pass either score_groups (single-task) or score_matrices "
            "(multi-task), not both"
        )
    if score_matrices is not None:
        return _stack_score_matrices(score_matrices)
    if score_groups is not None:
        return _stack_scores(score_groups)
    raise ValueError("one of score_groups or score_matrices is required")


def aggregate_metrics_table(
    score_groups: Optional[Mapping[str, Sequence[float]]] = None,
    *,
    score_matrices: Optional[Mapping[str, np.ndarray]] = None,
    reps: int = 2000,
    confidence_interval_size: float = 0.95,
    include_optimality_gap: Optional[bool] = None,
) -> Any:
    """rliable aggregate scores (mean, median, IQM, optimality gap).

    Either pass ``score_groups`` (``{algo: [seed_scores...]}``,
    single-task) **or** ``score_matrices`` (``{algo: ndarray[n_runs,
    n_tasks]}``, multi-task). Returns a :class:`pandas.DataFrame` with
    one row per algorithm and columns ``mean``, ``median``, ``iqm`` (and
    ``optimality_gap`` when included), each accompanied by ``_lo`` /
    ``_hi`` for the bootstrap CI.

    Parameters
    ----------
    include_optimality_gap:
        Whether to compute ``optimality_gap``. rliable's optimality gap
        ``gamma - mean(min(scores, gamma))`` is only meaningful on
        scores normalised so ``1.0`` is optimal; on raw scores ``>= 1``
        it collapses to a silent ``0.0``. When ``None`` (default) it is
        **omitted for the single-task** ``score_groups`` path (raw,
        un-normalised) and **included for the multi-task**
        ``score_matrices`` path (which is normalised before this call).
        Pass an explicit bool to override.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    matrices = _resolve_matrices(score_groups, score_matrices)
    if not matrices:
        return pd.DataFrame()

    if include_optimality_gap is None:
        # Single-task scores are raw, so the optimality gap is meaningless
        # there; the multi-task path normalises upstream, so keep it.
        include_optimality_gap = score_matrices is not None

    funcs = [
        metrics.aggregate_mean,
        metrics.aggregate_median,
        metrics.aggregate_iqm,
    ]
    cols = ["mean", "median", "iqm"]
    if include_optimality_gap:
        funcs.append(metrics.aggregate_optimality_gap)
        cols.append("optimality_gap")

    def aggregate_func(scores: np.ndarray) -> np.ndarray:
        return np.array([fn(scores) for fn in funcs])

    point, intervals = library.get_interval_estimates(
        matrices,
        aggregate_func,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )

    rows = []
    for algo in matrices:
        row = {"algorithm": algo}
        for i, name in enumerate(cols):
            row[name] = float(point[algo][i])
            row[f"{name}_lo"] = float(intervals[algo][0, i])
            row[f"{name}_hi"] = float(intervals[algo][1, i])
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm")


def aggregate_metrics_forest(
    score_matrices: Mapping[str, np.ndarray],
    *,
    reps: int = 2000,
    confidence_interval_size: float = 0.95,
) -> Any:
    """Tidy long-form aggregate-metrics frame for the rliable forest plot.

    Returns one row per ``(algorithm, metric)`` pair with columns
    ``algorithm``, ``metric``, ``point``, ``lo``, ``hi``. ``metric``
    cycles through ``mean``, ``median``, ``iqm``, ``optimality_gap``.
    Designed to feed straight into
    :func:`deap_pettingzoo.analysis.plots.plot_aggregate_metrics_forest`.
    """
    import pandas as pd  # noqa: WPS433

    table = aggregate_metrics_table(
        score_matrices=score_matrices,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )
    if table.empty:
        return pd.DataFrame(columns=["algorithm", "metric", "point", "lo", "hi"])

    metric_names = [c for c in table.columns if not c.endswith(("_lo", "_hi"))]
    rows = []
    for algo, row in table.iterrows():
        for name in metric_names:
            rows.append(
                {
                    "algorithm": algo,
                    "metric": name,
                    "point": float(row[name]),
                    "lo": float(row[f"{name}_lo"]),
                    "hi": float(row[f"{name}_hi"]),
                }
            )
    return pd.DataFrame(rows)


def performance_profile(
    score_groups: Optional[Mapping[str, Sequence[float]]] = None,
    *,
    score_matrices: Optional[Mapping[str, np.ndarray]] = None,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    n_tau: int = 50,
    reps: int = 2000,
    confidence_interval_size: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute performance profiles ``P(score >= tau)`` per algorithm.

    Either ``score_groups`` (single-task seeds) or ``score_matrices``
    (canonical multi-task ``(n_runs, n_tasks)`` layout) must be passed.
    Returns ``(taus, point_estimates, intervals)`` where
    ``point_estimates[algo]`` is a 1-D array of length ``n_tau`` and
    ``intervals[algo]`` is a ``(2, n_tau)`` array of lo/hi bounds.
    """
    library, _ = _require_rliable()
    matrices = _resolve_matrices(score_groups, score_matrices)
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
    score_groups: Optional[Mapping[str, Sequence[float]]] = None,
    *,
    score_matrices: Optional[Mapping[str, np.ndarray]] = None,
    reps: int = 2000,
) -> Any:
    """Pairwise P(X > Y) matrix with bootstrap CIs.

    Either ``score_groups`` (single-task) or ``score_matrices``
    (multi-task) must be supplied. Returns a :class:`pandas.DataFrame`
    with one row per ordered pair ``(algo_x, algo_y)``: ``probability``,
    ``ci_lo``, ``ci_hi``.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    matrices = _resolve_matrices(score_groups, score_matrices)
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
    for label in pairs:
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
    y_col: Optional[str] = None,
    n_points: int = 25,
    reps: int = 1000,
    confidence_interval_size: float = 0.95,
) -> Any:
    """IQM-vs-budget curves with bootstrap CI bands.

    ``runs_by_algo`` maps an algorithm name to an iterable of
    :class:`~deap_pettingzoo.analysis.loaders.RunRecord`. Returns a
    long-form :class:`pandas.DataFrame` with columns ``algorithm``,
    ``x``, ``iqm``, ``lo``, ``hi``.

    When ``y_col`` is ``None`` (default) the value column is
    auto-detected (``best`` -> ``mean`` -> ``max``) across all runs, so
    product-path runs that log ``mean`` but never ``best`` yield a
    non-empty curve. Mirrors
    :func:`deap_pettingzoo.analysis.aggregate.learning_curve_with_ci`.
    """
    library, metrics = _require_rliable()
    import pandas as pd  # noqa: WPS433

    from deap_pettingzoo.analysis.aggregate import resolve_y_col  # noqa: WPS433

    if y_col is None:
        all_runs = [r for runs in runs_by_algo.values() for r in runs]
        y_col = resolve_y_col(all_runs, x_col=x_col)
        if y_col is None:
            return pd.DataFrame()

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
