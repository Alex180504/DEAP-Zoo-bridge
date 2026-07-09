"""Aggregate statistics: IQM, bootstrap CI, learning curves with CI bands.

Numerically self-contained (numpy only) so the analysis package works
even without ``rliable`` or ``scipy`` installed. The implementations
match ``rliable.metrics.aggregate_iqm`` and the percentile-bootstrap
recipe from Agarwal et al. 2021 (Deep RL at the Edge of the
Statistical Precipice) to within sampling noise.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.analysis.loaders import RunRecord

logger = logging.getLogger(__name__)

#: Candidate per-iteration value columns, in auto-detect priority order.
#: ``best`` is written only by the personal ``examples/`` scripts; the
#: library trainers (EA/CMA/PSO/AgileRL) write ``mean``/``min``/``max``.
_Y_COL_CANDIDATES: Tuple[str, ...] = ("best", "mean", "max")


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: average of the middle 50 % of ``values``.

    Returns ``nan`` when ``values`` is empty. Drops the bottom and top
    25 % of samples **by count** (not by a quantile mask), matching
    :func:`rliable.metrics.aggregate_iqm`, i.e.
    :func:`scipy.stats.trim_mean` with ``proportiontocut=0.25``. A
    quantile mask keeps the wrong number of samples on ties (e.g.
    ``[1,1,1,1,2,3,4,5]`` -> 1.5 instead of the correct 1.75).
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    try:
        from scipy import stats  # noqa: WPS433  (optional dependency)

        return float(stats.trim_mean(arr, 0.25))
    except ImportError:
        # numpy fallback that is numerically identical to
        # scipy.stats.trim_mean(arr, 0.25): sort, cut floor(n * 0.25)
        # samples from each end by count, average the rest.
        arr = np.sort(arr)
        k = int(arr.size * 0.25)
        trimmed = arr[k : arr.size - k] if arr.size - 2 * k > 0 else arr
        return float(np.mean(trimmed))


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval for ``statistic``.

    Parameters
    ----------
    values:
        1-D sample.
    statistic:
        Callable mapping ``np.ndarray`` to ``float``. Default is
        :func:`numpy.mean`. Pass :func:`iqm` for IQM bootstrap.
    n_bootstrap:
        Number of resamples. 2000 is a good compromise between speed
        and CI precision; raise for tighter intervals on small samples.
    ci:
        Coverage in ``(0, 1)``. ``0.95`` -> 95 % CI.
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    (lo, hi):
        Lower and upper bounds of the CI. Returns ``(nan, nan)`` when
        ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(statistic(arr))
        return v, v
    rng = rng if rng is not None else np.random.default_rng(0)
    n = arr.size
    samples = np.empty(n_bootstrap, dtype=float)
    # Vectorised resampling: draw all indices at once for speed.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    for i in range(n_bootstrap):
        samples[i] = float(statistic(arr[idx[i]]))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def mean_with_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` with bootstrap CI."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    lo, hi = bootstrap_ci(
        arr, statistic=np.mean, n_bootstrap=n_bootstrap, ci=ci, rng=rng
    )
    return mean, lo, hi


def iqm_with_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return ``(iqm, lo, hi)`` with bootstrap CI on the IQM statistic."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = iqm(arr)
    lo, hi = bootstrap_ci(
        arr,
        statistic=lambda a: iqm(a),
        n_bootstrap=n_bootstrap,
        ci=ci,
        rng=rng,
    )
    return point, lo, hi


def resolve_y_col(
    runs: Sequence[RunRecord],
    *,
    x_col: str = "env_steps",
    candidates: Sequence[str] = _Y_COL_CANDIDATES,
) -> Optional[str]:
    """Pick the first candidate value column actually present in ``runs``.

    Returns the first column from ``candidates`` (default
    ``best`` -> ``mean`` -> ``max``) that co-occurs with ``x_col`` in at
    least one run's ``metrics`` frame, or ``None`` when none do. This
    lets :func:`learning_curve_with_ci` plot product-path runs (which
    write ``mean``/``min``/``max`` but no ``best``) without the caller
    having to know which column a backend logged.
    """
    available: set = set()
    for r in runs:
        if r.metrics is None:
            continue
        cols = set(r.metrics.columns)
        if x_col in cols:
            available |= cols
    for cand in candidates:
        if cand in available:
            return cand
    return None


def learning_curve_with_ci(
    runs: Sequence[RunRecord],
    *,
    x_col: str = "env_steps",
    y_col: Optional[str] = None,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Aggregate per-iteration metrics across seeds with a CI band.

    Aligns each run's ``(x_col, y_col)`` series onto the union of x
    values via piecewise-constant forward-fill (treats curves as
    step functions, suitable for env_steps / iteration counters). At
    each x, computes ``statistic`` ± bootstrap CI across runs.

    Parameters
    ----------
    runs:
        Iterable of :class:`RunRecord`. Runs lacking ``metrics`` or the
        requested columns are silently skipped.
    y_col:
        Value column to aggregate. When ``None`` (default) it is
        auto-detected via :func:`resolve_y_col` (``best`` -> ``mean`` ->
        ``max``) so the library's product-path runs — which log
        ``mean`` but never ``best`` — produce a non-empty curve instead
        of raising.
    statistic:
        ``"mean"`` or ``"iqm"``.
    n_bootstrap, ci, rng:
        Forwarded to :func:`bootstrap_ci`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``x_col``, ``mean`` (or ``iqm``), ``lo``, ``hi``,
        ``n_runs``.

    Raises
    ------
    ValueError
        If no run contributes data.
    """
    import pandas as pd  # noqa: WPS433  (lazy)

    if statistic not in {"mean", "iqm"}:
        raise ValueError(f"statistic must be 'mean' or 'iqm', got {statistic!r}")

    runs = list(runs)
    if y_col is None:
        y_col = resolve_y_col(runs, x_col=x_col)
        if y_col is None:
            raise ValueError(
                f"No run exposed any of {list(_Y_COL_CANDIDATES)!r} alongside "
                f"'{x_col}' in metrics.csv; pass an explicit y_col."
            )

    series: list = []
    for r in runs:
        if r.metrics is None:
            continue
        cols = set(r.metrics.columns)
        if x_col not in cols or y_col not in cols:
            continue
        df = r.metrics[[x_col, y_col]].dropna()
        if df.empty:
            continue
        df = df.sort_values(x_col)
        # collapse duplicate x values (keep last)
        df = df.groupby(x_col, as_index=False).last()
        series.append(df.set_index(x_col)[y_col])

    if not series:
        raise ValueError(
            f"No runs contained both '{x_col}' and '{y_col}' in metrics.csv"
        )

    # Align onto union of x values; forward-fill so a slow run keeps its
    # last known value rather than introducing NaN-masked outliers.
    aligned = pd.concat(series, axis=1).sort_index().ffill()
    point_fn = iqm if statistic == "iqm" else (lambda a: float(np.mean(a)))

    rows: list = []
    for x_val, row in aligned.iterrows():
        sample = row.dropna().to_numpy(dtype=float)
        if sample.size == 0:
            continue
        point = point_fn(sample)
        if sample.size >= 2:
            lo, hi = bootstrap_ci(
                sample,
                statistic=(iqm if statistic == "iqm" else np.mean),
                n_bootstrap=n_bootstrap,
                ci=ci,
                rng=rng,
            )
        else:
            lo, hi = point, point
        rows.append(
            {
                x_col: x_val,
                statistic: point,
                "lo": lo,
                "hi": hi,
                "n_runs": int(sample.size),
            }
        )

    return pd.DataFrame(rows)


def aggregate_summary(
    score_groups: Dict[str, Sequence[float]],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    require_equal_n: bool = False,
) -> Any:
    """Per-algorithm summary table: mean, median, IQM with bootstrap CIs.

    ``score_groups`` is the dict returned by
    :meth:`Experiment.final_scores`. Returns a
    :class:`pandas.DataFrame` indexed by algorithm with an ``n_seeds``
    column surfacing each algorithm's run count.

    Aggregation is **pooled, not paired**: each algorithm is summarised
    over whatever seeds it has. When those counts differ across
    algorithms a :class:`UserWarning` is emitted (and logged), because a
    headline comparison resting on, e.g., n=3 for one algorithm and n=10
    for another is easy to misread as equally well-supported.

    Parameters
    ----------
    require_equal_n:
        When ``True``, raise :class:`ValueError` instead of warning if
        the per-algorithm run counts are not all equal.
    """
    import pandas as pd  # noqa: WPS433

    rows: list = []
    counts: Dict[str, int] = {}
    for algo, vals in score_groups.items():
        arr = np.asarray(list(vals), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            # No finite scores -> not a participant; excluded from both the
            # table and the unequal-n check (it can neither trigger nor mask it).
            continue
        counts[algo] = int(arr.size)
        mean, mlo, mhi = mean_with_ci(arr, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        i_pt, ilo, ihi = iqm_with_ci(arr, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        rows.append(
            {
                "algorithm": algo,
                "n_seeds": int(arr.size),
                "mean": mean,
                "mean_lo": mlo,
                "mean_hi": mhi,
                "median": float(np.median(arr)),
                "iqm": i_pt,
                "iqm_lo": ilo,
                "iqm_hi": ihi,
                "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
            }
        )

    if len(set(counts.values())) > 1:
        msg = (
            "aggregate_summary pools unequal run counts per algorithm "
            f"({counts}); the comparison is pooled, not paired. Pass "
            "require_equal_n=True to enforce equal n, or down-sample seeds."
        )
        if require_equal_n:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    return pd.DataFrame(rows).set_index("algorithm") if rows else pd.DataFrame()
