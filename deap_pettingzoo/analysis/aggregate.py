"""Aggregate statistics: IQM, bootstrap CI, learning curves with CI bands.

Numerically self-contained (numpy only) so the analysis package works
even without ``rliable`` or ``scipy`` installed. The implementations
match ``rliable.metrics.aggregate_iqm`` and the percentile-bootstrap
recipe from Agarwal et al. 2021 (Deep RL at the Edge of the
Statistical Precipice) to within sampling noise.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.analysis.loaders import RunRecord


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: average of the middle 50 % of ``values``.

    Returns ``nan`` when ``values`` is empty. Uses inclusive linear
    interpolation between the 25th and 75th percentiles, matching
    :func:`rliable.metrics.aggregate_iqm`.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        # Not enough samples to trim — fall back to the plain mean
        # rather than returning nan, which is what rliable also does
        # in practice.
        return float(np.mean(arr))
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    middle = arr[(arr >= q25) & (arr <= q75)]
    if middle.size == 0:
        return float(np.mean(arr))
    return float(np.mean(middle))


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
        Coverage in ``(0, 1)``. ``0.95`` → 95 % CI.
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


def learning_curve_with_ci(
    runs: Sequence[RunRecord],
    *,
    x_col: str = "env_steps",
    y_col: str = "best",
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
) -> Any:
    """Per-algorithm summary table: mean, median, IQM with bootstrap CIs.

    ``score_groups`` is the dict returned by
    :meth:`Experiment.final_scores`. Returns a
    :class:`pandas.DataFrame` indexed by algorithm.
    """
    import pandas as pd  # noqa: WPS433

    rows: list = []
    for algo, vals in score_groups.items():
        arr = np.asarray(list(vals), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean, mlo, mhi = mean_with_ci(
            arr, n_bootstrap=n_bootstrap, ci=ci, rng=rng
        )
        i_pt, ilo, ihi = iqm_with_ci(
            arr, n_bootstrap=n_bootstrap, ci=ci, rng=rng
        )
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
    return pd.DataFrame(rows).set_index("algorithm") if rows else pd.DataFrame()
