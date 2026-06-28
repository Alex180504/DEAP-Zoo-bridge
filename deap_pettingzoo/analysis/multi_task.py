"""Multi-task score collection helpers.

These utilities turn a heterogeneous :class:`~deap_pettingzoo.analysis.
loaders.Experiment` (one whose runs span multiple environments) into the
canonical ``{algorithm: ndarray[n_runs, n_tasks]}`` matrix layout that
``rliable`` expects for cross-task aggregation.

The single-task path used elsewhere in the analysis package
(``Experiment.final_scores`` -> ``Dict[str, List[float]]``) remains
unchanged; multi-task callers reach for :func:`collect_score_matrices`
or :func:`final_score_table` here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.analysis.loaders import Experiment, RunRecord

__all__ = [
    "collect_score_matrices",
    "final_score_table",
]


def _run_score(run: RunRecord, metric: str) -> Optional[float]:
    if metric in ("final_fitness", "comparison_score"):
        value: Any = run.comparison_score
    elif metric == "eval_mean_reward":
        value = run.eval_mean_reward
    else:
        value = run.summary.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_score_matrices(
    experiment: Experiment,
    *,
    metric: str = "final_fitness",
    algorithms: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    min_seeds: int = 1,
    fill: str = "nan",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Stack per-(algorithm, env) final scores into rliable-ready matrices.

    Parameters
    ----------
    experiment:
        The hydrated experiment to mine.
    metric:
        Scalar key to extract from each :class:`RunRecord`. Defaults to
        ``"final_fitness"`` which resolves through
        :attr:`RunRecord.comparison_score` (held-out ``eval_mean_reward``
        preferred); any other value is fetched from
        :attr:`RunRecord.summary`.
    algorithms, envs:
        Optional explicit ordering / filter. When ``None`` the natural
        iteration order from :meth:`Experiment.by_algorithm` /
        :meth:`Experiment.by_env` is preserved.
    min_seeds:
        Drop any algorithm whose **maximum** seed count across the kept
        environments is below this threshold.
    fill:
        Strategy when seed counts differ across environments for the
        same algorithm:

        - ``"nan"`` (default): pad shorter columns with ``np.nan`` so
          every algorithm's matrix has shape ``(max_seeds, n_tasks)``.
          ``rliable`` handles NaNs in stratified bootstraps.
        - ``"truncate"``: keep only the first ``min(seeds_per_env)``
          rows per environment.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], List[str]]
        Mapping ``{algorithm: matrix[n_runs, n_tasks]}`` plus the
        ordered list of environment names that index the columns.
        Algorithms missing every kept environment are omitted.
    """
    if fill not in {"nan", "truncate"}:
        raise ValueError(f"fill must be 'nan' or 'truncate', got {fill!r}")
    if min_seeds < 1:
        raise ValueError("min_seeds must be >= 1")

    by_algo = experiment.by_algorithm()
    by_env = experiment.by_env()

    if algorithms is None:
        algo_order: List[str] = list(by_algo.keys())
    else:
        algo_order = [a for a in algorithms if a in by_algo]

    if envs is None:
        env_order: List[str] = list(by_env.keys())
    else:
        env_order = [e for e in envs if e in by_env]

    if not env_order:
        return {}, []

    # First pass: per-(algo, env) sorted score lists.
    per_pair: Dict[str, Dict[str, List[float]]] = {}
    for algo in algo_order:
        runs = by_algo.get(algo, [])
        # Sort runs by seed (then run_id) for stable column-row pairing.
        runs_sorted = sorted(
            runs,
            key=lambda r: (r.seed if r.seed is not None else 1 << 30, r.run_id),
        )
        per_env: Dict[str, List[float]] = {env: [] for env in env_order}
        for run in runs_sorted:
            if run.env_name not in per_env:
                continue
            score = _run_score(run, metric)
            if score is None:
                continue
            per_env[run.env_name].append(score)
        per_pair[algo] = per_env

    matrices: Dict[str, np.ndarray] = {}
    for algo in algo_order:
        per_env = per_pair[algo]
        seed_counts = [len(per_env[env]) for env in env_order]
        if max(seed_counts) < min_seeds:
            continue

        if fill == "truncate":
            keep = min(seed_counts)
            if keep <= 0:
                continue
            mat = np.array(
                [[per_env[env][i] for env in env_order] for i in range(keep)],
                dtype=float,
            )
        else:  # "nan"
            n_rows = max(seed_counts)
            if n_rows <= 0:
                continue
            mat = np.full((n_rows, len(env_order)), np.nan, dtype=float)
            for j, env in enumerate(env_order):
                col = per_env[env]
                if col:
                    mat[: len(col), j] = col
        matrices[algo] = mat

    return matrices, env_order


def final_score_table(
    experiment: Experiment,
    *,
    metric: str = "final_fitness",
    algorithms: Optional[Iterable[str]] = None,
    envs: Optional[Iterable[str]] = None,
):
    """Return a long-form ``DataFrame`` of final scores.

    Columns: ``algorithm``, ``env_name``, ``seed``, ``run_id``,
    ``score``. Convenient for box-plot / strip-plot inspection of
    multi-task experiments.
    """
    import pandas as pd  # local import keeps package import light

    algo_filter = set(algorithms) if algorithms is not None else None
    env_filter = set(envs) if envs is not None else None

    rows: List[Dict[str, Any]] = []
    for run in experiment.runs:
        if algo_filter is not None and run.algorithm not in algo_filter:
            continue
        if env_filter is not None and run.env_name not in env_filter:
            continue
        score = _run_score(run, metric)
        if score is None:
            continue
        rows.append(
            {
                "algorithm": run.algorithm,
                "env_name": run.env_name,
                "seed": run.seed,
                "run_id": run.run_id,
                "score": score,
            }
        )
    return pd.DataFrame(rows, columns=["algorithm", "env_name", "seed", "run_id", "score"])
