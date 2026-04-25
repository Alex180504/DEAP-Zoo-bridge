"""Hydrate on-disk run artefacts into Python objects.

The loaders are deliberately tolerant: missing optional artefacts
(``episodes.csv``, ``trajectories.parquet``) silently degrade to
``None`` rather than raising, so analysis code can branch on
availability. Only :file:`manifest.json` is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

# pandas is heavy; lazy-import to keep ``import deap_pettingzoo.analysis`` cheap.


def _lazy_pd():
    import pandas as pd  # noqa: WPS433
    return pd


def _read_csv_if_exists(path: Path):
    if not path.is_file():
        return None
    pd = _lazy_pd()
    return pd.read_csv(path)


def _read_parquet_if_exists(path: Path):
    if not path.is_file():
        return None
    pd = _lazy_pd()
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError):  # pragma: no cover - missing pyarrow
        return None


@dataclass
class RunRecord:
    """A single hydrated training run.

    Attributes
    ----------
    run_dir:
        Directory the run was loaded from.
    manifest:
        Parsed ``manifest.json`` content (always present).
    metrics:
        ``metrics.csv`` as a :class:`pandas.DataFrame`, or ``None`` if
        the run did not record per-iteration metrics.
    episodes:
        ``episodes.csv`` as a :class:`pandas.DataFrame`, or ``None``.
    trajectories:
        ``trajectories.parquet`` as a :class:`pandas.DataFrame`, or
        ``None`` if the file is missing or pyarrow is not installed.
    summary:
        Final-summary dict (``manifest.final_summary`` if present, else
        the parsed ``summary.json`` if it exists, else an empty dict).
    """

    run_dir: Path
    manifest: Dict[str, Any]
    metrics: Optional[Any] = None
    episodes: Optional[Any] = None
    trajectories: Optional[Any] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return str(self.manifest.get("algorithm", "unknown"))

    @property
    def env_name(self) -> str:
        return str(self.manifest.get("env_name", "unknown"))

    @property
    def seed(self) -> Optional[int]:
        s = self.manifest.get("seed")
        try:
            return int(s) if s is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", self.run_dir.name))

    @property
    def wall_time_s(self) -> Optional[float]:
        v = self.manifest.get("wall_time_s")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def final_fitness(self) -> Optional[float]:
        for key in ("final_fitness", "best_fitness", "mean_episode_reward"):
            if key in self.summary:
                try:
                    return float(self.summary[key])
                except (TypeError, ValueError):
                    continue
        return None


def load_run(run_dir: str | Path) -> RunRecord:
    """Load a single run directory.

    Raises
    ------
    FileNotFoundError
        If ``run_dir/manifest.json`` does not exist.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found under {run_dir}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    metrics = _read_csv_if_exists(run_dir / "metrics.csv")
    episodes = _read_csv_if_exists(run_dir / "episodes.csv")
    trajectories = _read_parquet_if_exists(run_dir / "trajectories.parquet")

    summary: Dict[str, Any] = dict(manifest.get("final_summary") or {})
    if not summary:
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                summary = {}

    return RunRecord(
        run_dir=run_dir,
        manifest=manifest,
        metrics=metrics,
        episodes=episodes,
        trajectories=trajectories,
        summary=summary,
    )


@dataclass
class Experiment:
    """Collection of :class:`RunRecord` objects sharing a root directory.

    The class is intentionally a thin wrapper around ``runs`` so that
    callers can ``for run in experiment`` or ``len(experiment)`` without
    learning a new API.
    """

    root: Path
    runs: List[RunRecord] = field(default_factory=list)

    # ----- container-ish behaviour -----------------------------------

    def __iter__(self):
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, idx: int) -> RunRecord:
        return self.runs[idx]

    # ----- selection ---------------------------------------------------

    def filter(
        self,
        *,
        algorithm: Optional[str] = None,
        env_name: Optional[str] = None,
        predicate: Optional[Callable[[RunRecord], bool]] = None,
    ) -> "Experiment":
        """Return a new :class:`Experiment` with the matching runs."""
        out = list(self.runs)
        if algorithm is not None:
            out = [r for r in out if r.algorithm == algorithm]
        if env_name is not None:
            out = [r for r in out if r.env_name == env_name]
        if predicate is not None:
            out = [r for r in out if predicate(r)]
        return Experiment(root=self.root, runs=out)

    def by_algorithm(self) -> Dict[str, List[RunRecord]]:
        groups: Dict[str, List[RunRecord]] = {}
        for r in self.runs:
            groups.setdefault(r.algorithm, []).append(r)
        return groups

    def by_env(self) -> Dict[str, List[RunRecord]]:
        groups: Dict[str, List[RunRecord]] = {}
        for r in self.runs:
            groups.setdefault(r.env_name, []).append(r)
        return groups

    # ----- summary table ----------------------------------------------

    def summary_table(self):
        """Return a :class:`pandas.DataFrame` with one row per run.

        Columns: ``run_id``, ``algorithm``, ``env_name``, ``seed``,
        ``wall_time_s``, ``final_fitness``, plus all keys present in
        ``RunRecord.summary`` (numeric only).
        """
        pd = _lazy_pd()
        rows: List[Dict[str, Any]] = []
        for r in self.runs:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "algorithm": r.algorithm,
                "env_name": r.env_name,
                "seed": r.seed,
                "wall_time_s": r.wall_time_s,
                "final_fitness": r.final_fitness,
            }
            for k, v in r.summary.items():
                if k in row:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def final_scores(
        self,
        metric: str = "final_fitness",
        *,
        algorithm: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Group final scalar scores by algorithm.

        Returns a mapping ``{algorithm: [score_seed_0, score_seed_1, ...]}``
        suitable for feeding into :mod:`compare` / :mod:`rliable_bridge`.
        """
        groups: Dict[str, List[float]] = {}
        for r in self.runs:
            if algorithm is not None and r.algorithm != algorithm:
                continue
            value: Any
            if metric == "final_fitness":
                value = r.final_fitness
            else:
                value = r.summary.get(metric)
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            groups.setdefault(r.algorithm, []).append(fv)
        return groups


def load_experiment(
    root: str | Path,
    *,
    pattern: str = "**/manifest.json",
) -> Experiment:
    """Recursively load every run under ``root``.

    Any subdirectory containing a ``manifest.json`` is treated as a run.
    Runs whose manifest fails to parse are skipped silently so a single
    corrupted run does not block analysis of the rest.
    """
    root = Path(root)
    runs: List[RunRecord] = []
    for manifest_path in sorted(root.glob(pattern)):
        try:
            runs.append(load_run(manifest_path.parent))
        except (OSError, json.JSONDecodeError):
            continue
    return Experiment(root=root, runs=runs)


# Mapping helpers used by other analysis modules ---------------------


def metrics_aligned_to_x(
    runs: Sequence[RunRecord],
    *,
    x_col: str = "env_steps",
    y_col: str = "best",
) -> Mapping[str, Any]:
    """Return aligned per-run y-values keyed by run_id.

    Used internally by :func:`learning_curve_with_ci`. Skips runs that
    don't contain the requested columns.
    """
    out: Dict[str, Any] = {}
    for r in runs:
        if r.metrics is None:
            continue
        cols = set(r.metrics.columns)
        if x_col not in cols or y_col not in cols:
            continue
        out[r.run_id] = r.metrics[[x_col, y_col]].copy()
    return out
