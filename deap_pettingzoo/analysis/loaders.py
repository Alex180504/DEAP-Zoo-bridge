"""Convert on-disk run artefacts into Python objects.

The loaders are deliberately tolerant to missing imports: missing optional artifacts
(``episodes.csv``, ``trajectories.parquet``) silently degrade to
``None`` rather than raising, so analysis code can branch on
availability. Only :file:`manifest.json` is required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


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


def _synthesise_metrics_from_episodes(episodes, summary: Dict[str, Any]):
    """Build a ``metrics``-shaped frame from a per-episode reward log.

    Produces columns ``iteration``, ``env_steps``, ``wall_time``,
    ``mean``, ``min``, ``max`` so RL runs (which log per-episode rewards
    in ``episodes.csv``) can share aggregation utilities with EA runs
    (which log per-generation aggregates in ``metrics.csv``).

    Aggregation is a running window of size ``max(1, n_episodes // 50)``
    so the curve is smoothed but still has ~50 points.
    """
    pd = _lazy_pd()
    df = episodes.copy()
    if "reward" not in df.columns:
        return None
    n = len(df)
    if n == 0:
        return None

    # iteration / env_steps axes
    if "episode_idx" in df.columns:
        df = df.sort_values("episode_idx").reset_index(drop=True)
    iters = pd.Series(range(n), name="iteration")
    if "length" in df.columns and (df["length"] > 0).all():
        env_steps = df["length"].cumsum().astype(int)
    else:
        # Fall back to summary totals (num_episodes -> env_steps).
        total_env_steps = summary.get("env_steps")
        total_episodes = summary.get("num_episodes") or n
        steps_per_ep = (
            float(total_env_steps) / max(1, int(total_episodes))
            if total_env_steps is not None
            else 1.0
        )
        env_steps = ((iters + 1) * steps_per_ep).round().astype(int)

    # Smoothed reward: rolling window over per-episode rewards.
    window = max(1, n // 50)
    reward = df["reward"].astype(float)
    rolling = reward.rolling(window=window, min_periods=1)

    out = pd.DataFrame(
        {
            "iteration": iters,
            "env_steps": env_steps.values,
            "nominal_env_steps": env_steps.values,
            "wall_time": [0.0] * n,
            "mean": rolling.mean().values,
            "min": rolling.min().values,
            "max": rolling.max().values,
        }
    )
    return out


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

    # Convenience accessors

    @property
    def algorithm(self) -> str:
        return str(self.manifest.get("algorithm", "unknown"))

    @property
    def env_name(self) -> str:
        return str(self.manifest.get("env_name", "unknown"))

    @property
    def action_mode(self) -> Optional[str]:
        """``"continuous"`` / ``"discrete"`` for this run, or ``None``.

        Resolved from the explicit ``action_mode`` hyperparameter when the
        trainer recorded one, else derived from the env config's
        ``continuous_actions`` flag (present on every aquarium run). Lets the
        Browse/Compare tables distinguish otherwise identically-named runs
        that differ only by action space.
        """
        hp = self.manifest.get("hyperparams") or {}
        mode = hp.get("action_mode")
        if isinstance(mode, str) and mode:
            return mode
        env_config = self.manifest.get("env_config") or {}
        if "continuous_actions" in env_config:
            return "continuous" if env_config["continuous_actions"] else "discrete"
        return None

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
        """Genuine optimisation fitness for this run.

        Resolves ``final_fitness`` -> ``best_fitness`` -> (co-evolution
        ``final_best``). Deliberately **excludes** ``mean_episode_reward``:
        a self-reported RL training reward is not a fitness, and silently
        coalescing it here mixed incomparable metrics into one column.
        Use :attr:`comparison_score` for the provenance-tagged
        cross-backend comparison metric.
        """
        for key in ("final_fitness", "best_fitness"):
            if key in self.summary:
                try:
                    return float(self.summary[key])
                except (TypeError, ValueError):
                    continue
        # Coevolution summary: {"final_best": {side_name: float}, "reporting_side": str}
        final_best = self.summary.get("final_best")
        if isinstance(final_best, dict) and final_best:
            reporting_side = self.summary.get("reporting_side")
            if reporting_side and reporting_side in final_best:
                try:
                    v = final_best[reporting_side]
                    if v is not None:
                        return float(v)
                except (TypeError, ValueError):
                    pass
            # Fallback: max across all sides
            try:
                valid = [float(v) for v in final_best.values() if v is not None]
                if valid:
                    return max(valid)
            except (TypeError, ValueError):
                pass
        return None

    @property
    def eval_mean_reward(self) -> Optional[float]:
        """Held-out re-evaluation reward, the canonical comparison metric.

        Written by the subprocess runner / :func:`compare_baselines` after
        training: a fixed-seed :class:`UnifiedEvaluator` rollout of the
        trained policy, recorded for **every** backend (EA, CMA-ES, PSO and
        the RL learners). This is the only metric on which EA and RL are
        directly comparable; ``None`` for legacy runs predating re-eval.
        """
        v = self.summary.get("eval_mean_reward")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def score_provenance(self) -> str:
        """Where :attr:`comparison_score` was sourced from.

        One of ``"held_out"`` (the held-out :attr:`eval_mean_reward`),
        ``"eval_failed"`` (re-evaluation was attempted but raised — the run
        has no comparable score and is excluded), ``"training_fitness"`` (the
        optimiser's own :attr:`final_fitness`), ``"training_reward"`` (only a
        self-reported ``mean_episode_reward``), or ``"none"``. Scores of
        differing provenance are not comparable; consumers surface this tag so
        a mixed comparison is blocked rather than rendered as a wrong verdict.

        An explicit ``eval_failed`` marker is honoured **before** any training
        fallback so a run whose held-out eval crashed is not silently
        re-admitted on its (incomparable) self-reported metric.
        """
        if self.summary.get("eval_metric_provenance") == "eval_failed":
            return "eval_failed"
        if self.eval_mean_reward is not None:
            return "held_out"
        if self.final_fitness is not None:
            return "training_fitness"
        if self.summary.get("mean_episode_reward") is not None:
            return "training_reward"
        return "none"

    @property
    def comparison_score(self) -> Optional[float]:
        """Provenance-aware scalar used for cross-backend comparison.

        Prefers the held-out :attr:`eval_mean_reward`; falls back to the
        genuine :attr:`final_fitness`, then to a self-reported
        ``mean_episode_reward`` (legacy runs with no re-eval). The chosen
        source is reported by :attr:`score_provenance`, so unlike the old
        ``final_fitness`` chain the fallback is never silent.

        Returns ``None`` when the run is tagged ``eval_failed``: a run whose
        canonical held-out eval crashed has no comparable score, so it is
        dropped from comparisons rather than re-admitted on a training metric.
        """
        if self.summary.get("eval_metric_provenance") == "eval_failed":
            return None
        if self.eval_mean_reward is not None:
            return self.eval_mean_reward
        genuine = self.final_fitness
        if genuine is not None:
            return genuine
        v = self.summary.get("mean_episode_reward")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None


def expand_coevolution_run(record: RunRecord) -> List[RunRecord]:
    """Expand a coevolution ``RunRecord`` into one ``RunRecord`` per side.

    When ``metrics.csv`` contains a ``side`` discriminator column (written
    by :class:`CoEvolutionTrainer`) this function returns a list of virtual
    ``RunRecord`` objects — one per unique side value — so that the analysis
    tools can treat each side as an independent algorithm without any
    special-casing.  Each virtual record has:

    * ``algorithm`` set to ``"<base_algo>/<side>"`` (e.g.
      ``"coevolution/predator_ea"``)
    * ``metrics`` filtered to that side's rows with the ``side`` column
      removed
    * ``summary['final_fitness']`` set to ``final_best[side]`` from the
      original coevolution summary, so pairwise tables and the Browse page
      can show a meaningful scalar

    If there is no ``side`` column the original record is returned
    unchanged (in a one-element list) so call sites can always do
    ``records = expand_coevolution_run(rec)``.
    """
    if record.metrics is None or "side" not in record.metrics.columns:
        return [record]
    sides = sorted(record.metrics["side"].dropna().unique().tolist())
    if len(sides) <= 1:
        return [record]

    base_algo = record.manifest.get("algorithm", "coevolution")
    base_run_id = record.run_id
    final_best: Dict[str, Any] = record.summary.get("final_best") or {}

    result: List[RunRecord] = []
    for side in sides:
        side_metrics = (
            record.metrics[record.metrics["side"] == side]
            .drop(columns=["side"])
            .reset_index(drop=True)
        )
        new_manifest = dict(record.manifest)
        new_manifest["algorithm"] = f"{base_algo}/{side}"
        new_manifest["run_id"] = f"{base_run_id}/{side}"

        side_summary = dict(record.summary)
        side_final = final_best.get(side)
        if side_final is not None:
            try:
                side_summary["final_fitness"] = float(side_final)
            except (TypeError, ValueError):
                pass

        result.append(
            RunRecord(
                run_dir=record.run_dir,
                manifest=new_manifest,
                metrics=side_metrics,
                episodes=record.episodes,
                trajectories=record.trajectories,
                summary=side_summary,
            )
        )
    return result


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

    # Held-out probe checkpoints (probe.csv, written by ProbeController or by
    # experiment-script benchmark callbacks) are merged into the metrics
    # frame as extra rows: probe rows carry `env_steps` plus `probe_*`
    # columns, trainer rows keep theirs, and each column pair dropna()s to
    # its own rows downstream (learning_curve_with_ci, the Compare page).
    # This makes probe curves selectable in the UI without a dedicated loader path.
    probe = _read_csv_if_exists(run_dir / "probe.csv")
    if probe is not None and len(probe) > 0:
        import pandas as pd

        if metrics is None:
            metrics = probe
        else:
            metrics = pd.concat([metrics, probe], ignore_index=True, sort=False)

    summary: Dict[str, Any] = dict(manifest.get("final_summary") or {})
    if not summary:
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                summary = {}

    # When a trainer (typically RL) logs per-episode rewards instead of
    # aggregated per-iteration metrics, synthesise a ``metrics``-shaped
    # frame so cross-algorithm tools (learning_curve_with_ci, the Compare
    # page) can plot EA and RL on the same axes.
    if metrics is None and episodes is not None and len(episodes) > 0:
        metrics = _synthesise_metrics_from_episodes(episodes, summary)

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

    def __iter__(self):
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, idx: int) -> RunRecord:
        return self.runs[idx]

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

        Columns: ``run_id``, ``algorithm``, ``action_mode``, ``env_name``,
        ``seed``, ``wall_time_s``, ``final_fitness``, plus all keys present in
        ``RunRecord.summary`` (numeric only).
        """
        pd = _lazy_pd()
        rows: List[Dict[str, Any]] = []
        for r in self.runs:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "algorithm": r.algorithm,
                "action_mode": r.action_mode,
                "env_name": r.env_name,
                "seed": r.seed,
                "wall_time_s": r.wall_time_s,
                "final_fitness": r.final_fitness,
                "eval_mean_reward": r.eval_mean_reward,
                "score_provenance": r.score_provenance,
            }
            for k, v in r.summary.items():
                if k in row:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _metric_value(r: RunRecord, metric: str) -> Optional[float]:
        """Resolve one scalar score for ``r`` under ``metric``.

        ``"final_fitness"`` and ``"comparison_score"`` both resolve to the
        held-out-preferred :attr:`RunRecord.comparison_score` (so every
        consumer that asks for the default metric automatically gets the
        honest cross-backend value); ``"eval_mean_reward"`` resolves to the
        held-out reward only; any other key is read verbatim from the run
        summary.
        """
        if metric in ("final_fitness", "comparison_score"):
            return r.comparison_score
        if metric == "eval_mean_reward":
            return r.eval_mean_reward
        value = r.summary.get(metric)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _provenance_for(r: RunRecord, metric: str) -> str:
        """Provenance tag for ``r`` under ``metric`` (see :meth:`score_provenances`)."""
        if metric in ("final_fitness", "comparison_score", "eval_mean_reward"):
            return r.score_provenance
        return "explicit_metric"

    def final_scores(
        self,
        metric: str = "final_fitness",
        *,
        algorithm: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Group final scalar scores by algorithm.

        Returns a mapping ``{algorithm: [score_seed_0, score_seed_1, ...]}``
        suitable for feeding into :mod:`compare` / :mod:`rliable_bridge`. The
        default ``metric="final_fitness"`` resolves through
        :attr:`RunRecord.comparison_score`, i.e. it prefers the held-out
        ``eval_mean_reward`` and never silently substitutes a training reward
        without it being reflected in :meth:`score_provenances`.

        Args:
            provenance: When given (e.g. ``"held_out"``), restrict the returned
                scores to runs of exactly that provenance. Leave ``None`` to
                pool every available score; in that case, if the pooled scores
                span more than one (non-``"none"``) provenance a **warning is
                logged** — pooling held-out and self-reported metrics yields a
                non-comparable result, so direct programmatic callers should
                either pass ``provenance="held_out"`` or gate on
                :meth:`score_provenances`.
        """
        groups: Dict[str, List[float]] = {}
        group_provs: Dict[str, set] = {}
        for r in self.runs:
            if algorithm is not None and r.algorithm != algorithm:
                continue
            prov = self._provenance_for(r, metric)
            if provenance is not None and prov != provenance:
                continue
            value = self._metric_value(r, metric)
            if value is None:
                continue
            groups.setdefault(r.algorithm, []).append(float(value))
            group_provs.setdefault(r.algorithm, set()).add(prov)

        if provenance is None:
            pooled = {p for provs in group_provs.values() for p in provs} - {"none"}
            if len(pooled) > 1:
                logger.warning(
                    "final_scores(metric=%r) pools scores of mixed provenance %s; "
                    "these are not directly comparable. Pass provenance='held_out' "
                    "to restrict, or gate on Experiment.score_provenances(). "
                    "Per-group provenance: %s",
                    metric,
                    sorted(pooled),
                    {a: sorted(p) for a, p in group_provs.items()},
                )
        return groups

    def per_agent_table(self):
        """Per-agent reward decomposition for each run.

        Returns a :class:`pandas.DataFrame` with one row per
        ``(run, agent)`` carrying ``total_reward``,
        ``mean_episode_reward``, ``n_steps`` and a ``source`` column.
        Prefers per-step ``trajectories.parquet``; falls back to the
        held-out ``eval_agent_rewards`` summary when no trajectories
        frame is loaded (see :func:`per_agent_table` for precedence).

        This is the only view that escapes the single-team-scalar
        collapse: ``episodes.csv`` and the aggregate metric both sum
        rewards across agents (degenerate on zero-sum competitive envs),
        whereas this carries per-agent rewards.
        """
        return per_agent_table(self.runs)

    def to_frame(
        self,
        metric: str = "final_fitness",
        *,
        include_config: bool = False,
    ):
        """Tidy per-(algorithm, env, seed) frame of scores for export.

        One row per run with ``run_id``, ``algorithm``, ``env_name``,
        ``seed``, ``score`` (resolved through the same provenance-aware
        path as :meth:`final_scores`), ``score_provenance`` and
        ``wall_time_s``.

        Parameters
        ----------
        include_config:
            When ``True``, also emit ``budget_mode``, ``budget_value``,
            ``git_sha`` and each scalar hyperparameter (prefixed
            ``hp.``) pulled from the run's ``manifest.json``, so a
            plotted result can be tied back to the seed, budget, code
            version and configuration that produced it.
        """
        pd = _lazy_pd()
        rows: List[Dict[str, Any]] = []
        for r in self.runs:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "algorithm": r.algorithm,
                "env_name": r.env_name,
                "seed": r.seed,
                "score": self._metric_value(r, metric),
                "score_provenance": self._provenance_for(r, metric),
                "wall_time_s": r.wall_time_s,
            }
            if include_config:
                row["budget_mode"] = r.manifest.get("budget_mode")
                row["budget_value"] = r.manifest.get("budget_value")
                env_meta = r.manifest.get("environment") or {}
                row["git_sha"] = env_meta.get("git_sha")
                hyperparams = r.manifest.get("hyperparams") or {}
                for k, v in hyperparams.items():
                    if v is None or isinstance(v, (str, int, float, bool)):
                        row[f"hp.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def export(
        self,
        path: str | Path,
        *,
        fmt: Optional[str] = None,
        metric: str = "final_fitness",
        include_config: bool = True,
    ) -> Path:
        """Write :meth:`to_frame` to ``path`` for reproducible re-plotting.

        ``fmt`` is one of ``"csv"`` / ``"json"`` / ``"parquet"``; when
        ``None`` it is inferred from ``path``'s suffix (defaulting to
        ``csv``). ``include_config`` defaults to ``True`` here so an
        exported campaign carries the seed/budget/git_sha/hyperparams
        needed to reproduce it. Returns ``path``.
        """
        path = Path(path)
        if fmt is None:
            fmt = path.suffix.lstrip(".").lower() or "csv"
        fmt = fmt.lower()
        df = self.to_frame(metric=metric, include_config=include_config)
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", indent=2)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"fmt must be 'csv', 'json' or 'parquet', got {fmt!r}")
        return path

    def score_provenances(
        self,
        metric: str = "final_fitness",
        *,
        algorithm: Optional[str] = None,
    ) -> Dict[str, set]:
        """Group score provenances by algorithm, aligned with :meth:`final_scores`.

        Returns ``{algorithm: {provenance, ...}}`` over the same runs
        :meth:`final_scores` would include. A group whose set is not a single
        ``{"held_out"}`` (or, across groups, a non-singleton union) signals a
        comparison that mixes held-out and self-reported metrics and must not
        be reported as a significance verdict.
        """
        out: Dict[str, set] = {}
        for r in self.runs:
            if algorithm is not None and r.algorithm != algorithm:
                continue
            if self._metric_value(r, metric) is None:
                continue
            out.setdefault(r.algorithm, set()).add(self._provenance_for(r, metric))
        return out


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
            rec = load_run(manifest_path.parent)
            runs.extend(expand_coevolution_run(rec))
        except (OSError, json.JSONDecodeError):
            continue
    return Experiment(root=root, runs=runs)


# Mapping helpers used by other analysis modules


def per_agent_table(runs: Sequence[RunRecord]):
    """Per-agent reward decomposition over ``runs``.

    Prefers per-step ``trajectories.parquet`` (full ``total_reward`` /
    ``mean_episode_reward`` / ``n_steps`` with ``source="trajectories"``).
    When a run has no trajectories frame it falls back to the held-out
    re-evaluation's per-agent means persisted in
    ``summary['eval_agent_rewards']`` (``source="held_out_eval"``,
    ``mean_episode_reward`` only; ``total_reward``/``n_steps`` are
    ``NaN``) so the comparable every-backend decomposition is still
    visible without the opt-in parquet sink.

    See :meth:`Experiment.per_agent_table`. Accepts any iterable of
    :class:`RunRecord` so UI code can decompose an arbitrary run subset
    without constructing an :class:`Experiment`.
    """
    pd = _lazy_pd()
    rows: List[Dict[str, Any]] = []
    for r in runs:
        traj = r.trajectories
        if traj is not None and {"agent", "reward"} <= set(traj.columns):
            if "episode_idx" in traj.columns:
                n_eps = max(1, int(traj["episode_idx"].nunique()))
            else:
                n_eps = 1
            for agent, series in traj.groupby("agent")["reward"]:
                total = float(series.sum())
                rows.append(
                    {
                        "run_id": r.run_id,
                        "algorithm": r.algorithm,
                        "env_name": r.env_name,
                        "seed": r.seed,
                        "agent": str(agent),
                        "total_reward": total,
                        "mean_episode_reward": total / n_eps,
                        "n_steps": int(series.size),
                        "source": "trajectories",
                    }
                )
            continue
        held_out = r.summary.get("eval_agent_rewards")
        if isinstance(held_out, dict):
            for agent, reward in held_out.items():
                try:
                    mean_reward = float(reward)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "run_id": r.run_id,
                        "algorithm": r.algorithm,
                        "env_name": r.env_name,
                        "seed": r.seed,
                        "agent": str(agent),
                        "total_reward": float("nan"),
                        "mean_episode_reward": mean_reward,
                        "n_steps": float("nan"),
                        "source": "held_out_eval",
                    }
                )
    return pd.DataFrame(rows)


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
