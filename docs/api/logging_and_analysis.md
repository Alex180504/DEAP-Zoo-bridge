# Logging & Analysis API

This document describes the experiment-logging foundation
(`deap_pettingzoo.utils.logging`) and the offline analysis library
(`deap_pettingzoo.analysis`). Both layers are **opt-in**: when
`config.output_dir is None` every trainer falls back to a `NoOpLogger`
so the package has no runtime dependency on disk artifacts or on the
analysis stack.

The artifacts written by these modules are also the contract consumed by
the Streamlit UI.

---

## Table of contents

1. [Disk layout](#disk-layout)
2. [Schema (`utils.logging.schema`)](#schema)
3. [Manifest (`utils.logging.manifest`)](#manifest)
4. [Sinks (`utils.logging.sinks`)](#sinks)
5. [`ExperimentLogger`](#experimentlogger)
6. [`FrameRecorder` (replay)](#framerecorder)
7. [Analysis loaders](#analysis-loaders)
8. [Aggregation & confidence intervals](#aggregation)
9. [Pairwise comparison](#pairwise-comparison)
10. [`rliable` bridge](#rliable-bridge)
11. [Plots & reports](#plots--reports)

---

## Disk layout

When a trainer is configured with `output_dir=Path("results/exp1")`, each
run produces a directory:

```
results/exp1/
└── ea_simple_tag_v3_seed0_2025-...-uuid/
    ├── manifest.json          # RunManifest, schema-versioned
    ├── metrics.csv            # streaming per-iteration scalars
    ├── probe.csv              # opt-in held-out champion probes (ProbeConfig)
    ├── episodes.csv           # per-episode results (optional)
    ├── trajectories.parquet   # opt-in (pyarrow)
    ├── summary.json           # final scores / wall-time / artifact map
    ├── tb/                    # opt-in TensorBoard event files
    └── replays/
        └── episode_0000.mp4   # opt-in (imageio[ffmpeg])
```

Run-id format (default): `{algorithm}_{env}_seed{seed}_{ISO-ts}_{shortuuid}`.

---

## Schema

`deap_pettingzoo.utils.logging.schema`

```python
SCHEMA_VERSION: int = 1
METRICS_REQUIRED_COLUMNS: tuple[str, ...] = ("iteration", "env_steps", "wall_time")
EPISODES_COLUMNS:        tuple[str, ...] = (...)
TRAJECTORIES_COLUMNS:    tuple[str, ...] = (...)
```

Schema is intentionally permissive (extra columns are allowed). The
required columns are validated by `CsvScalarSink` against the first
row — adding a new column on a later row raises `ValueError` to catch
schema drift early.

---

## Manifest

`deap_pettingzoo.utils.logging.manifest`

### `capture_environment_metadata() -> dict`

Best-effort host/runtime snapshot. Always returns a dict with keys:

- `python` – `sys.version.split()[0]`
- `platform` – `platform.platform()`
- `hostname`, `git_sha` (or `None`)
- `package_versions` – dict mapping the project's pinned packages to
  their installed version (or `None` if not importable):
  `deap-pettingzoo`, `deap`, `pettingzoo`, `mpe2`, `numpy`,
  `gymnasium`, `agilerl`, `torch`, `jax`, `qdax`, `rliable`.

### `class RunManifest`

`@dataclass` persisted as `<run_dir>/manifest.json`.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `int` | bumped if the layout changes |
| `run_id` | `str` | recommended `{algo}_{env}_{seed}_{ts}_{uuid}` |
| `algorithm` | `str` | e.g. `"ea"`, `"cma-es"`, `"maddpg"` |
| `env_name` | `str` | registered PettingZoo env id |
| `env_config` | `dict` | env config used |
| `seed` | `int \| None` | master seed |
| `budget_mode` | `str \| None` | `"env_steps"`, `"eval_episodes"`, `"wall_clock"` |
| `budget_value` | `Any` | as supplied |
| `hyperparams` | `dict` | algorithm config, json-safe |
| `notes` | `str` | free-form |
| `created_at` | `str` | ISO-8601 UTC |
| `wall_time_s` | `float \| None` | total training time |
| `artifacts` | `dict[str, str | dict]` | name -> relative path **or** structured entry (see below) |
| `environment` | `dict` | from `capture_environment_metadata()` |

`RunManifest.write(path: Path) -> None` writes pretty JSON via a `.tmp`
file + atomic replace.

### `artifacts["policy"]` (policy checkpoint contract)

Trainers register their final policy via
`deap_pettingzoo.utils.logging.checkpoints` so that
`deap_pettingzoo.analysis.load_policy` can rehydrate it. The entry has
the shape:

```json
{
  "kind": "ea_genome_v1",
  "path": "checkpoints/policy.json",
  ...extra
}
```

`path` is relative to the run directory when the file lives under it,
and absolute otherwise. Supported `kind` values (constants in
`deap_pettingzoo.utils.logging.checkpoints`):

| Constant                    | Value                    | Producer                         | Reload                      |
|-----------------------------|--------------------------|----------------------------------|-----------------------------|
| `KIND_EA_GENOME`            | `"ea_genome_v1"`         | `baselines.ea_trainer`           | ✅ `load_policy`             |
| `KIND_CMA_GENOME`           | `"cma_genome_v1"`        | `baselines.cma_trainer`          | ✅ `load_policy`             |
| `KIND_COEVOLUTION`          | `"coevolution_v1"`       | `baselines.coevolution.trainer`  | ✅ `load_policy` -> `CoevolutionPolicyBundle` |
| `KIND_AGILERL`              | `"agilerl_v1"`           | `baselines.agilerl_trainer`      | ⚠️ `PolicyReloadUnavailable` |
| `KIND_QDAX_REPERTOIRE`      | `"qdax_repertoire_v1"`   | `contrib.qdax.trainer`           | ⚠️ `PolicyReloadUnavailable` |

Helpers (all no-ops on `NoOpLogger`):

- `save_ea_genome(logger, genome, *, filename, kind, extra) -> Path | None`
- `save_agilerl_agent(logger, agent, *, filename, extra) -> Path | None`
- `register_existing_artifact(logger, *, kind, path, extra) -> Path | None`

---

## Sinks

`deap_pettingzoo.utils.logging.sinks`

All sinks implement the `Sink` Protocol:

```python
class Sink(Protocol):
    def open(self, run_dir: Path) -> None: ...
    def log_row(self, table: str, row: Mapping[str, Any]) -> None: ...
    def log_scalar(self, name: str, value: float, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

### Concrete sinks

| Class | Behaviour |
|---|---|
| `NoOpSink` | discards everything; used by `NoOpLogger`. |
| `JsonManifestSink(tables=("summary",))` | writes `<table>.json`. The row **overwrites** the file — used for one-shot artifacts. Tables not in `tables` are ignored. |
| `CsvScalarSink()` | append-only `<table>.csv`; header taken from first row; extra columns on later rows raise `ValueError`. |
| `ParquetTrajectorySink()` | opt-in (pyarrow); row-group flush per N rows. |
| `TensorBoardSink()` | opt-in (`torch.utils.tensorboard.SummaryWriter`). |
| `MultiSink(sinks)` | fans `log_row` / `log_scalar` to a list of sinks. |

---

## ExperimentLogger

`deap_pettingzoo.utils.logging.experiment_logger`

### Construction

```python
ExperimentLogger.create(
    output_dir: Path | str | None,
    *,
    algorithm: str,
    env_name: str,
    seed: int | None,
    env_config: dict | None = None,
    budget_mode: str | None = None,
    budget_value: Any = None,
    hyperparams: dict | None = None,
    notes: str = "",
    run_id: str | None = None,
    enable_tensorboard: bool = False,
    enable_parquet: bool = False,
    extra_sinks: Sequence[Sink] | None = None,
) -> ExperimentLogger
```

If `output_dir is None`, returns `NoOpLogger()` instead.

```python
ExperimentLogger.from_config(config, *, algorithm, notes="") -> ExperimentLogger
```

Convenience helper that pulls `output_dir`, `env_name`, `seed`,
`budget_*`, `env_config`, and `algorithm_params` from any config object
that satisfies the unified-trainer protocol (`EAConfig`, `CMAESConfig`,
`AgileRLConfig`, `CoEvolutionConfig`, QDax configs, ...).
Honours `algorithm_params["logging_tensorboard"]` and `["logging_parquet"]`.

### Attributes

- `logger.run_dir: Path` — absolute directory containing the artifacts.
- `logger.manifest: RunManifest` — in-memory manifest; mutate freely
  before `close()`.

### Methods

```python
logger.row(table: str, row: Mapping[str, Any]) -> None
logger.scalar(name: str, value: float, step: int) -> None
logger.add_artifact(name: str, path: Path | str) -> None
logger.close() -> None      # flushes, writes manifest.json
```

`ExperimentLogger` is also a context manager (`with` records wall-time
between `__enter__` and `__exit__`/`close()`).

### `NoOpLogger`

A drop-in replacement with the same surface; `run_dir = Path()`,
`manifest.run_id = "noop"`. All write methods are no-ops. Returned by
`ExperimentLogger.create(output_dir=None)` and used by every trainer
when `config.output_dir is None`.

### Trainer integration helpers

`deap_pettingzoo.baselines._logging_helpers`:

- `build_logger(config, *, algorithm, notes="")` ->
  `ExperimentLogger.from_config(...)` (or `NoOpLogger`).
- `dump_fitness_history(logger, fitness_history, *, env_steps_per_iteration, iteration_key="generation", extra_columns_per_row=None)`
  – emits one `metrics` row per iteration.
- `write_summary(logger, summary, *, training_time_s=None)` – writes
  `summary.json` and overrides `manifest.wall_time_s` so the trainer's
  measured time is preserved.

All baseline trainers (`EATrainer`, `CMAESTrainer`, `AgileRLTrainer`,
`CoEvolutionTrainer`, `MAPElitesTrainer`, `CMAMETrainer`) use these
helpers — see `_persist_results()` / `_persist_qdax_results()`.

---

## FrameRecorder

`deap_pettingzoo.utils.logging.replay`

```python
FrameRecorder(
    run_dir: Path,
    *,
    fps: int = 20,
    fmt: str = "mp4",
    max_frames_per_episode: int | None = None,
    max_dim: int | None = None,
    episode_prefix: str = "",
)
```

| Parameter | Default | Description |
|---|---|---|
| `fps` | `20` | Playback frame rate written to the video container. |
| `fmt` | `"mp4"` | `"mp4"` (requires `imageio[ffmpeg]`) or `"gif"`. Falls back to gif on ffmpeg failure. |
| `max_frames_per_episode` | `None` | Drop frames once this many have been buffered, capping memory and encode time. |
| `max_dim` | `None` | Downscale each frame so its larger dimension is at most this many pixels before buffering. Uses numpy index subsampling (no PIL/cv2). Essential for envs with large `width`/`height` (e.g. 6000 px) to stay within H.264 macroblock limits and avoid RAM exhaustion. |
| `episode_prefix` | `""` | Prepended to output filenames: `"ea_"` -> `frames/ea_episode_0001.mp4`. |

- `start_episode(idx: int)`
- `add_frame(rgb: np.ndarray)` — `(H, W, 3)` uint8. Downscales to `max_dim` before buffering if set; drops silently once `max_frames_per_episode` is reached.
- `end_episode() -> Path | None` — writes
  `frames/{episode_prefix}episode_{idx:04d}.{mp4|gif}`. Falls back to
  `.gif` if `imageio[ffmpeg]` is unavailable. Returns `None` when no
  frames were recorded. **There is no `close()` method**; each
  `end_episode()` finalises one artefact independently.

### Frame recording from trainers

Frame recording is **opt-in** via `TrainingConfig.recording`:

```python
from deap_pettingzoo.baselines import (
    TrainingConfig, RecordingConfig, RecordingMode, BudgetMode, EATrainer,
)

cfg = TrainingConfig(
    env_name="simple_spread_v3",
    budget_mode=BudgetMode.GENERATIONS, budget_value=10,
    output_dir="results/with_video",
    recording=RecordingConfig(
        enabled=True,
        mode=RecordingMode.COUNT,  # N videos spread evenly over the run
        max_recordings=10,
        fps=20, fmt="mp4", max_frames_per_episode=None,
        episode_prefix="",         # "" -> trainer uses "train_"
    ),
)
EATrainer().train(cfg)
# -> results/with_video/<run_id>/frames/train_episode_0001.mp4 ...
```

**Cadence is algorithm-agnostic.** `RecordingConfig.mode` selects one policy that
means the same thing for every trainer, expressed on the shared normalized
progress axis (`ProgressEvent.fraction` / `.env_steps`) rather than per-generation
vs. per-iteration units:

| `mode` | Behavior | Companion field |
|---|---|---|
| `COUNT` (default) | N captures spread evenly across the run (first -> start, last -> end) | `max_recordings` (default 10) |
| `EVERY` | One capture per generation / iteration, no cap | — |
| `INTERVAL` | One capture every N env steps | `every_n_steps` |
| `FINAL` | Only the end-of-run champion rollout | — |

The shared `RecordingController` (`deap_pettingzoo/baselines/recording.py`) owns the
recorder, `UnifiedEvaluator`, schedule, and sidecar metadata. Trainers whose loop
can't be hooked mid-run (QDax, the DEAP-internal EA μ±λ strategies) naturally fall
back to a single end-of-run capture via `RecordingController.finalize`. Co-evolution
records a heterogeneous rollout (each side's best policy) through the evaluator's
per-agent `policy_map` path. Each recording is associated with its iteration and
`env_steps` in the `frames/recordings.json` sidecar.

**Filename convention.** The `episode_prefix` is written into the
filename as `frames/{prefix}episode_{NNNN}.{ext}`. Trainers default to
`"train_"`; future live-rollout capture will use `"live_"`. This lets
training and live captures coexist in the same `frames/` directory and
cluster naturally in the Streamlit Replay page's source filter.

---

## Held-out champion probe

`deap_pettingzoo.baselines.probing`

Opt-in via `TrainingConfig.probe` — the measurement companion to frame
recording. Every `every_n_steps` cumulative env steps (plus a baseline
probe at the first progress event and a guaranteed end-of-run probe), the
frozen current champion plays `n_episodes` held-out episodes and one row is
appended to `<run_dir>/probe.csv`:

```text
iteration, env_steps, probe_reward_mean, probe_reward_std,
probe_episodes[, probe_<info_key>_mean, probe_<info_key>_std, ...]
```

```python
from deap_pettingzoo.baselines.base import ProbeConfig

cfg = TrainingConfig(
    env_name="simple_spread_v3",
    budget_mode=BudgetMode.ENV_STEPS, budget_value=1_000_000,
    output_dir="results/with_probe",
    probe=ProbeConfig(
        enabled=True,
        every_n_steps=50_000,          # required when enabled
        n_episodes=16,
        info_keys=("prey_lost_total",),  # env-info task metrics (episode-max)
        env_name=None,                 # optional held-out env override
        seed=123,
    ),
)
```

Key properties:

- **Off-budget by construction.** Probe episodes run on the
  `ProbeController`'s own env instance and are never counted into any
  trainer's step budget, so equal-budget comparisons stay fair when the
  probe is enabled with identical cadence/size for every algorithm.
- **One yardstick for every backend.** RL training returns (on-policy,
  exploration noise, shaping) and EA fitness are not comparable; the probe
  gives all trainers the same frozen-policy held-out score on the shared
  `env_steps` axis — the input `learning_curve_with_ci` / the Compare page
  and `analysis/rliable_bridge.py` need for cross-algorithm
  sample-efficiency curves.
- **Common random numbers.** Episode seeds are fixed within a run, so
  successive checkpoints differ by policy, not episode draw.
- **Supported trainers**: AgileRL (MADDPG/MATD3/IPPO/GroupedIPPO), EA
  `simple`, CMA-ES, NEAT/HyperNEAT/ES-HyperNEAT, and QDax MAP-Elites
  (buffered, flushed with the run's logger). Co-evolution instead exposes
  the run logger + pooled consumed steps to three-argument
  `generation_callback`s so experiment-specific benchmarks (e.g. a
  scripted-opponent probe) can write the same `probe.csv` schema with a
  `side` column.
- **Error isolation** mirrors recording: a failing probe emits one
  `RuntimeWarning` and never interrupts training.

---

## Analysis loaders

`deap_pettingzoo.analysis.loaders`

### `class RunRecord`

`@dataclass(frozen=True)` view over one `<run_dir>/`:

| Attribute | Type | Source |
|---|---|---|
| `run_dir` | `Path` | constructor |
| `manifest` | `dict` | `manifest.json` |
| `metrics` | `pd.DataFrame` | `metrics.csv`, with any `probe.csv` rows concatenated in (probe rows carry `env_steps` + `probe_*` columns; each column pair `dropna()`s to its own rows downstream) |
| `episodes` | `pd.DataFrame` | `episodes.csv` (or empty) |
| `trajectories` | `pd.DataFrame` | `trajectories.parquet` (lazy) |
| `summary` | `dict` | `summary.json` |

Convenience properties: `algorithm`, `env_name`, `seed`, `run_id`,
`wall_time_s`, `final_fitness` (looks at `summary["final_fitness"]`,
then `summary["best_fitness"]`, then `summary["mean_episode_reward"]`).

### `class Experiment`

```python
Experiment(root: Path, runs: list[RunRecord])
```

- `Experiment.filter(*, algorithm=None, env=None, seed=None) -> Experiment`
- `.by_algorithm() -> dict[str, list[RunRecord]]`
- `.by_env() -> dict[str, list[RunRecord]]`
- `.summary_table() -> pd.DataFrame` — one row per run with
  `algorithm, env, seed, run_id, wall_time_s, final_fitness`.
- `.final_scores(metric="final_fitness") -> dict[str, np.ndarray]` —
  groups scores by algorithm, returns a numpy array of length
  `n_runs(algorithm)` per group.

### Module-level helpers

```python
load_run(run_dir: Path) -> RunRecord                          # FileNotFoundError if no manifest.json
load_experiment(root: Path, pattern="**/manifest.json") -> Experiment   # corrupted runs are skipped
```

---

## Aggregation

`deap_pettingzoo.analysis.aggregate`

| Function | Returns | Notes |
|---|---|---|
| `iqm(scores)` | `float` | mean of middle 50% (Q1–Q3); falls back to plain mean for `n < 4`. Matches `rliable.metrics.aggregate_iqm` to 1e-9 in tests. |
| `bootstrap_ci(scores, *, statistic=np.mean, n_resamples=2000, ci=0.95, seed=None)` | `(low, high)` | vectorised resampling; ties broken numerically. |
| `mean_with_ci(scores, **kw)` | `(mean, low, high)` | thin wrapper. |
| `iqm_with_ci(scores, **kw)` | `(iqm, low, high)` | thin wrapper. |
| `learning_curve_with_ci(runs, *, x_col="env_steps", y_col="best", statistic="mean" \| "iqm", n_resamples=2000, ci=0.95)` | `pd.DataFrame[x, y, low, high]` | concatenates per-run series, forward-fills, then bootstraps per `x`. |
| `aggregate_summary(score_groups: dict[str, np.ndarray])` | `pd.DataFrame` | indexed by algorithm; columns `n, mean, mean_ci_low, mean_ci_high, iqm, iqm_ci_low, iqm_ci_high, median`. |

---

## Pairwise comparison

`deap_pettingzoo.analysis.compare`

| Function | Returns |
|---|---|
| `cohens_d(a, b)` | float — pooled-SD effect size |
| `cliffs_delta(a, b)` | float in [-1, 1] — vectorised non-parametric effect size |
| `mann_whitney(a, b, alternative="two-sided")` | `(U, p)` — uses scipy if available, else internal fallback |
| `welch_t(a, b, alternative="two-sided")` | `(t, p)` — Welch's two-sample t-test |
| `pairwise_table(experiment, *, metric="final_fitness", test="mann_whitney" \| "welch_t")` | long-form `pd.DataFrame[algo_a, algo_b, n_a, n_b, statistic, p_value, cliffs_delta, cohens_d]` |

---

## rliable bridge

`deap_pettingzoo.analysis.rliable_bridge` — thin wrappers around
`rliable.library.get_interval_estimates`. All functions raise
`ImportError` with an install hint if `rliable` is missing.

```python
aggregate_metrics_table(score_groups=None, *, score_matrices=None, reps=2000) -> pd.DataFrame
performance_profile(score_groups=None, *, score_matrices=None, taus, reps=2000) -> (taus, point, intervals)
probability_of_improvement(score_groups=None, *, score_matrices=None, reps=2000) -> pd.DataFrame
sample_efficiency_curve(runs_by_algo, *, x_col="env_steps", y_col="best", x_grid=None, reps=2000) -> pd.DataFrame
aggregate_metrics_forest(score_matrices, *, reps=2000, confidence_interval_size=0.95) -> pd.DataFrame
```

`score_groups` follows the same shape as `Experiment.final_scores()`:
`dict[str, np.ndarray[shape=(n_runs,)]]` (single-task).

`score_matrices` is the multi-task counterpart:
`dict[str, np.ndarray[shape=(n_runs, n_envs)]]` — exactly one of
`score_groups` or `score_matrices` must be supplied.

`aggregate_metrics_forest` returns long-form
`DataFrame[algorithm, metric, point, lo, hi]` with
`metric ∈ {mean, median, iqm, optimality_gap}`, ready to feed
`plots.plot_aggregate_metrics_forest`.

---

## Multi-task aggregation

When an `Experiment` spans **two or more environments**, use
`deap_pettingzoo.analysis.multi_task` + `deap_pettingzoo.analysis.normalize`
to build per-task score matrices and a normalisation spec, then forward
to the `rliable_bridge` multi-task functions.

```python
from deap_pettingzoo.analysis import (
    load_experiment,
    collect_score_matrices,
    NormalizationSpec,
    normalize_score_matrices,
    aggregate_metrics_forest,
)

exp = load_experiment("results/multi_env")
matrices, env_names = collect_score_matrices(exp, metric="final_fitness")
spec = NormalizationSpec.from_minmax(matrices, env_names)
normed = normalize_score_matrices(matrices, env_names, spec)
forest = aggregate_metrics_forest(normed, reps=2000)
```

API surface:

- `collect_score_matrices(experiment, *, metric="final_fitness", algorithms=None, envs=None, min_seeds=1, fill="nan" | "truncate") -> (dict[algo, ndarray(n_runs, n_envs)], list[env_names])`
- `final_score_table(experiment, *, metric="final_fitness", algorithms=None, envs=None) -> DataFrame[algorithm, env_name, seed, run_id, score]`
- `NormalizationSpec(low: dict[str, float], high: dict[str, float])` with classmethod factories:
  - `from_dict(low, high)` — explicit bounds
  - `from_minmax(matrices, env_names)` — per-env min/max from the supplied matrices (skips all-NaN and constant envs)
  - `from_reference_algorithm(matrices, env_names, *, algorithm, baseline_low=0.0)` — divide by reference-algo per-env mean
- `normalize_score_matrices(matrices, env_names, spec) -> dict[algo, ndarray]` — preserves NaN, leaves missing envs unchanged

---

## Plots & reports

`deap_pettingzoo.analysis.plots` — matplotlib (Agg backend, lazy import):

- `plot_learning_curves(runs_by_algo, *, x_col="env_steps", y_col="best", statistic="mean", ax=None) -> Figure`
- `plot_eval_distribution(score_groups, *, ax=None) -> Figure` — box + strip
- `plot_performance_profile(score_groups, *, taus=None, ax=None) -> Figure`
- `plot_pi_matrix(score_groups, *, ax=None) -> Figure` — annotated heatmap
- `plot_aggregate_metrics_forest(forest_df, *, title=None, figsize=(10, 4), metric_order=("mean", "median", "iqm", "optimality_gap")) -> Figure` — 4-panel forest from `aggregate_metrics_forest`
- `plot_normalized_performance_profile(taus, point, intervals=None, *, xlim=(0.0, 1.5), ax=None) -> Figure` — variant of the performance profile with a normalised x-axis label
- `plot_per_env_eval(score_matrices, env_names, *, ax=None) -> Figure` — small-multiples boxplot, one panel per env

`build_report` automatically detects multi-env experiments
(`len(experiment.by_env()) >= 2`) and adds a **Multi-task aggregate**
section with auto min-max normalisation, an `aggregate_metrics_forest`
table + figure, and a per-env panel.

`deap_pettingzoo.analysis.report.build_report(experiment, output_dir, *, metric="final_fitness", fmt="md" | "html", include_figures=True, title=None) -> Path`

Renders summary table + aggregate CIs + pairwise table; optionally
saves `fig_eval_distribution.png` and `fig_learning_curves.png`
alongside the report.

