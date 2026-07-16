# Experiment Configuration Schema

The **experiment spec** is the JSON contract for submitting an experiment to the
service. A client (the Setup UI, or any third party) builds one, posts it to
[`POST /experiments`](service_api.md#post-experiments) or
[`POST /experiments/validate`](service_api.md#post-experimentsvalidate), and the
runner turns it into the `TrainingConfig` / `CoEvolutionConfig` objects the
trainers consume.

Two representations back the same schema:

- **Pydantic** — `deap_pettingzoo.service.models.ExperimentSpecModel` parses and
  type-checks the request body and drives the live OpenAPI schema at `/docs`
  (`/openapi.json`).
- **Dataclass** — `deap_pettingzoo.baselines.experiment_spec.ExperimentSpec` is
  the JSON-round-trippable form the runner validates (`validate_spec`) and
  builds (`build_configs`).

The two mirror each other field for field; the pydantic model converts to the
dataclass via `ExperimentSpecModel.to_dataclass()`.

## Three-layer configuration merge

`env_config` and `hyperparameters` carry **only the values a client explicitly
set**. Everything else is filled in from the catalog when the runner resolves
the spec, across three layers of increasing precedence:

1. **Catalog defaults** — every `ParamSpec.default` for the chosen environment
   and algorithms.
2. **Raw-config overlay** — the free-form `raw_config` dict, deep-merged on top.
3. **Explicit values** — `env_config` / `hyperparameters`, merged last, so one
   tweaked field wins over the raw config while the rest of it stays intact.

So a form submits only the values it actually touched, and the catalog supplies
the rest.

## Top-level: `ExperimentSpec`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `environment` | `str` | **required** | Registered environment name (e.g. `"simple_tag_v3"`). |
| `algorithms` | `List[str]` | `[]` | Catalog algorithm names to run (e.g. `["ea", "maddpg"]`). Leave empty for a co-evolution experiment. |
| `env_config` | `Dict[str, Any]` | `{}` | Explicitly-set environment parameters only. Valid keys come from `GET /catalog/environments/{name}`. |
| `hyperparameters` | `Dict[str, Dict[str, Any]]` | `{}` | Per-algorithm overrides, keyed by algorithm name. Valid keys and ranges come from `GET /catalog/algorithms/{name}`. |
| `budget` | `BudgetSpec` | see below | Training budget and seed count. |
| `fitness` | `FitnessSettings` | see below | Fitness aggregation (population-based trainers). |
| `recording` | `RecordingSettings` | see below | Champion-rollout video recording. |
| `parallelism` | `ParallelismSettings` | see below | Parallel evaluation (population-based trainers). |
| `coevolution` | `CoevolutionSettings` | see below | Multi-side co-evolution; when `enabled`, `algorithms` is ignored. |
| `raw_config` | `Dict[str, Any]` | `{}` | Advanced free-form overlay (the middle layer of the merge above). |
| `label` | `str` | `""` | Human label. Sanitised to `[A-Za-z0-9 _\-,.:]` and truncated to 200 chars. |

### `budget` — `BudgetSpec`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | `str` | `"env_steps"` | One of `"env_steps"`, `"eval_episodes"`, `"generations"`. Must be supported by every selected algorithm (e.g. `"generations"` is rejected for the step-based AgileRL trainers). |
| `value` | `int` | `50000` | Budget amount; must be > 0. |
| `num_seeds` | `int` | `1` | Independent seeds run per algorithm; must be >= 1. |
| `budget_on_consumed` | `bool` | `false` | Opt-in real-tick budgeting (`env_steps` only). Supported by EA `simple`, CMA-ES, PSO, co-evolution, NEAT/HyperNEAT/ES-HyperNEAT, and QDax on the numpy/PettingZoo scoring path; rejected by EA `mu±lambda` and the QDax JAX-native scoring path. |

### `fitness` — `FitnessSettings`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `aggregation_method` | `str` | `"sum"` | One of `"sum"`, `"mean"`, `"min"`, `"max"`, `"team"`. Ignored by MARL trainers, whose reward comes straight from the environment. |
| `team_mapping` | `Dict[str, str] \| null` | `null` | `agent -> team` map; required when `aggregation_method="team"`. |

### `recording` — `RecordingSettings`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `false` | Turn champion-rollout recording on. |
| `fps` | `int` | `20` | Frames per second; must be > 0. |
| `fmt` | `str` | `"mp4"` | `"mp4"` (needs `imageio[ffmpeg]`) or `"gif"`. |
| `mode` | `str` | `"count"` | Cadence policy: `"count"`, `"every"`, `"interval"`, or `"final"`. |
| `every_n_steps` | `int \| null` | `null` | Env-step interval; required when `mode="interval"`, must be >= 1. |
| `max_recordings` | `int \| null` | `null` | Cap for `mode="count"` (10 captures when unset); must be >= 1. |
| `max_dim` | `int \| null` | `null` | Downscale each frame's larger axis to this many pixels before encoding. |

See [`logging_and_analysis.md`](logging_and_analysis.md#framerecorder) for how
these map onto `FrameRecorder` and the shared recording cadence.

### `parallelism` — `ParallelismSettings`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `num_workers` | `int` | `1` | Parallel evaluation workers; must be >= 0 (`0`/`1` = serial). Applies to population-based trainers; MARL trainers ignore it and parallelise via `num_envs`. |
| `parallel_backend` | `str` | `"process"` | `"process"` or `"dask"`. |
| `scheduler_address` | `str \| null` | `null` | Remote Dask scheduler (`tcp://...` URL or a scheduler-file path). Requires `parallel_backend="dask"`; empty strings normalise to unset. |

### `coevolution` — `CoevolutionSettings`

When `enabled=true` the experiment runs as a multi-side co-evolution and the
top-level `algorithms` list is ignored — each side names its own optimizer.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | `bool` | `false` | Switch on co-evolution mode. |
| `mode` | `str` | `"competitive_2pop"` | One of `"competitive_2pop"`, `"self_play_hof"`, `"cooperative"`. |
| `schedule` | `str` | `"synchronous"` | `"synchronous"` or `"alternating"`. |
| `k_opponents` | `int` | `3` | Opponents sampled per evaluation; must be >= 1. |
| `aggregation` | `str` | `"mean"` | One of `"mean"`, `"min"`, `"max"`, `"sum"`. |
| `alternating_generations` | `int` | `1` | Generations per side under the `"alternating"` schedule; must be >= 1. |
| `reporting_side` | `str \| null` | `null` | Side whose champion becomes the result policy (defaults to the first side). |
| `sides` | `List[CoevolutionSideSpec]` | `[]` | At least two sides are required. |

#### `coevolution.sides[]` — `CoevolutionSideSpec`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | `str` | **required** | Side name; must be unique within the spec. |
| `algorithm` | `str` | **required** | Catalog algorithm name, or `"frozen"`. Must be eligible as a co-evolution optimizer. |
| `role_prefixes` | `List[str]` | `[]` | Env-agent prefixes this side controls (e.g. `["adversary"]`). Every team must be covered by some side. |
| `hyperparameters` | `Dict[str, Any]` | `{}` | Per-side optimizer overrides. |

## Examples

### Standard experiment (two algorithms, three seeds)

```json
{
  "environment": "simple_tag_v3",
  "algorithms": ["ea", "maddpg"],
  "env_config": {
    "num_good": 1,
    "num_adversaries": 3,
    "num_obstacles": 2
  },
  "hyperparameters": {
    "ea": { "population_size": 100, "mutation_sigma": 0.2 },
    "maddpg": { "batch_size": 64 }
  },
  "budget": { "mode": "env_steps", "value": 200000, "num_seeds": 3 },
  "fitness": {
    "aggregation_method": "team",
    "team_mapping": {
      "adversary_0": "pursuers",
      "adversary_1": "pursuers",
      "adversary_2": "pursuers",
      "agent_0": "evaders"
    }
  },
  "recording": { "enabled": true, "mode": "count", "max_recordings": 5, "fps": 20, "fmt": "mp4" },
  "parallelism": { "num_workers": 8, "parallel_backend": "process" },
  "label": "EA vs MADDPG on simple_tag"
}
```

### Minimal experiment

Everything except `environment` has a default, so the smallest valid spec is:

```json
{ "environment": "simple_spread_v3", "algorithms": ["ea"] }
```

### Co-evolution experiment

```json
{
  "environment": "simple_tag_v3",
  "algorithms": [],
  "budget": { "mode": "generations", "value": 50, "num_seeds": 1 },
  "coevolution": {
    "enabled": true,
    "mode": "competitive_2pop",
    "schedule": "synchronous",
    "k_opponents": 3,
    "aggregation": "mean",
    "reporting_side": "pursuers",
    "sides": [
      { "name": "pursuers", "algorithm": "ea", "role_prefixes": ["adversary"] },
      { "name": "evaders", "algorithm": "cma-es", "role_prefixes": ["agent"] }
    ]
  },
  "label": "coevolution pursuers vs evaders"
}
```

## Validation

A spec is validated in two layers (defense in depth):

1. **Pydantic** at the HTTP boundary — types, required fields, and the field
   constraints above (`fmt`, `mode`, numeric bounds).
2. **`validate_spec`** against the catalog — environment / algorithm existence
   and availability, budget-mode support per algorithm, aggregation support,
   env-config parameter types and ranges, co-evolution team coverage and
   optimizer eligibility, plus the numeric guards on budget / seeds / recording
   / parallelism.

The Setup page runs both client-side before enabling submit; the service runs
`validate_spec` again server-side on `POST /experiments`. Call
[`POST /experiments/validate`](service_api.md#post-experimentsvalidate) for a
dry run. The report is field-addressed:

```json
{
  "ok": false,
  "issues": [
    {
      "path": "budget.mode",
      "code": "budget_mode_unsupported",
      "message": "'maddpg' does not support budget mode 'generations'.",
      "severity": "error"
    }
  ]
}
```

Issues with `severity: "warning"` are non-blocking — for example an unknown
env-config key, or heterogeneous observations that will be auto-padded with
SuperSuit.

See also [`service_api.md`](service_api.md) for the endpoints that consume this
schema, and [`api_specification.md`](api_specification.md) for the
`TrainingConfig` / `CoEvolutionConfig` objects the runner builds from it.
