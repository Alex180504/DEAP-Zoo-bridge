# Experiment Service API Reference

The experiment service exposes a REST API over HTTP (default
`http://127.0.0.1:8100`). The Streamlit frontend consumes this API;
third-party clients can use it as well.

**Base URL**: `http://127.0.0.1:8100` (override via `--host` / `--port`
CLI flags or the `FASTAPI_URL` environment variable).

**Interactive docs**: the running service serves live OpenAPI docs at `/docs`
(Swagger UI) and `/redoc`, with the raw schema at `/openapi.json`. This
reference mirrors that schema and adds the behavioural notes it cannot express.

**Content type**: `application/json` for all request and response bodies,
except the export endpoints (`application/zip` download) and
`POST /experiments/import` (`multipart/form-data` upload).

**Error model**: All error responses follow the shape:

```json
{
  "detail": "human-readable message"
}
```

Validation errors (422) include per-field details:

```json
{
  "detail": [
    {
      "loc": ["body", "env_name"],
      "msg": "unknown environment 'nonexistent_env'",
      "type": "value_error"
    }
  ]
}
```

---

## Catalog endpoints (read-only — no auth)

### `GET /health`

Service liveness check. Docker HEALTHCHECK uses this.

**Response** (200):
```json
{
  "status": "ok",
  "version": "0.1.0",
  "catalog_version": 1,
  "advanced_enabled": true
}
```

`advanced_enabled` reflects the `DEAP_ENABLE_ADVANCED` env var (default `true`).
The Setup -> Advanced UI section uses it to show/hide the dependency-install panel
and custom-script editor.

### `GET /catalog`

Full catalog snapshot — algorithms, environments, coevolution options,
and module availability. The frontend fetches this once on first render.

**Response** (200):
```json
{
  "catalog_version": 1,
  "algorithms": [ { "name": "ea", "aliases": [], "family": "evolutionary", ... } ],
  "environments": [ { "name": "simple_tag_v3", "teams": [...], "env_config": [...], ... } ],
  "coevolution": { "modes": [...], "schedules": [...], "samplers": [...], "eligible_optimizers": [...] },
  "budget_modes": ["eval_episodes", "env_steps", "generations"],
  "availability": { "rl-baselines": true, "jax": false, "neat": false, ... }
}
```

### `GET /catalog/algorithms`

List all registered algorithms.

**Response** (200): array of `AlgorithmSpec` objects.

### `GET /catalog/algorithms/{name}`

Get a single algorithm with its full hyperparameter schema.

**Path parameters**: `name` — canonical algorithm name (e.g. `ea`,
`cma-es`, `pso`, `maddpg`, `matd3`, `ippo`, `ippo-grouped`).

**Response** (200): `AlgorithmSpec` object.
**Response** (404): algorithm not found.

### `GET /catalog/environments`

List all registered environments.

**Response** (200): array of `EnvironmentSpec` objects.

### `GET /catalog/environments/{name}`

Get a single environment with its config parameters and team metadata.

**Response** (200): `EnvironmentSpec` object.
**Response** (404): environment not found.

### `GET /catalog/coevolution`

Coevolution options — modes, schedules, samplers, aggregation methods,
and per-side optimizer eligibility.

**Response** (200): coevolution options object.

### `GET /catalog/compatibility`

Algorithm<->environment compatibility check.

**Query parameters**:
- `env` (required) — environment name.
- `algorithms` (optional) — comma-separated algorithm names. If omitted,
  returns compatibility for all registered algorithms.

**Response** (200):
```json
{
  "matrix": { "ea": true, "maddpg": true, "ippo": true, ... },
  "compatible_algorithms": ["ea", "cma-es", "pso", "maddpg", ...],
  "incompatible_algorithms": [],
  "warnings": []
}
```

---

## Experiment endpoints

### `POST /experiments`

Submit a new experiment to the queue. The spec is validated server-side
(defense in depth — the Setup page also validates client-side before
calling this).

**Request body**: `ExperimentSpecModel` (see [configuration_schemas.md](configuration_schemas.md)).

**Response** (201):
```json
{
  "id": "a1b2c3d4-...",
  "status": "queued",
  "position": 3
}
```

**Response** (422): validation failed — field-level errors returned.

### `POST /experiments/validate`

Dry-run validation. Same input as `POST /experiments` but does not
enqueue. The Setup page calls this when the user clicks "Validate".

**Request body**: `ExperimentSpecModel`.

**Response** (200):
```json
{
  "ok": true,
  "errors": [],
  "warnings": ["raw-config import 'my.custom.wrapper' not verifiable"]
}
```

### `GET /experiments`

List experiments, ordered by submission time (newest first).

**Query parameters**:
- `status` — filter: `queued`, `running`, `completed`, `failed`, `cancelled`.
  Omit for all.
- `limit` — max results (default 50).
- `offset` — pagination offset (default 0).

**Response** (200): array of `ExperimentRecord`:
```json
[
  {
    "id": "a1b2c3d4-...",
    "label": "EA simple_tag test",
    "status": "completed",
    "created_at": 1718467200.0,
    "started_at": 1718467210.0,
    "finished_at": 1718467330.0,
    "run_dirs": ["results/ea_simple_tag_s1_..."],
    "algorithms": ["ea"],
    "environment": "simple_tag_v3",
    "error": null
  }
]
```

### `GET /experiments/{id}`

Get a single experiment by ID.

**Response** (200): `ExperimentRecord`.
**Response** (404): experiment not found.

### `POST /experiments/{id}/cancel`

Cancel an experiment. Queued experiments are marked `cancelled` immediately.
Running experiments have their runner subprocess terminated.

**Response** (200):
```json
{ "status": "cancelled" }
```

**Response** (404): experiment not found.
**Response** (409): experiment already in a terminal state.

### `DELETE /experiments/{id}`

Remove an experiment record. Only terminal states (`completed`, `failed`,
`cancelled`) and `queued` experiments can be deleted. Running experiments
must be cancelled first.

**Response** (200): `{ "deleted": true }`
**Response** (409): experiment is running — cancel first.

### `GET /experiments/{id}/spec`

Return the raw stored `ExperimentSpec` JSON. The UI uses this to pre-populate
the Setup form for a re-run.

**Response** (200):
```json
{ "spec": { "environment": "simple_tag_v3", "algorithms": ["ea"], "budget": { "mode": "env_steps", "value": 50000 } } }
```

**Response** (404): experiment not found.

---

## Progress & stdout

### `GET /experiments/{id}/progress`

Live progress snapshot. The Queue page polls this every ~2s.

**Response** (200):
```json
{
  "run_index": 0,
  "total_runs": 3,
  "algo": "ea",
  "seed": 42,
  "unit": "generation",
  "current": 45,
  "total": 100,
  "best_fitness": 12.34,
  "mean_fitness": 8.91,
  "wall_time_s": 67.2,
  "extra": {}
}
```

**Response** (404): experiment not found.
**Response** (200, no running run): `null`.

### `GET /experiments/{id}/stdout`

Tail the runner subprocess stdout log. Supports both byte-offset and
line-based access.

**Query parameters**:
- `offset` — byte offset to start reading from (default 0).
- `limit` — max bytes to return (default 65536).
- `lines` — if set, return the last N lines instead (overrides offset/limit).

**Response** (200):
```json
{
  "text": "[deap-bridge] training ea on simple_tag_v3 seed=42\nGen 1/100 best=3.2 mean=1.8 ...\n",
  "next_offset": 512,
  "eof": false
}
```

**Response** (404): experiment not found, or stdout log not yet created.

### `GET /logs`

Tail the backend **service** diagnostic log (`<DEAP_LOG_DIR>/service.log`,
default `<DEAP_RESULTS_ROOT>/_logs/service.log`). Same tailing contract as
`GET /experiments/{id}/stdout` — use `lines` for the last N lines, or
`offset`/`limit` to stream forward. This surfaces queue lifecycle, warnings,
and exceptions from the app process; per-experiment trainer output stays on
`GET /experiments/{id}/stdout`.

**Query parameters**:
- `offset` — byte offset to start reading from (default 0).
- `limit` — max bytes to return (default 65536).
- `lines` — if set, return the last N lines instead (overrides offset/limit).

**Response** (200):
```json
{
  "text": "2026-06-17 14:03:21 INFO  deap_pettingzoo.service.dispatcher  Experiment a1b2 running (pid=4711)\n",
  "next_offset": 4096,
  "eof": true
}
```

Verbosity is controlled by the `--verbose` flag / `DEAP_LOG_LEVEL` env var
(`warning` default -> `info` -> `debug`); format by `DEAP_LOG_FORMAT`
(`plain` default / `json`). A missing log file returns `{ "text": "",
"next_offset": 0, "eof": true }` (200), not 404.

---

## Export, import & run management

### `GET /experiments/{id}/export`

Stream a ZIP bundle of an experiment: its spec, run manifests, metrics, and —
when requested — recorded frames, TensorBoard logs, and trajectories.

**Query parameters**:
- `include_frames` — include recorded video frames (default `false`).
- `include_tb` — include TensorBoard event files (default `false`).
- `include_trajectories` — include trajectory / replay data (default `false`).

**Response** (200): `application/zip` attachment
(`Content-Disposition: attachment; filename="deap_<label>_<id>.zip"`).

**Response** (404): experiment not found.

### `GET /runs/export`

Stream a ZIP bundle of one or more run directories, addressed directly by path
(no experiment record required).

**Query parameters**:
- `run_dirs` (required) — comma-separated run directory paths.
- `include_frames`, `include_tb`, `include_trajectories` — as above.

**Response** (200): `application/zip` attachment
(`filename="deap_runs_export.zip"`).

**Response** (400): no `run_dirs` provided, or a path resolves outside the
results root.

### `POST /experiments/import`

Upload a previously exported ZIP, validate it, and restore its experiment
records and run directories.

**Request**: `multipart/form-data` with a single `file` field (the ZIP).

**Query parameters**:
- `overwrite_existing` — overwrite run directories that already exist
  (default `false`).

**Response** (200):
```json
{
  "ok": true,
  "imported": [
    {
      "original_exp_id": "a1b2c3d4-...",
      "new_exp_id": "f9e8d7c6-...",
      "label": "EA simple_tag test",
      "run_count": 3,
      "warnings": []
    }
  ]
}
```

**Response** (422): the archive failed validation — the body lists the issues.

### `POST /runs/delete`

Delete one or more run directories from disk. Each path must resolve under the
results root and contain a `manifest.json`; per-path failures are reported
individually rather than failing the whole request.

**Query parameters**:
- `run_dirs` (required) — comma-separated run directory paths.

**Response** (200):
```json
{
  "ok": true,
  "deleted": ["results/ea_simple_tag_s1_..."],
  "errors": []
}
```

`ok` is `true` only when every path was deleted. Failures appear in `errors`
as `{ "path": "...", "error": "..." }` objects (not found, missing
`manifest.json`, or outside the results root).

**Response** (400): no `run_dirs` provided.

---

## Queue control

### `GET /queue`

Queue state — which experiment is running, what's waiting, and whether
the dispatcher is paused.

**Response** (200):
```json
{
  "paused": false,
  "running_id": "a1b2c3d4-...",
  "queued_ids": ["e5f6g7h8-...", "i9j0k1l2-..."],
  "worker_limit": 1
}
```

### `POST /queue/pause`

Pause the dispatcher. The currently-running experiment continues;
no new experiments are dequeued until resumed.

**Response** (200): `{ "paused": true }`

### `POST /queue/resume`

Resume the dispatcher. Queued experiments begin dequeueing again.

**Response** (200): `{ "paused": false }`

---

## Advanced endpoints (arbitrary code/command execution — gated)

These power the **Setup -> Advanced** tab. They let an authorised user install a
custom environment package and submit a hand-written experiment script. They are
arbitrary code/command execution **by design** (trusted-input model) and are
gated behind `DEAP_ENABLE_ADVANCED` (default on). When disabled, all four return
**403**.

### `POST /env/install`

Start a dependency-install job in the backend environment. Commands are spawned
with an explicit argument list (never a shell); only an allowlist is accepted.

**Request body**:
```json
{ "manager": "pip", "action": "install", "args": ["marl-aquarium"] }
```
Allowed `(manager, action)`: `pip install`, `pip uninstall`, `git clone`.
`git clone` always writes under `<DEAP_RESULTS_ROOT>/_custom_envs/`.

**Response** (201): `{ "job_id": "…", "status": "running" }`
**Response** (400): disallowed operation or argument contains a forbidden character.
**Response** (403): advanced features disabled.

> **Caveat (containers):** packages installed this way live in the backend
> container's site-packages and are lost on rebuild. Add permanent deps to
> `Dockerfile.server`.

### `GET /env/install/{job_id}`

Poll an install job's status and tail its output (same tailing contract as
`GET /experiments/{id}/stdout`).

**Query parameters**: `offset`, `limit`, `lines` (as for stdout).

**Response** (200):
```json
{ "job_id": "…", "status": "running", "returncode": null,
  "command": "pip install marl-aquarium", "text": "Collecting marl-aquarium…",
  "next_offset": 512, "eof": false }
```
`status` is `running` | `completed` | `failed`.
**Response** (404): unknown job. **Response** (403): disabled.

### `POST /experiments/script/validate`

Static validation of a custom experiment script. **Static only** — confirms the
script parses and looks runnable; it cannot prove the script trains correctly.

**Request body**: `{ "script": "…python source…", "label": "", "entry": "script" }`

**Response** (200): a `ValidationReport` — `{ "ok": true, "issues": [...] }`.
Syntax errors -> blocking `error` issues; missing entrypoint / `examples.*` imports
-> non-blocking `warning` issues. **Response** (403): disabled.

### `POST /experiments/script`

Submit a custom script as a queued experiment. The dispatcher runs it as
`python user_script.py` in the experiment working directory (so relative
`results/…` outputs are linked to the experiment and discovered by Browse).

**Request body**: same as `…/validate`.

**Response** (201): `{ "id": "…", "status": "queued", "position": 0 }`
**Response** (422): script failed static validation (body is the `ValidationReport`).
**Response** (403): disabled.

The resulting `ExperimentRecord` reports `environment: "(custom script)"`.

---

## Data types

### `ParamSpec`

```json
{
  "name": "population_size",
  "type": "int",
  "default": 100,
  "min": 10,
  "max": 5000,
  "step": 1,
  "choices": null,
  "group": "Population",
  "label": "Population Size",
  "help": "Number of individuals per generation.",
  "advanced": false,
  "widget": "number_input"
}
```

### `AlgorithmSpec`

```json
{
  "name": "ea",
  "aliases": [],
  "family": "evolutionary",
  "hyperparameters": [ /* ParamSpec[] */ ],
  "capabilities": { "parallel": true, "coevolution_optimizer": true },
  "coevolution_optimizer": true,
  "required_extras": [],
  "available": true,
  "supported_budget_modes": ["eval_episodes", "env_steps", "generations"],
  "supported_aggregations": ["sum", "mean", "min", "max", "team"]
}
```

### `EnvironmentSpec`

```json
{
  "name": "simple_tag_v3",
  "env_config": [ /* ParamSpec[] */ ],
  "teams": [ { "name": "pursuers", "agent_prefixes": ["adversary"] },
             { "name": "evaders", "agent_prefixes": ["agent"] } ],
  "api": "parallel",
  "continuous_default": false,
  "tags": ["mpe2"],
  "required_extras": [],
  "available": true
}
```

### `ExperimentSpecModel` (request body)

See [configuration_schemas.md](configuration_schemas.md) for the full
field reference and JSON schema.

### `ExperimentRecord` (response)

| Field | Type | Description |
|---|---|---|
| `id` | string (UUID) | Unique experiment identifier. |
| `label` | string | Optional human label. |
| `status` | enum | `queued`, `running`, `completed`, `failed`, `cancelled`. |
| `created_at` | float | Unix timestamp of submission. |
| `started_at` | float? | Unix timestamp when dequeued. |
| `finished_at` | float? | Unix timestamp when completed/failed/cancelled. |
| `run_dirs` | string[]? | Discovered run directories (after completion). |
| `algorithms` | string[] | Algorithms in this experiment. |
| `environment` | string | Environment name. |
| `error` | string? | Error message if status is `failed`. |

### `ProgressSnapshot` (response)

| Field | Type | Description |
|---|---|---|
| `run_index` | int | Zero-based index of the current sub-run. |
| `total_runs` | int | Total sub-runs (algo × seed combinations). |
| `algo` | string | Algorithm for the current sub-run. |
| `seed` | int | Seed for the current sub-run. |
| `unit` | enum | `generation` or `env_step`. |
| `current` | int | Progress counter. |
| `total` | int | Budget (generations or steps). |
| `best_fitness` | float? | Best fitness so far. |
| `mean_fitness` | float? | Mean fitness so far. |
| `wall_time_s` | float | Elapsed wall time in seconds. |
| `extra` | object | Algorithm-specific extras. |
