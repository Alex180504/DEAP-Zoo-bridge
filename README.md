# DEAP-bridge

`deap_pettingzoo` (**DEAP-bridge**) is a research library for running and evaluating
**evolutionary algorithms** against **multi-agent reinforcement learning** policies
in the same enviroments. It lets [DEAP](https://github.com/DEAP/deap) evolutionary
optimisers drive [PettingZoo](https://pettingzoo.farama.org/) / MPE2
environments, and benchmarks them head-to-head against
[AgileRL](https://github.com/AgileRL/AgileRL) MARL learners — all behind one
unified trainer API, so an EA and a deep-RL run are configured, budgeted, and
evaluated the same way.

It ships as a plain Python library, and also as a two-process app (FastAPI
service + Streamlit UI) that you can bring up with a single `docker compose up`.

> Not yet publicly released. Backward compatibility is not a constraint.

## What it does

- **Evolve neural policies** — a genome is a flat weight vector that builds an
  MLP policy with a per-action-space head (Discrete / masked / MultiDiscrete /
  Box). Fitness comes from environment rollouts.
- **One API for EA and RL** — `TrainingConfig` / `TrainingResult` /
  `BudgetMode` give every backend the same surface. Budget can be expressed as
  generations, evaluation episodes, or environment steps, so comparisons are
  fair by construction.
- **Head-to-head comparisons** — `compare_baselines` trains several algorithms
  under a matched budget across seeds and scores them all on one shared
  evaluator.
- **Co-evolution** — pit populations (or an RL learner) against each other with
  `synchronous` / `alternating` schedules, opponent pools, and hall-of-fame
  archives.
- **Rollout recording** — opt-in champion-rollout capture (`RecordingConfig` /
  `RecordingMode`) renders `env.render('rgb_array')` frames to mp4/gif on a
  chosen cadence, replayable later in the UI.
- **Analysis & reporting** — aggregation, normalisation, plots, an
  [rliable](https://github.com/google-research/rliable) bridge, and
  self-contained HTML/CSV reports.
- **Interactive UI** — a Streamlit app to browse runs, compare learning curves
  with confidence intervals and pairwise significance tests, sweep
  hyperparameters, replay recorded rollouts, and generate reports.
- **Parallelism** — population trainers run single-process, in parallel across
  local workers, on a local Dask cluster, or across a multi-node Slurm
  allocation via a remote scheduler.

## Supported algorithms

| Family | Algorithms | Backend |
| --- | --- | --- |
| Evolutionary | EA (genetic algorithm), CMA-ES, PSO | DEAP |
| Neuroevolution | NEAT, HyperNEAT, ES-HyperNEAT | neat-python (optional) |
| MARL (off-policy) | MADDPG, MATD3 | AgileRL |
| MARL (on-policy) | IPPO, IPPO (grouped) | AgileRL |
| MARL + evolutionary HPO | MADDPG / MATD3 / IPPO / IPPO-grouped `*-hpo` | AgileRL |
| Quality-diversity | MAP-Elites | QDax (optional, JAX) |

The evolutionary and RL trainers can also serve as **co-evolution sides**, so
you can, for example, evolve a predator population against an RL-trained prey.

## Included environments

- **MPE2** (built-in): `simple_v3`, `simple_adversary_v3`, `simple_crypto_v3`,
  `simple_push_v3`, `simple_reference_v3`, `simple_speaker_listener_v3`,
  `simple_spread_v3`, `simple_tag_v3`, `simple_world_comm_v3`.
- **Aquarium** (`aquarium_v0`, optional `[aquarium]` extra) — a predator/prey
  fish-swarm env with a dynamic agent set.
- **OpenSpiel / JAXMarl** via the optional `contrib/` integrations
  (`[shimmy]`, `[jaxmarl]` extras).

New environments register through
[`environments/registry.py`](deap_pettingzoo/environments/registry.py); the
registry normalises AEC / parallel / JAX APIs to one parallel-shaped facade.
Any PettingZoo-compatible enviroments should be compatible.

## Install

Requires Python 3.11+. Core install:

```bash
pip install -e .
```

The RL baselines need AgileRL, which pins an older PettingZoo, so install it
without its deps first, then pull the `service` extra:

```bash
pip install --no-deps "agilerl>=2.6,<3"
pip install -e ".[service]"
```

On Windows there's a one-command helper that does exactly this:

```powershell
.\install.ps1 service     # backend + RL trainers
.\install.ps1 ui-full     # backend + Streamlit UI
.\install.ps1 all         # every optional extra
```

Useful extras: `[analysis]` (plots, reports, rliable), `[ui]` (Streamlit
frontend, no ML deps), `[dask]` (distributed execution), `[aquarium]`,
`[jax]` / `[jaxmarl]` / `[qdax]`. See
[`pyproject.toml`](pyproject.toml) for the full list.

## Quick start — as a library

Train an evolutionary baseline on an MPE2 environment:

```python
from deap_pettingzoo.baselines import BudgetMode, EATrainer, TrainingConfig

config = TrainingConfig(
    env_name="simple_spread_v3",
    budget_mode=BudgetMode.EVAL_EPISODES,
    budget_value=64,
    seed=0,
    output_dir="results/quickstart_demo",
)
trainer = EATrainer(hidden_size=8, population_size=4)
result = trainer.train(config)
print(result.metrics["final_fitness"])
```

Compare an EA against a deep-RL learner under a matched budget:

```python
from deap_pettingzoo.baselines import ComparisonConfig, compare_baselines

config = ComparisonConfig(
    algorithms=["ea", "maddpg"],
    env_name="simple_spread_v3",
    budget_mode="env_steps",
    budget_value=50_000,
    num_seeds=3,
    output_dir="results/ea_vs_maddpg",
)
results = compare_baselines(config)
```

Load a finished run and render a self-contained HTML report:

```python
from deap_pettingzoo.analysis import build_report, load_experiment

experiment = load_experiment("results/quickstart_demo")
report_path = build_report(experiment, "results/quickstart_demo/report")
print(report_path)
```

More runnable scripts live in [`examples/`](examples) — including parallel evaluation,
continuous control, turn-based self-play and co-evolution EA-vs-RL experiment suites.

## Quick start — via Docker

The app runs as two containers: a FastAPI **service** (owns the task queue and runs
the backend) and a Streamlit-based **UI** service.

```bash
docker compose up
```

Then open the UI at **http://localhost:8501**. The service listens on
**127.0.0.1:8100** and results persist in a named Docker volume.

Other modes:

```bash
# UI container against a backend running natively on the host
docker compose --profile native-backend up

# GPU-enabled service
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Prefer to run the two processes yourself, no Docker? Start the service and let
the UI launch it as a sidecar:

```bash
python -m deap_pettingzoo.ui --with-service
```

> The service executes Python import paths from experiment specs and (under the
> Setup -> Advanced tab) can install dependencies and run pasted scripts.

## Distributed / HPC runs

Population trainers scale from one machine (`parallel_backend="process"`, or an
auto-built local Dask cluster) to a multi-node Slurm allocation by pointing
`TrainingConfig` at a remote Dask scheduler via `scheduler_address` (or the
`DEAP_DASK_SCHEDULER` env var). See [`docs/hpc_guide.md`](docs/hpc_guide.md)
for the walkthrough and [`examples/hpc/`](examples/hpc) for launch scripts and
Slurm `.sbatch` templates.

## Documentation
- [docs/architecture.md](docs/architecture.md) — system overview
- [docs/api/](docs/api) — API + REST service reference
- [docs/ui/](docs/ui) — UI architecture and user guide
- [docs/uml/](docs/uml) — Full UML modelling of the system

Auto-generated HTML docs are available at https://alex180504.github.io/DEAP-Zoo-bridge/.
## Testing

```bash
python -m pytest -q --no-cov
```
