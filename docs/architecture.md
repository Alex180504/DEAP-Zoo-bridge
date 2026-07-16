# Architecture Documentation

## Overview

This document provides an overview of the architectural design for the DEAP-PettingZoo Bridge library. The architecture follows object-oriented design principles with clear separation of concerns, enabling evolutionary algorithms to seamlessly work with multi-agent reinforcement learning environments.

## Architecture Diagrams

All diagrams are created using PlantUML and located in `docs/UML/`.

### 1. Class Diagram (`class_diagram.puml`)

**Purpose**: Shows the overall system architecture, class relationships, and main components.

**Key Components**:

#### Genome Package (`deap_pettingzoo.genomes`)
- **GenomeBase**: Abstract base class defining the genome interface
- **FixedTopologyGenome**: Concrete implementation for fixed neural network architectures
  - Stores network architecture specification
  - Manages weight vectors as flat arrays
  - Provides mutation and validation methods
  - Compatible with DEAP's genetic operators

#### Policy Package (`deap_pettingzoo.policies`)
- **BasePolicy**: Abstract interface for all policy types
- **FeedforwardNeuralPolicy**: Converts genome to executable neural network
  - Builds network layers from genome weights
  - Handles observation preprocessing
  - Produces actions matching environment's action space
  - Supports both discrete and continuous action spaces

#### Environment Package (`deap_pettingzoo.environments`)
- **PettingZooWrapper**: Standardized interface for PettingZoo parallel-API environments
  - Manages environment lifecycle (reset, step, close)
  - Provides observation/action space information
  - Environment lookup is driven by the open **env registry**
    (`environments.registry`): `env_name` strings resolve to an
    `EnvSpec(factory, default_episode_len, ...)` entry.  Third-party
    envs are plugged in via `register_env(name, factory, ...)`.
- **Registry / builtin**: `environments.registry` holds the open
  dispatch table; `environments.builtin` auto-registers every installed
  MPE2 env (tagged `mpe2`) and `aquarium_v0` when `marl-aquarium` is
  available.  Experiment-side research code can register further envs via
  the same `register_env` call (e.g. the aquarium suite registers its
  wrapped `aquarium_coevolution_v0` variant).
- **SeedManager**: Ensures reproducible evaluations
  - Generates deterministic seeds for episodes
  - Manages RNG state
  - Critical for research reproducibility

#### Evaluator Package (`deap_pettingzoo.evaluators`)
- **BaseEvaluator**: Abstract evaluator interface
- **SingleEpisodeEvaluator**: Runs one episode and calculates fitness
  - Creates policies from genomes
  - Executes episode rollout
  - Aggregates rewards into fitness
- **MultiEpisodeEvaluator**: Runs multiple episodes for robust fitness estimation
  - Uses SeedManager for reproducibility
  - Calculates statistical measures (mean, std, min, max)
  - Provides configurable aggregation methods
- **ParallelEvaluator**: Parallelizes population evaluation
  - Uses multiprocessing for speedup
  - Distributes individuals across worker processes

#### Integration Package (`deap_pettingzoo.integration`)
- **DEAPToolboxBuilder**: Creates and configures DEAP toolbox
  - Registers individual/population creators
  - Sets up genetic operators
  - Integrates evaluation function
- **EvolutionRunner**: Manages the evolutionary process
  - Executes generation loop
  - Handles logging and checkpointing
  - Manages statistics and hall of fame

#### Utils Package (`deap_pettingzoo.utils`)
- **Logger**: Records experiment data
- **Visualizer**: Creates plots and renders behaviors
- **Checkpointer**: Saves/loads evolution state

#### Baselines Package (`deap_pettingzoo.baselines`)
- **BaselineTrainer / BaselineEvaluator**: Unified abstract API for all
  algorithms.
- **EATrainer**: DEAP-backed evolutionary trainer.
- **AgileRLTrainer** (+ `MADDPGTrainer`, `MATD3Trainer`, `IPPOTrainer`,
  `GroupedIPPOTrainer`): In-process multi-agent RL trainers backed by
  [AgileRL](https://github.com/AgileRL/AgileRL). Emit per-training-interval
  fitness rows to `metrics.csv` (via `ExperimentLogger.row("metrics", ...)`)
  alongside the existing `episodes.csv`, `summary.json`, and agent checkpoint.
  Exceptions in `agent.learn()` are dual-emitted to both the Python
  `logging` root logger (with `exc_info=True` for stack traces in
  `service.log`) and `warnings.warn` (for user-visible console output).
  The dependency is optional (`pip install 'deap-pettingzoo[rl-baselines]'`)
  and imported lazily inside `train()`.
- **EvolveHPOTrainer** (`MADDPGHPOTrainer`, `MATD3HPOTrainer`,
  `IPPOHPOTrainer`, `GroupedIPPOHPOTrainer`): Adds evolutionary HPO to
  `AgileRLTrainer` — a population of agents evolved by tournament selection
  and AgileRL `Mutations`. Off-policy variants use a replay-buffer rollout
  loop; the on-policy IPPO variants share the
  `agilerl_onpolicy.train_agent_on_policy` rollout helper with the
  co-evolution RL-HPO side. Threads `progress_callback` through to the live
  UI feed and has a full `ExperimentLogger` lifecycle: per-generation
  `metrics.csv` rows, `summary.json`, and agent checkpoint. All four are
  registered in the catalog (`*-hpo`) and route to `RLEvolveHPOSideState`
  (`coevolution_optimizer="rl_hpo"`) when used as a co-evolution side.
- **AgileRLPolicyWrapper**: Adapts trained AgileRL algorithms to
  `PolicyProtocol` using a rotation counter over `possible_agents`.
- **UnifiedEvaluator**: Single evaluator that works with any
  `PolicyProtocol` implementation. Optionally tracks env-info task metrics
  per episode (`info_keys`, episode-max aggregation) on
  `EvalResult.info_metrics`.
- **RecordingController / ProbeController** (`recording.py` / `probing.py`):
  Cross-cutting champion observers riding the shared
  `BaselineTrainer._on_progress(callback, event, champion)` hook. The
  recorder captures rollout videos on an algorithm-agnostic cadence; the
  prober runs `ProbeConfig.n_episodes` held-out episodes of the frozen
  champion every `every_n_steps` cumulative env steps — off-budget, on a
  separate env — and appends rows to `<run_dir>/probe.csv`. Probe rows are
  merged into `RunRecord.metrics` by the analysis loaders, which is what
  puts frozen-policy learning curves (`probe_reward_mean` vs `env_steps`)
  on the Compare page / rliable bridge for every algorithm family on one
  yardstick.
- **ComparisonRunner**: Orchestrates matched-budget EA-vs-RL studies.

**Design Patterns Used**:
- **Abstract Factory**: GenomeBase, BasePolicy, BaseEvaluator
- **Strategy**: Different fitness aggregation methods
- **Builder**: DEAPToolboxBuilder
- **Wrapper**: PettingZooWrapper, AgileRLPolicyWrapper
- **Singleton-like**: SeedManager (per evaluator)

### Catalog Package (`deap_pettingzoo.catalog`)

The catalog is the **single source of truth** for algorithm, environment,
coevolution, and compatibility metadata. It is dependency-free at import time
(uses `importlib.util.find_spec` probes — never imports torch/agilerl/jax).

**Key components**:
- **`schema.py`**: `ParamSpec`, `AlgorithmSpec`, `EnvironmentSpec`, `TeamSpec`,
  `ValidationReport` — frozen dataclasses with `coerce`/`validate`/`to_dict`.
- **`algorithms.py`**: Registry of all algorithm definitions (EA, CMA-ES, PSO,
  MADDPG, MATD3, IPPO, GroupedIPPO) with canonical names, aliases, hyperparameter
  schemas, capabilities, and coevolution eligibility. Trainer-class resolution
  is lazy (only imported when actually training).
- **`environments.py`**: Wraps the env registry, adding `ParamSpec` schemas for
  `env_config` and declarative `TeamSpec` metadata for multi-team envs.
- **`coevolution.py`**: Enumerated modes, schedules, aggregations, samplers, and
  per-side optimizer eligibility.
- **`compatibility.py`**: Algorithm↔environment compatibility matrix sourced
  from catalog metadata (not trainer instances).
- **`introspection.py`**: Hybrid schema discovery — `signature_defaults` +
  declarative overlay.
- **`availability.py`**: `module_available` via `find_spec` probe.

### Service Package (`deap_pettingzoo.service`)

The service package implements the experiment lifecycle backend:

- **`app.py`**: FastAPI application with lifespan-managed dispatcher. Serves the
  catalog as JSON (so the frontend never imports ML deps) and exposes the full
  REST surface for experiment submission, queue management, progress polling,
  and stdout streaming.
- **`models.py`**: Pydantic v2 models (`ExperimentSpecModel`, `BudgetSpecModel`,
  etc.) with `to_dataclass()` bridge methods to the baseline layer.
- **`db.py`**: Async SQLite (WAL mode) — experiment CRUD, FIFO queue ordering
  via an atomic `dequeue_next` claim (one `UPDATE ... RETURNING` flips
  `queued` -> `running`, no two-step race), crash recovery (stale `running` ->
  re-queued `queued`, not `failed` — see `docs/planned_features.md`). All I/O
  through `asyncio.to_thread`.
- **`dispatcher.py`**: Async dispatch loop — dequeues experiments, spawns the
  runner as a subprocess in its own process group/session, monitors
  completion against a per-experiment timeout derived from the spec's
  budget, and handles cancel/terminate by killing the whole process group
  (not just the immediate process). The loop body and pre-spawn filesystem
  work are wrapped in `try/except` so one hung/erroring experiment can't wedge
  the queue. `worker_limit=1` (sequential) with a seam for future concurrency.
- **`runner.py`**: Subprocess entry point — loads spec -> re-validates ->
  `build_configs` -> iterates `SubRun`s -> calls `trainer.train(progress_callback=...)`
  -> writes `_progress.json` and `_result.json`. Exits 0 on success, non-zero on
  failure.
- **`logging_config.py`**: Centralised *diagnostic* logging (exceptions,
  warnings, lifecycle, component activation) — distinct from the
  *experiment-results* logging in `deap_pettingzoo.utils.logging`. See the
  "Logging & observability" subsection below.

#### Logging & observability

Diagnostic logging is configured once per process via
`logging_config.configure_logging` (stdlib `logging` + `dictConfig`). Two
processes configure independently:

- the **app process** (`service.run`) logs to **stdout** (captured by
  `docker logs`) *and* a rotating `<DEAP_LOG_DIR>/service.log` file on the
  results volume;
- the **runner subprocess** (`runner.main`) logs to stdout only — the dispatcher
  already redirects that to `<experiment-dir>/_stdout.log`, so per-experiment
  diagnostics are retrievable per experiment.

`logging.captureWarnings(True)` routes every `warnings.warn(...)` in the stack
(parallel/device checks, coevolution callbacks, catalog availability, …) to the
`py.warnings` logger. Module loggers (`logging.getLogger(__name__)`) give the
natural `deap_pettingzoo.service.dispatcher` / `deap_pettingzoo.baselines.*`
hierarchy.

**The verbose switch** (`--verbose` / `DEAP_LOG_LEVEL`):

| Mode | Level | Logs |
|------|-------|------|
| default | `WARNING` | warnings (incl. captured `warnings.warn`) + exceptions only |
| `--verbose` / `-v` | `INFO` | + normal operation: experiment lifecycle, which components are active (env, algorithms, budget, parallelism, coevolution) |
| `-vv` / `--log-level debug` | `DEBUG` | + resolved hyperparameters, per-request / config-merge detail |

Format is plain text by default; set `DEAP_LOG_FORMAT=json` for structured
single-line JSON records. Retrieval: `GET /logs` (tails `service.log`),
`GET /experiments/{id}/stdout` (per-experiment), `docker compose logs service`,
or the volume file directly.

**Data flow (two-process architecture)**:

```
Streamlit UI                    FastAPI Service
(no torch/agilerl/jax)          (full ML stack)
       │                              │
       │  GET /catalog                │
       ├─────────────────────────────>│  catalog.to_json()
       │<─────────────────────────────┤
       │                              │
       │  POST /experiments           │
       ├─────────────────────────────>│  validate -> db.insert -> queue
       │<─────────────────────────────┤  201 {id, status}
       │                              │
       │  GET /experiments/{id}/progress
       ├─────────────────────────────>│  db.get -> _progress.json
       │<─────────────────────────────┤  {current, total, ...}
       │                              │
       │                              │  dispatcher dequeue
       │                              │  Popen runner.py
       │                              │  trainer.train(callback)
       │                              │  -> _progress.json
       │                              │  -> _result.json
```

---

### 2. Genome -> Policy -> Action Sequence (`sequence_genome_to_action.puml`)

**Purpose**: Shows how a genome is converted into a policy and how actions are generated from observations.

**Flow**:

1. **Initialization Phase**:
   - Evolution algorithm creates DEAP individual
   - Individual contains FixedTopologyGenome
   - Genome validates architecture and initializes weight vector

2. **Policy Creation Phase**:
   - Policy receives genome and environment specification
   - Extracts weight vector from genome
   - Parses network architecture
   - Splits weights into layer matrices and bias vectors
   - Configures activation functions
   - Retrieves action/observation spaces from environment

3. **Episode Execution Phase**:
   - Environment resets with seed
   - For each step:
     - Agent receives observation
     - Policy processes observation (normalize, reshape)
     - Forward pass through network layers
     - Apply activation functions (ReLU, tanh, sigmoid)
     - Process output to match action space format
     - Return action to environment

4. **Action Space Handling** (via `policies.action_heads`):
   - **Discrete** (`DiscreteHead` / `MaskedDiscreteHead`): argmax over `n` logits
   - **MultiDiscrete** (`MultiDiscreteHead`): split `sum(nvec)` logits into one
     chunk per sub-action and argmax each -> integer vector
   - **Continuous** (`BoxHead`): apply tanh and rescale to `[low, high]`
   - **Stochastic (gradient-free search)** (`SoftmaxSamplingHead` /
     `MultiDiscreteSoftmaxSamplingHead`, via `make_head(..., stochastic=True)`):
     sample from `softmax(logits / temperature)` instead of argmax, so EA/CMA/PSO/QD
     see a smooth fitness landscape rather than an argmax step function; scored
     evaluation switches back to the deterministic argmax heads

**Key Insights**:
- Genome is passive data structure
- Policy is active behavioral component
- Clear separation allows genome to be evolved while policy handles execution
- Observation/action processing ensures compatibility with any environment

---

### 3. Single Episode Evaluation Sequence (`sequence_single_episode_evaluation.puml`)

**Purpose**: Details the process of evaluating a single individual in one episode.

**Flow**:

1. **Evaluation Start**:
   - Evaluator receives individual from DEAP
   - Creates policies for all agents from the genome

2. **Episode Reset**:
   - Gets episode seed from SeedManager
   - Resets environment with seed
   - Initializes agent positions and states
   - Receives initial observations

3. **Episode Loop** (until done or max steps):
   - Collect actions from all policies
   - Execute step in environment
   - Agents update physics, check collisions, calculate rewards
   - Accumulate rewards per agent
   - Increment step counter

4. **Fitness Calculation**:
   - Aggregate rewards based on strategy:
     - **Sum**: Total team performance
     - **Mean**: Average agent performance
     - **Team-specific**: Competitive scenarios (team_A - team_B)
     - **Min**: Worst-case (for cooperative tasks)
   - Return fitness as tuple (DEAP requirement)

**Configuration Options**:
- `max_steps`: Episode length limit
- `fitness_aggregation`: How to combine agent rewards
- `seed`: Optional deterministic seed

**Key Insights**:
- Single episode can be noisy due to environment stochasticity
- Fitness aggregation method depends on task type
- Step limit prevents infinite episodes
- Tuple return format required by DEAP

---

### 4. Multi-Episode Evaluation Sequence (`sequence_multi_episode_evaluation.puml`)

**Purpose**: Shows robust fitness estimation through multiple episodes with seed management.

**Flow**:

1. **Initialization**:
   - Creates SeedManager with base seed
   - Creates SingleEpisodeEvaluator
   - Configures number of episodes and aggregation method

2. **Multi-Episode Evaluation**:
   - Generates deterministic seeds for all episodes
   - For each episode:
     - Get episode-specific seed
     - Run SingleEpisodeEvaluator with that seed
     - Log fitness value
     - Accumulate episode fitnesses

3. **Statistical Aggregation**:
   - Calculate mean, std, min, max across episodes
   - Select final fitness based on aggregation method:
     - **Mean**: Most common, balances performance
     - **Median**: Robust to outliers
     - **Worst-case**: Conservative estimate
     - **Best-case**: Optimistic estimate
   - Return final fitness to DEAP

**Reproducibility Guarantees**:
- Base seed -> deterministic episode seeds
- Same individual + same base seed -> identical fitness
- All individuals evaluated on same set of seeds (fair comparison)
- Seeds logged with results for reproducibility

**Key Insights**:
- Reduces fitness variance from environment stochasticity
- Critical for fair individual comparison
- Trade-off: Computational cost vs. fitness accuracy
- Typical: 3-10 episodes for good balance

---

### 5. Full Evolutionary Loop Sequence (`sequence_full_evolutionary_loop.puml`)

**Purpose**: Complete workflow from setup through evolution to results analysis.

**Flow**:

1. **Setup Phase**:
   - User configures experiment
   - EvolutionRunner creates DEAP toolbox
   - Registers genome, individual, population creators
   - Registers genetic operators (selection, crossover, mutation)
   - Creates ParallelEvaluator with MultiEpisodeEvaluator
   - Initializes Logger and Checkpointer
   - Generates initial random population

2. **Evolution Phase** (for each generation):

   a. **Parallel Evaluation**:
      - Distribute population across worker processes
      - Each worker runs MultiEpisodeEvaluator
      - Collect and assign fitnesses

   b. **Statistics & Logging**:
      - Compile generation statistics (mean, std, min, max fitness)
      - Log to CSV/JSON
      - Update hall of fame

   c. **Checkpointing** (periodic):
      - Save population state
      - Save evolution metadata
      - Enable resume capability

   d. **Visualization** (periodic):
      - Plot fitness evolution
      - Plot population diversity

   e. **Selection**:
      - Tournament, roulette, or other selection method
      - Select individuals for next generation

   f. **Crossover**:
      - Apply crossover with probability
      - Invalidate fitness of modified individuals

   g. **Mutation**:
      - Apply mutation with probability
      - Invalidate fitness of modified individuals

   h. **Population Replacement**:
      - Replace old population with offspring

3. **Results Phase**:
   - Extract best individual from hall of fame
   - Save final results and statistics
   - Create final visualizations
   - Save final checkpoint

4. **Post-Analysis** (optional):
   - Test best policy
   - Compare with RL baselines
   - Statistical significance testing
   - Ablation studies

**Key Components Interaction**:
- **EvolutionRunner** orchestrates entire process
- **ParallelEvaluator** provides speedup
- **Logger** ensures data persistence
- **Checkpointer** enables long-running experiments
- **Visualizer** provides insight during evolution

**Research Workflow**:
1. Multiple independent runs (different seeds)
2. Statistical significance testing
3. Comparison with RL baselines
4. Ablation studies (test component contributions)
5. Generalization testing (different environments)

---

## Design Decisions

### 1. Genome as Data, Policy as Behavior
- **Rationale**: Clear separation of what is evolved (genome) from what executes (policy)
- **Benefit**: Same genome can create different policy types
- **Trade-off**: Slight overhead in policy creation

### 2. Fixed Topology and Neuroevolution
The library supports both fixed- and dynamic-topology representations behind the
same `PolicyProtocol` + `BaselineTrainer` + `SideState` contracts:

- **Fixed topology** — EA / CMA-ES / PSO evolve a flat weight vector over a
  predefined MLP. Simple and fast, and sufficient for most pursuit-evasion tasks.
- **Dynamic-topology neuroevolution** (in `deap_pettingzoo/contrib/neat/`, lazy —
  requires `pip install deap-pettingzoo[neat]`):
  - **NEAT** — neuroevolution of augmenting topologies via `neat-python`
    (genomes add/remove nodes and connections during evolution).
    Supports parallel genome evaluation via `num_workers` and
    champion-rollout recording via `RecordingConfig` on `TrainingConfig`.
  - **HyperNEAT** — CPPN (Compositional Pattern Producing Network) evolved
    via NEAT, queried at fixed substrate coordinates to build a larger
    phenotype ANN. Geometric regularities in the substrate maps are
    exploited by the CPPN's periodic/Gaussian activations.
  - **ES-HyperNEAT** — Evolvable-Substrate HyperNEAT: the CPPN is queried
    via a quadtree variance-analysis algorithm that **discovers** where
    to place hidden nodes, rather than requiring them to be pre-specified
    in the substrate. The result is a network whose topology can vary
    from generation to generation.
- **AgileRL HPO** — MADDPG/MATD3 with evolvable network architectures
  and hyperparameter mutation via `agilerl.hpo`. Population-based training
  with tournament selection and architecture/parameter mutations. Champion
  exposed as opponent in co-evolution (`RLEvolveHPOSideState`).
- **Rationale**: Fixed topology is sufficient for most pursuit-evasion tasks;
  dynamic topology is available as an advanced research option for
  investigating the bias-variance trade-off of architecture search in
  multi-agent settings.

### 3. Multi-objective neuroevolution
- Support for NSGA-II and SPEA2 via `FitnessSpec` / `ObjectiveSpec`
  (reusable across evaluators).
- Built-in parsimony objective (`compute_parsimony`) and configurable
  reward-component decomposition.
- Pareto-front tracking in `EATrainer` and `EASideState`, with
  champion selection by first-objective dominance.
- Lexicase and ε-lexicase selection available for per-episode or
  per-agent test-case vectors.

### 4. Particle Swarm Optimisation (PSO)
- Classic PSO with inertia, cognitive, and social components.
- Same `FixedTopologyGenome` representation as EA/CMA-ES.
- Fully integrated: standalone `PSOTrainer`, co-evolution `PSOSideState`,
  parallel evaluation, and recording support.
- Mirrors the CMA-ES `generate/update` toolbox pattern.

### 5. Multi-Episode Evaluation
- **Required**: Environment stochasticity makes single episode unreliable
- **Configuration**: User chooses number of episodes (computational budget)
- **Reproducibility**: Seed management is critical

### 6. Parallel Evaluation
- **Speedup**: Population-level parallelism (embarrassingly parallel)
- **Alternative**: Could parallelize episodes, but more complex
- **Trade-off**: Memory overhead for multiple environment copies

### 7. Modular Evaluator Hierarchy
- **Single -> Multi -> Parallel**: Composition pattern
- **Benefit**: Can use any level depending on needs
- **Flexibility**: Easy to add new evaluation strategies

### 8. Environment Wrapper
- **Standardization**: Hide PettingZoo API differences (AEC vs Parallel)
- **Benefit**: Library code doesn't need to handle both APIs
- **Extension**: Easy to add custom environment preprocessing

---

## Interface Contracts

### Genome Interface
```python
class GenomeBase:
    def get_weights() -> np.ndarray
    def set_weights(weights: np.ndarray) -> None
    def validate() -> bool
    def clone() -> GenomeBase
```

### Policy Interface
```python
class BasePolicy:
    def get_action(observation) -> action
    def reset() -> None
    def seed(seed: int) -> None
```

### Evaluator Interface
```python
class BaseEvaluator:
    def evaluate(individual) -> tuple
    # Returns tuple for DEAP compatibility
```

---

## Extension Points

1. **New Genome Types**: Inherit from GenomeBase
2. **New Policy Types**: Inherit from BasePolicy (e.g., recurrent networks)
3. **New Evaluation Strategies**: Inherit from BaseEvaluator
4. **New Fitness Functions**: Implement custom aggregation
5. **New Environments**: Use PettingZooWrapper or create custom wrapper
6. **New Opponent Sources (co-evolution)**: Implement the
   `OpponentPool` Protocol (`sample_opponent(rng) -> PolicyProtocol`,
   `sample_opponents(rng, k)`, `update(population_like)`, `__len__`).
   This is the plug-in point for QDax repertoires, PSRO meta-solvers,
   or cross-framework frozen policies. See
   `docs/JAX_QDAX_INTEGRATION_NOTES.md`.
7. **New Co-evolution Optimisers**: Subclass
   `baselines.coevolution.sides.SideState`
   (see `EASideState`, `CMAESSideState`, `FrozenSideState`,
   `RLSideState`) and register it under a `Side.optimizer` token with
   `register_side_state(token, MySideState)`. `make_side_state(side)` and
   `Side.__post_init__` validation are both driven by the open
   `_SIDE_STATE_FACTORIES` registry, so the token is immediately a valid
   `Side.optimizer`.
8. **New Algorithms (plugin seam)**: Call
   `deap_pettingzoo.register_algorithm(name, *, label, family, trainer,
   capabilities, ...)` (re-exported from the package root and
   `deap_pettingzoo.catalog`). The algorithm becomes resolvable through the
   whole stack — `catalog.get_algorithm`, `list_algorithm_names`,
   `catalog.to_json`, and (via the self-healing trainer registry)
   `comparison.get_trainer`. Pass `trainer=` as the class or a dotted
   `"module:QualName"` string; add `coevolution_optimizer=` +
   `side_state=MySideState` to feed the co-evolution side factory from the
   same call. See **Adding an algorithm / side (plugin seam)** below.

### Adding an algorithm / side (plugin seam)

The catalog algorithm table (`catalog/algorithms.py::_REGISTRY`) and the
co-evolution side factory (`coevolution/sides.py::_SIDE_STATE_FACTORIES`) are
the two authoritative **open registries**; everything else — the trainer
registry, `Side` validation, `eligible_side_optimizers`, `to_json` — derives
from them. There are two registration paths:

* **Programmatic** (serial). Call `register_algorithm(...)` /
  `register_side_state(...)` at import/startup in your own process. Like a
  programmatic `register_env`, this does **not** auto-replay into spawned
  workers — it only affects the process that calls it. Use it for
  single-process runs, notebooks, and tests.
* **Entry-point** (robust for parallel/worker runs). Ship a zero-arg
  `register()` callable under the `deap_pettingzoo.algorithms` (or
  `deap_pettingzoo.coevolution_sides`) entry-point group. It is discovered
  and invoked once at catalog import **in every process** (main and workers),
  so a plugin-provided algorithm/side is available under `num_workers > 1`
  without any snapshot plumbing. A broken plugin degrades to a
  `RuntimeWarning` and never breaks catalog import.

```python
# my_pkg/plugin.py
from deap_pettingzoo import register_algorithm
from my_pkg.trainer import MyTrainer
from my_pkg.side import MySideState

def register():
    register_algorithm(
        "myalgo", label="My Algorithm", family="evolutionary",
        trainer=MyTrainer, capabilities=MyTrainer.CAPS,
        coevolution_optimizer="myalgo", side_state=MySideState,
    )
```

```toml
# my_pkg/pyproject.toml
[project.entry-points."deap_pettingzoo.algorithms"]
myalgo = "my_pkg.plugin:register"
```

### Co-evolution architecture

`CoEvolutionTrainer` is a single trainer driving 1..N co-evolving
*sides*, each owning its optimiser (`ea`, `cma`, `frozen`, `pso`, `rl`,
`rl_hpo`, `qdax_map_elites`, `qdax_cma_me`, `neat`, `hyperneat`,
`es_hyperneat`), optional Hall-of-Fame archive, and sampler policy. Three
modes are supported:

* `competitive_2pop` — classic adversarial co-evolution (e.g.
  pursuers vs evaders).
* `self_play_hof` — each side keeps its own HoF; when sampled as an
  opponent by another side, draws exclusively from that archive.
  Homogeneous-within-side obs/action topology is required because
  HoF snapshots must remain compatible with the side's policy
  factory.
* `cooperative` — teammates co-adapt; opponent sampling is optional.

**Bidirectional EA-RL co-evolution.** `Side(optimizer="rl", ...)`
plugs an AgileRL multi-agent algorithm (MADDPG / MATD3) into
the same generation loop. The RL side owns one algorithm object
(no "population"), exposes its current weights as a degenerate size-1
`RLPolicyPool` to other sides, and runs
`rollout_episodes_per_gen` rollouts + `learning_steps_per_gen` learn
steps per generation. Supported pairings: EA×EA, EA×RL, RL×RL,
plus any combination with frozen / CMA / QDax sides. Both off-policy
(MADDPG / MATD3, replay-buffer learn) and on-policy (IPPO / GroupedIPPO,
fresh-rollout PPO update via `agilerl_onpolicy.OnPolicyRolloutBuffer`)
learners are supported; on-policy sides keep no replay buffer and
collect serially. Caveat: `k_opponents` is auto-treated as 1 per RL
rollout (the RL side samples one opponent per episode). A GPU-contention
detector (`utils.device_check.warn_on_gpu_contention`) emits a warning
when a torch-CUDA RL side coexists with a JAX-GPU QDax side.

**Parallel RL co-evolution.** EA/CMA learner sides can
evaluate against RL opponents in parallel. When `num_workers > 1` and
an RL side is present, `RLPolicyPool.snapshot_opponents(rng, k)` calls
`AgileRLPolicyWrapper.make_frozen_cpu_snapshot()` once, producing a
`FrozenRLPolicyWrapper` (all actor `nn.Module`s moved to CPU, set to
`eval()`, detached — fully picklable). Workers receive one pickled
copy per episode; they do not share mutable state. An informational
`RuntimeWarning` is emitted to document the behaviour.

Parallel dispatch also extends to the RL learner itself: when
`_RLDriverContext.executor` is not `None`, `RLSideState.step_generation`
builds `rollout_episodes_per_gen` picklable `_RLRolloutPayload` objects
(one per episode), dispatches them to workers via
`executor.map(_rl_rollout_worker_fn, payloads)`, receives
`_RLTrajectory` objects back, and pushes each trajectory's
`_RLStepData` list into the local replay buffer before calling
`agent.learn()` on the driver. The worker function is a module-level
top-level callable so it is picklable for `ProcessPoolExecutor`.

Per-side-per-episode opponent sampling (`OpponentPool.sample_opponent`
returns **one** policy per side, shared across that side's team
agents) mirrors canonical self-play, reduces variance, and matches
the Protocol's contract for QDax-style single-policy repertoires.

The `SideState` abstract base isolates optimiser lifecycle
(`initialise`, `step_generation`, `current_pool`, `best_policy`,
`snapshot_best`) from the trainer's orchestration, which handles
scheduling (`synchronous` vs `alternating`), budget translation
(`ENV_STEPS` / `EVAL_EPISODES`), metrics aggregation, and the
reporting-side policy extraction.

---

## Custom Environments and Parallel Evaluation

### Registering a custom environment

Third-party or user-written environments must be registered before any
trainer or wrapper is instantiated:

```python
from deap_pettingzoo.environments.registry import register_env
from my_pkg import my_env_module  # exposes .parallel_env(**cfg)

register_env("my_env_v0", my_env_module, default_episode_len=200)
```

The `factory` argument can be either a module with a `parallel_env`
attribute or a plain callable `(render_mode=None, **cfg) -> ParallelEnv`.

### Worker subprocess propagation (`num_workers > 1`)

On **Windows** and on **macOS** with Python ≥ 3.8, `ProcessPoolExecutor`
uses the `spawn` start method: each worker is a fresh interpreter that
imports the library from scratch, so `_REGISTRY` is populated only with
the built-in environments.  A custom environment registered in the main
process is **unknown** to worker subprocesses unless explicitly
propagated.

All trainers that accept `num_workers > 1`
(`EATrainer`, `CMAESTrainer`, `CoEvolutionTrainer`) now do this
automatically:

1. Immediately before opening the worker pool they call
   `get_registry_snapshot()` to capture the current registry state as a
   picklable `list[(name, EnvSpec)]`.
2. That snapshot is passed as the `initializer` / `initargs` to
   `make_executor`, which forwards it to `ProcessPoolExecutor`.
3. Each worker calls `replay_registry_snapshot(snapshot)` once at
   startup, re-registering every entry with `overwrite=True`.

**Constraint**: every `factory` stored in `EnvSpec` must be picklable
(importable module, top-level callable, or a class instance with
`__reduce__`).  Lambdas and locally-defined closures are **not**
picklable.  `get_registry_snapshot()` now emits a `RuntimeWarning`
**naming each dropped env** whose factory is a non-module unpicklable
(lambda/closure/local class), instead of silently omitting it — so the
old cryptic "unregistered env" failure surfaces at snapshot time with the
cause. Built-in envs registered with a bare **module** object are the one
documented exception (they are re-registered in every worker at import),
so their omission stays silent. Use a module-level class or a
`functools.partial` for custom envs. The contrib `JaxmarlEnvFactory` /
`OpenSpielGameFactory` are module-level picklable classes for exactly this
reason.

**Dask**: that backend has no per-worker `initializer` hook, so the
registry snapshot rides on each worker spec/payload instead and is replayed
lazily (`replay_registry_snapshot_once`) — this works for both the
auto-built local cluster and a **remote** cluster reached via
`TrainingConfig.scheduler_address` / `DEAP_DASK_SCHEDULER`. On a remote
cluster the requirement tightens: any worker function a trainer hands to
`executor.map` (not just env factories) must resolve on the worker, so the
remote `_DaskExecutor.map` path raises immediately if
`fn.__module__ == "__main__"` rather than failing opaquely mid-cluster. See
`docs/hpc_guide.md` for the full remote-cluster prerequisites (shared
filesystem, matching library version on every node, Slurm/Docker
walkthroughs).

### Summary table

| Backend | Worker init hook | Auto-propagated |
|---------|-----------------|-----------------|
| `"process"` (default) | `ProcessPoolExecutor(initializer=…)` | Yes |
| `"dask"` (local or remote via `scheduler_address`) | None | Yes, via worker-spec snapshot replay (local + remote); custom `executor.map` functions still need to be importable, not `__main__`-defined |

---

## Docker Deployment

The two-process architecture (ML-heavy FastAPI service + dependency-light
Streamlit UI) maps onto two container images orchestrated by
`docker-compose.yml`. The split keeps ML dependencies (torch / agilerl / jax)
out of the UI image, which talks to the backend exclusively over HTTP.

### Images

| Image | Dockerfile | Contents | Entry point |
|-------|-----------|----------|-------------|
| `deap-bridge-service` | `Dockerfile.server` | Full ML stack (`.[service]` extra) + trainers + example scripts | `python -m deap_pettingzoo.service --host 0.0.0.0 --port 8100` |
| `deap-bridge-ui` | `Dockerfile.ui` | Streamlit + httpx + plotly (`.[ui]` extra, no ML deps) | `python -m streamlit run deap_pettingzoo/ui/app.py` |

Both images are built `FROM python:3.11-slim-bookworm`, run as a non-root
`deap` user, and expose a container `HEALTHCHECK` (`/health` for the service,
`/_stcore/health` for the UI). The service image installs torch from a
build-time wheel index (`TORCH_INDEX_URL`, CPU by default) and reinstalls
AgileRL's runtime dependencies explicitly, because AgileRL is installed
`--no-deps` to keep mpe2's newer pinned `pettingzoo`.

### Compose modes

Two run modes are defined in `docker-compose.yml`:

- **Mode 1 — full stack (default)**: `docker compose up`
  Both the FastAPI service and the Streamlit UI run as containers on an
  internal `deap-net` bridge network. The UI reaches the service at
  `http://service:8100`; the service port is also published to
  `127.0.0.1:8100` on the host so browser-facing export links
  (`st.link_button`) resolve. Results persist in the named `results` volume.
- **Mode 2 — native backend**: `docker compose --profile native-backend up`
  Only the UI is containerised (the `ui-native` service); it connects to a
  FastAPI backend running natively on the host via
  `host.docker.internal:8100`. Point it at your results directory with
  `RESULTS_HOST_DIR`.

The UI is published on `${STREAMLIT_PORT:-8501}` and waits for the service's
health check (`depends_on: condition: service_healthy`) before starting.

### Volumes and state

- **`results`** (named volume): mounted at `/app/results` — read/write for the
  service, read-only for the UI. Holds experiment outputs and the rotating
  `_logs/service.log`.
- **SQLite queue DB**: kept at `/app/data/_service.db` (`DEAP_SERVICE_DB`),
  *outside* the results volume, so it stays writable even when the volume is
  root-owned.

### Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVICE_PORT` | `8100` | Host port the service publishes to `127.0.0.1` |
| `STREAMLIT_PORT` | `8501` | Host port the UI publishes |
| `DEAP_RESULTS_ROOT` | `/app/results` | Results/output root |
| `FASTAPI_URL` | `http://service:8100` | Backend URL the UI calls |
| `DEAP_BROWSER_API_URL` | `http://localhost:${SERVICE_PORT}` | Backend URL used in browser-facing links |
| `DEAP_LOG_LEVEL` | `warning` | Diagnostic log level (`warning`/`info`/`debug`) |
| `DEAP_LOG_FORMAT` | `plain` | `plain` or `json` structured logs |
| `DEAP_ENABLE_ADVANCED` | `1` | Enables the Setup -> Advanced surface; set `0` to 403 the install/script endpoints |

### GPU (CUDA) deployment

The default build is CPU-only. Layer the `docker-compose.gpu.yml` override to
run the service on the GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

This builds a `deap-bridge-service:cuda` image with a CUDA torch wheel index
(`TORCH_INDEX_URL`, default `cu121`) and reserves all host GPUs for the service
container. Prerequisites: an NVIDIA GPU + recent driver, the NVIDIA Container
Toolkit configured for Docker (Docker Desktop with the WSL2 backend on
Windows). See the "GPU support" section of the README for setup and checks.

### Logging retrieval

Service logs go to container stdout *and* the rotating
`$DEAP_RESULTS_ROOT/_logs/service.log` file on the results volume. Retrieve
them via:

```bash
docker compose logs -f service                       # live stdout
docker exec deap-service cat /app/results/_logs/service.log
curl 'http://127.0.0.1:8100/logs?lines=200'          # via the API
```

### Distributed backends

For multi-node execution, `docker/dask-cluster/` and `docker/slurm-cluster/`
provide compose stacks / images for a Dask scheduler + workers and a Slurm
cluster (CPU and GPU variants). These pair with `TrainingConfig.scheduler_address`
/ `DEAP_DASK_SCHEDULER` — see `docs/hpc_guide.md` for the full walkthrough.

---

## Validation & Testing Strategy

1. **Unit Tests**: Each class in isolation
2. **Integration Tests**: Component interactions
3. **End-to-End Tests**: Full evolutionary runs (small scale)
4. **Reproducibility Tests**: Same seeds -> same results
5. **Performance Tests**: Benchmark evaluation throughput
