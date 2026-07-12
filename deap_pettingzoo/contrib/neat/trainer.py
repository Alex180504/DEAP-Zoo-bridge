"""NEAT trainer — wraps ``neat.Population`` behind the :class:`BaselineTrainer` API.

:class:`NEATTrainer` is a standard :class:`~deap_pettingzoo.baselines.base.BaselineTrainer`
implementation that uses NEAT to evolve a population of small
feed-forward neural networks.

Budget translation
------------------
NEAT's generation count is the natural budget unit.  The trainer maps
:attr:`~deap_pettingzoo.baselines.base.BudgetMode.EVAL_EPISODES` and
:attr:`~deap_pettingzoo.baselines.base.BudgetMode.ENV_STEPS` to
generations using the population size and evaluation episodes.

The native-generation :class:`~deap_pettingzoo.baselines.base.BudgetMode.GENERATIONS`
mode is not yet wired through the NEAT back-derivation.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines._logging_helpers import (
    build_logger,
    dump_fitness_history,
    write_summary,
)
from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
    derive_generations,
)
from deap_pettingzoo.baselines.recording import RecordingController
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size
from deap_pettingzoo.utils.parallel import (
    resolve_num_workers,
    resolve_scheduler_address,
)

from .config import build_neat_config
from .policy import NeatPolicy


class _RecordingReporter(neat.reporting.BaseReporter):
    """NEAT reporter that forwards each generation's champion to the
    recording controller, which decides (per RecordingMode) whether to
    capture it.

    Parameters
    ----------
    record_fn : callable
        ``record_fn(generation: int, best_genome) -> None`` — called once
        per generation with the current best genome.
    """

    def __init__(self, record_fn):
        self._record_fn = record_fn
        self._gen = 0

    def start_generation(self, generation: int) -> None:
        self._gen = generation

    def post_evaluate(self, config, population, species_set, best_genome) -> None:
        if self._gen <= 0:
            return
        try:
            self._record_fn(self._gen, best_genome)
        except Exception:
            pass


def _action_space_hint(action_space: Any) -> Dict[str, Any]:
    """Serialize a gymnasium action space to a picklable dict for workers."""
    import gymnasium.spaces as spaces

    if isinstance(action_space, spaces.Discrete):
        return {"type": "discrete", "args": {"n": int(action_space.n)}}
    if isinstance(action_space, spaces.MultiDiscrete):
        return {
            "type": "multidiscrete",
            "args": {"nvec": [int(x) for x in action_space.nvec]},
        }
    if isinstance(action_space, spaces.Box):
        return {
            "type": "continuous",
            "args": {
                "shape": tuple(int(s) for s in action_space.shape),
                "low": np.asarray(action_space.low, dtype=np.float32).tolist(),
                "high": np.asarray(action_space.high, dtype=np.float32).tolist(),
            },
        }
    raise NotImplementedError(
        f"NEAT parallel evaluation does not support action space "
        f"{type(action_space).__name__}. Only Discrete, MultiDiscrete and Box "
        "are supported."
    )


class NEATTrainer(BaselineTrainer):
    """Train a NEAT population on a PettingZoo environment.

    Orchestrates the neat-python ``Population.run()`` loop inside the
    :class:`BaselineTrainer` protocol.

    Algorithm-specific params accepted in ``TrainingConfig.algorithm_params``:

    * ``pop_size`` (int, default 50) — NEAT population size.
    * ``n_generations`` (int, optional) — explicit generation budget
      (bypasses the budget_mode translation when set).
    * ``neat_config_overrides`` (dict) — section -> key -> value overrides
      passed to :func:`build_neat_config`.
    * ``eval_episodes_per_genome`` (int, default 1) — number of
      episodes used to evaluate each genome's fitness.
    * ``action_aggregator`` (str, default "mean") — how to combine
      per-episode rewards into a scalar fitness: ``"mean"`` or
      ``"sum"``.

    ``TrainingConfig.seed`` and ``TrainingConfig.max_steps_per_episode`` are
    primary fields (read directly, not via ``algorithm_params``); see
    :class:`~deap_pettingzoo.baselines.base.TrainingConfig`.
    ``num_workers``/``parallel_backend`` are also primary fields — the
    constructor values below are only the fallback used when the config
    leaves them at their defaults (mirrors ``EATrainer``).
    """

    def __init__(
        self,
        num_workers: int = 1,
        parallel_backend: str = "process",
        scheduler_address: Optional[str] = None,
    ) -> None:
        self._fitness_history: List[float] = []
        self._env_steps_history: List[int] = []
        self._generation_num: int = 0
        # Cumulative real env ticks across the whole run. Incremented per-tick
        # on the serial path and aggregated from worker step-deltas on the
        # parallel path (NEAT runs its own rollout loop, not SingleEpisodeEvaluator).
        self._env_steps_consumed: int = 0
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if scheduler_address is not None and parallel_backend != "dask":
            raise ValueError(
                f"scheduler_address={scheduler_address!r} requires "
                f"parallel_backend='dask', got {parallel_backend!r}."
            )
        self._num_workers = int(num_workers)
        self._parallel_backend = str(parallel_backend)
        self._scheduler_address = scheduler_address

    # BaselineTrainer interface

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in (BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "multidiscrete_action": True,
            "multi_agent": True,
            "multi_objective": False,
            "parallel_training": self._num_workers > 1,
            "recurrent": False,
            "population_based": True,
            "algorithm": "NEAT",
        }

    def _resolve_parallelism(
        self, config: TrainingConfig
    ) -> tuple[int, str, Optional[str], bool]:
        """Resolve ``(effective_workers, backend, scheduler_address, remote)``.

        ``config.num_workers``/``config.parallel_backend``/
        ``config.scheduler_address`` are primary ``TrainingConfig`` fields; the
        trainer-constructor value is the fallback used only when the config
        leaves them at their defaults (mirrors ``EATrainer``). Shared by
        ``NEATTrainer``/``HyperNEATTrainer``/``ESHyperNEATTrainer``.

        ``remote`` is ``True`` when a remote Dask scheduler is in play
        (explicit address or the ``DEAP_DASK_SCHEDULER`` env var): the
        driver CPU cap is skipped and callers must take the parallel
        dispatch path even for ``effective_workers <= 1``.
        """
        requested_workers = (
            config.num_workers if config.num_workers != 1 else self._num_workers
        )
        backend = (
            config.parallel_backend
            if config.parallel_backend != "process"
            else self._parallel_backend
        )
        scheduler_address = (
            config.scheduler_address
            if config.scheduler_address is not None
            else self._scheduler_address
        )
        remote = (
            backend == "dask"
            and resolve_scheduler_address(scheduler_address) is not None
        )
        effective_workers = resolve_num_workers(
            requested_workers, cap_to_cpu=not remote
        )
        return effective_workers, backend, scheduler_address, remote

    @staticmethod
    def _resolve_max_steps(config: TrainingConfig, default: int = 500) -> int:
        """Resolve the per-episode step cap, first-class field over ``default``."""
        return (
            config.max_steps_per_episode
            if config.max_steps_per_episode is not None
            else default
        )

    def train(self, config: TrainingConfig) -> TrainingResult:
        t0 = time.perf_counter()
        # Opt-in consumed-step budgeting: run until the real env-tick budget
        # is exhausted instead of the nominal generation count.
        consumed_budget: Optional[int] = None
        if config.budget_on_consumed and config.budget_mode == BudgetMode.ENV_STEPS:
            consumed_budget = int(config.budget_value)
        # Reset per-run state so reusing a trainer instance does not carry a
        # previous run's consumed-step count into this run's metrics.
        self._env_steps_consumed = 0
        self._env_steps_history = []
        params = dict(config.algorithm_params)
        max_steps = self._resolve_max_steps(config)

        # Resolve budget to generations
        pop_size = int(params.get("pop_size", 50))
        n_generations = params.get("n_generations", None)
        if n_generations is None:
            n_generations = self._budget_to_generations(config, pop_size)

        # Build env
        env = PettingZooWrapper(
            config.env_name,
            env_config=config.env_config,
            render_mode="rgb_array" if config.render else None,
        )
        agents = list(env.possible_agents)
        reference_space = env.action_space(agents[0])
        divergent = [a for a in agents[1:] if env.action_space(a) != reference_space]
        if divergent:
            raise ValueError(
                "NEATTrainer requires a single shared action space across all "
                "agents (homogeneous-team neuroevolution). Diverging agents: "
                f"{divergent}. Use MixedPolicyEvaluator or an AgileRL trainer "
                "for heterogeneous-action envs."
            )
        shared_action_space = reference_space

        # Build NEAT config
        obs_shape = env.observation_space(env.possible_agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(shared_action_space)

        overrides = dict(params.get("neat_config_overrides", {}))
        seed = config.seed if config.seed is not None else params.get("seed", None)
        neat_config = build_neat_config(
            input_size=input_size,
            output_size=output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        # Logger + unified champion-rollout recording
        log = build_logger(config, algorithm="neat")
        self._recorder = RecordingController(
            config,
            log,
            max_steps=max_steps,
            seed=seed,
        )

        # Create initial population
        pop = neat.Population(neat_config, seed=seed)

        # Evaluation callback for neat
        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        aggregator = params.get("action_aggregator", "mean")

        # Build action-space hint for worker spec.
        _act_hint = _action_space_hint(shared_action_space)

        # Recording reporter
        def _neat_champion(best_genome):
            return NeatPolicy(
                best_genome, neat_config, action_space=shared_action_space
            )

        if self._recorder.enabled:

            def _record_champion(gen: int, best_genome) -> None:
                self._recorder.observe(
                    ProgressEvent(
                        unit="generation",
                        current=gen,
                        total=n_generations,
                        env_steps=gen * max_steps,
                    ),
                    lambda: _neat_champion(best_genome),
                )

            pop.add_reporter(_RecordingReporter(_record_champion))

        effective_workers, backend, scheduler_address, remote = (
            self._resolve_parallelism(config)
        )

        # 1-element sink so worker step-deltas aggregate across generations
        # (stays at 0 on the serial path, which counts on the trainer itself).
        _parallel_steps = [0]

        # Parallel or sequential eval
        if effective_workers > 1 or remote:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor

            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate

            snapshot = get_registry_snapshot(target_remote=remote)
            worker_spec = _NeatWorkerSpec(
                env_name=config.env_name,
                env_config=dict(config.env_config),
                input_size=input_size,
                output_size=output_size,
                action_space_type=_act_hint["type"],
                action_space_args=_act_hint.get("args", {}),
                eval_episodes_per_genome=eval_eps,
                seed=seed if seed is not None else 0,
                max_steps=max_steps,
                registry_snapshot=snapshot,
            )
            _executor = make_executor(
                effective_workers,
                backend=backend,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
                scheduler_address=scheduler_address,
            )
            _executor_ctx = _executor

            def _on_generation(best_fitness: float, gen_steps: int) -> None:
                # Parallel path: neat.Population.run() never touches these
                # driver-side counters (only the fitness assignment happens in
                # the worker-shipped callable), so they'd otherwise stay at 0
                # regardless of how many generations actually ran.
                self._generation_num += 1
                self._fitness_history.append(best_fitness)
                self._env_steps_history.append(gen_steps)

            _eval_genomes = make_neat_parallel_evaluate(
                _executor,
                worker_spec,
                step_sink=_parallel_steps,
                on_generation=_on_generation,
            )
            _is_parallel = True
        else:
            _executor_ctx = None
            _is_parallel = False

            _base_seed = seed if seed is not None else 0

            def _eval_genomes(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
                """neat fitness-function contract: assign .fitness to each genome."""
                self._generation_num += 1
                gen_fitnesses: list[float] = []
                _before = int(self._env_steps_consumed)

                for _genome_id, genome in genomes_and_ids:
                    try:
                        policy = NeatPolicy(
                            genome, neat_cfg, action_space=shared_action_space
                        )
                        rewards: list[float] = []
                        for ep_i in range(eval_eps):
                            ep_seed = (
                                _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            )
                            observations = env.reset(seed=ep_seed)
                            episode_reward = 0.0
                            done = False
                            _steps = 0
                            while not done and _steps < max_steps:
                                actions = {}
                                for agent in observations:
                                    actions[agent] = policy.get_action(
                                        observations[agent]
                                    )
                                (
                                    observations,
                                    step_rewards,
                                    dones,
                                    truncateds,
                                    _infos,
                                ) = env.step(actions)
                                episode_reward += float(sum(step_rewards.values()))
                                done = all(dones.values()) or all(truncateds.values())
                                _steps += 1
                                self._env_steps_consumed += 1
                            rewards.append(episode_reward)

                        if aggregator == "sum":
                            fitness = float(np.sum(rewards))
                        else:
                            fitness = float(np.mean(rewards))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        fitness = -1e9

                    genome.fitness = fitness
                    gen_fitnesses.append(fitness)

                if gen_fitnesses:
                    self._fitness_history.append(float(np.max(gen_fitnesses)))
                # Per-generation real env steps (serial path).
                self._env_steps_history.append(int(self._env_steps_consumed) - _before)

        # Add reporters
        pop.add_reporter(neat.StdOutReporter(show_species_detail=False))

        # Run NEAT
        def _consumed_total() -> int:
            return int(self._env_steps_consumed) + int(_parallel_steps[0])

        def _run_population() -> Any:
            if consumed_budget is None:
                return pop.run(_eval_genomes, n=n_generations)
            # Consumed budgeting: step one generation at a time until the
            # real env-tick budget is exhausted. ``Population`` carries all
            # state across ``run()`` calls and the NEAT config pins
            # ``no_fitness_termination=True``, so n=1 stepping is
            # semantically identical to a single n=N call. The safety cap
            # mirrors the EA loop: each generation consumes at least one
            # tick per genome, so nominal_gens * max_steps always suffices.
            safety_cap = n_generations * max(1, max_steps) + 10
            best = None
            for _ in range(safety_cap):
                if _consumed_total() >= consumed_budget:
                    break
                best = pop.run(_eval_genomes, n=1)
            return best if best is not None else pop.best_genome

        try:
            if _is_parallel and _executor_ctx is not None:
                with _executor_ctx:
                    best_genome = _run_population()
                # Fold the aggregated worker step-deltas into the run total.
                self._env_steps_consumed += int(_parallel_steps[0])
            else:
                best_genome = _run_population()
        except neat.CompleteExtinctionException:
            warnings.warn(
                "NEAT population went extinct before completing the budget. "
                "The best genome from the last surviving generation is used.",
                stacklevel=2,
            )
            best_genome = pop.best_genome

        # Build result policy
        champion = NeatPolicy(
            best_genome, neat_config, action_space=shared_action_space
        )

        elapsed = time.perf_counter() - t0

        # Realized count: equals n_generations under nominal budgeting, but
        # consumed budgeting (and extinction) can realize a different number.
        realized_gens = int(self._generation_num)

        # Guarantee a final capture (dedups against the last per-gen one).
        self._recorder.finalize(
            lambda: _neat_champion(best_genome),
            iteration=realized_gens,
            env_steps=realized_gens * max_steps,
        )

        try:
            env.close()
        except Exception:
            pass
        self._recorder.close()

        self._log(
            config,
            best_genome,
            pop_size,
            elapsed,
            algorithm="NEAT",
            nominal_env_steps_per_iteration=pop_size * eval_eps * max_steps,
        )
        if config.recording.enabled:
            log.close()

        return TrainingResult(
            algorithm="NEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
                # Nominal ceiling: every genome evaluated for ``eval_eps``
                # episodes of up to ``max_steps`` ticks, across the
                # generations that actually ran.
                "env_steps_nominal": int(
                    realized_gens * pop_size * eval_eps * max_steps
                ),
                "env_steps_consumed": int(self._env_steps_consumed),
            },
            metadata={
                "genome": best_genome,
                "config": neat_config,
            },
            seed=seed,
            training_time=elapsed,
        )

    # Helpers

    def _log(
        self,
        config: TrainingConfig,
        best_genome: Any,
        pop_size: int,
        elapsed: float,
        *,
        algorithm: str = "NEAT",
        nominal_env_steps_per_iteration: int = 0,
    ) -> None:
        """Persist fitness history and summary to ``config.output_dir``."""
        output_dir = config.output_dir
        if output_dir is None:
            return
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger(config, algorithm=algorithm)
        # Build per-generation entries with real env_steps (cumulative).
        _cumulative = 0
        _entries: list = []
        _step_hist = getattr(self, "_env_steps_history", [])
        for i, f in enumerate(self._fitness_history):
            if i < len(_step_hist):
                _cumulative += _step_hist[i]
            else:
                # Parallel path or missing data: apportion evenly.
                _cumulative = (
                    (
                        int(self._env_steps_consumed)
                        * (i + 1)
                        // max(1, len(self._fitness_history))
                    )
                    if self._fitness_history
                    else 0
                )
            _entries.append(
                {
                    "generation": i,
                    "best_fitness": f,
                    "env_steps": _cumulative,
                }
            )
        dump_fitness_history(
            logger,
            _entries,
            env_steps_per_iteration=nominal_env_steps_per_iteration or pop_size,
            nominal_env_steps_per_iteration=nominal_env_steps_per_iteration or pop_size,
        )
        write_summary(
            logger,
            {
                "algorithm": algorithm,
                "best_fitness": float(best_genome.fitness if best_genome else -np.inf),
                "generations": self._generation_num,
                "pop_size": pop_size,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    def _budget_to_generations(self, config: TrainingConfig, pop_size: int) -> int:
        """Translate ``config.budget_mode`` / ``.budget_value`` to generations.

        Uses the shared :func:`~deap_pettingzoo.baselines.base.derive_generations`
        helper so NEAT shares the round-to-nearest + warn-on-collapse behaviour
        of the other population backends.
        """
        eval_eps = int(config.algorithm_params.get("eval_episodes_per_genome", 1))
        if config.budget_mode == BudgetMode.EVAL_EPISODES:
            # Each generation evaluates pop_size genomes × eval_eps episodes.
            return derive_generations(
                config.budget_value,
                pop_size * eval_eps,
                1,
                label="NEAT",
                unit="eval episodes",
            )

        if config.budget_mode == BudgetMode.ENV_STEPS:
            # Resolve episode length from env config: MPE uses "max_cycles",
            # aquarium and other custom envs may use "max_time_steps".
            avg_steps_per_episode = int(
                config.env_config.get(
                    "max_cycles",
                    config.env_config.get("max_time_steps", 25),
                )
            )
            return derive_generations(
                config.budget_value,
                pop_size * eval_eps,
                avg_steps_per_episode,
                label="NEAT",
            )

        # Fallback — should not happen (supports_budget_mode gate).
        return 10
