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

Only :class:`~deap_pettingzoo.baselines.base.BudgetMode.WALL_CLOCK` is
not supported (NEAT does not expose a suspend/resume hook).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

import neat  # guarded by _require_neat

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
)
from deap_pettingzoo.baselines.recording import RecordingController
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

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

    def post_evaluate(
        self, config, population, species_set, best_genome
    ) -> None:
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
    * ``neat_config_overrides`` (dict) — section→key→value overrides
      passed to :func:`build_neat_config`.
    * ``eval_episodes_per_genome`` (int, default 1) — number of
      episodes used to evaluate each genome's fitness.
    * ``seed`` (int, optional) — RNG seed passed to both the NEAT
      config and the env reset.
    * ``action_aggregator`` (str, default "mean") — how to combine
      per-episode rewards into a scalar fitness: ``"mean"`` or
      ``"sum"``.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._fitness_history: List[float] = []
        self._generation_num: int = 0
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._num_workers = int(num_workers)

    # ------------------------------------------------------------------
    # BaselineTrainer interface
    # ------------------------------------------------------------------

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

    def train(self, config: TrainingConfig) -> TrainingResult:
        t0 = time.perf_counter()
        params = dict(config.algorithm_params)

        # --- Resolve budget to generations -----------------------------------
        pop_size = int(params.get("pop_size", 50))
        n_generations = params.get("n_generations", None)
        if n_generations is None:
            n_generations = self._budget_to_generations(config, pop_size)

        # --- Build env -------------------------------------------------------
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

        # --- Build NEAT config -----------------------------------------------
        obs_shape = env.observation_space(env.possible_agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(shared_action_space)

        overrides = dict(params.get("neat_config_overrides", {}))
        seed = params.get("seed", None)
        neat_config = build_neat_config(
            input_size=input_size,
            output_size=output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        # --- Logger + unified champion-rollout recording ---------------------
        log = build_logger(config, algorithm="neat")
        self._recorder = RecordingController(
            config, log, max_steps=500, seed=seed,
        )

        # --- Create initial population ---------------------------------------
        pop = neat.Population(neat_config, seed=seed)

        # --- Evaluation callback for neat ------------------------------------
        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        aggregator = params.get("action_aggregator", "mean")

        # Build action-space hint for worker spec.
        _act_hint = _action_space_hint(shared_action_space)

        # --- Recording reporter ----------------------------------------------
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
                        env_steps=gen * 500,
                    ),
                    lambda: _neat_champion(best_genome),
                )

            pop.add_reporter(_RecordingReporter(_record_champion))

        # --- Parallel or sequential eval -------------------------------------
        if self._num_workers > 1:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor
            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate

            snapshot = get_registry_snapshot()
            worker_spec = _NeatWorkerSpec(
                env_name=config.env_name,
                env_config=dict(config.env_config),
                input_size=input_size,
                output_size=output_size,
                action_space_type=_act_hint["type"],
                action_space_args=_act_hint.get("args", {}),
                eval_episodes_per_genome=eval_eps,
                seed=seed if seed is not None else 0,
            )
            _executor = make_executor(
                self._num_workers,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
            )
            _executor_ctx = _executor
            _eval_genomes = make_neat_parallel_evaluate(_executor, worker_spec)
            _is_parallel = True
        else:
            _executor_ctx = None
            _is_parallel = False

            _base_seed = seed if seed is not None else 0

            def _eval_genomes(
                genomes_and_ids: list, neat_cfg: neat.Config
            ) -> None:
                """neat fitness-function contract: assign .fitness to each genome."""
                self._generation_num += 1
                gen_fitnesses: list[float] = []

                for _genome_id, genome in genomes_and_ids:
                    try:
                        policy = NeatPolicy(
                            genome, neat_cfg, action_space=shared_action_space
                        )
                        rewards: list[float] = []
                        for ep_i in range(eval_eps):
                            ep_seed = _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            observations = env.reset(seed=ep_seed)
                            episode_reward = 0.0
                            done = False
                            _steps = 0
                            while not done and _steps < 500:
                                actions = {}
                                for agent in observations:
                                    actions[agent] = policy.get_action(
                                        observations[agent]
                                    )
                                observations, step_rewards, dones, truncateds, _infos = (
                                    env.step(actions)
                                )
                                episode_reward += float(sum(step_rewards.values()))
                                done = all(dones.values()) or all(truncateds.values())
                                _steps += 1
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

        # --- Add reporters ---------------------------------------------------
        pop.add_reporter(neat.StdOutReporter(show_species_detail=False))

        # --- Run NEAT --------------------------------------------------------
        try:
            if _is_parallel and _executor_ctx is not None:
                with _executor_ctx:
                    best_genome = pop.run(_eval_genomes, n=n_generations)
                if not self._fitness_history:
                    self._fitness_history.append(
                        float(best_genome.fitness if best_genome else 0.0)
                    )
            else:
                best_genome = pop.run(_eval_genomes, n=n_generations)
        except neat.CompleteExtinctionException:
            warnings.warn(
                "NEAT population went extinct before completing the budget. "
                "The best genome from the last surviving generation is used.",
                stacklevel=2,
            )
            best_genome = pop.best_genome

        # --- Build result policy --------------------------------------------
        champion = NeatPolicy(
            best_genome, neat_config, action_space=shared_action_space
        )

        elapsed = time.perf_counter() - t0

        # Guarantee a final capture (dedups against the last per-gen one).
        self._recorder.finalize(
            lambda: _neat_champion(best_genome),
            iteration=n_generations,
            env_steps=n_generations * 500,
        )

        try:
            env.close()
        except Exception:
            pass
        self._recorder.close()

        self._log(config, best_genome, pop_size, elapsed, algorithm="NEAT")
        if config.recording.enabled:
            log.close()

        return TrainingResult(
            algorithm="NEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
            },
            metadata={
                "genome": best_genome,
                "config": neat_config,
            },
            seed=seed,
            training_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        config: TrainingConfig,
        best_genome: Any,
        pop_size: int,
        elapsed: float,
        *,
        algorithm: str = "NEAT",
    ) -> None:
        """Persist fitness history and summary to ``config.output_dir``."""
        output_dir = config.output_dir
        if output_dir is None:
            return
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger(config, algorithm=algorithm)
        dump_fitness_history(
            logger,
            [
                {"generation": i, "best_fitness": f}
                for i, f in enumerate(self._fitness_history)
            ],
            env_steps_per_iteration=pop_size,
        )
        write_summary(
            logger,
            {
                "algorithm": algorithm,
                "best_fitness": float(
                    best_genome.fitness if best_genome else -np.inf
                ),
                "generations": self._generation_num,
                "pop_size": pop_size,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    def _budget_to_generations(
        self, config: TrainingConfig, pop_size: int
    ) -> int:
        """Translate ``config.budget_mode`` / ``.budget_value`` to generations."""
        if config.budget_mode == BudgetMode.EVAL_EPISODES:
            # Each generation evaluates pop_size genomes × eval_eps episodes.
            eval_eps = int(
                config.algorithm_params.get("eval_episodes_per_genome", 1)
            )
            eps_per_gen = pop_size * eval_eps
            return max(1, config.budget_value // max(1, eps_per_gen))

        if config.budget_mode == BudgetMode.ENV_STEPS:
            # Resolve episode length from env config: MPE uses "max_cycles",
            # aquarium and other custom envs may use "max_time_steps".
            avg_steps_per_episode = int(
                config.env_config.get(
                    "max_cycles",
                    config.env_config.get("max_time_steps", 25),
                )
            )
            eval_eps = int(
                config.algorithm_params.get("eval_episodes_per_genome", 1)
            )
            steps_per_gen = pop_size * eval_eps * avg_steps_per_episode
            n_gens = config.budget_value // max(1, steps_per_gen)
            if n_gens == 0:
                import warnings
                warnings.warn(
                    f"NEAT budget ({config.budget_value:,} steps) is smaller "
                    f"than one generation ({steps_per_gen:,} steps). "
                    "Running 1 generation only.",
                    stacklevel=2,
                )
                return 1
            return n_gens

        # Fallback — should not happen (supports_budget_mode gate).
        return 10
