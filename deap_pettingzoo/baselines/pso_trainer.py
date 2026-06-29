"""
PSO trainer mirroring :class:`CMAESTrainer`.

Drives a particle swarm over the flat weight vector of a
:class:`FixedTopologyGenome`.  The best-found genome is wrapped as a
:class:`PolicyProtocol` via :class:`GenomePolicyWrapper`.
"""

import time
from typing import Any, ClassVar, Dict, Optional

import numpy as np
from deap import tools

from deap_pettingzoo.baselines._logging_helpers import (
    build_logger,
    dump_fitness_history,
    write_summary,
)
from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    ProgressCallback,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
    derive_generations,
)
from deap_pettingzoo.baselines.ea_trainer import (
    GenomePolicyWrapper,
    _resolve_shared_action_space,
)
from deap_pettingzoo.baselines.recording import RecordingController
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration.pso_integration import (
    make_pso_ind_init,
    make_pso_stats,
    register_pso_toolbox,
    setup_pso_swarm,
)
from deap_pettingzoo.integration.deap_integration import (
    _EvalWorkerSpec,
    evaluate_population_parallel,
)
from deap_pettingzoo.policies.action_heads import required_output_size
from deap_pettingzoo.environments.registry import (
    get_registry_snapshot,
    replay_registry_snapshot,
)
from deap_pettingzoo.utils.parallel import make_executor, resolve_num_workers


class PSOTrainer(BaselineTrainer):
    """Particle Swarm Optimisation trainer.

    Same genome representation as :class:`EATrainer` and
    :class:`CMAESTrainer`, so all three can be compared head-to-head.
    """

    @staticmethod
    def describe_hyperparameters():
        """Declarative hyperparameter overlay consumed by the catalog."""
        from deap_pettingzoo.catalog.schema import (
            ParamSpec,
            ParamTarget,
            ParamType,
        )

        I, F, E = ParamType.INT, ParamType.FLOAT, ParamType.ENUM
        TR, AP = ParamTarget.TRAINER, ParamTarget.ALGORITHM_PARAMS
        return (
            ParamSpec("hidden_size", I, 64, minimum=1, step=1, target=TR,
                      group="Architecture",
                      help="Hidden layer width of the evolved MLP."),
            ParamSpec("population_size", I, 50, minimum=2, step=1, target=TR,
                      group="Population",
                      help="Number of particles in the swarm."),
            ParamSpec("phi1", F, 2.0, minimum=0.0, step=0.1, target=TR,
                      group="Strategy",
                      help="Cognitive coefficient (pull toward personal best)."),
            ParamSpec("phi2", F, 2.0, minimum=0.0, step=0.1, target=TR,
                      group="Strategy",
                      help="Social coefficient (pull toward global best)."),
            ParamSpec("weight", F, 0.7298, minimum=0.0, step=0.05, target=TR,
                      group="Strategy", help="Inertia weight."),
            ParamSpec("smin", F, None, target=TR, group="Strategy",
                      advanced=True, nullable=True,
                      help="Lower clamp on particle velocity (None = unset)."),
            ParamSpec("smax", F, None, target=TR, group="Strategy",
                      advanced=True, nullable=True,
                      help="Upper clamp on particle velocity (None = unset)."),
            ParamSpec("activation", E, "tanh", target=TR, group="Architecture",
                      advanced=True,
                      choices=("tanh", "relu", "sigmoid", "linear"),
                      help="MLP activation function."),
            ParamSpec("num_workers", I, 1, minimum=0, step=1, target=TR,
                      group="Parallelism",
                      help="Parallel evaluation workers (1 = serial)."),
            ParamSpec("parallel_backend", E, "process", target=TR,
                      group="Parallelism", advanced=True,
                      choices=("process", "scoop", "dask"),
                      help="Parallel executor backend."),
            ParamSpec("seed", I, None, minimum=0, step=1, target=AP,
                      group="Run", nullable=True,
                      help="Random seed for this run (None = unseeded)."),
            ParamSpec("max_steps_per_episode", I, 200, minimum=1, step=1,
                      target=AP, group="Run",
                      help="Hard cap on environment steps per rollout."),
        )

    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        phi1: float = 2.0,
        phi2: float = 2.0,
        weight: float = 0.7298,
        smin: Optional[float] = None,
        smax: Optional[float] = None,
        activation: str = "tanh",
        weight_range: tuple = (-1.0, 1.0),
        num_workers: int = 1,
        parallel_backend: str = "process",
    ):
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.phi1 = phi1
        self.phi2 = phi2
        self.weight = weight
        self.smin = smin
        self.smax = smax
        self.activation = activation
        self.weight_range = weight_range

        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
        from deap_pettingzoo.utils.parallel import SUPPORTED_BACKENDS
        if parallel_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown parallel_backend {parallel_backend!r}. "
                f"Expected one of {SUPPORTED_BACKENDS}."
            )
        self.num_workers = int(num_workers)
        self.parallel_backend = str(parallel_backend)

    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        start_time = time.time()
        params = self._translate_budget(config)
        num_generations = params["num_generations"]
        max_steps = params["max_steps_per_episode"]

        # Opt-in consumed-step budgeting: run until the real env-step budget is
        # exhausted instead of a precomputed generation count (manual-loop
        # trainer — PSO supports it natively).
        consumed_budget: Optional[int] = None
        if config.budget_on_consumed and config.budget_mode == BudgetMode.ENV_STEPS:
            consumed_budget = int(config.budget_value)

        env = PettingZooWrapper(config.env_name, env_config=config.env_config or None)
        sample_obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]
        input_size = len(sample_obs[agent_id])
        action_space = _resolve_shared_action_space(env)
        output_size = required_output_size(action_space)

        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env,
            max_steps=max_steps,
            aggregation_method=config.fitness.aggregation_method,
            team_mapping=config.fitness.team_mapping,
        )

        seed = config.algorithm_params.get("seed", None)
        # Seed global RNGs
        if seed is not None:
            from deap_pettingzoo.utils.rng import seed_everything
            seed_everything(int(seed))

        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "weight_range": self.weight_range,
        }

        pso_state, dimension = setup_pso_swarm(
            genome_config=genome_config,
            population_size=self.population_size,
            phi1=self.phi1,
            phi2=self.phi2,
            weight=self.weight,
            smin=self.smin,
            smax=self.smax,
            seed=seed,
        )
        ind_init = make_pso_ind_init(genome_config, pso_state)
        toolbox = register_pso_toolbox(pso_state, ind_init, evaluator)

        stats = make_pso_stats()
        hof = tools.HallOfFame(1)
        fitness_history: list = []

        log = build_logger(config, algorithm="pso")
        env_steps_per_iteration = self.population_size * max_steps
        self._recorder = RecordingController(
            config, log, max_steps=max_steps,
            seed=config.algorithm_params.get("seed"),
        )

        def _pso_champion():
            if len(hof) == 0:
                return None
            return GenomePolicyWrapper(
                hof[0].genome, config.env_name, action_space=action_space
            )

        # num_workers / parallel_backend are first-class TrainingConfig fields;
        # the trainer-constructor value is the fallback used only when the
        # config leaves them at their defaults.
        requested = (
            config.num_workers if config.num_workers != 1 else self.num_workers
        )
        effective_workers = resolve_num_workers(requested)
        backend = (
            config.parallel_backend
            if config.parallel_backend != "process"
            else self.parallel_backend
        )
        # One snapshot, reused for the process-pool initializer and shipped on
        # the worker spec so SCOOP/Dask workers can re-register custom envs too.
        registry_snapshot = get_registry_snapshot()
        worker_spec = _EvalWorkerSpec(
            env_name=config.env_name,
            env_config=dict(config.env_config) if config.env_config else None,
            max_steps=max_steps,
            aggregation_method=config.fitness.aggregation_method,
            team_mapping=config.fitness.team_mapping,
            registry_snapshot=registry_snapshot,
        )

        with make_executor(
            effective_workers, backend=backend,
            initializer=replay_registry_snapshot,
            initargs=(registry_snapshot,),
        ) as executor:
            base_eval_seed = config.algorithm_params.get("seed")
            _eval_counter = [0]
            # Aggregates real env-step deltas from parallel workers; stays at
            # 0 on the serial path where the evaluator counter is authoritative.
            _parallel_steps = [0]

            def _next_seeds(n):
                if base_eval_seed is None:
                    return [None] * n
                lo = int(base_eval_seed) + _eval_counter[0]
                _eval_counter[0] += n
                return list(range(lo, lo + n))

            # Nominal budgeting runs exactly ``num_generations``; consumed
            # budgeting runs until the real budget is exhausted, bounded by a safety cap.
            if consumed_budget is not None:
                max_iters = num_generations * max(1, max_steps) + 10
            else:
                max_iters = num_generations

            population = toolbox.generate()
            gen = 0
            while gen < max_iters:
                seeds = _next_seeds(len(population))
                if effective_workers > 1:
                    fitnesses = evaluate_population_parallel(
                        population, executor=executor, spec=worker_spec,
                        eval_seeds=seeds,
                        step_sink=_parallel_steps,
                    )
                else:
                    fitnesses = [
                        evaluator.evaluate(ind, seed=s)
                        for ind, s in zip(population, seeds)
                    ]
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                toolbox.update(population)
                hof.update(population)
                gen += 1
                record = stats.compile(population)

                # Extract gbest from toolbox context for champion tracking.
                gbest_ctx = toolbox._pso_gbest_ctx
                gbest_fit = gbest_ctx.get("fitness")
                fitness_history.append({
                    "generation": gen,
                    "mean": record["avg"],
                    "std": record["std"],
                    "min": record["min"],
                    "max": record["max"],
                    "gbest": gbest_fit,
                })
                consumed = (
                    int(evaluator.env_steps_consumed) + int(_parallel_steps[0])
                )
                if consumed_budget is not None:
                    self._on_progress(
                        progress_callback,
                        ProgressEvent(
                            unit="env_step",
                            current=consumed,
                            total=consumed_budget,
                            env_steps=consumed,
                            best_fitness=record["max"],
                            mean_fitness=record["avg"],
                            wall_time_s=time.time() - start_time,
                        ),
                        champion=_pso_champion,
                    )
                    if consumed >= consumed_budget:
                        break
                else:
                    self._on_progress(
                        progress_callback,
                        ProgressEvent(
                            unit="generation",
                            current=gen,
                            total=num_generations,
                            env_steps=gen * env_steps_per_iteration,
                            best_fitness=record["max"],
                            mean_fitness=record["avg"],
                            wall_time_s=time.time() - start_time,
                        ),
                        champion=_pso_champion,
                    )

        # Realized generation count (== num_generations on the nominal path).
        num_generations = gen
        env.close()
        self._recorder.finalize(
            _pso_champion, iteration=num_generations,
            env_steps=num_generations * env_steps_per_iteration,
        )
        self._recorder.close()
        training_time = time.time() - start_time

        best_ind = hof[0]
        best_genome = best_ind.genome
        policy = GenomePolicyWrapper(best_genome, config.env_name,
                                     action_space=action_space)

        try:
            with log:
                dump_fitness_history(log, fitness_history,
                    env_steps_per_iteration=self.population_size * max_steps)
                from deap_pettingzoo.utils.logging import save_ea_genome
                save_ea_genome(log, best_genome, kind="pso_genome", extra={
                    "final_fitness": float(best_ind.fitness.values[0]),
                    "env_name": config.env_name,
                })
                write_summary(log, {
                    "algorithm": "pso",
                    "final_fitness": float(best_ind.fitness.values[0]),
                    "num_generations": num_generations,
                    "population_size": self.population_size,
                    "dimension": dimension,
                }, training_time_s=training_time)
        except Exception:
            pass

        return TrainingResult(
            algorithm="pso",
            policy=policy,
            metrics={
                "fitness_history": fitness_history,
                "final_fitness": best_ind.fitness.values[0],
                "num_generations": num_generations,
                "population_size": self.population_size,
                "dimension": dimension,
                # PSO evaluates ``num_generations`` populations with no separate
                # gen-0 eval, so nominal is num_generations * per-iter.
                "env_steps_nominal": num_generations * env_steps_per_iteration,
                "env_steps_consumed": (
                    int(evaluator.env_steps_consumed) + int(_parallel_steps[0])
                ),
            },
            metadata={
                "genome_config": genome_config,
                "pso_config": {
                    "hidden_size": self.hidden_size,
                    "population_size": self.population_size,
                    "phi1": self.phi1, "phi2": self.phi2,
                    "weight": self.weight,
                    "activation": self.activation,
                },
                "num_workers": effective_workers,
                "parallel_backend": backend,
                "run_dir": str(log.run_dir) if config.output_dir is not None else None,
            },
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {
            BudgetMode.ENV_STEPS,
            BudgetMode.EVAL_EPISODES,
            BudgetMode.GENERATIONS,
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "multidiscrete_action": True,
            "needs_global_state": False,
            "on_policy": False,
            "supports_turn_based": True,
        }

    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        max_steps = config.algorithm_params.get("max_steps_per_episode", 200)
        if config.budget_mode == BudgetMode.GENERATIONS:
            gens = max(1, int(config.budget_value))
        elif config.budget_mode == BudgetMode.ENV_STEPS:
            gens = derive_generations(
                config.budget_value, self.population_size, max_steps, label="PSO"
            )
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            gens = derive_generations(
                config.budget_value,
                self.population_size,
                1,
                label="PSO",
                unit="eval episodes",
            )
        else:
            raise ValueError(f"PSOTrainer does not support {config.budget_mode}")
        return {
            "num_generations": max(1, int(gens)),
            "max_steps_per_episode": int(max_steps),
        }
