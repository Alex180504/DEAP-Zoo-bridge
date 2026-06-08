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
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.ea_trainer import (
    GenomePolicyWrapper,
    _resolve_shared_action_space,
)
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
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

    _default_recording_every_n: ClassVar[int] = 1

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

    def train(self, config: TrainingConfig) -> TrainingResult:
        start_time = time.time()
        params = self._translate_budget(config)
        num_generations = params["num_generations"]
        max_steps = params["max_steps_per_episode"]

        env = PettingZooWrapper(config.env_name, env_config=config.env_config or None)
        sample_obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]
        input_size = len(sample_obs[agent_id])
        action_space = _resolve_shared_action_space(env)
        output_size = required_output_size(action_space)

        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env,
            max_steps=max_steps,
            aggregation_method="sum",
        )

        seed = config.algorithm_params.get("seed", None)
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
        recording_cfg = config.recording
        recording_cadence = self._resolve_recording_every_n(config)
        recording_evaluator: Optional[UnifiedEvaluator] = None
        if recording_cfg.enabled:
            recorder = log.get_frame_recorder(
                fps=recording_cfg.fps, fmt=recording_cfg.fmt,
                max_frames_per_episode=recording_cfg.max_frames_per_episode,
                max_dim=recording_cfg.max_dim,
                episode_prefix=recording_cfg.episode_prefix or "train_",
            )
            if recorder.enabled:
                recording_evaluator = UnifiedEvaluator(
                    max_steps=max_steps, frame_recorder=recorder,
                )

        requested = config.algorithm_params.get("num_workers", self.num_workers)
        effective_workers = resolve_num_workers(requested)
        backend = config.algorithm_params.get("parallel_backend", self.parallel_backend)
        worker_spec = _EvalWorkerSpec(
            env_name=config.env_name,
            env_config=dict(config.env_config) if config.env_config else None,
            max_steps=max_steps,
            aggregation_method="sum",
            team_mapping=None,
        )

        with make_executor(
            effective_workers, backend=backend,
            initializer=replay_registry_snapshot,
            initargs=(get_registry_snapshot(),),
        ) as executor:
            base_eval_seed = config.algorithm_params.get("seed")
            _eval_counter = [0]

            def _next_seeds(n):
                if base_eval_seed is None:
                    return [None] * n
                lo = int(base_eval_seed) + _eval_counter[0]
                _eval_counter[0] += n
                return list(range(lo, lo + n))

            population = toolbox.generate()
            for gen in range(num_generations):
                seeds = _next_seeds(len(population))
                if effective_workers > 1:
                    fitnesses = evaluate_population_parallel(
                        population, executor=executor, spec=worker_spec,
                        eval_seeds=seeds,
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

                if recording_evaluator and recording_cfg.enabled and (
                    (gen + 1) % recording_cadence == 0
                ):
                    if len(hof) > 0:
                        try:
                            best = hof[0]
                            champ = GenomePolicyWrapper(
                                best.genome, config.env_name,
                                action_space=action_space,
                            )
                            rs = config.algorithm_params.get("seed")
                            recording_evaluator.evaluate(
                                policy=champ, env_name=config.env_name,
                                num_episodes=1,
                                seed=int(rs) if rs is not None else None,
                                env_config=dict(config.env_config)
                                if config.env_config else None,
                            )
                        except Exception:
                            pass

        env.close()
        if recording_evaluator is not None:
            recording_evaluator.close()
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
            },
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": False,
            "supports_turn_based": True,
        }

    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        max_steps = config.algorithm_params.get("max_steps_per_episode", 200)
        if config.budget_mode == BudgetMode.ENV_STEPS:
            gens = config.budget_value // (self.population_size * max_steps)
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            gens = config.budget_value // self.population_size
        else:
            raise ValueError(f"PSOTrainer does not support {config.budget_mode}")
        return {
            "num_generations": max(1, int(gens)),
            "max_steps_per_episode": int(max_steps),
        }
