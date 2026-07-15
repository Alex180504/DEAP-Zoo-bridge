"""
CMA-ES trainer implementation using :class:`deap.cma.Strategy`.

This module mirrors :mod:`deap_pettingzoo.baselines.ea_trainer` but drives
an (μ/μ_w, λ)-CMA-ES search over the flat weight vector of a
:class:`FixedTopologyGenome`. The resulting best-found genome is wrapped
as a :class:`PolicyProtocol` via :class:`GenomePolicyWrapper` so that it
is drop-in compatible with :class:`UnifiedEvaluator` and
:func:`compare_baselines`.
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
from deap_pettingzoo.environments.registry import (
    get_registry_snapshot,
    replay_registry_snapshot,
)
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration.cma_integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_one_plus_lambda_strategy,
    setup_cma_strategy,
)
from deap_pettingzoo.integration.deap_integration import (
    _EvalWorkerSpec,
    evaluate_population_parallel,
)
from deap_pettingzoo.policies.action_heads import required_output_size
from deap_pettingzoo.utils.parallel import (
    make_executor,
    resolve_num_workers,
    resolve_scheduler_address,
)


class CMAESTrainer(BaselineTrainer):
    """CMA-ES trainer built on top of :class:`deap.cma.Strategy`.

    Uses the same :class:`FixedTopologyGenome` representation as
    :class:`EATrainer`, so the two trainers can be compared head-to-head
    under identical budget/environment configurations via
    :func:`compare_baselines`.
    """

    @staticmethod
    def describe_hyperparameters():
        """Declarative hyperparameter overlay consumed by the catalog.

        ``weight_range`` is a tuple and is therefore omitted from the form
        surface (it is reachable through raw config / the Python API).
        """
        from deap_pettingzoo.catalog.schema import (
            ParamSpec,
            ParamTarget,
            ParamType,
        )

        I, F, E = ParamType.INT, ParamType.FLOAT, ParamType.ENUM
        TR, RC = ParamTarget.TRAINER, ParamTarget.RUN_CONFIG
        return (
            ParamSpec(
                "hidden_size",
                I,
                64,
                minimum=1,
                step=1,
                target=TR,
                group="Architecture",
                help="Hidden layer width of the evolved MLP.",
            ),
            ParamSpec(
                "population_size",
                I,
                50,
                minimum=2,
                step=1,
                target=TR,
                group="Population",
                help="lambda - samples drawn per generation.",
            ),
            ParamSpec(
                "sigma0",
                F,
                0.3,
                minimum=1e-6,
                step=0.05,
                target=TR,
                group="Strategy",
                help="Initial step size of the CMA search distribution.",
            ),
            ParamSpec(
                "variant",
                E,
                "mu_mu_lambda",
                target=TR,
                group="Strategy",
                choices=("mu_mu_lambda", "one_plus_lambda"),
                help="CMA-ES flavour: rank-mu update or (1+lambda) elitist.",
            ),
            ParamSpec(
                "activation",
                E,
                "tanh",
                target=TR,
                group="Architecture",
                advanced=True,
                choices=("tanh", "relu", "sigmoid", "linear"),
                help="MLP activation function.",
            ),
            ParamSpec(
                "num_workers",
                I,
                1,
                minimum=0,
                step=1,
                target=TR,
                group="Parallelism",
                help="Parallel evaluation workers (1 = serial).",
            ),
            ParamSpec(
                "parallel_backend",
                E,
                "process",
                target=TR,
                group="Parallelism",
                advanced=True,
                choices=("process", "dask"),
                help="Parallel executor backend.",
            ),
            ParamSpec(
                "seed",
                I,
                None,
                minimum=0,
                step=1,
                target=RC,
                group="Run",
                nullable=True,
                help="Random seed for this run (None = unseeded).",
            ),
            ParamSpec(
                "max_steps_per_episode",
                I,
                200,
                minimum=1,
                step=1,
                target=RC,
                group="Run",
                help="Hard cap on environment steps per rollout.",
            ),
        )

    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        sigma0: float = 0.3,
        activation: str = "tanh",
        weight_range: tuple = (-1.0, 1.0),
        variant: str = "mu_mu_lambda",
        num_workers: int = 1,
        parallel_backend: str = "process",
        scheduler_address: Optional[str] = None,
    ):
        """Initialise the CMA-ES trainer.

        Args:
            hidden_size: Hidden layer size for the evolved MLP.
            population_size: ``lambda_`` — number of samples per
                generation. Defaulted to ``50`` to match
                :class:`EATrainer` rather than DEAP's dimension-dependent
                default, so head-to-head comparisons evaluate an equal
                number of rollouts per generation.
            sigma0: Initial step size of the CMA search distribution.
            activation: Activation function for the MLP.
            weight_range: Probed bounds used when materialising genomes
                from sampled vectors (CMA itself is unbounded — this
                only affects the genome's bookkeeping).
            variant: CMA-ES flavour. ``"mu_mu_lambda"`` (default) uses
                :class:`deap.cma.Strategy` — the standard (μ/μ_w, λ)
                rank-μ update. ``"one_plus_lambda"`` uses
                :class:`deap.cma.StrategyOnePlusLambda` — a single-parent
                elitist variant that is typically more sample-efficient
                on unimodal objectives.
            num_workers: Parallel evaluation workers (1 = serial); the
                ``TrainingConfig`` field wins unless left at its default.
            parallel_backend: Executor backend (``"process"`` / ``"dask"``);
                the ``TrainingConfig`` field wins unless left at its default.
            scheduler_address: Remote Dask scheduler (``tcp://...`` URL or
                scheduler-file path); requires ``parallel_backend='dask'``.
                The ``TrainingConfig`` field wins unless left at ``None``;
                the ``DEAP_DASK_SCHEDULER`` env var is the final fallback.

        Raises:
            ValueError: If ``variant`` is not ``'mu_mu_lambda'`` or
                ``'one_plus_lambda'``.
            ValueError: If ``num_workers`` is negative.
            ValueError: If ``parallel_backend`` is not in
                :data:`~deap_pettingzoo.utils.parallel.SUPPORTED_BACKENDS`.
            ValueError: If ``scheduler_address`` is set with a non-Dask
                ``parallel_backend``.
        """
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.sigma0 = sigma0
        self.activation = activation
        self.weight_range = weight_range
        if variant not in {"mu_mu_lambda", "one_plus_lambda"}:
            raise ValueError(
                f"Unknown CMA-ES variant {variant!r}. "
                "Expected 'mu_mu_lambda' or 'one_plus_lambda'."
            )
        self.variant = variant

        if num_workers < 0:
            raise ValueError(
                f"num_workers must be >= 0 (or 1 for serial), got {num_workers}."
            )
        from deap_pettingzoo.utils.parallel import SUPPORTED_BACKENDS

        if parallel_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown parallel_backend {parallel_backend!r}. "
                f"Expected one of {SUPPORTED_BACKENDS}."
            )
        if scheduler_address is not None and parallel_backend != "dask":
            raise ValueError(
                f"scheduler_address={scheduler_address!r} requires "
                f"parallel_backend='dask', got {parallel_backend!r}."
            )
        self.num_workers = int(num_workers)
        self.parallel_backend = str(parallel_backend)
        self.scheduler_address = scheduler_address

    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Train CMA-ES with the given configuration.

        Args:
            config: Unified training configuration.
            progress_callback: Optional per-generation progress hook.

        Returns:
            :class:`TrainingResult` whose ``policy`` wraps the best-found
            genome as a :class:`PolicyProtocol`.

        Raises:
            ValueError: If ``config.budget_mode`` is not supported
                (propagated from :meth:`_translate_budget`).
            ValueError: If the environment uses heterogeneous action
                spaces (propagated from
                :func:`~deap_pettingzoo.baselines.ea_trainer._resolve_shared_action_space`).
        """
        start_time = time.time()

        params = self._translate_budget(config)
        num_generations = params["num_generations"]
        max_steps_per_episode = params["max_steps_per_episode"]

        # Opt-in consumed-step budgeting: run until the real env-step budget is
        # exhausted instead of a precomputed generation count (manual-loop
        # trainer — CMA-ES supports it natively).
        consumed_budget: Optional[int] = None
        if config.budget_on_consumed and config.budget_mode == BudgetMode.ENV_STEPS:
            consumed_budget = int(config.budget_value)

        # Environment + evaluator (homogeneous agents share one genome).
        env = PettingZooWrapper(config.env_name, env_config=config.env_config or None)
        sample_obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]

        input_size = len(sample_obs[agent_id])
        action_space = _resolve_shared_action_space(env)
        output_size = required_output_size(action_space)

        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env,
            max_steps=max_steps_per_episode,
            aggregation_method=config.fitness.aggregation_method,
            team_mapping=config.fitness.team_mapping,
        )

        # Seed global RNGs
        seed = config.seed
        if seed is not None:
            from deap_pettingzoo.utils.rng import seed_everything

            seed_everything(int(seed))

        is_box = hasattr(action_space, "low") and hasattr(action_space, "high")
        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "output_activation": "linear" if is_box else self.activation,
            "weight_range": self.weight_range,
        }

        strategy, dimension = self._build_strategy(genome_config)
        ind_init = make_cma_ind_init(genome_config)
        toolbox = register_cma_toolbox(strategy, ind_init, evaluator)

        stats = make_cma_stats()
        hof = tools.HallOfFame(1)

        fitness_history = []

        # Logger + unified champion-rollout recording.
        log = build_logger(config, algorithm="cma-es")
        env_steps_per_iteration = self.population_size * max_steps_per_episode
        self._recorder = RecordingController(
            config,
            log,
            max_steps=max_steps_per_episode,
            seed=seed,
        )
        from deap_pettingzoo.baselines.probing import ProbeController

        # Periodic held-out champion probe (off-budget: runs on its own env).
        self._prober = ProbeController(
            config,
            log,
            max_steps=max_steps_per_episode,
            seed=seed,
        )

        def _cma_champion():
            if len(hof) == 0:
                return None
            return GenomePolicyWrapper(
                hof[0].genome, config.env_name, action_space=action_space
            )

        # Parallel evaluation setup. num_workers / parallel_backend /
        # scheduler_address are first-class TrainingConfig fields; the
        # trainer-constructor value is the fallback used only when the config
        # leaves them at their defaults.
        requested = config.num_workers if config.num_workers != 1 else self.num_workers
        backend = (
            config.parallel_backend
            if config.parallel_backend != "process"
            else self.parallel_backend
        )
        scheduler_address = (
            config.scheduler_address
            if config.scheduler_address is not None
            else self.scheduler_address
        )
        # Remote scheduler (arg or DEAP_DASK_SCHEDULER env var): the workers
        # live elsewhere, so skip the driver CPU cap and force the parallel
        # dispatch path even for num_workers <= 1 (advisory there).
        effective_scheduler = (
            resolve_scheduler_address(scheduler_address) if backend == "dask" else None
        )
        remote = effective_scheduler is not None
        effective_workers = resolve_num_workers(requested, cap_to_cpu=not remote)
        use_parallel = effective_workers > 1 or remote
        # One snapshot, reused for the process-pool initializer and shipped on
        # the worker spec so Dask workers can re-register custom envs too.
        registry_snapshot = get_registry_snapshot(target_remote=remote)
        worker_spec = _EvalWorkerSpec(
            env_name=config.env_name,
            env_config=dict(config.env_config) if config.env_config else None,
            max_steps=max_steps_per_episode,
            aggregation_method=config.fitness.aggregation_method,
            team_mapping=config.fitness.team_mapping,
            registry_snapshot=registry_snapshot,
        )

        # Observed Dask worker hosts, captured before the executor closes.
        dask_workers: Optional[list] = None

        with make_executor(
            effective_workers,
            backend=backend,
            initializer=replay_registry_snapshot,
            initargs=(registry_snapshot,),
            scheduler_address=scheduler_address,
        ) as executor:
            base_eval_seed = seed
            _eval_counter = [0]
            # Aggregates real env-step deltas from parallel workers; stays at
            # 0 on the serial path where the evaluator counter is authoritative.
            _parallel_steps = [0]

            def _next_seeds(n: int):
                if base_eval_seed is None:
                    return [None] * n
                lo = int(base_eval_seed) + _eval_counter[0]
                _eval_counter[0] += n
                return list(range(lo, lo + n))

            # Nominal budgeting runs exactly ``num_generations``; consumed
            # budgeting runs until the real budget is exhausted, bounded by a
            # safety cap (each gen consumes >= 1 tick per individual
            # and episodes run at most ``max_steps`` ticks).
            if consumed_budget is not None:
                max_iters = num_generations * max(1, max_steps_per_episode) + 10
            else:
                max_iters = num_generations

            gen = 0
            while gen < max_iters:
                population = toolbox.generate()
                seeds = _next_seeds(len(population))
                if use_parallel:
                    fitnesses = evaluate_population_parallel(
                        population,
                        executor=executor,
                        spec=worker_spec,
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
                consumed = int(evaluator.env_steps_consumed) + int(_parallel_steps[0])
                fitness_history.append(
                    {
                        "generation": gen,
                        "mean": record["avg"],
                        "std": record["std"],
                        "min": record["min"],
                        "max": record["max"],
                        "env_steps": consumed,
                    }
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
                        champion=_cma_champion,
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
                        champion=_cma_champion,
                    )

            # Capture at the end of the run so late-joining workers show up.
            if hasattr(executor, "worker_info"):
                dask_workers = sorted({str(w["host"]) for w in executor.worker_info()})

        # Realized generation count (== num_generations on the nominal path).
        num_generations = gen
        env.close()
        self._recorder.finalize(
            _cma_champion,
            iteration=num_generations,
            env_steps=num_generations * env_steps_per_iteration,
        )
        self._recorder.close()
        self._prober.finalize(
            _cma_champion,
            iteration=num_generations,
            env_steps=num_generations * env_steps_per_iteration,
        )
        self._prober.close()
        training_time = time.time() - start_time

        best_individual = hof[0]
        best_genome = best_individual.genome
        policy = GenomePolicyWrapper(
            best_genome, config.env_name, action_space=action_space
        )

        # Persist results via the logger
        try:
            with log:
                dump_fitness_history(
                    log,
                    fitness_history,
                    env_steps_per_iteration=self.population_size
                    * max_steps_per_episode,
                    gen_offset=-1,
                    nominal_env_steps_per_iteration=self.population_size
                    * max_steps_per_episode,
                )
                from deap_pettingzoo.utils.logging import (
                    KIND_CMA_GENOME,
                    save_ea_genome,
                )

                save_ea_genome(
                    log,
                    best_genome,
                    kind=KIND_CMA_GENOME,
                    extra={
                        "final_fitness": float(best_individual.fitness.values[0]),
                        "env_name": config.env_name,
                        "variant": self.variant,
                    },
                )
                write_summary(
                    log,
                    {
                        "algorithm": "cma-es",
                        "final_fitness": float(best_individual.fitness.values[0]),
                        "num_generations": num_generations,
                        "population_size": self.population_size,
                        "dimension": dimension,
                        "variant": self.variant,
                    },
                    training_time_s=training_time,
                )
        except Exception:
            pass

        return TrainingResult(
            algorithm="cma-es",
            policy=policy,
            metrics={
                "fitness_history": fitness_history,
                "final_fitness": best_individual.fitness.values[0],
                "num_generations": num_generations,
                "population_size": self.population_size,
                "dimension": dimension,
                # CMA-ES evaluates ``num_generations`` populations with no
                # separate gen-0 eval, so nominal is num_generations * per-iter
                # (unlike EA's (num_generations + 1) * per-iter).
                "env_steps_nominal": num_generations * env_steps_per_iteration,
                "env_steps_consumed": (
                    int(evaluator.env_steps_consumed) + int(_parallel_steps[0])
                ),
            },
            metadata={
                "genome_config": genome_config,
                "cma_config": {
                    "hidden_size": self.hidden_size,
                    "population_size": self.population_size,
                    "sigma0": self.sigma0,
                    "activation": self.activation,
                    "weight_range": list(self.weight_range),
                    "variant": self.variant,
                },
                "num_workers": effective_workers,
                "parallel_backend": backend,
                "scheduler_address": effective_scheduler,
                "dask_workers": dask_workers,
                "run_dir": str(log.run_dir) if config.output_dir is not None else None,
            },
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """CMA-ES supports env-step, eval-episode, and generation budgets."""
        return mode in {
            BudgetMode.ENV_STEPS,
            BudgetMode.EVAL_EPISODES,
            BudgetMode.GENERATIONS,
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Report CMA-ES capability flags (matches EA's)."""
        return {
            "discrete_action": True,
            "continuous_action": True,
            "multidiscrete_action": True,
            "needs_global_state": False,
            "on_policy": False,
            "supports_turn_based": True,
        }

    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        """Translate a budget into ``num_generations`` and episode length.

        Matches :meth:`EATrainer._translate_budget` so identical budget
        configurations yield the same number of generations for both
        trainers.

        Raises:
            ValueError: If ``config.budget_mode`` is not
                ``BudgetMode.ENV_STEPS``, ``BudgetMode.EVAL_EPISODES``, or
                ``BudgetMode.GENERATIONS``.
        """
        max_steps_per_episode = (
            config.max_steps_per_episode
            if config.max_steps_per_episode is not None
            else 200
        )

        if config.budget_mode == BudgetMode.GENERATIONS:
            num_generations = max(1, int(config.budget_value))
        elif config.budget_mode == BudgetMode.ENV_STEPS:
            num_generations = derive_generations(
                config.budget_value,
                self.population_size,
                max_steps_per_episode,
                label="CMA-ES",
            )
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            num_generations = derive_generations(
                config.budget_value,
                self.population_size,
                1,
                label="CMA-ES",
                unit="eval episodes",
            )
        else:
            raise ValueError(
                f"CMAESTrainer does not support budget mode {config.budget_mode}"
            )

        return {
            "num_generations": int(num_generations),
            "max_steps_per_episode": int(max_steps_per_episode),
        }

    def _build_strategy(self, genome_config: Dict[str, Any]):
        """Dispatch to the CMA-ES flavour selected by ``self.variant``."""
        if self.variant == "mu_mu_lambda":
            return setup_cma_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        if self.variant == "one_plus_lambda":
            return setup_cma_one_plus_lambda_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        raise ValueError(  # pragma: no cover — guarded in __init__.
            f"Unknown CMA-ES variant {self.variant!r}"
        )
