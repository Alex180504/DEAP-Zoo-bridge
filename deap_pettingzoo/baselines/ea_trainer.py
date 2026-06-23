"""
EA trainer implementation using DEAP.

This module wraps the existing DEAP integration to conform to the
unified BaselineTrainer interface.
"""

import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
import numpy as np
from deap import algorithms, tools

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
    PolicyProtocol,
)
from deap_pettingzoo.baselines.recording import RecordingController
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration.deap_integration import (
    _EvalWorkerSpec,
    create_toolbox,
    evaluate_population_parallel,
    make_parallel_evaluate_map,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.policies.action_heads import required_output_size
from deap_pettingzoo.environments.registry import (
    get_registry_snapshot,
    replay_registry_snapshot,
)
from deap_pettingzoo.utils.parallel import make_executor, resolve_num_workers


def _resolve_shared_action_space(env: PettingZooWrapper) -> Any:
    """Return the single shared action space across ``env.possible_agents``.

    deap-bridge's EA/CMA-ES stack is built around homogeneous-team
    neuroevolution: every agent is driven by the same genome, so the
    action decoding has to be identical across agents.  This helper
    enforces that contract.

    Args:
        env: Environment wrapper.

    Returns:
        The shared :class:`gymnasium.spaces.Space` instance (the
        space object belonging to the first agent; all others compare
        equal via :meth:`gymnasium.spaces.Space.__eq__`).

    Raises:
        ValueError: If any agent's action space differs from the first
            agent's.  The error message lists the divergent agents so
            callers can see which env needs a per-role (e.g.
            :class:`MixedPolicyEvaluator`) or AgileRL trainer setup
            instead.
    """
    agents = list(env.possible_agents)
    reference = env.action_space(agents[0])
    divergent = [a for a in agents[1:] if env.action_space(a) != reference]
    if divergent:
        raise ValueError(
            "EA/CMA-ES trainers require a single shared action space across "
            "all agents (homogeneous-team neuroevolution). "
            f"Agents diverging from {agents[0]!r}'s space: {divergent}. "
            "Use MixedPolicyEvaluator (with per-role policies) or an "
            "AgileRL trainer for heterogeneous-action envs."
        )
    return reference


class GenomePolicyWrapper:
    """Wraps a genome to satisfy PolicyProtocol.

    Converts genome evaluation calls into the policy interface
    expected by UnifiedEvaluator. For homogeneous multi-agent
    environments, all agents use the same policy.

    When ``action_space`` is supplied, the inner
    :class:`FeedforwardNeuralPolicy` attaches a matching
    :class:`~deap_pettingzoo.policies.action_heads.ActionHead`, so
    :meth:`get_action` returns a value that is valid for the env's
    action space (``int`` for Discrete, ``np.ndarray`` for Box).
    """

    def __init__(self, genome, env_name: str, action_space: Optional[Any] = None):
        """Initialize policy wrapper.

        Args:
            genome: FixedTopologyGenome instance.
            env_name: Environment name (stored for reference).
            action_space: Optional shared action space; when supplied,
                the inner :class:`FeedforwardNeuralPolicy` decodes raw
                network output into an env-valid action.
        """
        self._genome = genome
        self._env_name = env_name
        self._action_space = action_space

        # Create single policy for all agents (homogeneous)
        self._policy = FeedforwardNeuralPolicy(genome, action_space=action_space)
    
    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        """Get action for observation.

        Args:
            observation: Environment observation.
            action_mask: Optional legal-action mask forwarded to the
                underlying :class:`FeedforwardNeuralPolicy` (used by
                turn-based AEC adapters).

        Returns:
            Action for the given observation.
        """
        return self._policy.get_action(observation, action_mask=action_mask)
    
    def reset(self) -> None:
        """Reset policy state (no-op for stateless feedforward policies)."""
    
    @property
    def genome(self):
        """Access underlying genome."""
        return self._genome


class EATrainer(BaselineTrainer):
    """Evolutionary Algorithm trainer using DEAP.
    
    Wraps the existing DEAP integration to provide a unified
    training interface compatible with BaselineTrainer.
    """

    @staticmethod
    def describe_hyperparameters():
        """Declarative hyperparameter overlay consumed by the catalog.

        Returns the :class:`~deap_pettingzoo.catalog.schema.ParamSpec` tuple
        describing every UI-configurable knob of this trainer. The
        :class:`~deap_pettingzoo.catalog.schema.ParamTarget` of each spec tells
        :func:`~deap_pettingzoo.baselines.experiment_spec.build_configs` whether
        the value is a trainer-constructor argument (``trainer_params``) or a
        per-run ``algorithm_params`` entry.

        Dict-valued knobs (``selection_params`` / ``crossover_params`` /
        ``mutation_params``) and ``fitness_spec`` are intentionally omitted —
        they are raw-config / Python-API territory (see the UI-overhaul spec).
        """
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
                      help="Number of individuals evaluated per generation."),
            ParamSpec("strategy", E, "simple", target=TR, group="Strategy",
                      choices=("simple", "mu_plus_lambda", "mu_comma_lambda"),
                      help="Evolutionary driver / generational loop."),
            ParamSpec("mu", I, None, minimum=1, step=1, target=TR,
                      group="Strategy", advanced=True, nullable=True,
                      help="Parents kept between generations "
                           "(mu_*_lambda strategies). Defaults to population."),
            ParamSpec("lambda_", I, None, minimum=1, step=1, target=TR,
                      group="Strategy", advanced=True, nullable=True,
                      help="Offspring per generation (mu_*_lambda strategies)."),
            ParamSpec("selection_method", E, "tournament", target=TR,
                      group="Selection", advanced=True,
                      choices=("tournament", "roulette", "best", "random",
                               "nsga2", "spea2"),
                      help="Parent-selection operator."),
            ParamSpec("tournament_size", I, 3, minimum=1, step=1, target=TR,
                      group="Selection", advanced=True,
                      help="Tournament size (tournament selection only)."),
            ParamSpec("crossover_method", E, "blend", target=TR,
                      group="Operators", advanced=True,
                      choices=("blend", "one_point", "two_point", "uniform",
                               "sbx", "es_blend"),
                      help="Crossover operator."),
            ParamSpec("crossover_rate", F, 0.7, minimum=0.0, maximum=1.0,
                      step=0.05, target=TR, group="Operators",
                      help="Probability of applying crossover."),
            ParamSpec("crossover_alpha", F, 0.5, minimum=0.0, maximum=1.0,
                      step=0.05, target=TR, group="Operators", advanced=True,
                      help="Blend-crossover alpha (blend / es_blend only)."),
            ParamSpec("mutation_method", E, "gaussian", target=TR,
                      group="Operators", advanced=True,
                      choices=("gaussian", "polynomial_bounded"),
                      help="Mutation operator."),
            ParamSpec("mutation_rate", F, 0.3, minimum=0.0, maximum=1.0,
                      step=0.05, target=TR, group="Operators",
                      help="Probability of mutating a child."),
            ParamSpec("mutation_sigma", F, 0.2, minimum=0.0, step=0.05,
                      target=TR, group="Operators", advanced=True,
                      help="Gaussian mutation standard deviation."),
            ParamSpec("mutation_indpb", F, 0.1, minimum=0.0, maximum=1.0,
                      step=0.05, target=TR, group="Operators", advanced=True,
                      help="Per-weight independent mutation probability."),
            ParamSpec("num_workers", I, 1, minimum=0, step=1, target=TR,
                      group="Parallelism",
                      help="Parallel evaluation workers (1 = serial)."),
            ParamSpec("parallel_backend", E, "process", target=TR,
                      group="Parallelism", advanced=True,
                      choices=("process", "scoop", "dask"),
                      help="Parallel executor backend."),
            ParamSpec("seed", I, None, minimum=0, step=1,
                      target=AP, group="Run", nullable=True,
                      help="Random seed for this run (None = unseeded)."),
            ParamSpec("max_steps_per_episode", I, 200, minimum=1, step=1,
                      target=AP, group="Run",
                      help="Hard cap on environment steps per rollout."),
        )

    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_alpha: float = 0.5,
        mutation_sigma: float = 0.2,
        mutation_indpb: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        strategy: str = "simple",
        mu: Optional[int] = None,
        lambda_: Optional[int] = None,
        selection_method: str = "tournament",
        crossover_method: str = "blend",
        mutation_method: str = "gaussian",
        selection_params: Optional[Dict[str, Any]] = None,
        crossover_params: Optional[Dict[str, Any]] = None,
        mutation_params: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        parallel_backend: str = "process",
        # --- multi-objective (Phase 1.3) ---
        fitness_spec: Optional[Any] = None,
    ):
        """Initialize EA trainer.

        Args:
            hidden_size: Hidden layer size for neural network.
            population_size: Number of individuals in population.
            tournament_size: Tournament selection size (ignored unless
                ``selection_method == 'tournament'``).
            crossover_alpha: Blend crossover alpha parameter (used only
                when ``crossover_method in {'blend', 'es_blend'}``).
            mutation_sigma: Gaussian mutation standard deviation (used
                only when ``mutation_method == 'gaussian'``).
            mutation_indpb: Independent mutation probability per weight.
            crossover_rate: Probability of applying crossover per pair
                in the ``"simple"`` strategy (DEAP's own ``cxpb`` for the
                ``eaMu*Lambda`` strategies).
            mutation_rate: Probability of applying mutation per child
                (``mutpb`` for the ``eaMu*Lambda`` strategies).
            strategy: Evolutionary driver. One of ``"simple"`` (current
                in-house generational loop), ``"mu_plus_lambda"``
                (:func:`deap.algorithms.eaMuPlusLambda`) or
                ``"mu_comma_lambda"``
                (:func:`deap.algorithms.eaMuCommaLambda`).
            mu: Parents kept between generations in the
                ``mu_*_lambda`` strategies. Defaults to
                ``population_size``.
            lambda_: Offspring produced per generation in the
                ``mu_*_lambda`` strategies. Defaults to
                ``population_size``.
            selection_method: ``create_toolbox`` selection method —
                one of ``'tournament'``, ``'roulette'``, ``'best'``,
                ``'random'``.
            crossover_method: ``create_toolbox`` crossover method —
                one of ``'blend'``, ``'one_point'``, ``'two_point'``,
                ``'uniform'``, ``'sbx'``, ``'es_blend'``.
            mutation_method: ``create_toolbox`` mutation method —
                one of ``'gaussian'``, ``'polynomial_bounded'``.
            selection_params: Extra kwargs merged into the selection
                config (e.g. ``{'tournament_size': 5}``).
            crossover_params: Extra kwargs merged into the crossover
                config (e.g. ``{'eta': 20.0}`` for SBX).
            mutation_params: Extra kwargs merged into the mutation
                config.

        Raises:
            ValueError: If ``strategy`` is not one of the three recognised
                driver names.
            ValueError: If ``strategy == 'mu_comma_lambda'`` and
                ``lambda_ < mu``.
            ValueError: If ``num_workers`` is negative.
            ValueError: If ``parallel_backend`` is not in
                :data:`~deap_pettingzoo.utils.parallel.SUPPORTED_BACKENDS`.
        """
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_alpha = crossover_alpha
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        if strategy not in {"simple", "mu_plus_lambda", "mu_comma_lambda"}:
            raise ValueError(
                f"Unknown EA strategy {strategy!r}. "
                "Expected 'simple', 'mu_plus_lambda' or 'mu_comma_lambda'."
            )
        self.strategy = strategy
        self.mu = mu if mu is not None else population_size
        self.lambda_ = lambda_ if lambda_ is not None else population_size
        if strategy == "mu_comma_lambda" and self.lambda_ < self.mu:
            raise ValueError(
                f"mu_comma_lambda requires lambda_ >= mu; "
                f"got mu={self.mu}, lambda_={self.lambda_}."
            )

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.selection_params = dict(selection_params or {})
        self.crossover_params = dict(crossover_params or {})
        self.mutation_params = dict(mutation_params or {})
        self.fitness_spec = fitness_spec

        # Multi-objective guard: NSGA-II / SPEA2 require the "simple"
        # strategy.  eaMu*Lambda uses DEAP builtins that cannot dispatch
        # multi-objective selection internally.
        if self._is_multiobjective and strategy != "simple":
            raise ValueError(
                "Multi-objective selection (nsga2/spea2) requires "
                f"strategy='simple', got {strategy!r}."
            )

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
        self.num_workers = int(num_workers)
        self.parallel_backend = str(parallel_backend)

    @property
    def _is_multiobjective(self) -> bool:
        """``True`` when a multi-objective selection method is configured."""
        return self.selection_method in {"nsga2", "spea2"}

    @staticmethod
    def _mo_champion(pareto_or_hof: Any) -> Any:
        """Return a single champion from a ParetoFront or HallOfFame.

        For a Pareto front (multi-objective) the individual with the
        best **first** objective value is returned (this is typically
        the reward axis).  For a standard HallOfFame ``[0]`` is used.
        """
        if isinstance(pareto_or_hof, tools.ParetoFront):
            if not pareto_or_hof.items:
                raise RuntimeError("ParetoFront is empty")
            return max(pareto_or_hof.items, key=lambda ind: ind.fitness.values[0])
        return pareto_or_hof[0]
    
    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Train EA with the given configuration.

        Args:
            config: Training configuration.
            progress_callback: Optional per-generation progress hook (fires for
                the ``"simple"`` strategy live; for ``mu_*_lambda`` strategies
                the events are emitted after DEAP's hidden loop returns).

        Returns:
            TrainingResult with best genome wrapped as policy.

        Raises:
            ValueError: If ``config.budget_mode`` is not supported by
                this trainer (propagated from :meth:`_translate_budget`).
            ValueError: If the environment uses heterogeneous action
                spaces (propagated from
                :func:`_resolve_shared_action_space`).
        """
        start_time = time.time()
        
        # Translate budget to EA parameters
        ea_params = self._translate_budget(config)
        num_generations = ea_params['num_generations']
        max_steps_per_episode = ea_params['max_steps_per_episode']
        
        # Setup environment and evaluator
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
            fitness_spec=self.fitness_spec,
        )
        
        # Setup DEAP — with multi-objective weights when appropriate.
        if self._is_multiobjective and self.fitness_spec is not None:
            setup_deap_creators(weights=self.fitness_spec.weights)
        else:
            setup_deap_creators()
        
        # Create genome configuration
        seed = config.algorithm_params.get('seed', None)
        genome_config = {
            'layer_sizes': [input_size, self.hidden_size, output_size],
            'activation': 'tanh',
            'weight_range': (-1.0, 1.0),
            'random_seed': seed if seed is not None else (config.num_seeds if config.num_seeds == 1 else None),
        }
        
        # Build operator configs (method + per-method params).
        selection_config: Dict[str, Any] = {
            "method": self.selection_method,
            "tournament_size": self.tournament_size,
        }
        selection_config.update(self.selection_params)

        crossover_config: Dict[str, Any] = {
            "method": self.crossover_method,
            "alpha": self.crossover_alpha,
        }
        crossover_config.update(self.crossover_params)

        mutation_config: Dict[str, Any] = {
            "method": self.mutation_method,
            "sigma": self.mutation_sigma,
            "indpb": self.mutation_indpb,
        }
        mutation_config.update(self.mutation_params)

        # Create toolbox
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config=selection_config,
            crossover_config=crossover_config,
            mutation_config=mutation_config,
        )
        
        # Override clone to use our clone_individual
        from deap_pettingzoo.integration.deap_integration import clone_individual
        toolbox.register("clone", clone_individual)
        
        # Create initial population
        population = toolbox.population(n=self.population_size)
        
        # Setup statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        if self._is_multiobjective:
            hof = tools.ParetoFront()
        else:
            hof = tools.HallOfFame(1)
        
        # Setup random number generator
        rng = np.random.default_rng(config.algorithm_params.get('seed'))
        
        # Track metrics
        fitness_history = []

        log = build_logger(config, algorithm="ea")
        env_steps_per_iteration = self.population_size * max_steps_per_episode
        self._recorder = RecordingController(
            config, log, max_steps=max_steps_per_episode,
            seed=config.algorithm_params.get('seed'),
        )

        def _ea_champion():
            """Lazy champion provider for the recording controller."""
            items = hof.items if isinstance(hof, tools.ParetoFront) else hof
            if not items:
                return None
            return GenomePolicyWrapper(
                self._mo_champion(hof).genome,
                config.env_name,
                action_space=action_space,
            )

        def _make_event(gen: int) -> Optional[ProgressEvent]:
            if not fitness_history:
                return None
            entry = fitness_history[-1]
            return ProgressEvent(
                unit="generation",
                current=gen,
                total=num_generations,
                env_steps=gen * env_steps_per_iteration,
                best_fitness=entry.get("max"),
                mean_fitness=entry.get("mean"),
                wall_time_s=time.time() - start_time,
            )

        def _emit_gen_progress(gen: int) -> None:
            """Emit progress only (no recording) — used at gen 0 and for the
            DEAP-internal μ±λ strategies whose hof is only final."""
            event = _make_event(gen)
            if event is not None:
                self._emit_progress(progress_callback, event)

        def _on_generation(gen: int) -> None:
            """Per-generation hook: emit progress **and** maybe record."""
            event = _make_event(gen)
            if event is not None:
                self._on_progress(progress_callback, event, champion=_ea_champion)

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
            max_steps=max_steps_per_episode,
            aggregation_method=config.fitness.aggregation_method,
            team_mapping=config.fitness.team_mapping,
            registry_snapshot=registry_snapshot,
        )

        with make_executor(
            effective_workers,
            backend=backend,
            initializer=replay_registry_snapshot,
            initargs=(registry_snapshot,),
        ) as executor:
            # Register toolbox.map so DEAP's built-in eaMu*Lambda algorithms dispatch through the pool.
            if effective_workers > 1:
                toolbox.register(
                    "map",
                    make_parallel_evaluate_map(executor, worker_spec),
                )
                
            base_eval_seed = config.algorithm_params.get("seed")
            _eval_counter = [0]

            def _next_seeds(n: int):
                if base_eval_seed is None:
                    return [None] * n
                lo = int(base_eval_seed) + _eval_counter[0]
                _eval_counter[0] += n
                return list(range(lo, lo + n))

            def _eval_population(inds):
                """Serial or parallel evaluation based on workers."""
                inds = list(inds)
                seeds = _next_seeds(len(inds))
                if effective_workers > 1:
                    return evaluate_population_parallel(
                        inds,
                        executor=executor,
                        spec=worker_spec,
                        eval_seeds=seeds,
                    )
                return [
                    evaluator.evaluate(ind, seed=s)
                    for ind, s in zip(inds, seeds)
                ]

            # Evaluate initial population
            fitnesses = _eval_population(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            hof.update(population)
            record = stats.compile(population)
            fitness_history.append({
                'generation': 0,
                'mean': record['avg'],
                'std': record['std'],
                'min': record['min'],
                'max': record['max'],
            })
            _emit_gen_progress(0)

            # Evolution loop — strategy-dependent.
            if self.strategy == "simple":
                self._run_simple(
                    population, toolbox, stats, hof, fitness_history,
                    num_generations, rng,
                    eval_population=_eval_population,
                    on_generation=_on_generation,
                )
                # Safety net: guarantee a final capture even if the last
                # per-gen hook didn't land exactly on fraction 1.0 (dedups).
                self._recorder.finalize(
                    _ea_champion, iteration=num_generations,
                    env_steps=num_generations * env_steps_per_iteration,
                )
            elif self.strategy == "mu_plus_lambda":
                _, logbook = algorithms.eaMuPlusLambda(
                    population, toolbox,
                    mu=self.mu, lambda_=self.lambda_,
                    cxpb=self.crossover_rate, mutpb=self.mutation_rate,
                    ngen=num_generations,
                    stats=stats, halloffame=hof, verbose=False,
                )
                for record in logbook:
                    gen = record['gen']
                    if gen == 0:
                        continue  # already recorded from initial evaluation
                    fitness_history.append({
                        'generation': gen,
                        'mean': record['avg'],
                        'std': record['std'],
                        'min': record['min'],
                        'max': record['max'],
                    })
                    _emit_gen_progress(gen)
                # eaMu*Lambda doesn't expose a per-gen hook and hof holds only
                # the final champion; record once at the end.
                self._recorder.finalize(
                    _ea_champion, iteration=num_generations,
                    env_steps=num_generations * env_steps_per_iteration,
                )
            elif self.strategy == "mu_comma_lambda":
                _, logbook = algorithms.eaMuCommaLambda(
                    population, toolbox,
                    mu=self.mu, lambda_=self.lambda_,
                    cxpb=self.crossover_rate, mutpb=self.mutation_rate,
                    ngen=num_generations,
                    stats=stats, halloffame=hof, verbose=False,
                )
                for record in logbook:
                    gen = record['gen']
                    if gen == 0:
                        continue  # already recorded from initial evaluation
                    fitness_history.append({
                        'generation': gen,
                        'mean': record['avg'],
                        'std': record['std'],
                        'min': record['min'],
                        'max': record['max'],
                    })
                    _emit_gen_progress(gen)
                self._recorder.finalize(
                    _ea_champion, iteration=num_generations,
                    env_steps=num_generations * env_steps_per_iteration,
                )
            else:  # pragma: no cover — guarded in __init__.
                raise ValueError(f"Unknown strategy {self.strategy!r}")

        # Clean up
        env.close()
        self._recorder.close()
        
        training_time = time.time() - start_time
        
        # Get best individual (MO: champion from Pareto front).
        best_individual = self._mo_champion(hof)
        best_genome = best_individual.genome
        
        # Wrap genome as policy
        policy = GenomePolicyWrapper(best_genome, config.env_name, action_space=action_space)

        # For multi-objective runs, record the full Pareto front.
        pareto_front: Optional[List[Any]] = None
        if isinstance(hof, tools.ParetoFront):
            pareto_front = list(hof.items)

        # Persist results via the logger 
        # No-op when ``config.output_dir is None`` (NoOpLogger) so the backend stays usable without the analysis stack.
        try:
            with log:
                dump_fitness_history(
                    log,
                    fitness_history,
                    env_steps_per_iteration=self.population_size * max_steps_per_episode,
                )
                from deap_pettingzoo.utils.logging import save_ea_genome, KIND_EA_GENOME
                save_ea_genome(
                    log,
                    best_genome,
                    kind=KIND_EA_GENOME,
                    extra={
                        "final_fitness": float(best_individual.fitness.values[0]),
                        "env_name": config.env_name,
                    },
                )
                write_summary(
                    log,
                    {
                        "algorithm": "ea",
                        "final_fitness": float(best_individual.fitness.values[0]),
                        "num_generations": num_generations,
                        "population_size": self.population_size,
                        "strategy": self.strategy,
                    },
                    training_time_s=training_time,
                )
        except Exception:
            # Defensive: never fail training because of logging cleanup.
            pass

        # Create result
        return TrainingResult(
            algorithm='ea',
            policy=policy,
            metrics={
                'fitness_history': fitness_history,
                'final_fitness': best_individual.fitness.values[0],
                'num_generations': num_generations,
                'population_size': self.population_size,
            },
            metadata={
                'genome_config': genome_config,
                'ea_config': {
                    'hidden_size': self.hidden_size,
                    'population_size': self.population_size,
                    'tournament_size': self.tournament_size,
                    'crossover_alpha': self.crossover_alpha,
                    'mutation_sigma': self.mutation_sigma,
                    'mutation_indpb': self.mutation_indpb,
                    'crossover_rate': self.crossover_rate,
                    'mutation_rate': self.mutation_rate,
                    'strategy': self.strategy,
                    'mu': self.mu,
                    'lambda_': self.lambda_,
                    'selection_method': self.selection_method,
                    'crossover_method': self.crossover_method,
                    'mutation_method': self.mutation_method,
                },
                'num_workers': effective_workers,
                'parallel_backend': backend,
                'run_dir': str(log.run_dir) if config.output_dir is not None else None,
            },
            training_time=training_time,
            pareto_front=pareto_front,
        )
    
    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """Check if EA supports the given budget mode.
        
        Args:
            mode: Budget mode to check.
            
        Returns:
            True if supported.
        """
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get EA capabilities.
        
        Returns:
            Capabilities dictionary.
        """
        return {
            'discrete_action': True,
            'continuous_action': True,
            'multidiscrete_action': True,
            'needs_global_state': False,
            'on_policy': False,  # EA is not on/off-policy (different paradigm)
            'supports_turn_based': True,
        }
    
    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        """Translate budget config to EA parameters.

        Args:
            config: Training configuration.

        Returns:
            Dictionary with EA-specific parameters.

        Raises:
            ValueError: If ``config.budget_mode`` is not
                ``BudgetMode.ENV_STEPS`` or ``BudgetMode.EVAL_EPISODES``.
        """
        max_steps_per_episode = config.algorithm_params.get('max_steps_per_episode', 200)
        
        if config.budget_mode == BudgetMode.ENV_STEPS:
            # Calculate generations from environment steps
            # env_steps = population_size * num_generations * max_steps_per_episode
            total_env_steps = config.budget_value
            num_generations = total_env_steps // (self.population_size * max_steps_per_episode)
            num_generations = max(1, num_generations)  # At least 1 generation
            
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            # Each episode evaluates one individual
            # Total episodes = population_size * num_generations
            total_episodes = config.budget_value
            num_generations = total_episodes // self.population_size
            num_generations = max(1, num_generations)
            
        else:
            raise ValueError(f"EA does not support budget mode: {config.budget_mode}")
        
        return {
            'num_generations': num_generations,
            'max_steps_per_episode': max_steps_per_episode,
        }

    def _run_simple(
        self,
        population,
        toolbox,
        stats,
        hof,
        fitness_history,
        num_generations: int,
        rng,
        *,
        eval_population,
        on_generation=None,
    ) -> None:
        """In-house ``eaSimple``-style generational loop.

        Preserves the historical behaviour of :class:`EATrainer`:
        per-pair crossover probability ``self.crossover_rate`` and
        per-individual mutation probability ``self.mutation_rate``,
        with full replacement.

        Args:
            eval_population: Callable ``(inds) -> [fitness_tuple, ...]``
                that runs serial or parallel evaluation depending on
                the configured worker count.
            on_generation: Optional callable ``(gen: int) -> None``
                invoked after each generation completes (after the
                hall-of-fame and statistics are updated). Used by the
                opt-in frame-recording machinery to fire one extra
                champion-rollout episode.
        """
        for gen in range(1, num_generations + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if rng.random() < self.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = eval_population(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            hof.update(population)

            record = stats.compile(population)
            fitness_history.append({
                'generation': gen,
                'mean': record['avg'],
                'std': record['std'],
                'min': record['min'],
                'max': record['max'],
            })

            if on_generation is not None:
                on_generation(gen)
