"""
EA trainer implementation using DEAP.

This module wraps the existing DEAP integration to conform to the
unified BaselineTrainer interface.
"""

import time
from pathlib import Path
from typing import Dict, Any, ClassVar, Optional
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
    TrainingConfig,
    TrainingResult,
    PolicyProtocol,
)
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
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

    #: Per-generation cadence is the natural fit for EA — every
    #: generation produces one champion-rollout artefact when
    #: ``TrainingConfig.recording.enabled`` is set.
    _default_recording_every_n: ClassVar[int] = 1

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

        # Parallel evaluation defaults (overridable via
        # ``TrainingConfig.algorithm_params``). See
        # :mod:`deap_pettingzoo.utils.parallel` for semantics; we only
        # validate the range here. Windows ``spawn`` adds ~1-3s per
        # worker to the first evaluation — amortised over the whole
        # run, but noticeable for tiny populations.
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
    
    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train EA with the given configuration.
        
        Args:
            config: Training configuration.
            
        Returns:
            TrainingResult with best genome wrapped as policy.
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
            aggregation_method='sum'
        )
        
        # Setup DEAP
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
        
        hof = tools.HallOfFame(1)
        
        # Setup random number generator
        rng = np.random.default_rng(config.algorithm_params.get('seed'))
        
        # Track metrics
        fitness_history = []

        # ----- Logger + optional FrameRecorder ---------------------------
        # Logger is built up-front (run_dir is created eagerly in
        # ``ExperimentLogger.create``) so the FrameRecorder can write
        # ``frames/train_episode_NNNN.{mp4,gif}`` while training is
        # still running. Falls back to NoOpLogger when
        # ``config.output_dir is None``.
        log = build_logger(config, algorithm="ea")
        recording_cfg = config.recording
        recording_cadence = self._resolve_recording_every_n(config)
        recording_evaluator: Optional[UnifiedEvaluator] = None
        if recording_cfg.enabled:
            recorder = log.get_frame_recorder(
                fps=recording_cfg.fps,
                fmt=recording_cfg.fmt,
                max_frames_per_episode=recording_cfg.max_frames_per_episode,
                episode_prefix=recording_cfg.episode_prefix or "train_",
            )
            if recorder.enabled:
                recording_evaluator = UnifiedEvaluator(
                    max_steps=max_steps_per_episode,
                    frame_recorder=recorder,
                )

        def _record_champion_if_due(gen: int) -> None:
            """Record one champion-rollout episode when cadence hits.

            Best-effort: any failure is swallowed so recording can't
            break a long training run.
            """
            if recording_evaluator is None or hof.items == []:
                return
            if gen % recording_cadence != 0:
                return
            try:
                best = hof[0]
                champ_policy = GenomePolicyWrapper(
                    best.genome, config.env_name, action_space=action_space
                )
                rollout_seed = config.algorithm_params.get('seed')
                recording_evaluator.evaluate(
                    policy=champ_policy,
                    env_name=config.env_name,
                    num_episodes=1,
                    seed=int(rollout_seed) if rollout_seed is not None else None,
                    env_config=dict(config.env_config) if config.env_config else None,
                )
            except Exception:
                pass

        # --- Parallel evaluation setup ----------------------------------
        # ``algorithm_params["num_workers"]`` overrides the constructor
        # default. ``resolve_num_workers`` normalises 0/None to 1 and
        # caps at ``os.cpu_count()``.
        requested = config.algorithm_params.get("num_workers", self.num_workers)
        effective_workers = resolve_num_workers(requested)
        backend = config.algorithm_params.get(
            "parallel_backend", self.parallel_backend
        )
        worker_spec = _EvalWorkerSpec(
            env_name=config.env_name,
            env_config=dict(config.env_config) if config.env_config else None,
            max_steps=max_steps_per_episode,
            aggregation_method="sum",
            team_mapping=None,
        )

        with make_executor(
            effective_workers, backend=backend
        ) as executor:
            # Register toolbox.map so DEAP's built-in eaMu*Lambda
            # algorithms dispatch through the pool.
            if effective_workers > 1:
                toolbox.register(
                    "map",
                    make_parallel_evaluate_map(executor, worker_spec),
                )

            # Deterministic per-individual evaluation seeds when a
            # base seed is provided — makes serial and parallel runs
            # produce identical fitness arrays. Applied on the
            # in-house ``simple`` loop only; the ``eaMu*Lambda`` paths
            # go through DEAP's own ``toolbox.map`` with non-seeded
            # evaluation (matches prior behaviour).
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

            # Evolution loop — strategy-dependent.
            if self.strategy == "simple":
                self._run_simple(
                    population, toolbox, stats, hof, fitness_history,
                    num_generations, rng,
                    eval_population=_eval_population,
                    on_generation=_record_champion_if_due,
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
                # eaMu*Lambda doesn't expose a per-gen hook; record once
                # against the final hall-of-fame champion.
                _record_champion_if_due(num_generations)
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
                _record_champion_if_due(num_generations)
            else:  # pragma: no cover — guarded in __init__.
                raise ValueError(f"Unknown strategy {self.strategy!r}")
        # --- End parallel evaluation scope ------------------------------

        # Clean up
        env.close()
        if recording_evaluator is not None:
            recording_evaluator.close()
        
        training_time = time.time() - start_time
        
        # Get best individual
        best_individual = hof[0]
        best_genome = best_individual.genome
        
        # Wrap genome as policy
        policy = GenomePolicyWrapper(best_genome, config.env_name, action_space=action_space)

        # ----- Persist results via the (already-constructed) logger ------
        # No-op when ``config.output_dir is None`` (NoOpLogger) so the
        # backend stays usable without the analysis stack.
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
            },
            training_time=training_time,
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
