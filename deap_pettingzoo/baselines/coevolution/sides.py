"""Side abstractions — paradigm-agnostic runtime state per co-evolving group.

A :class:`Side` is the static *declaration* of a group: its name, the
role prefixes of the env agents it controls, and its optimiser choice.
A :class:`SideState` is the live runtime object that owns the
population / strategy, advances it one generation at a time, and
exposes its current members as an
:class:`~deap_pettingzoo.baselines.coevolution.pools.OpponentPool` so
*other* sides can sample it.

Three concrete implementations ship:

* :class:`EASideState` — classical GA over
  :class:`FixedTopologyGenome` via DEAP's toolbox.
* :class:`CMAESSideState` — CMA-ES via :class:`deap.cma.Strategy`.
* :class:`FrozenSideState` — non-evolving; wraps a pre-built
  :class:`FrozenPolicyPool` (use case: EA/CMA vs frozen pre-trained RL
  or scripted opponents).

All three satisfy the same :class:`SideState` protocol, which is what
the :class:`CoEvolutionTrainer` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from deap import tools

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.cma_integration import (
    make_cma_ind_init,
    register_cma_toolbox,
    setup_cma_one_plus_lambda_strategy,
    setup_cma_strategy,
)
from deap_pettingzoo.integration.deap_integration import (
    clone_individual,
    create_toolbox,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.policies.action_heads import required_output_size

from .archive import HallOfFame
from .pools import (
    ArchivePool,
    CompositePool,
    FrozenPolicyPool,
    OpponentPool,
    PolicyFactory,
    PopulationPool,
)
from .samplers import Sampler, uniform_sampler


# Side declaration


@dataclass
class Side:
    """Declarative description of one co-evolving group.

    Attributes:
        name: Unique identifier (appears in metrics / logs).
        role_prefixes: Agent-name prefixes this side controls (e.g.
            ``('adversary',)`` for MPE2's pursuers). Must be disjoint
            from all other sides in the run.
        optimizer: One of ``'ea' | 'cma' | 'frozen'``.
        optimizer_config: Per-optimizer keyword arguments. See the
            matching :class:`SideState` subclass for accepted keys.
        hof: Optional :class:`HallOfFame` attached to this side. When
            set, the trainer adds snapshots of the side's best
            individual at ``snapshot_every`` cadence. Other sides can
            opt into drawing opponents from it via
            ``use_archive_in_opponent_pool=True``.
        use_archive_in_opponent_pool: When ``True`` and ``hof`` is set,
            :meth:`SideState.current_pool` returns a
            :class:`CompositePool` blending the current generation with
            the HoF (weighted ``archive_mix``). Default ``False`` —
            current generation only.
        archive_mix: Weight on the HoF component of the composite pool
            (weight on the current-gen component is ``1 - archive_mix``).
            Ignored when ``use_archive_in_opponent_pool`` is False.
        opponent_sampler: Sampler other sides use when drawing from this
            side's current-generation pool. Default :func:`uniform_sampler`.
        archive_sampler: Sampler used to draw from this side's HoF.
            Default :func:`uniform_sampler`.
        frozen_policies: Pre-built policies for ``optimizer='frozen'``.
            Ignored for ``'ea'`` / ``'cma'``.
    """

    name: str
    role_prefixes: Tuple[str, ...]
    optimizer: str = "ea"
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    hof: Optional[HallOfFame] = None
    use_archive_in_opponent_pool: bool = False
    archive_mix: float = 0.5
    opponent_sampler: Sampler = uniform_sampler
    archive_sampler: Sampler = uniform_sampler
    frozen_policies: Optional[Sequence[PolicyProtocol]] = None

    def __post_init__(self) -> None:
        if self.optimizer not in {
            "ea",
            "cma",
            "frozen",
            "qdax_map_elites",
            "qdax_cma_me",
            "rl",
            "pso",
            "rl_hpo",
        }:
            raise ValueError(
                f"Side {self.name!r}: optimizer must be 'ea', 'cma', "
                "'frozen', 'qdax_map_elites', 'qdax_cma_me', 'rl', "
                f"'pso', or 'rl_hpo'; got {self.optimizer!r}."
            )
        if not self.role_prefixes:
            raise ValueError(
                f"Side {self.name!r}: role_prefixes must be non-empty."
            )
        self.role_prefixes = tuple(self.role_prefixes)
        if not (0.0 <= self.archive_mix <= 1.0):
            raise ValueError(
                f"Side {self.name!r}: archive_mix must be in [0, 1]; "
                f"got {self.archive_mix}."
            )
        if self.optimizer == "frozen" and not self.frozen_policies:
            raise ValueError(
                f"Side {self.name!r}: optimizer='frozen' requires a "
                "non-empty frozen_policies= list."
            )


# Runtime state — base class


@dataclass
class GenerationStats:
    """Per-generation summary emitted by :meth:`SideState.step_generation`."""

    generation: int
    side_name: str
    mean: float
    std: float
    min: float
    max: float
    best_fitness: float
    qd_score: Optional[float] = None
    coverage: Optional[float] = None
    archive_best_fitness: Optional[float] = None
    filled_niches: Optional[int] = None
    total_niches: Optional[int] = None


class SideState:
    """Base class for per-side runtime state.

    Subclasses must implement :meth:`initialise`, :meth:`step_generation`,
    :meth:`current_pool`, and :meth:`best_policy`.
    
    Notes:
        Subclasses that use ``np.random.*`` operators set this after
        seeding in :meth:`initialise` (via the trainer). When set,
        :meth:`step_generation` wraps operator calls in a save/set/restore
        block so each side advances its own RNG stream independently.
    """

    def __init__(self, side: Side) -> None:
        self.side = side
        self.generation = 0
        self.best_fitness: float = -np.inf
        self.history: List[GenerationStats] = []

        self._np_state: Optional[Any] = None

    # Lifecycle hooks — subclasses override

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:  # pragma: no cover - abstract
        raise NotImplementedError

    def current_pool(self) -> OpponentPool:  # pragma: no cover - abstract
        raise NotImplementedError

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:  # pragma: no cover
        raise NotImplementedError

    def best_snapshot(self) -> Any:  # pragma: no cover - abstract
        """Return the HoF-ready snapshot of the best individual."""
        raise NotImplementedError

    def snapshot_best(self) -> None:
        """Append the best current individual to this side's HoF (if any)."""
        if self.side.hof is None:
            return
        self.side.hof.add(self.best_snapshot(), float(self.best_fitness))


# EA side


def _make_feedforward_policy_factory(
    action_space: Any,
    masked: bool = False,
) -> PolicyFactory:
    """Build a :class:`PolicyFactory` that wraps a genome in an MLP policy.

    The factory accepts either a :class:`FixedTopologyGenome` or an
    individual with a ``.genome`` attribute (DEAP-style). Stateless,
    so a single sampled policy can safely be shared across multiple
    same-team agents in one episode.
    """

    def _factory(item: Any) -> PolicyProtocol:
        genome = item.genome if hasattr(item, "genome") else item
        if not isinstance(genome, FixedTopologyGenome):
            raise TypeError(
                "Feedforward policy factory expected FixedTopologyGenome "
                f"(or item.genome of that type); got {type(genome)}."
            )
        return FeedforwardNeuralPolicy(
            genome, action_space=action_space, masked=masked
        )

    return _factory


def _resolve_side_action_space(
    env: PettingZooWrapper, role_prefixes: Tuple[str, ...]
) -> Any:
    """Resolve the shared action space across the side's agents.

    EA/CMA sides assume a homogeneous team (one genome per individual
    controlling every agent on the side). This mirrors the contract
    already enforced by :func:`_resolve_shared_action_space` in the
    plain :class:`EATrainer`.
    """
    side_agents = [
        a for a in env.possible_agents if a.startswith(role_prefixes)
    ]
    if not side_agents:
        raise ValueError(
            f"No env agents match role_prefixes={role_prefixes!r}."
        )
    reference = env.action_space(side_agents[0])
    divergent = [a for a in side_agents[1:] if env.action_space(a) != reference]
    if divergent:
        raise ValueError(
            "EA/CMA co-evolution side requires a single shared action "
            f"space across its agents {side_agents!r}; divergent: "
            f"{divergent}. Use a FrozenSideState with per-agent policies "
            "or an AgileRL trainer for heterogeneous-action sides."
        )
    return reference


class EASideState(SideState):
    """EA side — classical GA over :class:`FixedTopologyGenome`.

    Accepted ``optimizer_config`` keys (all optional):

    * ``hidden_size`` (int, default ``32``).
    * ``population_size`` (int, default ``20``).
    * ``crossover_rate`` / ``mutation_rate`` (per-pair / per-child
      probability; defaults ``0.7`` / ``0.3``).
    * ``selection_method`` / ``crossover_method`` / ``mutation_method``
      (same dispatch as :class:`EATrainer`).
    * ``tournament_size`` (int, default ``3``).
    * ``crossover_alpha`` (float, default ``0.5``).
    * ``mutation_sigma`` / ``mutation_indpb`` (defaults ``0.2`` / ``0.1``).
    * ``weight_range`` (tuple, default ``(-1.0, 1.0)``).
    * ``activation`` (str, default ``'tanh'``).
    * ``masked`` (bool, default ``False``) — attach a
      :class:`MaskedDiscreteHead` (turn-based envs).
    * ``seed`` (int or None).
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self.hidden_size = int(cfg.get("hidden_size", 32))
        self.population_size = int(cfg.get("population_size", 20))
        self.crossover_rate = float(cfg.get("crossover_rate", 0.7))
        self.mutation_rate = float(cfg.get("mutation_rate", 0.3))
        self.tournament_size = int(cfg.get("tournament_size", 3))
        self.crossover_alpha = float(cfg.get("crossover_alpha", 0.5))
        self.mutation_sigma = float(cfg.get("mutation_sigma", 0.2))
        self.mutation_indpb = float(cfg.get("mutation_indpb", 0.1))
        self.selection_method = cfg.get("selection_method", "tournament")
        self.crossover_method = cfg.get("crossover_method", "blend")
        self.mutation_method = cfg.get("mutation_method", "gaussian")
        self.weight_range = tuple(cfg.get("weight_range", (-1.0, 1.0)))
        self.activation = cfg.get("activation", "tanh")
        self.masked = bool(cfg.get("masked", False))
        self.seed = cfg.get("seed", None)

        # Multi-objective support (Phase 1.3).
        self._is_multiobjective = self.selection_method in {"nsga2", "spea2"}

        self.toolbox = None
        self.population: List[Any] = []
        if self._is_multiobjective:
            self.hof = tools.ParetoFront()
        else:
            self.hof = tools.HallOfFame(1)
        self._policy_factory: Optional[PolicyFactory] = None
        self._action_space: Any = None
        #: Per-side deterministic RNG. Seeded from the trainer-provided
        #: ``rng`` (derived from :attr:`CoEvolutionConfig.seed`) in
        #: :meth:`initialise`, falling back to ``self.seed`` or
        #: ``np.random.default_rng()``. Used for all stochastic EA
        #: operations in :meth:`step_generation` so one side's RNG draws
        #: never influence another side's evolution.
        self._rng: np.random.Generator = np.random.default_rng()
        # _np_state inherited from SideState — set by the trainer after seeding.

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # Per-side RNG — prefer the trainer-provided rng, fall back to
        # the side's own optimizer_config seed, finally to a fresh one.
        if rng is not None:
            self._rng = rng
        elif self.seed is not None:
            self._rng = np.random.default_rng(int(self.seed))
        else:
            self._rng = np.random.default_rng()

        sample_obs = env.reset(seed=0)
        learner_agent = next(
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        )
        input_size = len(sample_obs[learner_agent])
        self._action_space = _resolve_side_action_space(env, self.side.role_prefixes)
        output_size = required_output_size(self._action_space)

        setup_deap_creators()
        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "weight_range": self.weight_range,
            "random_seed": self.seed,
        }

        # Placeholder evaluator — the trainer rewrites toolbox.evaluate
        # via register() before each step_generation call so the
        # opponent pools reflect the other sides' *current* state.
        def _noop_eval(_individual: Any) -> Tuple[float]:  # pragma: no cover
            return (0.0,)

        class _NoopEvaluator:  # lightweight shim conforming to BaseEvaluator.evaluate
            def evaluate(self, individual, **_kwargs):
                return _noop_eval(individual)

        selection_cfg = {
            "method": self.selection_method,
            "tournament_size": self.tournament_size,
        }
        crossover_cfg = {"method": self.crossover_method, "alpha": self.crossover_alpha}
        mutation_cfg = {
            "method": self.mutation_method,
            "sigma": self.mutation_sigma,
            "indpb": self.mutation_indpb,
        }

        self.toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=_NoopEvaluator(),
            selection_config=selection_cfg,
            crossover_config=crossover_cfg,
            mutation_config=mutation_cfg,
        )
        self.toolbox.register("clone", clone_individual)
        self.population = self.toolbox.population(n=self.population_size)
        self._policy_factory = _make_feedforward_policy_factory(
            self._action_space, masked=self.masked
        )

        # Initial fitness pass will happen in the trainer's "bootstrap"
        # step, which rebinds toolbox.evaluate against a fresh evaluator
        # sourced from the *other* sides' initial pools.

    def rebind_evaluator(self, evaluator: Any) -> None:
        """Swap ``toolbox.evaluate`` to use the given evaluator.

        Called by the trainer before every generation so the opponent
        pools reflect other sides' latest state.
        """
        if self.toolbox is None:  # pragma: no cover - defensive
            raise RuntimeError("EASideState.initialise() must be called first.")
        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", _evaluate_with, evaluator=evaluator)

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        """Re-evaluate every individual under the current toolbox.evaluate."""
        if self.toolbox is None:  # pragma: no cover
            raise RuntimeError("EASideState.initialise() must be called first.")
        if batch_eval_fn is not None:
            fitnesses = batch_eval_fn(self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in self.population:
                ind.fitness.values = self.toolbox.evaluate(ind)
        self.hof.update(self.population)
        if len(self.hof) > 0:
            self.best_fitness = float(self._mo_champion().fitness.values[0])

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        assert self.toolbox is not None
        self.generation += 1
        evaluator = evaluator_factory(self)
        self.rebind_evaluator(evaluator)

        # Invalidate all current individuals so parents are always re-evaluated against the current generation's opponents.
        for ind in self.population:
            del ind.fitness.values

        # Simple (mu, lambda)=(N, N) generational loop with full replacement - matches EATrainer's "simple" strategy.
        global_state_before: Optional[Any] = None
        if self._np_state is not None:
            global_state_before = np.random.get_state()
            np.random.set_state(self._np_state)

        offspring = [self.toolbox.clone(ind) for ind in self.population]
        for i in range(1, len(offspring), 2):
            if self._rng.random() < self.crossover_rate:
                self.toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values
        for ind in offspring:
            if self._rng.random() < self.mutation_rate:
                self.toolbox.mutate(ind)
                del ind.fitness.values

        if global_state_before is not None:
            self._np_state = np.random.get_state()
            np.random.set_state(global_state_before)

        # Evaluate invalid individuals.
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if batch_eval_fn is not None and invalid:
            fitnesses = batch_eval_fn(invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        # (mu, lambda) selection: choose next generation from offspring ONLY.
        # In competitive coevolution, (mu+lambda) elitism is harmful because
        # parent fitness values were earned against *previous* opponents; stale
        # high-scoring parents will dominate tournament selection over offspring
        # that are correctly evaluated against today's (harder) opponents,
        # causing the population to freeze at a local optimum forever.
        self.population = self.toolbox.select(offspring, self.population_size)
        self.hof.update(self.population)
        fitnesses = [ind.fitness.values[0] for ind in self.population]
        best = float(max(fitnesses))
        # Track running max of current-gen bests (not HoF, whose fitness is
        # stale — evaluated against a past opponent pool and not comparable
        # to current-gen offspring evaluated against current opponents).
        self.best_fitness = max(self.best_fitness, best)

        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(np.mean(fitnesses)),
            std=float(np.std(fitnesses)),
            min=float(np.min(fitnesses)),
            max=best,
            best_fitness=best,  # current-gen best, not stale HoF value
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        assert self._policy_factory is not None
        pop_pool: OpponentPool = PopulationPool(
            self.population,
            self._policy_factory,
            sampler=self.side.opponent_sampler,
        )
        if self.side.use_archive_in_opponent_pool and self.side.hof is not None:
            archive_pool = ArchivePool(
                self.side.hof,
                self._policy_factory,
                sampler=self.side.archive_sampler,
            )
            if len(self.side.hof) == 0:
                # Archive empty → behave as if pure current-gen pool.
                return pop_pool
            weights = (1.0 - self.side.archive_mix, self.side.archive_mix)
            return CompositePool([pop_pool, archive_pool], weights=weights)
        return pop_pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        assert len(self.hof) > 0, "No evaluated individuals yet."
        assert self._policy_factory is not None
        return self._policy_factory(self._mo_champion())

    def best_snapshot(self) -> Any:
        """Return a *cloned* genome of the current best (HoF-safe)."""
        assert len(self.hof) > 0
        return self._mo_champion().genome.clone()

    def _mo_champion(self) -> Any:
        """Return a champion from ParetoFront or HallOfFame.

        For a Pareto front the individual with the highest first
        objective value (typically reward) is returned.
        For standard HallOfFame ``[0]`` is returned.
        """
        if isinstance(self.hof, tools.ParetoFront):
            if not self.hof.items:
                raise RuntimeError("ParetoFront is empty")
            return max(self.hof.items, key=lambda ind: ind.fitness.values[0])
        return self.hof[0]


def _evaluate_with(individual: Any, evaluator: Any) -> Tuple[float, ...]:
    """Top-level adapter so DEAP can pickle toolbox.evaluate if needed."""
    return evaluator.evaluate(individual)


# 
# CMA-ES side


class CMAESSideState(SideState):
    """CMA-ES side — (μ/μ_w, λ) or (1+λ) over a flat weight vector.

    Accepted ``optimizer_config`` keys:

    * ``hidden_size``, ``population_size``, ``sigma0``, ``activation``,
      ``weight_range`` (see :class:`CMAESTrainer`).
    * ``variant``: ``'mu_mu_lambda'`` (default) or ``'one_plus_lambda'``.
    * ``masked``: attach :class:`MaskedDiscreteHead`.
    * ``seed``: int or None.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self.hidden_size = int(cfg.get("hidden_size", 32))
        self.population_size = int(cfg.get("population_size", 20))
        self.sigma0 = float(cfg.get("sigma0", 0.3))
        self.activation = cfg.get("activation", "tanh")
        self.weight_range = tuple(cfg.get("weight_range", (-1.0, 1.0)))
        self.variant = cfg.get("variant", "mu_mu_lambda")
        self.masked = bool(cfg.get("masked", False))
        self.seed = cfg.get("seed", None)

        self.strategy = None
        self.toolbox = None
        self.population: List[Any] = []
        self.hof = tools.HallOfFame(1)
        self._policy_factory: Optional[PolicyFactory] = None
        self._action_space: Any = None
        self._ind_init = None
        # _np_state inherited from SideState — set by the trainer after seeding.

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        sample_obs = env.reset(seed=0)
        learner_agent = next(
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        )
        input_size = len(sample_obs[learner_agent])
        self._action_space = _resolve_side_action_space(env, self.side.role_prefixes)
        output_size = required_output_size(self._action_space)

        # If a per-side rng is provided, derive a CMA seed from it so
        # multiple CMA sides stay independent under a single trainer seed.
        if rng is not None:
            cma_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            np.random.seed(cma_seed)
        elif self.seed is not None:
            np.random.seed(int(self.seed))

        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "weight_range": self.weight_range,
        }

        if self.variant == "mu_mu_lambda":
            self.strategy, _ = setup_cma_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        elif self.variant == "one_plus_lambda":
            self.strategy, _ = setup_cma_one_plus_lambda_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        else:
            raise ValueError(
                f"CMAESSideState: unknown variant {self.variant!r}. "
                "Expected 'mu_mu_lambda' or 'one_plus_lambda'."
            )

        self._ind_init = make_cma_ind_init(genome_config)

        class _NoopEvaluator:
            def evaluate(self, individual, **_kwargs):  # pragma: no cover
                return (0.0,)

        self.toolbox = register_cma_toolbox(self.strategy, self._ind_init, _NoopEvaluator())
        # Seed the population — needed so other sides can sample from
        # this side before first step_generation is completed.
        self.population = self.toolbox.generate()
        self._policy_factory = _make_feedforward_policy_factory(
            self._action_space, masked=self.masked
        )

    def rebind_evaluator(self, evaluator: Any) -> None:
        assert self.toolbox is not None
        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", _evaluate_with, evaluator=evaluator)

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        assert self.toolbox is not None
        if batch_eval_fn is not None:
            fitnesses = batch_eval_fn(self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in self.population:
                ind.fitness.values = self.toolbox.evaluate(ind)
        self.hof.update(self.population)
        if len(self.hof) > 0:
            self.best_fitness = float(self.hof[0].fitness.values[0])

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        assert self.toolbox is not None
        self.generation += 1
        evaluator = evaluator_factory(self)
        self.rebind_evaluator(evaluator)

        # Invalidate all current individuals so they are re-evaluated against
        # *this* generation's opponents (not the bootstrap pool).  Without this,
        # gen 1 silently reuses bootstrap-era fitness values for the CMA update.
        for ind in self.population:
            del ind.fitness.values

        # Re-use the currently-seeded population for this generation's
        # evaluation (it was generated during initialise() or the
        # previous update step), then update the strategy and resample.
        invalid_pop = [ind for ind in self.population if not ind.fitness.valid]
        if batch_eval_fn is not None and invalid_pop:
            fitnesses = batch_eval_fn(invalid_pop)
            for ind, fit in zip(invalid_pop, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in self.population:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)

        global_state_before: Optional[Any] = None
        if self._np_state is not None:
            global_state_before = np.random.get_state()
            np.random.set_state(self._np_state)

        self.toolbox.update(self.population)
        next_population = self.toolbox.generate()

        if global_state_before is not None:
            self._np_state = np.random.get_state()
            np.random.set_state(global_state_before)

        # Collect stats from the *evaluated* population before swapping.
        self.hof.update(self.population)
        fitnesses = [ind.fitness.values[0] for ind in self.population]
        best = float(max(fitnesses))
        # Track running max of current-gen bests (not HoF, whose fitness is
        # stale — evaluated against a past opponent pool and not comparable
        # to current-gen offspring evaluated against current opponents).
        self.best_fitness = max(self.best_fitness, best)

        # Pre-sample next generation's population so current_pool() is
        # populated even before evaluation begins.
        self.population = next_population

        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(np.mean(fitnesses)),
            std=float(np.std(fitnesses)),
            min=float(np.min(fitnesses)),
            max=best,
            best_fitness=best,  # current-gen best, not stale HoF value
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        assert self._policy_factory is not None
        pop_pool: OpponentPool = PopulationPool(
            self.population,
            self._policy_factory,
            sampler=self.side.opponent_sampler,
        )
        if self.side.use_archive_in_opponent_pool and self.side.hof is not None:
            if len(self.side.hof) == 0:
                return pop_pool
            archive_pool = ArchivePool(
                self.side.hof,
                self._policy_factory,
                sampler=self.side.archive_sampler,
            )
            weights = (1.0 - self.side.archive_mix, self.side.archive_mix)
            return CompositePool([pop_pool, archive_pool], weights=weights)
        return pop_pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        assert len(self.hof) > 0
        assert self._policy_factory is not None
        return self._policy_factory(self.hof[0])

    def best_snapshot(self) -> Any:
        assert len(self.hof) > 0
        return self.hof[0].genome.clone()


# ---------------------------------------------------------------------------
# PSO side
# ---------------------------------------------------------------------------


class PSOSideState(SideState):
    """PSO side — particle swarm over a flat weight vector.

    Accepted ``optimizer_config`` keys mirror :class:`PSOTrainer`:

    * ``hidden_size``, ``population_size``, ``phi1``, ``phi2``,
      ``weight``, ``smin``, ``smax``, ``activation``, ``weight_range``.
    * ``masked``: attach :class:`MaskedDiscreteHead`.
    * ``seed``: int or None.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self.hidden_size = int(cfg.get("hidden_size", 32))
        self.population_size = int(cfg.get("population_size", 20))
        self.phi1 = float(cfg.get("phi1", 2.0))
        self.phi2 = float(cfg.get("phi2", 2.0))
        self.weight = float(cfg.get("weight", 0.7298))
        self.smin = cfg.get("smin", None)
        self.smax = cfg.get("smax", None)
        self.activation = cfg.get("activation", "tanh")
        self.weight_range = tuple(cfg.get("weight_range", (-1.0, 1.0)))
        self.masked = bool(cfg.get("masked", False))
        self.seed = cfg.get("seed", None)

        self.pso_state: Optional[Dict[str, Any]] = None
        self.toolbox = None
        self.population: List[Any] = []
        self.hof = tools.HallOfFame(1)
        self._policy_factory: Optional[PolicyFactory] = None
        self._action_space: Any = None
        self._ind_init = None

    def initialise(
        self, env: PettingZooWrapper, rng: Optional[np.random.Generator] = None,
    ) -> None:
        sample_obs = env.reset(seed=0)
        learner_agent = next(
            a for a in env.possible_agents
            if a.startswith(self.side.role_prefixes)
        )
        input_size = len(sample_obs[learner_agent])
        self._action_space = _resolve_side_action_space(env, self.side.role_prefixes)
        output_size = required_output_size(self._action_space)

        if rng is not None:
            pso_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif self.seed is not None:
            pso_seed = int(self.seed)
        else:
            pso_seed = None

        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "weight_range": self.weight_range,
        }
        from deap_pettingzoo.integration.pso_integration import (
            setup_pso_swarm,
            make_pso_ind_init,
            register_pso_toolbox,
        )
        self.pso_state, _ = setup_pso_swarm(
            genome_config=genome_config,
            population_size=self.population_size,
            phi1=self.phi1, phi2=self.phi2, weight=self.weight,
            smin=self.smin, smax=self.smax,
            seed=pso_seed,
        )
        self._ind_init = make_pso_ind_init(genome_config, self.pso_state)

        class _NoopEvaluator:
            def evaluate(self, individual, **_kwargs):  # pragma: no cover
                return (0.0,)

        self.toolbox = register_pso_toolbox(
            self.pso_state, self._ind_init, _NoopEvaluator(),
        )
        self.population = self.toolbox.generate()
        self._policy_factory = _make_feedforward_policy_factory(
            self._action_space, masked=self.masked,
        )

    def rebind_evaluator(self, evaluator: Any) -> None:
        assert self.toolbox is not None
        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", _evaluate_with, evaluator=evaluator)

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        assert self.toolbox is not None
        if batch_eval_fn is not None:
            fitnesses = batch_eval_fn(self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in self.population:
                ind.fitness.values = self.toolbox.evaluate(ind)
        self.hof.update(self.population)
        if len(self.hof) > 0:
            self.best_fitness = float(self.hof[0].fitness.values[0])

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        assert self.toolbox is not None
        self.generation += 1
        evaluator = evaluator_factory(self)
        self.rebind_evaluator(evaluator)

        for ind in self.population:
            del ind.fitness.values

        invalid = [ind for ind in self.population if not ind.fitness.valid]
        if batch_eval_fn is not None and invalid:
            fitnesses = batch_eval_fn(invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in self.population:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)

        self.toolbox.update(self.population)
        self.hof.update(self.population)
        fitnesses = [ind.fitness.values[0] for ind in self.population]
        best = float(max(fitnesses))
        self.best_fitness = max(self.best_fitness, best)

        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(np.mean(fitnesses)),
            std=float(np.std(fitnesses)),
            min=float(np.min(fitnesses)),
            max=best,
            best_fitness=best,
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        assert self._policy_factory is not None
        pop_pool: OpponentPool = PopulationPool(
            self.population, self._policy_factory,
            sampler=self.side.opponent_sampler,
        )
        if self.side.use_archive_in_opponent_pool and self.side.hof is not None:
            if len(self.side.hof) == 0:
                return pop_pool
            archive_pool = ArchivePool(
                self.side.hof, self._policy_factory,
                sampler=self.side.archive_sampler,
            )
            weights = (1.0 - self.side.archive_mix, self.side.archive_mix)
            return CompositePool([pop_pool, archive_pool], weights=weights)
        return pop_pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        assert len(self.hof) > 0
        assert self._policy_factory is not None
        return self._policy_factory(self.hof[0])

    def best_snapshot(self) -> Any:
        assert len(self.hof) > 0
        return self.hof[0].genome.clone()


# ---------------------------------------------------------------------------
# Frozen side
# ---------------------------------------------------------------------------


class FrozenSideState(SideState):
    """Non-evolving side — wraps a static :class:`FrozenPolicyPool`.

    Used for EA/CMA vs frozen opponents (pre-trained RL policies,
    scripted benchmarks, or historical-champion playback). Its
    :meth:`step_generation` is a no-op that only records constant-zero
    stats for consistency with evolving sides.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        policies = list(side.frozen_policies or [])
        if not policies:
            raise ValueError(
                f"FrozenSideState({side.name!r}): frozen_policies must be "
                "non-empty."
            )
        self._pool = FrozenPolicyPool(
            policies, sampler=side.opponent_sampler
        )
        self._policies = policies

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # Nothing to probe; the user provided pre-built policies.
        return None

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
    ) -> GenerationStats:
        self.generation += 1
        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=0.0, std=0.0, min=0.0, max=0.0, best_fitness=0.0,
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        return self._pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        # "Best" is ill-defined for a frozen pool; return the first.
        return self._policies[0]

    def best_snapshot(self) -> Any:
        return self._policies[0]


# Factory


def make_side_state(side: Side) -> SideState:
    """Construct the appropriate :class:`SideState` subclass for ``side``."""
    if side.optimizer == "ea":
        return EASideState(side)
    if side.optimizer == "cma":
        return CMAESSideState(side)
    if side.optimizer == "frozen":
        return FrozenSideState(side)
    if side.optimizer in {"qdax_map_elites", "qdax_cma_me"}:
        try:
            from deap_pettingzoo.contrib.qdax.coevolution import QDaxSideState
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"Side {side.name!r}: optimizer={side.optimizer!r} requires "
                "the QDax extra. Install with `pip install "
                "deap-pettingzoo[qdax]`."
            ) from exc
        return QDaxSideState(side)
    if side.optimizer == "rl":
        from .rl_side import RLSideState

        return RLSideState(side)
    if side.optimizer == "pso":
        return PSOSideState(side)
    if side.optimizer == "rl_hpo":
        from .rl_hpo_side import RLEvolveHPOSideState
        return RLEvolveHPOSideState(side)
    raise ValueError(  # pragma: no cover — guarded by Side.__post_init__
        f"Unknown optimizer {side.optimizer!r}"
    )


__all__ = [
    "Side",
    "SideState",
    "EASideState",
    "CMAESSideState",
    "FrozenSideState",
    "PSOSideState",
    "GenerationStats",
    "make_side_state",
]
