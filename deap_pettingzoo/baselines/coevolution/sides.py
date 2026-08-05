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

import importlib
import random as _random
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
        anchor_policies: Fixed reference opponents other sides face
            alongside this side's evolving population — typically scripted
            controllers. Purely about what *opponents* see, exactly like
            ``archive_mix``: this side still evolves normally.

            The point is measurement. In pure co-evolution an individual's
            fitness is scored against a moving population, so a rising
            fitness can mean the individual improved or that its opponents
            got worse, and selection cannot tell those apart. Reserving part
            of every evaluation for an opponent that never changes gives the
            fitness a fixed reference and makes it comparable across
            generations.
        anchor_mix: Fraction of each individual's ``k_opponents`` drawn
            from ``anchor_policies``, rounded to a whole number of episodes
            and never rounded down to zero when non-zero. Default ``0.0``
            (no anchoring). ``1.0`` anchors every episode, which is training
            against the fixed opponent inside the co-evolution harness —
            valid as an ablation, not co-evolution.
        anchor_registry_names: Names under which ``anchor_policies`` are
            registered via
            :func:`~deap_pettingzoo.baselines.coevolution.pools.register_frozen_policy`.
            Required when anchors cannot be pickled to worker processes,
            which is the normal case for scripted policies holding an env
            handle.
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
    anchor_policies: Optional[Sequence[PolicyProtocol]] = None
    anchor_mix: float = 0.0
    anchor_registry_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        # The valid optimizer tokens are exactly the keys of the open
        # ``_SIDE_STATE_FACTORIES`` registry, so a ``register_side_state`` (or
        # ``register_algorithm(..., side_state=...)`` call makes a new token a
        # valid ``Side.optimizer`` automatically.
        valid = set(_SIDE_STATE_FACTORIES)
        if self.optimizer not in valid:
            raise ValueError(
                f"Side {self.name!r}: optimizer must be one of "
                f"{sorted(valid)}; got {self.optimizer!r}."
            )
        if not self.role_prefixes:
            raise ValueError(f"Side {self.name!r}: role_prefixes must be non-empty.")
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
        if not (0.0 <= self.anchor_mix <= 1.0):
            raise ValueError(
                f"Side {self.name!r}: anchor_mix must be in [0, 1]; "
                f"got {self.anchor_mix}."
            )
        if self.anchor_mix > 0.0 and not self.anchor_policies:
            raise ValueError(
                f"Side {self.name!r}: anchor_mix={self.anchor_mix} requires a "
                "non-empty anchor_policies= list."
            )
        if self.anchor_registry_names is not None:
            names = tuple(self.anchor_registry_names)
            if len(names) != len(self.anchor_policies or ()):
                raise ValueError(
                    f"Side {self.name!r}: anchor_registry_names has "
                    f"{len(names)} entries but anchor_policies has "
                    f"{len(self.anchor_policies or ())}."
                )
            self.anchor_registry_names = names


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
    #: Real env interactions this side performed *during* this generation
    #: (per-generation, not cumulative). Populated by evolving sides from
    #: their evaluator's ``env_steps_consumed`` counter; left at 0 for sides
    #: that cannot report real ticks. The trainer sums these across sides and
    #: generations for the top-level ``env_steps_consumed`` metric.
    env_steps: int = 0
    #: Wall-clock seconds this ``step_generation`` call took.
    wall_time_s: float = 0.0


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

        #: Whether this side's policies attach a
        #: :class:`MaskedDiscreteHead` (turn-based envs). Normalised here so
        #: the trainer, the learner-policy factory and the opponent pools all
        #: read one source rather than re-parsing ``optimizer_config``.
        self.masked: bool = bool(side.optimizer_config.get("masked", False))
        #: Per-side deterministic RNG. Replaced in :meth:`initialise` with the
        #: trainer-provided generator (derived from
        #: :attr:`CoEvolutionConfig.seed`) so one side's stochastic operators
        #: never influence another side's evolution.
        self._rng: np.random.Generator = np.random.default_rng()

        self._np_state: Optional[Any] = None
        self._py_state: Optional[Any] = None
        # True for RL sides; read by trainer GPU-contention and worker-fallback detectors.
        self.is_rl_side: bool = False
        # Torch device string; overwritten in initialise() by RL subclasses.
        self.resolved_device: str = "cpu"

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

    # Learner-policy hooks — how this side's genomes become live policies

    def learner_policy_factory(self) -> Callable[[Any, str, Any], PolicyProtocol]:
        """Return the ``learner_factory`` the co-evolution evaluator calls.

        The returned callable maps ``(genome, agent_id, action_space)`` to a
        live :class:`PolicyProtocol`, one per learner agent per episode. The
        default builds a :class:`FeedforwardNeuralPolicy`, which is what
        EA / CMA-ES / PSO need; paradigms with a different phenotype (NEAT,
        HyperNEAT) override this so they can drive the *shared*
        :class:`~deap_pettingzoo.baselines.coevolution.evaluator.CoEvolutionEvaluator`
        rather than rolling a bespoke rollout loop.
        """
        masked = self.masked

        def _factory(genome: Any, agent_id: str, action_space: Any) -> PolicyProtocol:
            # Homogeneous side — every agent shares one action space, so the
            # agent id does not affect how the policy is built.
            del agent_id
            return _make_feedforward_policy_factory(action_space, masked=masked)(genome)

        return _factory

    def learner_snapshot(self, genome: Any) -> Optional[Any]:
        """Return a picklable descriptor of *genome* for worker processes.

        ``None`` (the default) means "ship the genome itself and let the
        worker build a :class:`FeedforwardNeuralPolicy`" — the historical
        behaviour of the parallel path. Sides whose phenotype needs more than
        a genome (NEAT's config, HyperNEAT's substrate) return an
        :class:`~deap_pettingzoo.baselines.coevolution.pools.OpponentSnapshot`
        that
        :func:`~deap_pettingzoo.baselines.coevolution.pools.rebuild_opponent_policy`
        understands.

        Deliberately separate from :meth:`learner_policy_factory`: collapsing
        the two would rebuild this (potentially expensive) descriptor once per
        agent per episode instead of once per evaluation.
        """
        del genome
        return None


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
        return FeedforwardNeuralPolicy(genome, action_space=action_space, masked=masked)

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
    side_agents = [a for a in env.possible_agents if a.startswith(role_prefixes)]
    if not side_agents:
        raise ValueError(f"No env agents match role_prefixes={role_prefixes!r}.")
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
        self.seed = cfg.get("seed", None)
        self._is_multiobjective = self.selection_method in {"nsga2", "spea2"}

        self.toolbox = None
        self.population: List[Any] = []
        if self._is_multiobjective:
            self.hof = tools.ParetoFront()
        else:
            self.hof = tools.HallOfFame(1)
        self._policy_factory: Optional[PolicyFactory] = None
        self._action_space: Any = None
        # _rng / _np_state inherited from SideState — the trainer seeds
        # _rng in initialise() and captures _np_state after it.

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
        _is_box = hasattr(self._action_space, "low") and hasattr(
            self._action_space, "high"
        )
        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "output_activation": "linear" if _is_box else self.activation,
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
        # RNG isolation: save/restore both Python random and NumPy global state so each side advances its own stream independently
        py_before: Optional[Any] = None
        np_before: Optional[Any] = None
        try:
            if self._py_state is not None:
                py_before = _random.getstate()
                _random.setstate(self._py_state)
            if self._np_state is not None:
                np_before = np.random.get_state()
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

            # (mu, lambda) selection: choose next generation from offspring only
            self.population = self.toolbox.select(offspring, self.population_size)
        finally:
            if py_before is not None:
                self._py_state = _random.getstate()
                _random.setstate(py_before)
            if np_before is not None:
                self._np_state = np.random.get_state()
                np.random.set_state(np_before)

        # Evaluate invalid individuals.
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if batch_eval_fn is not None and invalid:
            fitnesses = batch_eval_fn(invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
        else:
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

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
            # Real env ticks this side consumed this generation. The trainer
            # builds a fresh evaluator per side per generation, so the
            # evaluator's cumulative counter equals this generation's total.
            env_steps=int(getattr(evaluator, "env_steps_consumed", 0)),
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
                # Archive empty -> behave as if pure current-gen pool.
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

        _is_box = hasattr(self._action_space, "low") and hasattr(
            self._action_space, "high"
        )
        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "output_activation": "linear" if _is_box else self.activation,
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

        self.toolbox = register_cma_toolbox(
            self.strategy, self._ind_init, _NoopEvaluator()
        )
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

        # RNG isolation: save/restore both Python random and NumPy global state so each side advances its own stream independently
        py_before: Optional[Any] = None
        np_before: Optional[Any] = None
        try:
            if self._py_state is not None:
                py_before = _random.getstate()
                _random.setstate(self._py_state)
            if self._np_state is not None:
                np_before = np.random.get_state()
                np.random.set_state(self._np_state)

            self.toolbox.update(self.population)
            next_population = self.toolbox.generate()
        finally:
            if py_before is not None:
                self._py_state = _random.getstate()
                _random.setstate(py_before)
            if np_before is not None:
                self._np_state = np.random.get_state()
                np.random.set_state(np_before)

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
            # Real env ticks consumed this generation (fresh evaluator/gen).
            env_steps=int(getattr(evaluator, "env_steps_consumed", 0)),
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


# PSO side


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
        self.seed = cfg.get("seed", None)

        self.pso_state: Optional[Dict[str, Any]] = None
        self.toolbox = None
        self.population: List[Any] = []
        self.hof = tools.HallOfFame(1)
        self._policy_factory: Optional[PolicyFactory] = None
        self._action_space: Any = None
        self._ind_init = None

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

        if rng is not None:
            pso_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        elif self.seed is not None:
            pso_seed = int(self.seed)
        else:
            pso_seed = None

        _is_box = hasattr(self._action_space, "low") and hasattr(
            self._action_space, "high"
        )
        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "output_activation": "linear" if _is_box else self.activation,
            "weight_range": self.weight_range,
        }
        from deap_pettingzoo.integration.pso_integration import (
            make_pso_ind_init,
            register_pso_toolbox,
            setup_pso_swarm,
        )

        self.pso_state, _ = setup_pso_swarm(
            genome_config=genome_config,
            population_size=self.population_size,
            phi1=self.phi1,
            phi2=self.phi2,
            weight=self.weight,
            smin=self.smin,
            smax=self.smax,
            seed=pso_seed,
        )
        self._ind_init = make_pso_ind_init(genome_config, self.pso_state)

        class _NoopEvaluator:
            def evaluate(self, individual, **_kwargs):  # pragma: no cover
                return (0.0,)

        self.toolbox = register_pso_toolbox(
            self.pso_state,
            self._ind_init,
            _NoopEvaluator(),
        )
        self.population = self.toolbox.generate()
        self._policy_factory = _make_feedforward_policy_factory(
            self._action_space,
            masked=self.masked,
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

        self.hof.update(self.population)

        # RNG isolation: save/restore both Python random and NumPy global state so each side advances its own stream independently
        py_before: Optional[Any] = None
        np_before: Optional[Any] = None
        try:
            if self._py_state is not None:
                py_before = _random.getstate()
                _random.setstate(self._py_state)
            if self._np_state is not None:
                np_before = np.random.get_state()
                np.random.set_state(self._np_state)

            self.toolbox.update(self.population)
        finally:
            if py_before is not None:
                self._py_state = _random.getstate()
                _random.setstate(py_before)
            if np_before is not None:
                self._np_state = np.random.get_state()
                np.random.set_state(np_before)

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
            # Real env ticks consumed this generation (fresh evaluator/gen).
            env_steps=int(getattr(evaluator, "env_steps_consumed", 0)),
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


# Frozen side


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
                f"FrozenSideState({side.name!r}): frozen_policies must be " "non-empty."
            )
        self._pool = FrozenPolicyPool(policies, sampler=side.opponent_sampler)
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
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            best_fitness=0.0,
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


# Open side-state factory registry


def _load_qdax_side_state() -> type:
    """Loader for the QDax co-evolution side (needs the ``qdax`` extra)."""
    try:
        from deap_pettingzoo.contrib.qdax.coevolution import QDaxSideState
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Co-evolution optimizer 'qdax_map_elites'/'qdax_cma_me' requires "
            "the QDax extra. Install with `pip install deap-pettingzoo[qdax]`."
        ) from exc
    return QDaxSideState


def _load_rl_side_state() -> type:
    """Loader for the AgileRL co-evolution side."""
    from .rl_side import RLSideState

    return RLSideState


def _load_rl_hpo_side_state() -> type:
    """Loader for the AgileRL evolutionary-HPO co-evolution side."""
    from .rl_hpo_side import RLEvolveHPOSideState

    return RLEvolveHPOSideState


def _load_neat_side_state() -> type:
    """Loader for the NEAT co-evolution side (needs the ``neat`` extra)."""
    try:
        from deap_pettingzoo.contrib.neat.coevolution import NeatSideState
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Co-evolution optimizer 'neat' requires the neat extra. Install "
            "with `pip install deap-pettingzoo[neat]`."
        ) from exc
    return NeatSideState


def _load_hyperneat_side_state() -> type:
    """Loader for the HyperNEAT co-evolution side (needs the ``neat`` extra)."""
    try:
        from deap_pettingzoo.contrib.neat.hyperneat_coevolution import (
            HyperNeatSideState,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Co-evolution optimizer 'hyperneat' requires the neat extra. "
            "Install with `pip install deap-pettingzoo[neat]`."
        ) from exc
    return HyperNeatSideState


def _load_es_hyperneat_side_state() -> type:
    """Loader for the ES-HyperNEAT side (needs the ``neat`` extra)."""
    try:
        from deap_pettingzoo.contrib.neat.hyperneat_coevolution import (
            ESHyperNeatSideState,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Co-evolution optimizer 'es_hyperneat' requires the neat extra. "
            "Install with `pip install deap-pettingzoo[neat]`."
        ) from exc
    return ESHyperNeatSideState


#: Open registry mapping a ``Side.optimizer`` token to a zero-arg **loader**
#: returning the matching :class:`SideState` subclass. Each loader owns any
#: lazy import + the friendly "install the X extra" :class:`ImportError`, so
#: :func:`make_side_state` stays a one-line lookup and third parties can add a
#: side without editing this module (see :func:`register_side_state`). This acts as
#: the authoritative source for :meth:`Side.__post_init__` validation too.
_SIDE_STATE_FACTORIES: Dict[str, Callable[[], type]] = {
    "ea": lambda: EASideState,
    "cma": lambda: CMAESSideState,
    "frozen": lambda: FrozenSideState,
    "pso": lambda: PSOSideState,
    "rl": _load_rl_side_state,
    "rl_hpo": _load_rl_hpo_side_state,
    "qdax_map_elites": _load_qdax_side_state,
    "qdax_cma_me": _load_qdax_side_state,
    "neat": _load_neat_side_state,
    "hyperneat": _load_hyperneat_side_state,
    "es_hyperneat": _load_es_hyperneat_side_state,
}


def _resolve_side_loader(loader_or_class: Any) -> Callable[[], type]:
    """Normalise ``register_side_state`` input into a zero-arg class loader.

    Accepts a :class:`SideState` subclass (used directly), a dotted
    ``"module:QualName"`` string (imported lazily — worker-safe), or an
    existing zero-arg loader callable (returned as-is).
    """
    if isinstance(loader_or_class, type):
        cls = loader_or_class
        return lambda: cls
    if isinstance(loader_or_class, str):
        dotted = loader_or_class

        def _loader() -> type:
            return _import_dotted(dotted)

        return _loader
    if callable(loader_or_class):
        return loader_or_class
    raise TypeError(
        "register_side_state expects a SideState subclass, a dotted "
        f"'module:QualName' string, or a zero-arg loader; got "
        f"{loader_or_class!r}."
    )


def _import_dotted(dotted: str) -> type:
    """Import and return the object named by a dotted ``module:QualName`` path.

    Also allows the ``module.QualName`` form (splits on the last dot).
    """
    module_path, sep, qualname = dotted.partition(":")
    if not sep:
        module_path, _, qualname = dotted.rpartition(".")
    if not module_path or not qualname:
        raise ValueError(
            f"Invalid dotted side-state binding {dotted!r}; expected "
            "'module:QualName'."
        )
    obj: Any = importlib.import_module(module_path)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def register_side_state(
    token: str,
    loader_or_class: Any,
    *,
    overwrite: bool = False,
) -> None:
    """Register a :class:`SideState` factory for a co-evolution optimizer token.

    Once registered, ``token`` is a valid :attr:`Side.optimizer` (validated
    against this registry) and :func:`make_side_state` builds it.

    Args:
        token: The ``Side.optimizer`` string this side answers to.
        loader_or_class: A :class:`SideState` subclass, a dotted
            ``"module:QualName"`` string (imported lazily on first use — the
            worker-safe form), or a zero-arg loader returning the subclass
            (which may own a lazy import + friendly ``ImportError``).
        overwrite: If ``False`` (default), re-registering an existing token
            raises :class:`ValueError`; set ``True`` to replace.

    Raises:
        ValueError: If ``token`` is already registered and ``overwrite`` is
            ``False``.
        TypeError: If ``loader_or_class`` is not a class, dotted string, or
            callable.
    """
    if not overwrite and token in _SIDE_STATE_FACTORIES:
        raise ValueError(
            f"Co-evolution side optimizer {token!r} is already registered. "
            "Pass overwrite=True to replace."
        )
    _SIDE_STATE_FACTORIES[token] = _resolve_side_loader(loader_or_class)


def make_side_state(side: Side) -> SideState:
    """Construct the appropriate :class:`SideState` subclass for ``side``.

    Looks ``side.optimizer`` up in the open :data:`_SIDE_STATE_FACTORIES`
    registry, resolves the concrete class through its loader (triggering any
    lazy optional-extra import), and instantiates ``cls(side)``.
    """
    try:
        loader = _SIDE_STATE_FACTORIES[side.optimizer]
    except KeyError:  # pragma: no cover — guarded by Side.__post_init__
        raise ValueError(
            f"Unknown optimizer {side.optimizer!r}. Known: "
            f"{sorted(_SIDE_STATE_FACTORIES)}"
        ) from None
    cls = loader()
    return cls(side)


__all__ = [
    "Side",
    "SideState",
    "EASideState",
    "CMAESSideState",
    "FrozenSideState",
    "PSOSideState",
    "GenerationStats",
    "make_side_state",
    "register_side_state",
]
