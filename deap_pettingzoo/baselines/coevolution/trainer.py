"""Co-evolution trainer — orchestrates multi-side generational evolution.

:class:`CoEvolutionTrainer` is a :class:`BaselineTrainer` subclass that
handles three modes via one config-driven entry point:

* ``"competitive_2pop"`` — two (or more) sides whose rewards are
  typically zero-sum or strongly opposed (e.g. pursuers vs evaders).
* ``"self_play_hof"`` — league-style self-play: every evolving side
  owns its own Hall-of-Fame, and when sampled *as an opponent* by
  another side, draws exclusively from that archive. Requires at
  least two sides with disjoint ``role_prefixes`` (in
  asymmetric-topology envs this is the natural generalisation of
  classical self-play, since a single HoF cannot span mismatched
  policy topologies).
* ``"cooperative"`` — N sides rewarded jointly; each side "opposes" no
  one but the partition lets us run heterogeneous optimisers on
  different agent groups (e.g. EA on ``agent_*``, CMA on
  ``adversary_*``) with the other side's current-gen policies as
  stand-ins.

Scheduling:

* ``schedule="synchronous"`` — every generation, each side advances
  one step against a snapshot of the other sides' **current** state.
* ``schedule="alternating"`` — one side advances ``alternating_generations``
  at a time while the rest stay frozen, then we rotate.

Mixed EA/CMA-ES vs frozen RL is supported natively: make the RL side a
``Side(optimizer='frozen', frozen_policies=[...])`` and it will be
sampled as an opponent by the other sides. The RL agent(s) can be
wrapped via :class:`AgileRLPolicyWrapper` (one wrapper per agent) to
expose a :class:`PolicyProtocol`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper

from .evaluator import CoEvolutionEvaluator
from .sides import GenerationStats, Side, SideState, make_side_state


VALID_MODES = ("competitive_2pop", "self_play_hof", "cooperative")
VALID_SCHEDULES = ("synchronous", "alternating")


@dataclass
class CoEvolutionConfig:
    """Configuration for :class:`CoEvolutionTrainer`.

    Attributes:
        mode: High-level variant — one of :data:`VALID_MODES`.
        sides: Ordered list of :class:`Side` declarations. In
            ``self_play_hof`` mode this must have at least two sides
            with disjoint ``role_prefixes``; every evolving side must
            provide its own :class:`HallOfFame`, and the trainer
            forces each side's opponents (as seen from other sides)
            to be drawn purely from that archive
            (``use_archive_in_opponent_pool=True``, ``archive_mix=1.0``).
            In ``cooperative`` mode all sides share the same reward
            sign and are jointly optimised.
        schedule: ``"synchronous"`` or ``"alternating"``.
        alternating_generations: Number of inner generations per side
            per outer rotation in alternating schedules. Ignored when
            ``schedule == "synchronous"``.
        k_opponents: How many opponent-tuples to sample per individual
            evaluation. Default ``3``.
        aggregation: How to collapse the ``k`` per-sample rewards.
            ``"mean" | "min" | "max" | "sum"``. Default ``"mean"``.
        snapshot_every: Append each side's current best to its HoF
            every ``N`` generations (only when the side has ``hof``
            set). Default ``1``.
        reporting_side: Which side's best individual becomes the
            :attr:`TrainingResult.policy`. Defaults to the first side.
        max_steps_per_episode: Hard cap on env steps per rollout.
            Default ``200``.
        seed: Base seed for trainer RNG (controls opponent sampling
            and EA RNGs for reproducibility).
    """

    mode: str
    sides: List[Side]
    schedule: str = "synchronous"
    alternating_generations: int = 1
    k_opponents: int = 3
    aggregation: str = "mean"
    snapshot_every: int = 1
    reporting_side: Optional[str] = None
    max_steps_per_episode: int = 200
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"CoEvolutionConfig.mode={self.mode!r} invalid; "
                f"must be one of {VALID_MODES}."
            )
        if self.schedule not in VALID_SCHEDULES:
            raise ValueError(
                f"CoEvolutionConfig.schedule={self.schedule!r} invalid; "
                f"must be one of {VALID_SCHEDULES}."
            )
        if not self.sides:
            raise ValueError("CoEvolutionConfig.sides must be non-empty.")

        names = [s.name for s in self.sides]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate side names in {names!r}.")

        if self.mode == "self_play_hof":
            # "Self-play with HoF" = each side maintains its own
            # archive of past champions, and when that side is sampled
            # as an opponent by OTHER sides, the sample is drawn from
            # its HoF instead of its current generation. In
            # symmetric-topology envs this recovers classical
            # self-play; in asymmetric-topology envs (e.g. simple_tag,
            # where pursuers and evaders have different obs/action
            # spaces) it's the natural generalisation — you cannot
            # share a single HoF across sides with mismatched policy
            # topologies.
            if len(self.sides) < 2:
                raise ValueError(
                    "self_play_hof mode requires at least 2 sides with "
                    "disjoint role_prefixes. For a single population "
                    "playing itself, declare two sides over the same "
                    "optimizer/config — the env's agents are then "
                    "partitioned into a learner subset and an opponent "
                    "subset."
                )
            # Every evolving side must have an archive, and other
            # sides sample it exclusively.
            for s in self.sides:
                if s.optimizer == "frozen":
                    continue
                if s.hof is None:
                    raise ValueError(
                        f"self_play_hof mode: side {s.name!r} must have "
                        "hof set (RingHallOfFame or ElitistHallOfFame)."
                    )
                s.use_archive_in_opponent_pool = True
                s.archive_mix = 1.0  # opponents come *only* from HoF
        if self.mode == "competitive_2pop" and len(self.sides) < 2:
            raise ValueError(
                "competitive_2pop mode requires at least 2 sides; got "
                f"{len(self.sides)}."
            )
        if self.mode == "cooperative" and len(self.sides) < 2:
            raise ValueError(
                "cooperative mode requires at least 2 sides; got "
                f"{len(self.sides)}."
            )
        if self.reporting_side is None:
            self.reporting_side = self.sides[0].name
        elif self.reporting_side not in names:
            raise ValueError(
                f"reporting_side={self.reporting_side!r} not in {names!r}."
            )
        if self.k_opponents < 1:
            raise ValueError(f"k_opponents must be >= 1, got {self.k_opponents}.")
        if self.alternating_generations < 1:
            raise ValueError(
                f"alternating_generations must be >= 1, got {self.alternating_generations}."
            )
        if self.snapshot_every < 1:
            raise ValueError(
                f"snapshot_every must be >= 1, got {self.snapshot_every}."
            )


class CoEvolutionTrainer(BaselineTrainer):
    """Unified co-evolution trainer.

    Consumes a :class:`CoEvolutionConfig` passed in
    ``TrainingConfig.algorithm_params['coevolution']`` and emits a
    :class:`TrainingResult` whose ``policy`` is the best individual of
    the designated :attr:`CoEvolutionConfig.reporting_side`.

    The trainer is registered in :mod:`comparison` as ``"coevolution"``
    so it plugs into :func:`compare_baselines` — but note that a
    meaningful co-evolution head-to-head against plain :class:`EATrainer`
    requires matching env / budget configuration by hand.
    """

    def train(self, config: TrainingConfig) -> TrainingResult:
        coev_cfg = config.algorithm_params.get("coevolution")
        if not isinstance(coev_cfg, CoEvolutionConfig):
            raise ValueError(
                "CoEvolutionTrainer requires "
                "config.algorithm_params['coevolution'] to be a "
                "CoEvolutionConfig instance."
            )

        start = time.time()
        num_generations = self._translate_budget(config, coev_cfg)

        env = PettingZooWrapper(
            config.env_name, env_config=config.env_config or None
        )

        # Seed RNGs *before* any side-state initialisation so that
        # side.initialise()'s randomness (CMA's np.random.seed, weight
        # sampling, etc.) is reproducible under coev_cfg.seed and does
        # not depend on side-iteration order.
        if coev_cfg.seed is not None:
            np.random.seed(int(coev_cfg.seed))
        self._rng = np.random.default_rng(coev_cfg.seed)

        states: Dict[str, SideState] = {
            s.name: make_side_state(s) for s in coev_cfg.sides
        }
        # Give every side its own deterministic child RNG, and isolate
        # initialise() from the global NumPy RNG so one side's seeding
        # calls (notably CMA) do not leak into the next side.
        for state in states.values():
            side_rng = np.random.default_rng(
                self._rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            previous_np_state = np.random.get_state()
            try:
                state.initialise(env, rng=side_rng)
            finally:
                np.random.set_state(previous_np_state)

        # Bootstrap: evaluate each evolving side's initial population
        # against the *initial* state of the other sides so fitnesses
        # exist before the first step_generation.
        self._bootstrap_populations(env, states, coev_cfg)

        # Generation loop.
        per_side_history: Dict[str, List[GenerationStats]] = {
            name: [] for name in states
        }
        if coev_cfg.schedule == "synchronous":
            for _g in range(num_generations):
                self._run_synchronous_generation(
                    env, states, coev_cfg, per_side_history
                )
        else:  # alternating
            total = 0
            rotation = [n for n in states]
            while total < num_generations:
                for name in rotation:
                    for _inner in range(coev_cfg.alternating_generations):
                        if total >= num_generations:
                            break
                        self._run_alternating_step(
                            env, states, coev_cfg, per_side_history, name
                        )
                        total += 1

        # Build reporting-side policy.
        reporting = states[coev_cfg.reporting_side]
        policy = reporting.best_policy(env)

        env.close()
        training_time = time.time() - start

        metrics = {
            "per_side_history": {
                name: [self._stats_to_dict(s) for s in hist]
                for name, hist in per_side_history.items()
            },
            "final_best": {
                name: (
                    None
                    if state.side.optimizer == "frozen"
                    else float(state.best_fitness)
                )
                for name, state in states.items()
            },
            "hof_sizes": {
                name: (len(state.side.hof) if state.side.hof is not None else 0)
                for name, state in states.items()
            },
            "num_generations": num_generations,
        }
        metadata = {
            "mode": coev_cfg.mode,
            "schedule": coev_cfg.schedule,
            "k_opponents": coev_cfg.k_opponents,
            "aggregation": coev_cfg.aggregation,
            "reporting_side": coev_cfg.reporting_side,
            "sides": [
                {
                    "name": s.name,
                    "role_prefixes": list(s.role_prefixes),
                    "optimizer": s.optimizer,
                    "hof": (
                        type(s.hof).__name__ if s.hof is not None else None
                    ),
                }
                for s in coev_cfg.sides
            ],
        }

        return TrainingResult(
            algorithm="coevolution",
            policy=policy,
            metrics=metrics,
            metadata=metadata,
            seed=coev_cfg.seed,
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_to_dict(s: GenerationStats) -> Dict[str, Any]:
        return {
            "generation": s.generation,
            "side": s.side_name,
            "mean": s.mean,
            "std": s.std,
            "min": s.min,
            "max": s.max,
            "best_fitness": s.best_fitness,
        }

    def _translate_budget(
        self, config: TrainingConfig, coev_cfg: CoEvolutionConfig
    ) -> int:
        """Translate a global budget into a generation count.

        Episodes per generation ≈
            Σ over evolving sides of (side.population_size * k_opponents).
        Env steps per generation ≈ episodes × max_steps_per_episode.
        """
        evolving = [s for s in coev_cfg.sides if s.optimizer != "frozen"]
        if not evolving:
            # Pathological — nothing is actually evolving — do at least
            # one pass to produce output.
            return 1
        ep_per_gen = sum(
            int(s.optimizer_config.get("population_size", 20)) * coev_cfg.k_opponents
            for s in evolving
        )
        ep_per_gen = max(1, ep_per_gen)
        if config.budget_mode == BudgetMode.ENV_STEPS:
            steps_per_gen = ep_per_gen * coev_cfg.max_steps_per_episode
            return max(1, config.budget_value // max(1, steps_per_gen))
        if config.budget_mode == BudgetMode.EVAL_EPISODES:
            return max(1, config.budget_value // ep_per_gen)
        raise ValueError(
            f"CoEvolutionTrainer does not support budget mode {config.budget_mode}."
        )

    def _collect_opponent_sides(
        self,
        states: Dict[str, SideState],
        learner_name: str,
        coev_cfg: CoEvolutionConfig,
        pool_snapshot: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Tuple[str, ...], Any]]:
        """Build the ``opponent_sides`` list for a given learner.

        All modes (``competitive_2pop``, ``self_play_hof``,
        ``cooperative``) share the same underlying pattern: opponents
        come from *other* sides. The difference is how those sides'
        pools are composed (self_play_hof forces
        ``use_archive_in_opponent_pool=True`` on every side in
        :meth:`CoEvolutionConfig.__post_init__`).

        When ``pool_snapshot`` is provided (non-empty dict), opponent
        pools are read from it instead of calling
        :meth:`SideState.current_pool` live. This preserves the
        "synchronous" schedule's contract: every side in a generation
        evaluates against opponents drawn from pools as they were at
        the *start* of the generation, not from pools that earlier
        sides in the loop have already advanced.
        """
        opp: List[Tuple[str, Tuple[str, ...], Any]] = []
        for name, state in states.items():
            if name == learner_name:
                continue
            if pool_snapshot is not None and name in pool_snapshot:
                pool = pool_snapshot[name]
            else:
                pool = state.current_pool()
            opp.append((name, state.side.role_prefixes, pool))
        return opp

    def _bootstrap_populations(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
    ) -> None:
        """Initial fitness evaluation for every evolving side."""
        for name, state in states.items():
            if state.side.optimizer == "frozen":
                continue
            evaluator = self._build_evaluator(env, state, states, coev_cfg)
            # Set toolbox.evaluate then evaluate the population.
            state.rebind_evaluator(evaluator)  # type: ignore[attr-defined]
            state.evaluate_population()  # type: ignore[attr-defined]
            # Snapshot initial best into HoF if configured.
            if state.side.hof is not None and coev_cfg.snapshot_every >= 1:
                state.snapshot_best()

    def _build_evaluator(
        self,
        env: PettingZooWrapper,
        learner_state: SideState,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        pool_snapshot: Optional[Dict[str, Any]] = None,
    ) -> CoEvolutionEvaluator:
        opp_sides = self._collect_opponent_sides(
            states, learner_state.side.name, coev_cfg,
            pool_snapshot=pool_snapshot,
        )
        # Fallback for self-play bootstrap: if learner's own pool would
        # be drawn from an empty archive, use pure current-gen pool to
        # avoid ArchivePool empty errors on the very first generation.
        assert (
            learner_state.side.optimizer != "frozen"
        ), "frozen sides never become learners."
        # Per-side RNG derived from trainer RNG (still deterministic
        # under a fixed base seed; differs per individual-episode).
        evaluator = CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=learner_state.side.role_prefixes,
            learner_factory=_build_learner_factory(learner_state, coev_cfg),
            opponent_sides=opp_sides,
            k_opponents=coev_cfg.k_opponents,
            aggregation=coev_cfg.aggregation,
            max_steps=coev_cfg.max_steps_per_episode,
        )
        # Inject a deterministic RNG so evaluations under a fixed seed
        # replay identically.  Wrap a closure so the evaluator always
        # receives a freshly-forked generator per call.
        trainer_rng = self._rng

        original_evaluate = evaluator.evaluate

        def _evaluate_with_trainer_rng(individual, seed=None, rng=None):
            if rng is None:
                # Fork a child Generator so concurrent evaluations (if
                # ever added) don't share state.
                rng = np.random.default_rng(trainer_rng.integers(0, 2**63 - 1))
            return original_evaluate(individual, seed=seed, rng=rng)

        evaluator.evaluate = _evaluate_with_trainer_rng  # type: ignore[assignment]
        return evaluator

    def _run_synchronous_generation(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        history: Dict[str, List[GenerationStats]],
    ) -> None:
        # Snapshot every side's opponent pool at the *start* of the
        # generation. The synchronous schedule's contract is that all
        # evolving sides advance one generation simultaneously against
        # a frozen view of every other side's state. Reading
        # ``state.current_pool()`` live inside the per-side loop would
        # break that contract — earlier sides would have already
        # mutated their populations/HoFs by the time later sides are
        # evaluated.
        pool_snapshot: Dict[str, Any] = {
            name: state.current_pool() for name, state in states.items()
        }
        for name, state in states.items():
            if state.side.optimizer == "frozen":
                continue
            evaluator = self._build_evaluator(
                env, state, states, coev_cfg, pool_snapshot=pool_snapshot
            )
            stats = state.step_generation(lambda _s, _ev=evaluator: _ev)
            history[name].append(stats)
            if (
                state.side.hof is not None
                and stats.generation % coev_cfg.snapshot_every == 0
            ):
                state.snapshot_best()

    def _run_alternating_step(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        history: Dict[str, List[GenerationStats]],
        name: str,
    ) -> None:
        state = states[name]
        if state.side.optimizer == "frozen":
            return
        evaluator = self._build_evaluator(env, state, states, coev_cfg)
        stats = state.step_generation(lambda _s, _ev=evaluator: _ev)
        history[name].append(stats)
        if (
            state.side.hof is not None
            and stats.generation % coev_cfg.snapshot_every == 0
        ):
            state.snapshot_best()


def _build_learner_factory(
    learner_state: SideState, coev_cfg: CoEvolutionConfig
):
    """Build the ``learner_factory`` the evaluator will call.

    Returns ``(genome, agent_id, action_space) -> PolicyProtocol`` that
    delegates to the side's own policy factory.
    """
    from .sides import _make_feedforward_policy_factory  # local import to avoid cycle

    masked = bool(learner_state.side.optimizer_config.get("masked", False))

    def _factory(genome: Any, agent_id: str, action_space: Any):
        return _make_feedforward_policy_factory(action_space, masked=masked)(genome)

    return _factory


__all__ = [
    "CoEvolutionConfig",
    "CoEvolutionTrainer",
    "VALID_MODES",
    "VALID_SCHEDULES",
]
