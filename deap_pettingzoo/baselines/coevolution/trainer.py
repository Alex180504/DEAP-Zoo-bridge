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

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    ProgressCallback,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.contrib.qdax.progress import format_qdax_progress_fields
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.environments.registry import (
    get_registry_snapshot,
    replay_registry_snapshot,
)
from deap_pettingzoo.utils.parallel import (
    SUPPORTED_BACKENDS,
    make_executor,
    resolve_num_workers,
)

from .evaluator import CoEvolutionEvaluator
from .rl_side import RLSideState, _RLDriverContext
from .sides import GenerationStats, Side, SideState, make_side_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker initializer
# ---------------------------------------------------------------------------

def _prewarm_and_replay(snapshot) -> None:  # type: ignore[type-arg]
    """ProcessPoolExecutor worker initializer.

    1. Pre-warm the full ``agilerl.networks`` / ``torch._inductor`` import
       chain so that later ``pickle.loads`` calls inside
       ``FrozenRLPolicyWrapper.__setstate__`` find every module already in
       ``sys.modules``.  (On Windows spawn workers these modules are never
       pre-loaded; a race in ``_apply_agilerl_grouping_patch`` can leave
       ``torch._inductor`` partially initialised, causing the actor
       deserialisation to crash.)
    2. Call the real :func:`~deap_pettingzoo.environments.registry.replay_registry_snapshot`
       to register environment factories in this worker process.
    """
    from deap_pettingzoo.baselines.agilerl_policy import _ensure_agilerl_imports

    _ensure_agilerl_imports()

    from deap_pettingzoo.environments.registry import replay_registry_snapshot

    replay_registry_snapshot(snapshot)


def _format_progress_entry(stats: GenerationStats) -> str:
    """Format one side's progress entry for the console line."""
    if stats.qd_score is None or stats.coverage is None:
        return (
            f"{stats.side_name}: best={stats.best_fitness:.3f} "
            f"mean={stats.mean:.3f}"
        )

    archive_best = (
        stats.archive_best_fitness
        if stats.archive_best_fitness is not None
        else stats.best_fitness
    )
    summary = format_qdax_progress_fields(
        batch_best=stats.best_fitness,
        batch_mean=stats.mean,
        archive_best=archive_best,
        qd_score=stats.qd_score,
        coverage=stats.coverage,
        filled_niches=stats.filled_niches,
        total_niches=stats.total_niches,
    )
    return (
        f"{stats.side_name}: {summary}"
    )


# QDax-specific evaluator

class _QDaxPolicyEvaluator:
    """Thin evaluator that accepts a pre-constructed policy (FlaxPolicyAdapter)
    instead of a DEAP individual with a genome.

    QDaxSideState converts its JAX pytree genotype into a policy *before*
    calling ``evaluator.evaluate(policy)``, so the generic
    ``CoEvolutionEvaluator`` (which requires ``individual.genome``) cannot
    be used directly.  This wrapper runs one episode with the given policy
    against sampled opponents and returns a scalar fitness.
    """

    def __init__(self, env_wrapper, learner_role_prefixes, opponent_sides,
                 k_opponents, aggregation, max_steps, rng_factory,
                 executor=None, env_name=None, env_config=None):
        self._env = env_wrapper
        self._learner_prefixes = learner_role_prefixes
        self._opponent_sides = opponent_sides
        self._k = k_opponents
        self._aggregation = aggregation
        self._max_steps = max_steps
        self._rng_factory = rng_factory  # callable () -> Generator
        self._executor = executor        # ProcessPoolExecutor or None
        self._env_name = env_name
        self._env_config = env_config
        # Optional hook called at each episode step with (obs_dict, env_wrapper).
        # Set by QDaxSideState._evaluate_one_genotype to collect trajectory data.
        self.trajectory_hook = None  # type: ignore[assignment]

        self._learner_agents = [
            a for a in env_wrapper.possible_agents
            if any(a.startswith(p) for p in learner_role_prefixes)
        ]
        self._side_agents: Dict[str, list] = {}
        for side_name, prefixes, _pool in opponent_sides:
            self._side_agents[side_name] = [
                a for a in env_wrapper.possible_agents
                if any(a.startswith(p) for p in prefixes)
            ]

    def evaluate(self, policy) -> float:
        """Run k episodes with *policy* for all learner agents; return fitness."""
        rng = self._rng_factory()
        rewards = []
        for i in range(self._k):
            opp_policies: Dict[str, Any] = {}
            for side_name, _prefixes, pool in self._opponent_sides:
                side_policy = pool.sample_opponent(rng)
                for agent in self._side_agents[side_name]:
                    opp_policies[agent] = side_policy

            learner_total = self._run_episode(policy, opp_policies, seed=i)
            rewards.append(learner_total)

        arr = np.asarray(rewards, dtype=np.float64)
        if self._aggregation == "mean":
            return float(arr.mean())
        if self._aggregation == "min":
            return float(arr.min())
        if self._aggregation == "max":
            return float(arr.max())
        return float(arr.mean())

    def _run_episode(self, policy, opp_policies, seed=None) -> float:
        from deap_pettingzoo.baselines.coevolution.evaluator import _invoke_get_action

        observations = self._env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        for p in opp_policies.values():
            if hasattr(p, "reset"):
                p.reset()

        total = 0.0
        steps = 0
        while steps < self._max_steps and observations:
            if self.trajectory_hook is not None:
                try:
                    self.trajectory_hook(observations, self._env)
                except Exception:
                    logger.debug("trajectory_hook failed", exc_info=True)
            infos = getattr(self._env, "last_infos", {}) or {}
            actions: Dict[str, Any] = {}
            for agent, obs in observations.items():
                mask = infos.get(agent, {}).get("action_mask")
                if agent in self._learner_agents:
                    actions[agent] = _invoke_get_action(policy, obs, mask, deterministic=True)
                elif agent in opp_policies:
                    actions[agent] = _invoke_get_action(opp_policies[agent], obs, mask)
            observations, rew, terms, truncs, _ = self._env.step(actions)
            for agent in self._learner_agents:
                total += float(rew.get(agent, 0.0))
            steps += 1
            if terms and all(terms.values()):
                break
            if truncs and all(truncs.values()):
                break
        return total
from deap_pettingzoo.utils.device_check import (
    torch_uses_cuda,
    warn_on_gpu_contention,
)


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
    num_workers: int = 1
    parallel_backend: str = "process"
    generation_callback: Optional[Callable[[int, Dict[str, Any]], None]] = field(
        default=None, repr=False, compare=False
    )
    """Optional callable invoked after each generation with ``(gen, states)``.

    ``gen`` is the 1-based generation number.  ``states`` is a
    ``{side_name: SideState}`` dict — callers can read
    ``state.best_fitness`` or call ``state.best_policy(None)`` to obtain
    the current champion for each side.  Exceptions raised by the
    callback are silently swallowed (a ``RuntimeWarning`` is emitted)
    so they cannot interrupt training.
    """

    def __post_init__(self) -> None:
        """Validate config fields and enforce mode-specific invariants.

        Note:
            ``self_play_hof`` semantics: each side maintains its own archive
            of past champions; when sampled *as an opponent* by other sides
            the draw comes from its HoF rather than the current generation.
            In symmetric-topology envs this recovers classical self-play; in
            asymmetric-topology envs (e.g. ``simple_tag`` where pursuers and
            evaders have different obs/action spaces) it is the natural
            generalisation — a single HoF cannot span sides with mismatched
            policy topologies.
        """
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

        for i, side in enumerate(self.sides):
            for other_side in self.sides[i + 1 :]:
                for prefix in side.role_prefixes:
                    for other_prefix in other_side.role_prefixes:
                        if prefix.startswith(other_prefix) or other_prefix.startswith(
                            prefix
                        ):
                            raise ValueError(
                                "Overlapping role_prefixes across sides are not "
                                "allowed: "
                                f"{side.name!r} prefix {prefix!r} conflicts with "
                                f"{other_side.name!r} prefix {other_prefix!r}. "
                                "Prefixes must be disjoint so agent assignment "
                                "based on startswith is unambiguous."
                            )

        if self.mode == "self_play_hof":
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
        if self.parallel_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"parallel_backend={self.parallel_backend!r} invalid; "
                f"must be one of {SUPPORTED_BACKENDS}."
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

    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Execute the co-evolution training loop.

        Note:
            Seeding order: ``np.random.seed`` and the per-trainer
            ``default_rng`` are set *before* any side's ``initialise()``
            so results are reproducible regardless of side-iteration order.
            Each side's global NumPy state is saved to ``state._np_state``
            after ``initialise()`` (before restoring the previous state) so
            ``step_generation`` resumes from the same deterministic stream.

            GPU/RL contention: if any RL side resolved CUDA *and* JAX has
            GPU access a ``RuntimeWarning`` is emitted. Parallel
            co-evolution (``num_workers > 1``) is not supported alongside
            RL sides (torch state does not pickle across spawn) and falls
            back to serial automatically.

            Bootstrap: the initial population of every evolving side is
            evaluated against the other sides' initial state so fitnesses
            exist before the first ``step_generation`` call. Evaluator
            construction happens inside the ``make_executor`` context so
            EA/CMA sides share the worker pool for ``k_opponents`` episode
            dispatch.

            Frame recording is best-effort and handled by the unified
            :class:`~deap_pettingzoo.baselines.recording.RecordingController`
            (heterogeneous per-agent rollout); any error is swallowed so it
            cannot interrupt training.
        """
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

        # Stash env-construction metadata so _build_evaluator can pass
        # it down to CoEvolutionEvaluator workers.
        self._env_name: str = config.env_name
        self._env_config: Optional[Dict[str, Any]] = config.env_config

        # Resolve the executor up front so we can size pools sanely
        # and emit one warning if num_workers > 1 with k_opponents == 1.
        resolved_workers = resolve_num_workers(coev_cfg.num_workers)
        if resolved_workers > 1 and coev_cfg.k_opponents == 1:
            warnings.warn(
                "CoEvolutionConfig: num_workers="
                f"{coev_cfg.num_workers} but k_opponents=1; "
                "parallelism is a no-op and the serial path will be used.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._resolved_workers = resolved_workers

        with make_executor(
            resolved_workers,
            backend=coev_cfg.parallel_backend,
            initializer=_prewarm_and_replay,
            initargs=(get_registry_snapshot(),),
        ) as executor:
            self._executor = executor
            # Keep side initialisation inside the executor scope. This
            # preserves the last known-good ordering for QDax-backed
            # coevolution on Windows.
            states = self._initialize_sides(env, coev_cfg, resolved_workers)
            print(
                f"[CoEv] bootstrapping {len(states)} sides "
                f"({sum(1 for s in states.values() if s.side.optimizer != 'frozen')} active)",
                flush=True,
            )
            _bootstrap_t0 = time.time()
            self._bootstrap_populations(env, states, coev_cfg)
            print(
                f"[CoEv] bootstrap done in {time.time() - _bootstrap_t0:.1f}s",
                flush=True,
            )

            per_side_history: Dict[str, List[GenerationStats]] = {
                name: [] for name in states
            }
            self._setup_recording(config, coev_cfg)
            # Env used by the recording controller's champion provider to
            # build per-side policies and enumerate agents (read-only use).
            self._rec_env = env

            # Per-generation progress reporting (logger-independent).
            self._progress_callback = progress_callback
            self._progress_total = num_generations
            self._progress_start = start

            if coev_cfg.schedule == "synchronous":
                self._run_synchronous_loop(
                    env, states, coev_cfg, config,
                    num_generations, per_side_history,
                )
            else:  # alternating
                self._run_alternating_loop(
                    env, states, coev_cfg, config,
                    num_generations, per_side_history,
                )

            # Guarantee a final capture (dedups against the last per-gen one).
            self._recorder.finalize(
                lambda: self._coev_policy_map(states),
                iteration=num_generations,
                env_steps=num_generations * self._recording_max_steps,
            )

            # Build reporting-side policy *inside* the executor scope
            # in case best_policy() ever needs to evaluate (it does
            # not today, but cheap insurance).
            reporting = states[coev_cfg.reporting_side]
            policy = reporting.best_policy(env)

        env.close()
        self._recorder.close()
        self._rec_env = None
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
            "coevolution_num_workers": resolved_workers,
            "coevolution_parallel_backend": coev_cfg.parallel_backend,
            "run_dir": (
                str(self._recording_log.run_dir)
                if config.output_dir is not None
                and getattr(self, "_recording_log", None) is not None
                else None
            ),
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

        self._persist_results(
            config,
            coev_cfg,
            per_side_history,
            states,
            training_time,
            num_generations,
        )

        return TrainingResult(
            algorithm="coevolution",
            policy=policy,
            metrics=metrics,
            metadata=metadata,
            seed=coev_cfg.seed,
            training_time=training_time,
        )

    # Setup helpers

    def _initialize_sides(
        self,
        env: PettingZooWrapper,
        coev_cfg: CoEvolutionConfig,
        resolved_workers: int,
    ) -> Dict[str, SideState]:
        """Seed RNGs and construct/initialise every side's state.

        The global NumPy RNG and the per-trainer ``default_rng`` are
        seeded *before* any side's ``initialise()`` so results are
        reproducible regardless of side-iteration order. Each side's
        resulting global NumPy state is captured into ``state._np_state``
        (and the previous global state restored afterwards) so
        ``step_generation`` resumes from the same deterministic stream.
        Emits GPU-contention and RL-parallelism warnings.
        """
        if coev_cfg.seed is not None:
            np.random.seed(int(coev_cfg.seed))
        self._rng = np.random.default_rng(coev_cfg.seed)

        states: Dict[str, SideState] = {
            s.name: make_side_state(s) for s in coev_cfg.sides
        }
        for state in states.values():
            side_rng = np.random.default_rng(
                self._rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            previous_np_state = np.random.get_state()
            try:
                state.initialise(env, rng=side_rng)
            except Exception:
                np.random.set_state(previous_np_state)
                raise
            state._np_state = np.random.get_state()
            np.random.set_state(previous_np_state)

        rl_cuda_sides = [
            name for name, s in states.items()
            if isinstance(s, RLSideState)
            and torch_uses_cuda(s.resolved_device)
        ]
        qdax_sides = [
            name for name, s in states.items()
            if s.side.optimizer in {"qdax_map_elites", "qdax_cma_me"}
        ]
        warn_on_gpu_contention(rl_cuda_sides, qdax_sides)

        has_rl_side = any(isinstance(s, RLSideState) for s in states.values())
        if has_rl_side and resolved_workers > 1:
            has_onpolicy_rl = any(
                isinstance(s, RLSideState) and getattr(s, "is_on_policy", False)
                for s in states.values()
            )
            rl_detail = (
                "On-policy RL learners (IPPO/GroupedIPPO) collect their own "
                "rollouts serially — the live actor+critic must produce the "
                "log_prob/value the actor-only snapshot cannot."
                if has_onpolicy_rl
                else "Off-policy RL rollout collection also parallelizes."
            )
            warnings.warn(
                "CoEvolutionConfig: num_workers="
                f"{coev_cfg.num_workers} but an RL side is present. "
                "EA/CMA learner sides evaluate against RL opponents in "
                "parallel via FrozenRLPolicyWrapper CPU snapshots. "
                + rl_detail,
                RuntimeWarning,
                stacklevel=2,
            )
        return states

    def _setup_recording(
        self, config: TrainingConfig, coev_cfg: CoEvolutionConfig
    ) -> None:
        """Build the run logger and the unified recording controller.

        Stores ``self._recording_log`` (also reused for metrics/summary
        persistence) and ``self._recorder``. Co-evolution is heterogeneous,
        so recordings are produced via the controller's per-agent
        ``policy_map`` path (see :meth:`_coev_policy_map`).
        """
        from deap_pettingzoo.baselines._logging_helpers import build_logger
        from deap_pettingzoo.baselines.recording import RecordingController

        self._recording_log = build_logger(config, algorithm="coevolution")
        # Use the env's natural episode length, not max_steps_per_episode from
        # algorithm_params — that's a training knob, not the rollout duration.
        _env_cfg = config.env_config or {}
        self._recording_max_steps: int = int(
            config.recording.max_frames_per_episode
            or _env_cfg.get("max_cycles")
            or _env_cfg.get("max_time_steps")
            or 500
        )
        self._recorder = RecordingController(
            config, self._recording_log,
            max_steps=self._recording_max_steps,
            seed=coev_cfg.seed,
        )

    def _coev_policy_map(
        self, states: Dict[str, SideState]
    ) -> Optional[Dict[str, Any]]:
        """Build the per-agent ``{agent_id: policy}`` map for a recording.

        Each side's current best policy is mapped to the env agents whose
        names start with the side's ``role_prefixes``. Returns ``None`` when
        no policy can be built yet (e.g. before the first evaluation).
        """
        env = getattr(self, "_rec_env", None)
        if env is None:
            return None
        mapping: Dict[str, Any] = {}
        for st in states.values():
            try:
                pol = st.best_policy(env)
            except Exception:
                continue
            for agent in env.possible_agents:
                if agent.startswith(st.side.role_prefixes):
                    mapping[agent] = pol
        return mapping or None

    # Generation-loop helpers

    def _invoke_generation_callback(
        self,
        coev_cfg: CoEvolutionConfig,
        gen: int,
        states: Dict[str, SideState],
    ) -> None:
        """Call the user generation callback and emit a progress event.

        Both are error-isolated so neither can interrupt training. The
        progress event is logger-independent (built from in-loop state) and
        is wired into **both** the synchronous and alternating schedules.
        """
        if coev_cfg.generation_callback is not None:
            try:
                coev_cfg.generation_callback(gen, states)
            except Exception as _cb_exc:
                warnings.warn(
                    f"[CoEvolution] generation_callback failed at gen {gen}: "
                    f"{type(_cb_exc).__name__}: {_cb_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        best: Optional[float] = None
        side_best: Dict[str, float] = {}
        for nm, st in states.items():
            if st.side.optimizer == "frozen":
                continue
            try:
                bf = float(st.best_fitness)
            except (TypeError, ValueError):
                continue
            side_best[nm] = bf
            best = bf if best is None else max(best, bf)
        start = getattr(self, "_progress_start", None)
        max_steps = int(getattr(self, "_recording_max_steps", 0) or 0)
        # Single hook: emit progress and let the recording controller decide
        # (per RecordingMode) whether to capture a heterogeneous rollout built
        # from each side's current best policy.
        self._on_progress(
            getattr(self, "_progress_callback", None),
            ProgressEvent(
                unit="generation",
                current=gen,
                total=int(getattr(self, "_progress_total", 0)),
                env_steps=gen * max_steps,
                best_fitness=best,
                mean_fitness=None,
                wall_time_s=(time.time() - start) if start is not None else 0.0,
                extra={"side_best": side_best},
            ),
            champion=lambda: self._coev_policy_map(states),
        )

    def _run_synchronous_loop(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        config: TrainingConfig,
        num_generations: int,
        history: Dict[str, List[GenerationStats]],
    ) -> None:
        """Drive the synchronous schedule: every side advances each gen."""
        for _g in range(num_generations):
            _gt = time.time()
            self._run_synchronous_generation(env, states, coev_cfg, history)
            # Progress + champion recording happen together in the callback.
            self._invoke_generation_callback(coev_cfg, _g + 1, states)
            _side_info = "  ".join(
                _format_progress_entry(h[-1])
                for n, h in history.items()
                if h
            )
            print(
                f"[CoEv] gen {_g + 1}/{num_generations}  "
                f"{_side_info}  "
                f"({time.time()-_gt:.1f}s)",
                flush=True,
            )

    def _run_alternating_loop(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        config: TrainingConfig,
        num_generations: int,
        history: Dict[str, List[GenerationStats]],
    ) -> None:
        """Drive the alternating schedule: one side advances at a time."""
        total = 0
        rotation = list(states)
        while total < num_generations:
            for name in rotation:
                for _inner in range(coev_cfg.alternating_generations):
                    if total >= num_generations:
                        break
                    _gt = time.time()
                    self._run_alternating_step(
                        env, states, coev_cfg, history, name
                    )
                    total += 1
                    # Progress + champion recording happen in the callback.
                    self._invoke_generation_callback(coev_cfg, total, states)
                    _side_info = "  ".join(
                        _format_progress_entry(h[-1])
                        for n, h in history.items()
                        if h
                    )
                    print(
                        f"[CoEv] step {total}/{num_generations} ({name})  "
                        f"{_side_info}  "
                        f"({time.time()-_gt:.1f}s)",
                        flush=True,
                    )

    # Logging helpers

    def _persist_results(
        self,
        config: TrainingConfig,
        coev_cfg: "CoEvolutionConfig",
        per_side_history: Dict[str, List[GenerationStats]],
        states: Dict[str, Any],
        training_time: float,
        num_generations: int,
    ) -> None:
        """Write coevolution metrics + summary via :class:`ExperimentLogger`.

        Each side's per-generation stats land in ``metrics.csv`` with a
        ``side`` discriminator column; a single ``summary.json`` records
        per-side final bests.

        Note:
            Reuses the logger built in ``train()`` (so frame recording and
            metrics share a single run directory). Falls back to building a
            fresh one if the trainer is used outside ``train()``
            (defensive).
        """
        from deap_pettingzoo.baselines._logging_helpers import (
            build_logger,
            write_summary,
        )

        log = getattr(self, "_recording_log", None) or build_logger(
            config, algorithm="coevolution"
        )

        with log:
            for name, hist in per_side_history.items():
                for s in hist:
                    log.row(
                        "metrics",
                        {
                            "iteration": int(s.generation),
                            "env_steps": int(s.env_steps),
                            "wall_time": float(s.wall_time_s),
                            "side": name,
                            "mean": s.mean,
                            "std": s.std,
                            "min": s.min,
                            "max": s.max,
                            "best_fitness": s.best_fitness,
                        },
                    )
            write_summary(
                log,
                {
                    "algorithm": "coevolution",
                    "mode": coev_cfg.mode,
                    "schedule": coev_cfg.schedule,
                    "num_generations": num_generations,
                    "final_best": {
                        name: (
                            None
                            if state.side.optimizer == "frozen"
                            else float(state.best_fitness)
                        )
                        for name, state in states.items()
                    },
                    "reporting_side": coev_cfg.reporting_side,
                },
                training_time_s=training_time,
            )
            self._save_coevolution_policy(log, coev_cfg, states)

    def _save_coevolution_policy(
        self,
        log: Any,
        coev_cfg: "CoEvolutionConfig",
        states: Dict[str, Any],
    ) -> None:
        """Serialize per-side genomes to ``checkpoints/policy.json``.

        Only EA/CMA sides with a ``FixedTopologyGenome`` snapshot are
        persisted; frozen and QDax sides are skipped (frozen policies
        were supplied by the caller; QDax repertoires live in their
        own subprocesses and are out of scope for this hook).
        """
        from deap_pettingzoo.utils.logging import KIND_COEVOLUTION
        from deap_pettingzoo.utils.logging.checkpoints import (
            _checkpoints_dir,
            _is_real_logger,
            _register,
        )
        import json as _json

        if not _is_real_logger(log):
            return
        sides_payload: Dict[str, Any] = {}
        for name, state in states.items():
            if state.side.optimizer not in {"ea", "cma"}:
                continue
            try:
                snap = state.best_snapshot()
            except (AssertionError, AttributeError):
                continue
            if hasattr(snap, "to_dict"):
                sides_payload[name] = {
                    "kind": (
                        "ea_genome_v1"
                        if state.side.optimizer == "ea"
                        else "cma_genome_v1"
                    ),
                    "genome": snap.to_dict(),
                    "best_fitness": float(state.best_fitness),
                }
        if not sides_payload:
            return
        target = _checkpoints_dir(log.run_dir) / "policy.json"
        target.write_text(
            _json.dumps(
                {
                    "schema": "coevolution_v1",
                    "reporting_side": coev_cfg.reporting_side,
                    "sides": sides_payload,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _register(
            log,
            kind=KIND_COEVOLUTION,
            path=target,
            extra={
                "reporting_side": coev_cfg.reporting_side,
                "side_names": sorted(sides_payload.keys()),
            },
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # Evolutionary sides support MultiDiscrete; an RL side is
            # validated per-optimizer by the co-evolution validator.
            "multidiscrete_action": True,
            "needs_global_state": False,
            "on_policy": False,
            "supports_turn_based": True,
        }

    # Internals

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

        Note:
            Per-side episode counts: EA/CMA/QDax sides contribute
            ``population_size * k_opponents`` episodes (one rollout per
            individual per opponent sample); RL sides contribute
            ``rollout_episodes_per_gen`` (no population — ``k_opponents``
            does not apply, the RL side samples one opponent per rollout
            episode internally).
        """
        evolving = [s for s in coev_cfg.sides if s.optimizer != "frozen"]
        if not evolving:
            # Pathological — nothing is actually evolving — do at least
            # one pass to produce output.
            return 1
        ep_per_gen = 0
        for s in evolving:
            if s.optimizer == "rl":
                ep_per_gen += int(
                    s.optimizer_config.get("rollout_episodes_per_gen", 4)
                )
            elif s.optimizer in {"qdax_map_elites", "qdax_cma_me"}:
                ep_per_gen += (
                    int(s.optimizer_config.get("batch_size", 20))
                    * coev_cfg.k_opponents
                )
            else:
                ep_per_gen += (
                    int(s.optimizer_config.get("population_size", 20))
                    * coev_cfg.k_opponents
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

    def _make_batch_eval_fn(self, evaluator: Any) -> Optional[Any]:
        """Return a population-level parallel batch-evaluation function.

        When ``self._resolved_workers <= 1`` or the evaluator is not a
        :class:`CoEvolutionEvaluator`, returns ``None`` so callers fall
        back to the serial ``toolbox.evaluate`` path.

        The returned callable accepts a list of DEAP individuals and
        dispatches ``N × k_opponents`` :class:`_EpisodePayload` s via
        ``self._executor`` in a single ``executor.map`` call — giving
        true population-level parallelism instead of the episode-level
        dispatch that was previously wired inside the evaluator.

        All opponent sampling happens on the driver before any worker is
        launched, so pool RNGs mutate exactly once per batch call.
        """
        from .evaluator import CoEvolutionEvaluator, _EpisodePayload, _episode_rollout

        if self._resolved_workers <= 1:
            return None
        if not isinstance(evaluator, CoEvolutionEvaluator):
            return None

        env_name = self._env_name
        env_config = self._env_config
        executor = self._executor
        rng = self._rng
        k = evaluator.k_opponents
        aggregation = evaluator.aggregation
        max_steps = evaluator.max_steps
        learner_role_prefixes = evaluator.learner_role_prefixes
        learner_masked = evaluator._learner_masked
        opponent_sides = list(evaluator.opponent_sides)  # snapshot at creation time

        def _batch_eval(individuals: List[Any]) -> List[tuple]:
            n = len(individuals)
            if n == 0:
                return []
            # Sample all N×k opponents on the driver (preserves pool RNG state).
            per_side_snapshots: Dict[str, List[Any]] = {}
            side_role_prefixes: Dict[str, tuple] = {}
            for side_name, prefixes, pool in opponent_sides:
                per_side_snapshots[side_name] = pool.snapshot_opponents(rng, k * n)
                side_role_prefixes[side_name] = tuple(prefixes)
            # Build N×k payloads.
            payloads: List[_EpisodePayload] = []
            for i, ind in enumerate(individuals):
                for ep_i in range(k):
                    global_ep = i * k + ep_i
                    snaps = {
                        name: per_side_snapshots[name][global_ep]
                        for name in per_side_snapshots
                    }
                    payloads.append(
                        _EpisodePayload(
                            env_name=env_name,
                            env_config=env_config,
                            learner_genome=ind.genome,
                            learner_role_prefixes=learner_role_prefixes,
                            learner_masked=learner_masked,
                            opponent_snapshots=snaps,
                            side_role_prefixes=side_role_prefixes,
                            episode_seed=None,
                            max_steps=max_steps,
                        )
                    )
            # Dispatch all N×k episodes in parallel.
            results = list(executor.map(_episode_rollout, payloads))
            # Aggregate per individual.
            fitnesses: List[tuple] = []
            for i in range(n):
                ep_results = results[i * k : (i + 1) * k]
                learner_rewards = [r.learner_total for r in ep_results]
                arr = np.asarray(learner_rewards)
                if aggregation == "mean":
                    fit = float(arr.mean())
                elif aggregation == "min":
                    fit = float(arr.min())
                elif aggregation == "max":
                    fit = float(arr.max())
                else:  # "sum"
                    fit = float(arr.sum())
                fitnesses.append((fit,))
            return fitnesses

        return _batch_eval

    def _bootstrap_populations(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
    ) -> None:
        """Initial fitness evaluation for every evolving side.

        Note:
            QDax-backed sides build a temporary repertoire during
            ``initialise()`` so other sides can sample an initial opponent
            pool. They are bootstrapped first here so that temporary
            repertoire is replaced with real coevolution evaluations before
            EA/CMA sides evaluate against it.
        """
        ordered_states = sorted(
            states.items(),
            key=lambda item: (
                0 if item[1].side.optimizer in {"qdax_map_elites", "qdax_cma_me"}
                else 1
            ),
        )
        for name, state in ordered_states:
            if state.side.optimizer == "frozen":
                continue
            if not hasattr(state, "rebind_evaluator"):
                continue
            evaluator = self._build_evaluator(env, state, states, coev_cfg)
            # Set toolbox.evaluate then evaluate the population.
            state.rebind_evaluator(evaluator)  # type: ignore[attr-defined]
            batch_fn = self._make_batch_eval_fn(evaluator)
            state.evaluate_population(batch_eval_fn=batch_fn)  # type: ignore[call-arg]
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
    ) -> Any:
        """Construct the evaluator or driver context for *learner_state*.

        Note:
            RL learner sides bypass :class:`CoEvolutionEvaluator` entirely
            — they drive their own rollouts (with replay-buffer side effects)
            and consume an :class:`_RLDriverContext` directly inside
            ``step_generation``. The shared snapshot machinery is still used
            so the synchronous schedule's contract holds for the RL side too.

            Executor assignment: only EA/CMA learners receive the shared
            ``executor``; QDax learners stay serial (their inner numpy-pool
            already uses ``optimizer_config['num_workers']``) and frozen
            sides never call ``_build_evaluator`` at all.

            QDax sides drive their own JAX update loop; their scoring
            function receives a pre-built ``FlaxPolicyAdapter`` rather than
            a DEAP individual, so a lightweight
            :class:`_QDaxPolicyEvaluator` is returned instead of
            :class:`CoEvolutionEvaluator`.

            A deterministic RNG is injected via a closure so evaluations
            under a fixed seed replay identically; each call forks a fresh
            child generator to avoid shared state.
        """
        opp_sides = self._collect_opponent_sides(
            states, learner_state.side.name, coev_cfg,
            pool_snapshot=pool_snapshot,
        )
        if learner_state.side.optimizer == "rl":
            # Phase-2: pass executor + env identity so RLSideState can
            # dispatch rollout episodes to worker processes in parallel.
            # The executor is only handed over when resolved_workers > 1;
            # for serial runs None keeps the existing _collect_episode path.
            rl_executor = (
                self._executor if self._resolved_workers > 1 else None
            )
            return _RLDriverContext(
                env_wrapper=env,
                learner_role_prefixes=learner_state.side.role_prefixes,
                opponent_sides=opp_sides,
                max_steps=coev_cfg.max_steps_per_episode,
                rng=np.random.default_rng(
                    self._rng.integers(0, 2**63 - 1)
                ),
                executor=rl_executor,
                env_name=self._env_name,
                env_config=self._env_config,
            )
        assert (
            learner_state.side.optimizer != "frozen"
        ), "frozen sides never become learners."
        learner_optimizer = learner_state.side.optimizer
        # Hand the executor to EA/CMA/QDax evaluators whose parallel path is
        # implemented.  The resolved_workers > 1 guard avoids handing a
        # _SerialExecutor wrapper when the user set num_workers=1.
        executor_for_side = (
            self._executor
            if learner_optimizer in {"ea", "cma", "qdax_map_elites", "qdax_cma_me"}
            and self._resolved_workers > 1
            else None
        )

        if learner_optimizer in {"qdax_map_elites", "qdax_cma_me"}:
            trainer_rng = self._rng

            def _qdax_rng_factory():
                return np.random.default_rng(trainer_rng.integers(0, 2**63 - 1))

            return _QDaxPolicyEvaluator(
                env_wrapper=env,
                learner_role_prefixes=learner_state.side.role_prefixes,
                opponent_sides=opp_sides,
                k_opponents=coev_cfg.k_opponents,
                aggregation=coev_cfg.aggregation,
                max_steps=coev_cfg.max_steps_per_episode,
                rng_factory=_qdax_rng_factory,
                executor=executor_for_side,
                env_name=getattr(self, "_env_name", None),
                env_config=getattr(self, "_env_config", None),
            )

        evaluator = CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=learner_state.side.role_prefixes,
            learner_factory=_build_learner_factory(learner_state, coev_cfg),
            opponent_sides=opp_sides,
            k_opponents=coev_cfg.k_opponents,
            aggregation=coev_cfg.aggregation,
            max_steps=coev_cfg.max_steps_per_episode,
            executor=executor_for_side,
            env_name=getattr(self, "_env_name", None),
            env_config=getattr(self, "_env_config", None),
            learner_masked=bool(getattr(learner_state, "masked", False)),
        )
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

    def _step_one_side(
        self,
        env: PettingZooWrapper,
        state: SideState,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        history: Dict[str, List[GenerationStats]],
        pool_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Advance a single non-frozen side by one generation.

        Shared body for both schedules: build the side's evaluator
        (against a frozen ``pool_snapshot`` when supplied), run one
        ``step_generation``, append stats, and snapshot the best
        individual on the configured cadence. Frozen sides are skipped.
        """
        if state.side.optimizer == "frozen":
            return
        evaluator = self._build_evaluator(
            env, state, states, coev_cfg, pool_snapshot=pool_snapshot
        )
        batch_fn = self._make_batch_eval_fn(evaluator)
        stats = state.step_generation(
            lambda _s, _ev=evaluator: _ev, batch_eval_fn=batch_fn
        )
        history[state.side.name].append(stats)
        if (
            state.side.hof is not None
            and stats.generation % coev_cfg.snapshot_every == 0
        ):
            state.snapshot_best()

    def _run_synchronous_generation(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        history: Dict[str, List[GenerationStats]],
    ) -> None:
        """Advance every non-frozen side by one generation simultaneously.

        Note:
            Pool snapshot: all sides' opponent pools are captured at the
            *start* of the generation so every side evaluates against a
            frozen view of the other sides' state. Reading
            ``state.current_pool()`` live inside the per-side loop would
            break the contract — earlier sides would have already mutated
            their populations/HoFs.

            RL pools mutate in place each generation (both the pool object
            and the underlying torch agent). An optional
            ``snapshot_sync()`` hook is honoured so opponents see a frozen
            pre-learn view consistent with the synchronous-schedule
            contract.
        """
        pool_snapshot: Dict[str, Any] = {}
        for name, state in states.items():
            pool = state.current_pool()
            snap = getattr(pool, "snapshot_sync", None)
            if callable(snap):
                pool = snap()
            pool_snapshot[name] = pool
        for name, state in states.items():
            self._step_one_side(
                env, state, states, coev_cfg, history,
                pool_snapshot=pool_snapshot,
            )

    def _run_alternating_step(
        self,
        env: PettingZooWrapper,
        states: Dict[str, SideState],
        coev_cfg: CoEvolutionConfig,
        history: Dict[str, List[GenerationStats]],
        name: str,
    ) -> None:
        self._step_one_side(env, states[name], states, coev_cfg, history)


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
