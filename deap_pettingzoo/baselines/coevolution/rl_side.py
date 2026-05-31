"""RL side state for bidirectional EA-RL co-evolution.

Adds a fourth runtime-state class to the
:mod:`deap_pettingzoo.baselines.coevolution.sides` family: an
:class:`RLSideState` that wraps an AgileRL multi-agent algorithm
(MADDPG / MATD3 in this release) and trains it generation-by-generation
against opponents drawn from the *other* sides' current
:class:`OpponentPool`s.

The difference from EA / CMA / QDax sides:

* there is **no population of individuals** — the side owns one
  algorithm object whose internal networks improve continuously across
  generations;
* :meth:`current_pool` returns a degenerate size-1
  :class:`RLPolicyPool` wrapping the live agent, refreshed once per
  generation so other sides' next-gen evaluations face the latest RL
  policy;
* EA/CMA learner sides **can** evaluate against the RL opponent pool in
  parallel — :class:`~deap_pettingzoo.baselines.coevolution.pools.RLPolicyPool`
  produces picklable
  :class:`~deap_pettingzoo.baselines.agilerl_policy.FrozenRLPolicyWrapper`
  CPU snapshots via
  :meth:`~deap_pettingzoo.baselines.agilerl_policy.AgileRLPolicyWrapper.make_frozen_cpu_snapshot`;
* the RL learner's own rollout collection is still serial in this
  release; parallel dispatch is planned for Phase 2.

#TODO: implement opponent-aware adapters for on-policy algorithms (IPPO and variants) so the RL side can also use those.
#TODO (Phase 2): add executor-based parallel rollout collection for RLSideState.step_generation.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.evaluators.mixed_policy import _invoke_get_action

from .pools import OpponentPool, RLPolicyPool
from .sides import GenerationStats, Side, SideState

SUPPORTED_RL_ALGORITHMS: Dict[str, Tuple[str, str, bool]] = {
    "maddpg": ("agilerl.algorithms.maddpg", "MADDPG", False),
    "matd3":  ("agilerl.algorithms.matd3",  "MATD3",  False),
    "ippo":   ("agilerl.algorithms.ippo",   "IPPO",   True),
    "grouped_ippo": ("agilerl.algorithms.ippo", "IPPO", True),
    "ippo-grouped": ("agilerl.algorithms.ippo", "IPPO", True),
}


# Driver context — replaces the CoEvolutionEvaluator handed to other sides


@dataclass
class _RLDriverContext:
    """Per-call context object handed to :meth:`RLSideState.step_generation`.

    Mirrors the (non-public) shape the trainer was previously passing
    only to EA / CMA sides via :class:`CoEvolutionEvaluator`. RL sides
    don't reuse :class:`CoEvolutionEvaluator` because they drive their
    own rollouts (with replay-buffer side effects) and have no notion
    of "evaluating one individual".

    Constructed by :meth:`CoEvolutionTrainer._build_evaluator` whenever
    the learner side has ``optimizer == "rl"``.

    The optional ``executor``, ``env_name``, and ``env_config`` fields
    support Phase-2 parallel rollout collection: when ``executor`` is
    not ``None``, :meth:`RLSideState.step_generation` dispatches
    rollout episodes to worker processes via
    :func:`_rl_rollout_worker_fn` instead of calling
    :meth:`~RLSideState._collect_episode` serially.
    """

    env_wrapper: PettingZooWrapper
    learner_role_prefixes: Tuple[str, ...]
    opponent_sides: List[Tuple[str, Tuple[str, ...], OpponentPool]]
    max_steps: int
    rng: np.random.Generator
    #: Optional parallel executor (``ProcessPoolExecutor`` or compatible).
    #: ``None`` keeps the existing serial ``_collect_episode`` path.
    executor: Optional[Any] = None
    #: Registered env name — needed by workers to rebuild the env.
    env_name: Optional[str] = None
    #: Env kwargs forwarded to :class:`PettingZooWrapper`.
    env_config: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Phase-2 parallel rollout helpers (top-level so they are picklable)
# ---------------------------------------------------------------------------


@dataclass
class _RLStepData:
    """One transition step for all RL-side agents.

    Mirrors the arguments of
    ``MultiAgentReplayBuffer.save_to_memory(obs, actions, rewards,
    next_obs, done)``.
    """

    obs: Dict[str, np.ndarray]
    actions: Dict[str, np.ndarray]
    rewards: Dict[str, float]
    next_obs: Dict[str, np.ndarray]
    done: Dict[str, bool]


@dataclass
class _RLTrajectory:
    """Return value of :func:`_rl_rollout_worker_fn` — one episode."""

    steps: List[_RLStepData]
    episode_return: float


@dataclass
class _RLRolloutPayload:
    """Picklable payload for one RL rollout episode dispatched to a worker.

    The driver builds one payload per episode in
    :meth:`RLSideState.step_generation` and sends them to workers via
    the executor. Workers call :func:`_rl_rollout_worker_fn` with the
    payload and return :class:`_RLTrajectory`.

    All fields must be picklable (no live torch agents, no CUDA state).
    """

    env_name: str
    env_config: Optional[Dict[str, Any]]
    max_steps: int
    episode_seed: int
    #: CPU-only inference snapshot of the RL learner (picklable).
    frozen_rl_snapshot: Any  # FrozenRLPolicyWrapper
    rl_agent_ids: List[str]
    rl_role_prefixes: Tuple[str, ...]
    #: One snapshot per opponent side (keyed by side name).
    opponent_snapshots: Dict[str, Any]  # Dict[str, OpponentSnapshot]
    #: Maps opponent side name → role_prefixes.
    side_role_prefixes: Dict[str, Tuple[str, ...]]


def _rl_rollout_worker_fn(payload: _RLRolloutPayload) -> _RLTrajectory:
    """Top-level worker function: run one RL rollout episode.

    Lives at module top level so it is picklable for
    :class:`~concurrent.futures.ProcessPoolExecutor`.  All state is
    rebuilt from the picklable ``payload`` — nothing live crosses the
    process boundary.

    The RL-side policy is reconstructed from ``payload.frozen_rl_snapshot``
    (a :class:`~deap_pettingzoo.baselines.agilerl_policy.FrozenRLPolicyWrapper`).
    Opponent policies are rebuilt via
    :func:`~deap_pettingzoo.baselines.coevolution.pools.rebuild_opponent_policy`.
    The env is rebuilt (and cached per worker process) via the same
    :func:`~deap_pettingzoo.baselines.coevolution.evaluator._resolve_worker_env`
    used by EA/CMA parallel evaluation.
    """
    # Lazy imports — top-level imports would force all workers to import
    # the full coevolution stack at spawn time.
    from deap_pettingzoo.baselines.coevolution.evaluator import _resolve_worker_env
    from deap_pettingzoo.baselines.coevolution.pools import rebuild_opponent_policy
    from deap_pettingzoo.evaluators.mixed_policy import _invoke_get_action

    env = _resolve_worker_env(payload.env_name, payload.env_config)
    possible_agents = list(env.possible_agents)

    # RL-side agent IDs come directly from the payload (set at build time
    # from the driver-process _side_agent_ids list).
    rl_ids = list(payload.rl_agent_ids)

    # Rebuild opponent policies (one per side, shared by all side agents).
    opp_policies: Dict[str, Any] = {}
    for side_name, snapshot in payload.opponent_snapshots.items():
        prefixes = payload.side_role_prefixes.get(side_name, ())
        members = [a for a in possible_agents if any(a.startswith(p) for p in prefixes)]
        if not members:
            continue
        action_space = env.action_space(members[0])
        side_policy = rebuild_opponent_policy(snapshot, action_space=action_space)
        if hasattr(side_policy, "reset"):
            side_policy.reset()
        for a in members:
            opp_policies[a] = side_policy

    # The frozen RL policy — already a PolicyProtocol.
    frozen_rl = payload.frozen_rl_snapshot
    if hasattr(frozen_rl, "reset"):
        frozen_rl.reset()

    # Episode rollout.
    observations = env.reset(seed=payload.episode_seed)
    info: Dict[str, Any] = getattr(env, "last_infos", {}) or {}

    steps_data: List[_RLStepData] = []
    ep_reward = 0.0
    steps = 0

    while steps < payload.max_steps and observations:
        rl_obs = {a: observations[a] for a in rl_ids if a in observations}
        if not rl_obs:
            break

        env_actions: Dict[str, Any] = {}
        squeezed_actions: Dict[str, np.ndarray] = {}

        # RL-side actions via the frozen policy (round-robin dispatch).
        for aid in rl_ids:
            if aid not in observations:
                continue
            rl_mask = info.get(aid, {}).get("action_mask")
            act = _invoke_get_action(frozen_rl, observations[aid], rl_mask)
            arr = np.asarray(act)
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
            squeezed_actions[aid] = arr
            env_actions[aid] = arr.item() if arr.shape == () else arr

        # Zero-fill terminated RL agents so the buffer shape is consistent.
        if len(rl_obs) < len(rl_ids) and rl_obs and squeezed_actions:
            _ref_obs = next(iter(rl_obs.values()))
            _ref_act = np.asarray(next(iter(squeezed_actions.values())))
            for a in rl_ids:
                if a not in rl_obs:
                    rl_obs[a] = np.zeros_like(_ref_obs)
                if a not in squeezed_actions:
                    squeezed_actions[a] = np.zeros_like(_ref_act)

        # Opponent actions.
        for agent, obs in observations.items():
            if agent in env_actions:
                continue
            if agent in opp_policies:
                opp_mask = info.get(agent, {}).get("action_mask")
                env_actions[agent] = _invoke_get_action(
                    opp_policies[agent], obs, opp_mask
                )
            else:
                # No policy available — sample a random action so env.step
                # receives a complete action dict regardless.
                env_actions[agent] = env.action_space(agent).sample()

        next_obs, rewards, terms, truncs, next_info = env.step(env_actions)

        rl_next_obs = {a: next_obs[a] for a in rl_ids if a in next_obs}
        for a in rl_ids:
            if a not in rl_next_obs and a in rl_obs:
                rl_next_obs[a] = np.zeros_like(rl_obs[a])

        rl_reward = {a: float(rewards.get(a, 0.0)) for a in rl_ids}
        rl_done = {
            a: bool(terms.get(a, False) or truncs.get(a, False))
            for a in rl_ids
        }

        steps_data.append(_RLStepData(
            obs=rl_obs,
            actions=squeezed_actions,
            rewards=rl_reward,
            next_obs=rl_next_obs,
            done=rl_done,
        ))
        ep_reward += sum(rl_reward.values())
        steps += 1
        observations = next_obs
        info = next_info or {}

        if terms and all(terms.values()):
            break
        if truncs and all(truncs.values()):
            break

    return _RLTrajectory(steps=steps_data, episode_return=ep_reward)


# Side state


class RLSideState(SideState):
    """Co-evolving RL side backed by an AgileRL multi-agent algorithm.

    Accepted ``optimizer_config`` keys:

    * ``algorithm`` (str, **required**) — one of
      ``"maddpg" | "matd3"``. On-policy algorithms (``"ippo"``,
      ``"grouped_ippo"``) are accepted by name for forward
      compatibility but currently raise :class:`NotImplementedError`
      from :meth:`initialise`.

    Parallel co-evolution (``CoEvolutionConfig.num_workers > 1``) is
    supported for EA/CMA learner sides that face an RL opponent: the
    trainer ships :class:`~deap_pettingzoo.baselines.agilerl_policy.FrozenRLPolicyWrapper`
    CPU snapshots across worker boundaries.  The RL learner's own
    rollout collection is still serial; see the module TODO comment.
    * ``rollout_episodes_per_gen`` (int, default ``4``) — number of
      episodes collected per :meth:`step_generation` call against the
      current opponent-pool snapshot.
    * ``learning_steps_per_gen`` (int, default ``1``) — number of
      ``agent.learn(experiences)`` calls per generation (off-policy
      only; the buffer must hold at least ``batch_size`` transitions
      for any to fire).
    * ``batch_size`` (int, default ``64``).
    * ``memory_size`` (int, default ``10_000``).
    * ``learning_delay`` (int, default ``0``) — minimum number of total
      transitions before learning starts (matches AgileRL's trainer
      semantics).
    * ``device`` (str, default: auto — ``"cuda"`` if available, else
      ``"cpu"``). The trainer's GPU-contention detector reads
      :attr:`resolved_device` after :meth:`initialise`.
    * ``agent_ctor_kwargs`` (dict, default ``{}``) — forwarded
      verbatim to the AgileRL algorithm constructor.
    * ``seed`` (int or ``None``).

    The side controls **every** env agent matching ``role_prefixes``;
    one AgileRL algorithm object owns all of them (heterogeneous
    obs / action spaces are handled by AgileRL natively, so unlike
    EA / CMA sides no single-shared-space check is performed).
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config

        algo = cfg.get("algorithm")
        if algo is None:
            raise ValueError(
                f"RLSideState({side.name!r}): "
                "optimizer_config['algorithm'] is required. "
                f"Choose one of {sorted(SUPPORTED_RL_ALGORITHMS)}."
            )
        algo = str(algo).lower()
        if algo not in SUPPORTED_RL_ALGORITHMS:
            raise ValueError(
                f"RLSideState({side.name!r}): unknown algorithm "
                f"{algo!r}. Supported: {sorted(SUPPORTED_RL_ALGORITHMS)}."
            )
        self.algorithm: str = algo
        self.is_on_policy: bool = SUPPORTED_RL_ALGORITHMS[algo][2]

        self.rollout_episodes_per_gen = int(cfg.get("rollout_episodes_per_gen", 4))
        if self.rollout_episodes_per_gen < 1:
            raise ValueError(
                f"RLSideState({side.name!r}): rollout_episodes_per_gen "
                f"must be >= 1, got {self.rollout_episodes_per_gen}."
            )
        self.learning_steps_per_gen = int(cfg.get("learning_steps_per_gen", 1))
        if self.learning_steps_per_gen < 0:
            raise ValueError(
                f"RLSideState({side.name!r}): learning_steps_per_gen "
                f"must be >= 0, got {self.learning_steps_per_gen}."
            )
        self.batch_size = int(cfg.get("batch_size", 64))
        self.memory_size = int(cfg.get("memory_size", 10_000))
        self.learning_delay = int(cfg.get("learning_delay", 0))
        self.device: Optional[str] = cfg.get("device")  # None => auto
        self.agent_ctor_kwargs: Dict[str, Any] = dict(
            cfg.get("agent_ctor_kwargs", {})
        )
        self.seed = cfg.get("seed", None)

        # Filled in during :meth:`initialise`.
        self._side_agent_ids: List[str] = []
        self._agent: Any = None
        self._memory: Any = None
        self._wrapper: Optional[PolicyProtocol] = None
        self._pool: Optional[RLPolicyPool] = None
        #: Effective torch device after :meth:`initialise`. 
        #: Read by :func:`deap_pettingzoo.utils.device_check.warn_on_gpu_contention`.
        self.resolved_device: str = "cpu"
        self._last_returns: List[float] = []
        self._bootstrap_ctx: Optional[_RLDriverContext] = None
        self._total_transitions: int = 0
        #: Counters used by tests to assert that learning actually fired.
        #: ``_total_learn_calls`` increments for every successful
        #: ``agent.learn(experiences)`` invocation;
        #: ``_total_learn_failures`` increments for every caught
        #: failure inside :meth:`step_generation`.
        self._total_learn_calls: int = 0
        self._total_learn_failures: int = 0

    # Lifecycle

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if self.is_on_policy:
            raise NotImplementedError(
                f"RLSideState({self.side.name!r}): on-policy algorithms "
                f"({self.algorithm!r}) are not yet supported in "
                "bidirectional EA-RL co-evolution. Use 'maddpg' or "
                "'matd3', or wrap a pre-trained on-policy agent in "
                "FrozenSideState (one-directional opponent only)."
            )

        # Resolve device.
        if self.device is not None:
            self.resolved_device = str(self.device)
        else:
            try:
                import torch  # type: ignore

                self.resolved_device = (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            except ImportError:
                self.resolved_device = "cpu"

        # Seed.
        if self.seed is None and rng is not None:
            self.seed = int(
                rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)
            )
        seed = int(self.seed) if self.seed is not None else 0
        np.random.seed(seed)
        try:
            import torch  # type: ignore

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover
            pass

        # Restrict to side-controlled env agents.
        side_agents = [
            a for a in env.possible_agents
            if a.startswith(self.side.role_prefixes)
        ]
        if not side_agents:
            raise ValueError(
                f"RLSideState({self.side.name!r}): no env agents matched "
                f"role_prefixes={self.side.role_prefixes!r}."
            )
        self._side_agent_ids = side_agents

        obs_spaces = [env.observation_space(a) for a in side_agents]
        act_spaces = [env.action_space(a) for a in side_agents]

        # Lazy AgileRL import.
        module_path, class_name, _ = SUPPORTED_RL_ALGORITHMS[self.algorithm]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"RLSideState({self.side.name!r}): algorithm "
                f"{self.algorithm!r} requires AgileRL. Install with: "
                "pip install 'deap-pettingzoo[rl-baselines]'"
            ) from exc
        alg_cls = getattr(module, class_name)

        ctor_kwargs = dict(self.agent_ctor_kwargs)
        ctor_kwargs.setdefault("device", self.resolved_device)
        self._agent = alg_cls(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            agent_ids=side_agents,
            **ctor_kwargs,
        )

        # Off-policy replay buffer.
        from agilerl.components.multi_agent_replay_buffer import (
            MultiAgentReplayBuffer,
        )

        self._memory = MultiAgentReplayBuffer(
            memory_size=self.memory_size,
            field_names=["obs", "action", "reward", "next_obs", "done"],
            agent_ids=side_agents,
            device=self.resolved_device,
        )

        # Build wrapper + degenerate pool.
        from deap_pettingzoo.baselines.agilerl_policy import (
            AgileRLPolicyWrapper,
        )

        self._wrapper = AgileRLPolicyWrapper(
            self._agent, side_agents, self.algorithm
        )
        self._pool = RLPolicyPool(self._wrapper)

    def rebind_evaluator(self, evaluator: Any) -> None:
        """Stash the bootstrap-time driver context for :meth:`evaluate_population`.

        EA / CMA sides use :meth:`rebind_evaluator` to wire a
        :class:`CoEvolutionEvaluator` into ``toolbox.evaluate``. RL
        sides have no such DEAP-toolbox indirection — they consume the
        driver context directly inside :meth:`step_generation`. We
        keep this hook as a no-op-with-stash so the trainer's
        ``_bootstrap_populations`` loop can also work for RL.
        """
        if isinstance(evaluator, _RLDriverContext):
            self._bootstrap_ctx = evaluator

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        """One warm-up rollout to seed the replay buffer.

        Trainer contract: called from :meth:`_bootstrap_populations`
        right after :meth:`rebind_evaluator`. Without this, the first
        :meth:`step_generation` on the smallest reasonable budget would
        produce zero transitions before its learning attempt.
        """
        if self._bootstrap_ctx is None:
            return
        try:
            r = self._collect_episode(self._bootstrap_ctx)
        except (RuntimeError, ValueError, TypeError) as exc:
            warnings.warn(
                f"RLSideState({self.side.name!r}): bootstrap rollout "
                f"failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._last_returns = [r]
        self.best_fitness = float(r)

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        ctx = evaluator_factory(self)
        if not isinstance(ctx, _RLDriverContext):
            raise TypeError(
                f"RLSideState({self.side.name!r}).step_generation: "
                f"expected _RLDriverContext from the trainer, got "
                f"{type(ctx).__name__}."
            )
        self.generation += 1

        episode_returns: List[float] = []

        # --- Phase-2 parallel rollout path ---
        if (
            ctx.executor is not None
            and ctx.env_name is not None
            and self._wrapper is not None
        ):
            from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper

            if isinstance(self._wrapper, AgileRLPolicyWrapper):
                try:
                    episode_returns = self._collect_episodes_parallel(ctx)
                except Exception as exc:
                    warnings.warn(
                        f"RLSideState({self.side.name!r}): parallel rollout "
                        f"collection failed ({exc}); falling back to serial.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    episode_returns = []
                    for _ in range(self.rollout_episodes_per_gen):
                        episode_returns.append(self._collect_episode(ctx))
            else:
                for _ in range(self.rollout_episodes_per_gen):
                    episode_returns.append(self._collect_episode(ctx))
        else:
            # --- Serial rollout path (default) ---
            for _ in range(self.rollout_episodes_per_gen):
                episode_returns.append(self._collect_episode(ctx))

        # Off-policy learn step(s).
        for _ in range(self.learning_steps_per_gen):
            if (
                self._total_transitions >= self.learning_delay
                and len(self._memory) >= self.batch_size
            ):
                experiences = self._memory.sample(self.batch_size)
                try:
                    self._agent.learn(experiences)
                    self._total_learn_calls += 1
                except (RuntimeError, ValueError, TypeError) as exc:
                    self._total_learn_failures += 1
                    warnings.warn(
                        f"RLSideState({self.side.name!r}): "
                        f"agent.learn() failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

        # Refresh wrapper + pool — other sides next gen face the new RL policy.
        from deap_pettingzoo.baselines.agilerl_policy import (
            AgileRLPolicyWrapper,
        )

        self._wrapper = AgileRLPolicyWrapper(
            self._agent, self._side_agent_ids, self.algorithm
        )
        if self._pool is not None:
            self._pool.update(self._wrapper)

        arr = np.asarray(episode_returns or [0.0], dtype=np.float64)
        best = float(arr.max())
        self.best_fitness = best
        self._last_returns = list(episode_returns)
        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=best,
            best_fitness=best,
        )
        self.history.append(stats)
        return stats

    # --- Phase-2 helpers ---

    def _collect_episodes_parallel(
        self, ctx: _RLDriverContext
    ) -> List[float]:
        """Dispatch ``rollout_episodes_per_gen`` episodes to worker processes.

        Builds picklable :class:`_RLRolloutPayload` objects (one per
        episode), maps them through ``ctx.executor``, and pushes the
        returned :class:`_RLTrajectory` transitions to the local replay
        buffer on the driver.

        Returns a list of per-episode cumulative rewards.
        """
        from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper

        assert isinstance(self._wrapper, AgileRLPolicyWrapper)
        frozen = self._wrapper.make_frozen_cpu_snapshot()

        # One snapshot per opponent side (sampled once on the driver).
        opp_snapshots: Dict[str, Any] = {}
        for opp_name, _prefixes, opp_pool in ctx.opponent_sides:
            snaps = opp_pool.snapshot_opponents(ctx.rng, 1)
            opp_snapshots[opp_name] = snaps[0]

        side_role_prefixes = {
            name: prefixes for name, prefixes, _ in ctx.opponent_sides
        }

        payloads = [
            _RLRolloutPayload(
                env_name=ctx.env_name,  # type: ignore[arg-type]
                env_config=ctx.env_config,
                max_steps=ctx.max_steps,
                episode_seed=int(ctx.rng.integers(0, 2**31 - 1)),
                frozen_rl_snapshot=frozen,
                rl_agent_ids=list(self._side_agent_ids),
                rl_role_prefixes=self.side.role_prefixes,
                opponent_snapshots=opp_snapshots,
                side_role_prefixes=side_role_prefixes,
            )
            for _ in range(self.rollout_episodes_per_gen)
        ]

        trajectories: List[_RLTrajectory] = list(
            ctx.executor.map(_rl_rollout_worker_fn, payloads)
        )

        episode_returns: List[float] = []
        for traj in trajectories:
            episode_returns.append(traj.episode_return)
            for step in traj.steps:
                try:
                    self._memory.save_to_memory(
                        step.obs,
                        step.actions,
                        step.rewards,
                        step.next_obs,
                        step.done,
                    )
                    self._total_transitions += 1
                except (RuntimeError, ValueError, TypeError) as exc:
                    warnings.warn(
                        f"RLSideState({self.side.name!r}): "
                        f"save_to_memory failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return episode_returns

    # OpponentPool surface

    def current_pool(self) -> OpponentPool:
        if self._pool is None:
            raise RuntimeError(
                f"RLSideState({self.side.name!r}).current_pool: "
                "initialise() must be called first."
            )
        return self._pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        if self._wrapper is None:
            raise RuntimeError(
                f"RLSideState({self.side.name!r}).best_policy: "
                "initialise() must be called first."
            )
        return self._wrapper

    def best_snapshot(self) -> Any:
        """Return the live wrapper.

        Note: this returns a *reference* to the same underlying torch
        agent — subsequent generations will mutate it in place. For
        proper checkpointed self-play HoF, deep-copy via the wrapped
        agent's ``save_checkpoint`` API in user code; v1 stores live
        references, which is sufficient for the canonical
        "current-pool-only" co-evolution flow.
        """
        return self._wrapper

    # Rollout

    def _collect_episode(self, ctx: _RLDriverContext) -> float:
        """Run one rollout episode, push transitions, return cumulative reward.

        Uses ``self._agent.get_action`` for our side's agents (training
        mode = stochastic actions for exploration; the AgileRL agent
        chooses based on its own training/eval flag — we don't pass
        ``training=True`` because the API is inconsistent across
        algorithms; the buffer's diversity is driven by the
        algorithm's own action-noise schedule).

        Note:
            The leading vectorisation dim (size 1) is squeezed from
            ``raw_actions`` before both ``env.step`` and
            ``save_to_memory``.  Storing batched actions
            (shape ``(1, action_dim)``) alongside unbatched observations
            (shape ``(features,)``) causes a 2-vs-3 ndim mismatch in
            ``MADDPG.learn()`` when the buffer's sampled tensors are
            passed to actor/critic forward passes.  Keeping obs and
            action shapes rank-aligned at write time avoids this.
            :meth:`~deap_pettingzoo.baselines.agilerl_trainer.AgileRLTrainer._run_off_policy`
            applies the same fix.
        """
        env = ctx.env_wrapper

        # Sample one opponent policy per opponent side (shared across
        # that side's agents — same contract the EA evaluator uses).
        opp_policies: Dict[str, PolicyProtocol] = {}
        for _, prefixes, pool in ctx.opponent_sides:
            policy = pool.sample_opponent(ctx.rng)
            for a in env.possible_agents:
                if a.startswith(prefixes):
                    opp_policies[a] = policy

        episode_seed = int(ctx.rng.integers(0, 2**31 - 1))
        observations = env.reset(seed=episode_seed)
        info = getattr(env, "last_infos", {}) or {}

        for p in opp_policies.values():
            if hasattr(p, "reset"):
                p.reset()

        rl_ids = self._side_agent_ids
        ep_reward = 0.0
        steps = 0

        while steps < ctx.max_steps and observations:
            # Build RL-only obs / info dicts.  Track which agents are
            # still alive this step so env_actions can be restricted
            # to active agents only.
            active_obs = {a: observations[a] for a in rl_ids if a in observations}
            if not active_obs:
                break  # all RL agents terminated
            rl_info = {a: info.get(a, {}) for a in rl_ids if a in observations}

            # AgileRL agents (MATD3/MADDPG) iterate *all* entries in
            # possible_action_spaces in their get_action post-processing
            # loop and raise KeyError if any agent is absent.  Zero-fill
            # terminated agents so the full roster is always present.
            rl_obs = dict(active_obs)
            if len(rl_obs) < len(rl_ids):
                _ref_obs = next(iter(rl_obs.values()))
                for a in rl_ids:
                    if a not in rl_obs:
                        rl_obs[a] = np.zeros_like(_ref_obs)
                        rl_info[a] = {}

            # Actions for our side via AgileRL agent (full roster).
            try:
                action_result = self._agent.get_action(rl_obs, infos=rl_info)
            except TypeError:
                action_result = self._agent.get_action(rl_obs)
            actions, raw_actions = (
                action_result
                if isinstance(action_result, tuple)
                else (action_result, action_result)
            )

            # Build env_actions from processed `actions` (scalar for
            # Discrete, clipped float for Continuous), restricted to
            # *active* agents only — terminated agents must not be
            # included in env.step().
            env_actions: Dict[str, Any] = {}
            for aid in active_obs:
                act = actions.get(aid)
                if act is not None:
                    arr = np.asarray(act)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                    env_actions[aid] = arr.item() if arr.shape == () else arr

            # Build squeezed_actions from RAW actions for the replay
            # buffer.  MATD3/MADDPG require the full probability vector
            # (GumbelSoftmax for Discrete) or raw continuous value — NOT
            # the post-processed argmax integer.  Mirrors
            # agilerl_trainer._complete_action_dict which also uses
            # raw_actions.  rl_obs is already zero-filled above; guard
            # squeezed_actions defensively.
            squeezed_actions: Dict[str, Any] = {}
            for aid, act in raw_actions.items():
                arr = np.asarray(act)
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                squeezed_actions[aid] = arr
            if squeezed_actions and len(squeezed_actions) < len(rl_ids):
                _ref_act = np.asarray(next(iter(squeezed_actions.values())))
                for a in rl_ids:
                    if a not in squeezed_actions:
                        squeezed_actions[a] = np.zeros_like(_ref_act)

            # Opponent actions via PolicyProtocol.
            for agent, obs in observations.items():
                if agent in env_actions:
                    continue
                if agent in opp_policies:
                    mask = info.get(agent, {}).get("action_mask")
                    env_actions[agent] = _invoke_get_action(
                        opp_policies[agent], obs, mask
                    )

            next_obs, rewards, terms, truncs, next_info = env.step(env_actions)

            # Push transition (RL-side agents only).
            rl_next_obs = {
                a: next_obs[a]
                for a in rl_ids
                if a in next_obs
            }
            # If any RL agent terminated this step, fill its next_obs
            # with a zero-shaped placeholder of the correct dtype so the buffer accepts the dict.
            for a in rl_ids:
                if a not in rl_next_obs and a in rl_obs:
                    rl_next_obs[a] = np.zeros_like(rl_obs[a])
            rl_reward = {a: float(rewards.get(a, 0.0)) for a in rl_ids}
            rl_term = {
                a: bool(terms.get(a, False) or truncs.get(a, False))
                for a in rl_ids
            }

            try:
                self._memory.save_to_memory(
                    rl_obs, squeezed_actions, rl_reward, rl_next_obs, rl_term
                )
                self._total_transitions += 1
            except (RuntimeError, ValueError, TypeError) as exc:
                warnings.warn(
                    f"RLSideState({self.side.name!r}): "
                    f"save_to_memory failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            ep_reward += sum(rl_reward.values())
            steps += 1
            observations = next_obs
            info = next_info or {}

            if terms and all(terms.values()):
                break
            if truncs and all(truncs.values()):
                break

        return ep_reward


__all__ = [
    "RLSideState",
    "SUPPORTED_RL_ALGORITHMS",
    "_RLDriverContext",
    "_RLRolloutPayload",
    "_RLTrajectory",
    "_RLStepData",
    "_rl_rollout_worker_fn",
]
