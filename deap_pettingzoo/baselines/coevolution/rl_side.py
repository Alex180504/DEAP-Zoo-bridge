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
* parallel co-evolution (``CoEvolutionConfig.num_workers > 1``) is not
  yet supported on RL sides — the trainer falls back to serial with a
  one-shot warning.

#TODO: implement oponnent-aware adapters for on-policy algorithms (IPPO and variants) so the RL side can also use those.
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
    """

    env_wrapper: PettingZooWrapper
    learner_role_prefixes: Tuple[str, ...]
    opponent_sides: List[Tuple[str, Tuple[str, ...], OpponentPool]]
    max_steps: int
    rng: np.random.Generator


# Side state


class RLSideState(SideState):
    """Co-evolving RL side backed by an AgileRL multi-agent algorithm.

    Accepted ``optimizer_config`` keys:

    * ``algorithm`` (str, **required**) — one of
      ``"maddpg" | "matd3"``. On-policy algorithms (``"ippo"``,
      ``"grouped_ippo"``) are accepted by name for forward
      compatibility but currently raise :class:`NotImplementedError`
      from :meth:`initialise`.
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

    Parallel co-evolution (``CoEvolutionConfig.num_workers > 1``) is
    not supported on RL sides — :meth:`current_pool` returns an
    :class:`RLPolicyPool` whose
    :meth:`~RLPolicyPool.snapshot_opponents` raises. The trainer
    detects this case and falls back to serial evaluation with a
    one-shot :class:`RuntimeWarning`.
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
        for _ in range(self.rollout_episodes_per_gen):
            r = self._collect_episode(ctx)
            episode_returns.append(r)

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
            # Build RL-only obs / info dicts (the AgileRL agent was
            # constructed with only this side's agent_ids, so passing
            # only those keys is correct).
            rl_obs = {a: observations[a] for a in rl_ids if a in observations}
            if not rl_obs:
                break  # all RL agents terminated
            rl_info = {a: info.get(a, {}) for a in rl_ids if a in observations}

            # Actions for our side via AgileRL agent.
            try:
                action_result = self._agent.get_action(rl_obs, infos=rl_info)
            except TypeError:
                action_result = self._agent.get_action(rl_obs)
            actions, raw_actions = (
                action_result
                if isinstance(action_result, tuple)
                else (action_result, action_result)
            )

            env_actions: Dict[str, Any] = {}
            squeezed_actions: Dict[str, Any] = {}
            for aid, act in actions.items():
                arr = np.asarray(act)
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                squeezed_actions[aid] = arr
                if arr.shape == ():
                    env_actions[aid] = arr.item()
                else:
                    env_actions[aid] = arr

            # Zero-fill obs and squeezed_actions for any RL agent that has
            # already terminated (MultiAgentReplayBuffer requires the full agent_ids roster on every write).
            if len(rl_obs) < len(rl_ids) and rl_obs and squeezed_actions:
                _ref_obs = next(iter(rl_obs.values()))
                _ref_act = np.asarray(next(iter(squeezed_actions.values())))
                for a in rl_ids:
                    if a not in rl_obs:
                        rl_obs[a] = np.zeros_like(_ref_obs)
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
]
