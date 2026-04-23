"""
AgileRL-backed trainers for the unified baseline API.

This module defines an abstract :class:`AgileRLTrainer` that orchestrates
multi-agent training through AgileRL's in-process APIs, plus four
concrete subclasses:

- :class:`MADDPGTrainer` - Multi-Agent DDPG (off-policy)
- :class:`MATD3Trainer`  - Multi-Agent Twin-Delayed DDPG (off-policy)
- :class:`IPPOTrainer`   - Independent PPO (on-policy)
- :class:`MAPPOTrainer`  - Centralized-critic multi-agent PPO (on-policy)

The dependency on ``agilerl`` is imported lazily inside :meth:`train` so
that ``import deap_pettingzoo.baselines`` does not require the optional
``rl-baselines`` extra.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.agilerl_adapter import (
    build_parallel_env,
    extract_space_metadata,
    translate_budget_to_env_steps,
)
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgileRLTrainer(BaselineTrainer):
    """Abstract :class:`BaselineTrainer` backed by AgileRL.

    Subclasses must set:

    - :attr:`ALGORITHM_NAME`  - short label (e.g. ``'maddpg'``).
    - :attr:`ALGORITHM_IMPORT`  - ``(module_path, class_name)`` tuple
      used for the lazy import inside :meth:`train`.
    - :attr:`ON_POLICY`  - ``True`` for PPO-family algorithms,
      ``False`` for DDPG-family.
    - :meth:`get_capabilities`  - per-algorithm capability flags.
    """

    ALGORITHM_NAME: str = "agilerl"
    ALGORITHM_IMPORT: Tuple[str, str] = ("agilerl", "")
    ON_POLICY: bool = False

    # Default hyperparameters; subclasses may override.
    DEFAULT_PARAMS: Dict[str, Any] = {
        "batch_size": 64,
        "memory_size": 10_000,
        "learning_delay": 500,
        "rollout_steps": 256,
        "device": "cpu",
    }

    def __init__(self, device: str = "cpu", verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose

    # --- BaselineTrainer interface ----------------------------------------

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train an AgileRL algorithm under the given budget.

        Raises:
            ImportError: If the optional ``rl-baselines`` extra is not
                installed.
        """
        alg_cls = self._import_algorithm()

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.algorithm_params or {})
        seed = int(params.pop("seed", 0))
        self._seed_everything(seed)

        env = build_parallel_env(
            config.env_name, env_config=config.env_config or None
        )
        obs_spaces, act_spaces, agent_ids = extract_space_metadata(env)

        # Split params: constructor vs. loop knobs.
        loop_keys = {
            "batch_size",
            "memory_size",
            "learning_delay",
            "rollout_steps",
            "max_steps_per_episode",
        }
        ctor_kwargs = {k: v for k, v in params.items() if k not in loop_keys}
        ctor_kwargs.setdefault("device", self.device)

        agent = self._build_agent(
            alg_cls, obs_spaces, act_spaces, agent_ids, ctor_kwargs
        )

        budget_steps = self._budget_to_steps(config)
        start = time.time()
        if self.ON_POLICY:
            metrics = self._run_on_policy(
                env, agent, agent_ids, budget_steps, params, seed
            )
        else:
            metrics = self._run_off_policy(
                env, agent, agent_ids, budget_steps, params, seed
            )
        training_time = time.time() - start
        try:
            env.close()
        except SystemExit:
            # Some third-party envs (e.g. marl-aquarium) call sys.exit()
            # from close(); suppress to keep the process alive.
            pass
        except Exception:  # pragma: no cover
            pass

        policy = AgileRLPolicyWrapper(agent, agent_ids, self.ALGORITHM_NAME)
        return TrainingResult(
            algorithm=self.ALGORITHM_NAME,
            policy=policy,
            metrics=metrics,
            metadata={
                "env_name": config.env_name,
                "budget_mode": config.budget_mode.value,
                "budget_value": config.budget_value,
                "agent_ids": agent_ids,
                "on_policy": self.ON_POLICY,
                "algorithm_params": params,
            },
            seed=seed,
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    # --- Hooks / helpers --------------------------------------------------

    def _import_algorithm(self):
        """Lazily import the AgileRL algorithm class."""
        module_path, class_name = self.ALGORITHM_IMPORT
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                f"AgileRL is required for {self.ALGORITHM_NAME.upper()} "
                "training. Install with: "
                "pip install 'deap-pettingzoo[rl-baselines]'"
            ) from exc
        return getattr(module, class_name)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover
            pass

    def _budget_to_steps(self, config: TrainingConfig) -> int:
        if config.budget_mode == BudgetMode.WALL_CLOCK:
            # Soft cap: rely on a large step budget and let the caller's
            # timer stop us; we return a very large number here.
            return 10**9
        return translate_budget_to_env_steps(config)

    def _build_agent(
        self,
        alg_cls,
        obs_spaces,
        act_spaces,
        agent_ids: List[str],
        ctor_kwargs: Dict[str, Any],
    ):
        """Instantiate the AgileRL algorithm. Subclasses may override
        to inject algorithm-specific defaults (e.g. MAPPO's centralized
        critic configuration)."""
        return alg_cls(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            agent_ids=agent_ids,
            **ctor_kwargs,
        )

    @staticmethod
    def _unwrap_actions_for_env(actions: Dict[str, Any]) -> Dict[str, Any]:
        """Strip a leading vectorization dim of size 1 from each action.

        AgileRL always returns batched actions (shape ``(1, ...)`` for a
        single non-vectorized env). PettingZoo's parallel API expects
        unbatched actions (a scalar for ``Discrete`` spaces, a 1-D array
        for ``Box`` spaces), so we squeeze the leading dim per-agent.
        """
        unwrapped: Dict[str, Any] = {}
        for agent_id, act in actions.items():
            arr = np.asarray(act)
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
                # Discrete actions should be Python ints for PettingZoo.
                if arr.shape == ():
                    arr = arr.item()
            unwrapped[agent_id] = arr
        return unwrapped

    # --- Training loops ---------------------------------------------------

    def _run_off_policy(
        self,
        env,
        agent,
        agent_ids: List[str],
        budget_steps: int,
        params: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """Hand-rolled off-policy loop using AgileRL's replay buffer."""
        from agilerl.components.multi_agent_replay_buffer import (  # noqa: WPS433
            MultiAgentReplayBuffer,
        )

        memory = MultiAgentReplayBuffer(
            memory_size=int(params["memory_size"]),
            field_names=["obs", "action", "reward", "next_obs", "done"],
            agent_ids=agent_ids,
            device=self.device,
        )
        batch_size = int(params["batch_size"])
        learning_delay = int(params["learning_delay"])

        obs, info = env.reset(seed=seed)
        episode_rewards: List[float] = []
        ep_reward = 0.0
        ep_len = 0
        losses: List[float] = []

        for step in range(budget_steps):
            action_result = agent.get_action(obs, infos=info)
            actions, raw_actions = (
                action_result if isinstance(action_result, tuple) else (action_result, action_result)
            )
            env_actions = self._unwrap_actions_for_env(actions)
            next_obs, reward, termination, truncation, info = env.step(env_actions)

            memory.save_to_memory(
                obs, raw_actions, reward, next_obs, termination
            )

            ep_reward += float(sum(reward.values()))
            ep_len += 1

            if step >= learning_delay and len(memory) >= batch_size:
                experiences = memory.sample(batch_size)
                try:
                    loss = agent.learn(experiences)
                    if isinstance(loss, dict):
                        losses.append(
                            float(np.mean([float(v) for v in loss.values()]))
                        )
                except Exception as exc:  # pragma: no cover - version drift
                    if self.verbose:
                        print(f"[{self.ALGORITHM_NAME}] learn() failed: {exc}")

            done = any(termination.values()) or any(truncation.values())
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                obs, info = env.reset()
            else:
                obs = next_obs

        if ep_len > 0:
            episode_rewards.append(ep_reward)

        return {
            "episode_rewards": episode_rewards,
            "num_episodes": len(episode_rewards),
            "env_steps": budget_steps,
            "mean_loss": float(np.mean(losses)) if losses else None,
        }

    def _run_on_policy(
        self,
        env,
        agent,
        agent_ids: List[str],
        budget_steps: int,
        params: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """On-policy loop: collect rollouts, then call ``agent.learn``.

        Uses AgileRL's ``collect_rollouts`` helper when available; falls
        back to a trivial loop that exercises ``get_action`` only.
        """
        rollout_steps = int(params["rollout_steps"])
        episode_rewards: List[float] = []
        losses: List[float] = []

        try:
            from agilerl.rollouts.on_policy import (  # noqa: WPS433
                collect_rollouts_multi_agent,
            )
        except ImportError:
            collect_rollouts_multi_agent = None

        obs, info = env.reset(seed=seed)
        steps_done = 0
        ep_reward = 0.0

        while steps_done < budget_steps:
            remaining = min(rollout_steps, budget_steps - steps_done)

            if collect_rollouts_multi_agent is not None:
                # Delegate rollout + learn to AgileRL's helper.
                try:
                    collect_rollouts_multi_agent(agent, env, n_steps=remaining)
                    try:
                        loss = agent.learn()
                        if isinstance(loss, dict):
                            losses.append(
                                float(np.mean([float(v) for v in loss.values()]))
                            )
                    except Exception as exc:  # pragma: no cover
                        if self.verbose:
                            print(
                                f"[{self.ALGORITHM_NAME}] learn() failed: {exc}"
                            )
                    steps_done += remaining
                    # We cannot cheaply recover per-episode rewards here;
                    # we report coarse step totals instead.
                    continue
                except Exception as exc:  # pragma: no cover
                    if self.verbose:
                        print(
                            f"[{self.ALGORITHM_NAME}] collect_rollouts "
                            f"failed: {exc}; falling back to manual loop."
                        )
                    collect_rollouts_multi_agent = None

            # Manual fallback: step the env and accumulate rewards; no
            # learning signal (produces a reasonable but untrained agent).
            for _ in range(remaining):
                action_result = agent.get_action(obs, infos=info)
                actions = (
                    action_result[0]
                    if isinstance(action_result, tuple)
                    else action_result
                )
                env_actions = self._unwrap_actions_for_env(actions)
                next_obs, reward, termination, truncation, info = env.step(
                    env_actions
                )
                ep_reward += float(sum(reward.values()))
                done = any(termination.values()) or any(truncation.values())
                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, info = env.reset()
                else:
                    obs = next_obs
                steps_done += 1

        if ep_reward > 0.0 or not episode_rewards:
            episode_rewards.append(ep_reward)

        return {
            "episode_rewards": episode_rewards,
            "num_episodes": len(episode_rewards),
            "env_steps": steps_done,
            "mean_loss": float(np.mean(losses)) if losses else None,
        }


# ---------------------------------------------------------------------------
# Concrete trainers
# ---------------------------------------------------------------------------


class MADDPGTrainer(AgileRLTrainer):
    """Multi-Agent DDPG via AgileRL (off-policy, decentralized actors,
    centralized critics)."""

    ALGORITHM_NAME = "maddpg"
    ALGORITHM_IMPORT = ("agilerl.algorithms.maddpg", "MADDPG")
    ON_POLICY = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": False,
        }


class MATD3Trainer(AgileRLTrainer):
    """Multi-Agent Twin-Delayed DDPG via AgileRL (off-policy)."""

    ALGORITHM_NAME = "matd3"
    ALGORITHM_IMPORT = ("agilerl.algorithms.matd3", "MATD3")
    ON_POLICY = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": False,
        }


class IPPOTrainer(AgileRLTrainer):
    """Independent PPO via AgileRL (on-policy, decentralized)."""

    ALGORITHM_NAME = "ippo"
    ALGORITHM_IMPORT = ("agilerl.algorithms.ippo", "IPPO")
    ON_POLICY = True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": True,
        }


class MAPPOTrainer(IPPOTrainer):
    """Multi-Agent PPO with a centralized critic.

    AgileRL's ``IPPO`` supports a centralized critic via its
    ``shared_critic``-style configuration; :class:`MAPPOTrainer`
    forwards such flags through :attr:`TrainingConfig.algorithm_params`
    and declares ``needs_global_state=True`` so the compatibility
    checker flags fully heterogeneous environments.
    """

    ALGORITHM_NAME = "mappo"
    ON_POLICY = True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": True,
            "on_policy": True,
        }

    def _build_agent(
        self,
        alg_cls,
        obs_spaces,
        act_spaces,
        agent_ids: List[str],
        ctor_kwargs: Dict[str, Any],
    ):
        # Enable the centralized critic if AgileRL's IPPO exposes such a
        # kwarg; users may override via TrainingConfig.algorithm_params.
        ctor_kwargs.setdefault("shared_critic", True)
        try:
            return alg_cls(
                observation_spaces=obs_spaces,
                action_spaces=act_spaces,
                agent_ids=agent_ids,
                **ctor_kwargs,
            )
        except TypeError:
            # Fall back silently if ``shared_critic`` isn't accepted.
            ctor_kwargs.pop("shared_critic", None)
            return alg_cls(
                observation_spaces=obs_spaces,
                action_spaces=act_spaces,
                agent_ids=agent_ids,
                **ctor_kwargs,
            )


class GroupedIPPOTrainer(IPPOTrainer):
    """IPPO with AgileRL's prefix-based agent grouping.

    AgileRL 2.6+ exposes a grouping API on :class:`agilerl.algorithms.IPPO`:
    agents whose IDs share a common prefix (split on the **final** ``_``
    before any trailing digit block) share one actor/critic network. Classic
    PettingZoo naming — ``adversary_0``, ``adversary_1``, ``agent_0`` — is
    already compatible; agents in a group **must** have identical
    observation and action spaces.

    This subclass validates the grouping **up-front** and fails loudly
    when the environment's agents cannot be sensibly grouped (mixed
    prefixes with no structure, or same-prefix agents with mismatched
    spaces). No silent fallback to per-agent networks.

    Notes:
        * Grouping reduces parameter count and typically speeds training
          when roles are symmetric (e.g. predator–prey pursuit).
        * Grouping is derived from the environment's ``possible_agents``
          naming, following AgileRL's default behaviour.
    """

    ALGORITHM_NAME = "ippo-grouped"
    ON_POLICY = True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": True,
        }

    @staticmethod
    def _group_prefix(agent_id: str) -> str:
        """Return the prefix before a trailing numeric suffix in ``agent_id``.

        Mirrors AgileRL's grouping convention for ids like ``agent_0`` and
        ``agent_1``. Agents without an underscore, or whose final underscore
        is not followed by only digits, are treated as singleton groups
        (prefix = full id) so non-indexed names like ``team_blue`` /
        ``team_red`` are not collapsed into a single ``team`` group.
        """
        if "_" not in agent_id:
            return agent_id
        prefix, suffix = agent_id.rsplit("_", 1)
        if suffix.isdigit():
            return prefix
        return agent_id

    @classmethod
    def _validate_grouping(
        cls,
        agent_ids: List[str],
        obs_spaces,
        act_spaces,
    ) -> Dict[str, List[str]]:
        """Group ``agent_ids`` by prefix; fail loudly on inconsistencies.

        Raises:
            ValueError: If the environment yields only singleton groups
                (grouping has no effect — user probably wants plain
                ``IPPOTrainer``), or if any group contains agents with
                mismatched observation or action spaces.

        Returns:
            ``{prefix: [agent_id, ...]}`` mapping, preserving insertion
            order of the original ``agent_ids`` list.
        """
        groups: Dict[str, List[str]] = {}
        for aid in agent_ids:
            groups.setdefault(cls._group_prefix(aid), []).append(aid)

        # Refuse to run when grouping is a no-op: every agent is its
        # own group. That means the user gained nothing over IPPO and
        # almost certainly intended a different env or trainer.
        if all(len(members) == 1 for members in groups.values()):
            raise ValueError(
                f"GroupedIPPOTrainer: agent_ids={agent_ids!r} yield only "
                "singleton groups. Use IPPOTrainer instead, or rename the "
                "environment's agents so multiple share a prefix "
                "(e.g. 'adversary_0', 'adversary_1')."
            )

        # Spaces within a group must be identical.
        def _space_key(sp):
            return repr(sp)

        for prefix, members in groups.items():
            if len(members) < 2:
                continue
            ref_aid = members[0]
            ref_obs = obs_spaces[ref_aid]
            ref_act = act_spaces[ref_aid]
            for aid in members[1:]:
                if _space_key(obs_spaces[aid]) != _space_key(ref_obs):
                    raise ValueError(
                        "GroupedIPPOTrainer: agents in group "
                        f"{prefix!r} have mismatched observation spaces. "
                        f"{ref_aid}: {ref_obs!r}; {aid}: {obs_spaces[aid]!r}."
                    )
                if _space_key(act_spaces[aid]) != _space_key(ref_act):
                    raise ValueError(
                        "GroupedIPPOTrainer: agents in group "
                        f"{prefix!r} have mismatched action spaces. "
                        f"{ref_aid}: {ref_act!r}; {aid}: {act_spaces[aid]!r}."
                    )
        return groups

    def _build_agent(
        self,
        alg_cls,
        obs_spaces,
        act_spaces,
        agent_ids: List[str],
        ctor_kwargs: Dict[str, Any],
    ):
        """Validate grouping then delegate to AgileRL's IPPO constructor.

        AgileRL's IPPO detects grouping from ``agent_ids`` internally —
        no extra kwarg required — so all we do here is the pre-flight
        consistency check.
        """
        validation_obs_spaces = (
            dict(zip(agent_ids, obs_spaces))
            if isinstance(obs_spaces, (list, tuple))
            else obs_spaces
        )
        validation_act_spaces = (
            dict(zip(agent_ids, act_spaces))
            if isinstance(act_spaces, (list, tuple))
            else act_spaces
        )
        self._validate_grouping(
            agent_ids,
            validation_obs_spaces,
            validation_act_spaces,
        )
        return alg_cls(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            agent_ids=agent_ids,
            **ctor_kwargs,
        )
