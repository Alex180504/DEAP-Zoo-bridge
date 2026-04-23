"""Adapters that present **strictly turn-based** PettingZoo AEC envs as
parallel-shaped envs suitable for the deap-bridge stack.

Two modes are supplied:

* :class:`MaskedSimultaneousAdapter` — every ``reset``/``step`` returns
  dicts keyed by all currently-active agents. Only the current player's
  action is executed by the wrapped AEC env; other agents' actions are
  discarded. ``infos[agent]["action_mask"]`` is populated from upstream
  when available. This mode is **EA-friendly** — existing evaluators
  work unchanged because they already iterate ``obs.keys()``. Downside:
  N−1 phantom policy calls per turn and the per-agent trajectory is
  not semantically honest for value-based RL.

* :class:`SequentialTurnAdapter` — ``reset``/``step`` return
  *single-agent* dicts containing only the currently-acting player.
  Semantically honest (1 env_step = 1 game turn) and the right choice
  for future TD-learning / AlphaZero-style experiments.

Both adapters normalise the classic-games observation convention
(``observe(agent) -> {"observation": arr, "action_mask": mask}``) so
downstream policies see a plain array observation with the mask
surfaced in ``infos[agent]["action_mask"]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv, ParallelEnv


# ---------------------------------------------------------------------------
# Helpers shared by both adapters
# ---------------------------------------------------------------------------


def _unwrap_obs_space(raw_space):
    """Return the underlying observation space, stripping mask-dict wrappers.

    PettingZoo classic envs expose ``observation_space(agent)`` as a
    ``spaces.Dict({"observation": Box(...), "action_mask": Box(...)})``.
    Downstream MLPs need the plain ``Box`` half.
    """
    if isinstance(raw_space, spaces.Dict) and "observation" in raw_space.spaces:
        return raw_space.spaces["observation"]
    return raw_space


def _split_observation(raw_obs, action_space) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Split a possibly-masked raw observation into ``(obs_array, mask)``.

    Args:
        raw_obs: Output of ``aec_env.observe(agent)``. Either a plain
            array/tensor or the classic-games dict
            ``{"observation": arr, "action_mask": mask}``.
        action_space: Agent's action space; used to synthesise an
            all-ones mask when the env doesn't provide one but the
            action space is ``Discrete`` (so downstream mask-aware
            heads still see a valid mask shape).

    Returns:
        ``(array, mask_or_None)``.
    """
    if isinstance(raw_obs, dict) and "observation" in raw_obs:
        arr = np.asarray(raw_obs["observation"])
        mask = raw_obs.get("action_mask")
        mask = None if mask is None else np.asarray(mask)
        return arr, mask
    return np.asarray(raw_obs), None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _TurnBasedAdapterBase(ParallelEnv):
    """Shared state / space-exposure logic for the two adapters."""

    def __init__(self, aec_env: AECEnv, *, env_name: str):
        self.aec_env = aec_env
        self._env_name = env_name
        self.possible_agents: List[str] = list(aec_env.possible_agents)
        self.agents: List[str] = []
        # Keep PettingZoo's ``metadata`` visible to tooling.
        self.metadata = dict(getattr(aec_env, "metadata", {}) or {})

    # -- Space exposure -----------------------------------------------------

    def observation_space(self, agent: str):
        return _unwrap_obs_space(self.aec_env.observation_space(agent))

    def action_space(self, agent: str):
        return self.aec_env.action_space(agent)

    # -- Utility ------------------------------------------------------------

    def _coerce_action(self, action: Any) -> Any:
        """Convert numpy arrays of length 1 / scalars into the int the
        wrapped AEC env expects for ``Discrete`` action spaces."""
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return action.item()
            if action.size == 1:
                return int(action.item())
            # Multi-element: likely a logit vector — argmax.
            return int(np.argmax(action))
        return action

    def _observe(self, agent: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raw = self.aec_env.observe(agent)
        return _split_observation(raw, self.aec_env.action_space(agent))

    def render(self):
        return self.aec_env.render()

    def close(self):
        return self.aec_env.close()

    @property
    def unwrapped(self):  # pragma: no cover - convenience passthrough
        return self.aec_env


class MaskedSimultaneousAdapter(_TurnBasedAdapterBase):
    """Turn-based AEC env → parallel-shaped facade.

    Every ``reset`` / ``step`` returns dicts keyed by all currently-active
    agents. At each ``step`` only the current player's action is executed
    by the wrapped AEC env; other agents' actions are silently discarded.
    The action mask for the current player is populated from upstream;
    non-current agents receive an all-zero mask to signal "no-op turn".
    """

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        self.aec_env.reset(seed=seed, options=options)
        self.agents = list(self.aec_env.agents)
        return self._snapshot()

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError(
                f"MaskedSimultaneousAdapter({self._env_name!r}).step called "
                "after episode terminated; call reset() first."
            )
        current = self.aec_env.agent_selection
        if current not in actions:
            raise ValueError(
                f"MaskedSimultaneousAdapter({self._env_name!r}): no action "
                f"provided for current player {current!r}; got keys "
                f"{sorted(actions.keys())}."
            )
        action = self._coerce_action(actions[current])
        self.aec_env.step(action)
        # AEC envs accumulate into ``rewards`` dict per-step.
        rewards = {a: float(self.aec_env.rewards.get(a, 0.0)) for a in self.agents}
        terminations = {a: bool(self.aec_env.terminations.get(a, False)) for a in self.agents}
        truncations = {a: bool(self.aec_env.truncations.get(a, False)) for a in self.agents}
        # Now advance our view of active agents.
        self.agents = list(self.aec_env.agents)
        obs_dict, infos = self._snapshot_obs()
        # Preserve terminal entries for agents that *just* terminated.
        for a in list(rewards.keys()):
            if terminations[a] or truncations[a]:
                # Keep them in the return dicts one last time even if the
                # AEC env has already dropped them from ``.agents``.
                continue
        return obs_dict, rewards, terminations, truncations, infos

    def _snapshot(self):
        obs_dict, infos = self._snapshot_obs()
        return obs_dict, infos

    def _snapshot_obs(self):
        current = getattr(self.aec_env, "agent_selection", None)
        obs_dict: Dict[str, np.ndarray] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        for agent in self.agents:
            arr, mask = self._observe(agent)
            obs_dict[agent] = arr
            info: Dict[str, Any] = {}
            if mask is not None and agent == current:
                info["action_mask"] = mask
            elif mask is not None:
                # Waiting agents: all-zero mask signals "no-op this turn".
                info["action_mask"] = np.zeros_like(mask)
            infos[agent] = info
        return obs_dict, infos


class SequentialTurnAdapter(_TurnBasedAdapterBase):
    """Turn-based AEC env → single-agent-per-step facade.

    ``reset`` / ``step`` return dicts of length 1 keyed by the currently-
    acting player. When the episode terminates, the final ``step`` return
    carries reward / termination / truncation entries for every
    ``possible_agent`` so callers can read terminal rewards for all
    sides.
    """

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        self.aec_env.reset(seed=seed, options=options)
        current = self.aec_env.agent_selection
        self.agents = [current] if current is not None else []
        if current is None:
            return {}, {}
        arr, mask = self._observe(current)
        info: Dict[str, Any] = {}
        if mask is not None:
            info["action_mask"] = mask
        return {current: arr}, {current: info}

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError(
                f"SequentialTurnAdapter({self._env_name!r}).step called "
                "after episode terminated; call reset() first."
            )
        current_pre = self.aec_env.agent_selection
        if current_pre in actions:
            action = actions[current_pre]
        elif len(actions) == 1:
            action = next(iter(actions.values()))
        else:
            raise ValueError(
                f"SequentialTurnAdapter({self._env_name!r}): expected an "
                f"action for current player {current_pre!r}; got keys "
                f"{sorted(actions.keys())}."
            )
        action = self._coerce_action(action)
        self.aec_env.step(action)

        # Episode ended?
        remaining = list(self.aec_env.agents)
        if not remaining:
            # Final step — surface all agents' terminal bookkeeping.
            rewards = {
                a: float(self.aec_env.rewards.get(a, 0.0))
                for a in self.aec_env.possible_agents
            }
            terminations = {
                a: bool(self.aec_env.terminations.get(a, False))
                for a in self.aec_env.possible_agents
            }
            truncations = {
                a: bool(self.aec_env.truncations.get(a, False))
                for a in self.aec_env.possible_agents
            }
            self.agents = []
            return {}, rewards, terminations, truncations, {}

        # Normal turn: advance to the next player.
        next_agent = self.aec_env.agent_selection
        arr, mask = self._observe(next_agent)
        info: Dict[str, Any] = {}
        if mask is not None:
            info["action_mask"] = mask
        rewards = {current_pre: float(self.aec_env.rewards.get(current_pre, 0.0))}
        terminations = {current_pre: bool(self.aec_env.terminations.get(current_pre, False))}
        truncations = {current_pre: bool(self.aec_env.truncations.get(current_pre, False))}
        # Also surface next_agent's bookkeeping so the returned dicts align
        # with the obs dict.
        rewards.setdefault(next_agent, 0.0)
        terminations.setdefault(next_agent, False)
        truncations.setdefault(next_agent, False)
        self.agents = [next_agent]
        return {next_agent: arr}, rewards, terminations, truncations, {next_agent: info}


__all__ = [
    "MaskedSimultaneousAdapter",
    "SequentialTurnAdapter",
]
