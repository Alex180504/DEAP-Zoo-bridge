"""Adapter presenting a JAX-native multi-agent env as a PettingZoo parallel env.

Primary use case: let numpy / torch trainers (``EATrainer``,
``CMAESTrainer``, ``AgileRLTrainer``) drive environments from
`JaxMARL <https://github.com/FLAIROx/JaxMARL>`_ without changing any
downstream code.  Per-step numpy<->JAX orchestration is applied, which
makes this adapter *slow* (roughly at the speed of a native PettingZoo MPE env);
the fast QDax training path bypasses this adapter entirely via
:mod:`deap_pettingzoo.contrib.qdax.scoring_jax`.

The adapter expects the underlying env to satisfy :class:`JaxEnvProtocol`
(a structural type modelled after the JaxMARL ``MultiAgentEnv`` API).
It is intentionally permissive about return-tuple arities so JaxMARL's
minor API drifts across releases do not break the bridge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from pettingzoo.utils.env import ParallelEnv


@runtime_checkable
class JaxEnvProtocol(Protocol):
    """Structural type for a JaxMARL-compatible multi-agent env.

    The adapter only relies on the attributes and methods listed here.
    Every JaxMARL environment satisfies this shape.
    """

    agents: List[str]

    def observation_space(self, agent: str): ...
    def action_space(self, agent: str): ...
    def reset(self, key) -> Tuple[Dict[str, Any], Any]: ...
    def step(
        self, key, state: Any, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any], Any, Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]: ...


def _to_numpy_dict(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a dict of JAX arrays (or anything array-like) to numpy."""
    return {k: np.asarray(v) for k, v in d.items()}


class JaxEnvToPettingZooAdapter(ParallelEnv):
    """Wrap a JaxMARL-style env as a PettingZoo :class:`ParallelEnv`.

    The adapter owns a ``jax.random.PRNGKey`` internally and splits it
    on every ``reset`` / ``step`` call, so rollouts are deterministic
    given a fixed construction seed.  ``reset(seed=...)`` reseeds the
    internal key; otherwise the key advances monotonically.

    Args:
        jax_env: A JaxMARL-style env (see :class:`JaxEnvProtocol`).
        env_name: Identifier surfaced in error messages.
        seed: Initial PRNG seed.  Defaults to ``0``.
        render_mode: Forwarded to :meth:`render`; not used by the
            underlying JAX env in most cases.
    """

    metadata: Dict[str, Any] = {
        "name": "jax_parallel_adapter",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        jax_env: JaxEnvProtocol,
        *,
        env_name: str = "jax_parallel",
        seed: int = 0,
        render_mode: Optional[str] = None,
    ) -> None:
        # Lazy JAX import so the core package stays JAX-free.
        import jax

        self._jax = jax
        self._env = jax_env
        self._env_name = env_name
        self._render_mode = render_mode
        self._state: Any = None
        self._key = self._jax.random.PRNGKey(int(seed))
        self.possible_agents: List[str] = list(jax_env.agents)
        self.agents: List[str] = list(self.possible_agents)
        # Respect env-specific metadata if provided.
        env_meta = dict(getattr(jax_env, "metadata", {}) or {})
        merged = dict(self.metadata)
        merged.update(env_meta)
        merged.setdefault("name", env_name)
        merged.setdefault("is_parallelizable", True)
        self.metadata = merged

    # Space exposure

    def observation_space(self, agent: str):
        return self._env.observation_space(agent)

    def action_space(self, agent: str):
        return self._env.action_space(agent)

    # Lifecycle

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._key = self._jax.random.PRNGKey(int(seed))
        self._key, sub = self._jax.random.split(self._key)
        obs, self._state = self._env.reset(sub)
        self.agents = list(self.possible_agents)
        return _to_numpy_dict(obs), {agent: {} for agent in self.agents}

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
        if self._state is None:
            raise RuntimeError(
                f"JaxEnvToPettingZooAdapter({self._env_name!r}) stepped "
                "before reset()."
            )
        self._key, sub = self._jax.random.split(self._key)
        jax_actions = {k: self._jax.numpy.asarray(v) for k, v in actions.items()}
        obs, self._state, rewards, dones, infos = self._env.step(
            sub, self._state, jax_actions
        )
        obs_np = _to_numpy_dict(obs)
        rewards_np: Dict[str, float] = {
            k: float(np.asarray(v)) for k, v in rewards.items()
        }
        # JaxMARL collapses per-agent done into a ``__all__`` key + per-agent
        # flags.  ``__truncated__`` signals a time-limit truncation rather than
        # a natural termination (some JaxMARL envs set this in ``dones``).
        global_done = bool(np.asarray(dones.get("__all__", False)))
        global_truncated = bool(np.asarray(dones.get("__truncated__", False)))
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        for agent in self.agents:
            agent_done = bool(np.asarray(dones.get(agent, global_done)))
            # When the episode ends via time limit, classify as truncation.
            truncations[agent] = agent_done and global_truncated
            terminations[agent] = agent_done and not global_truncated
        infos_np: Dict[str, Dict[str, Any]] = {
            agent: dict(infos.get(agent, {})) for agent in self.agents
        }
        # Prune agents that are individually done from obs/infos (PettingZoo
        # parallel contract: done agents must not appear in the observation
        # dict even when the global episode is still running).
        # Snapshot required: self.agents is mutated inside the loop.
        active_snapshot = self.agents[:]
        for agent in active_snapshot:
            if terminations[agent] or truncations[agent]:
                self.agents.remove(agent)
                obs_np.pop(agent, None)
                infos_np.pop(agent, None)
        if not self.agents:
            return {}, rewards_np, terminations, truncations, {}
        return obs_np, rewards_np, terminations, truncations, infos_np

    def render(self):
        render_fn = getattr(self._env, "render", None)
        if render_fn is None:
            return None
        try:
            return render_fn(self._state)
        except TypeError:
            return render_fn()

    def close(self):
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


__all__ = ["JaxEnvProtocol", "JaxEnvToPettingZooAdapter"]
