"""
Policy wrapper for trained AgileRL multi-agent algorithms.

AgileRL algorithms expose :meth:`get_action` that takes a dict mapping
``agent_id -> observation`` and returns per-agent actions. The unified
baseline API, by contrast, is a stateless-looking :class:`PolicyProtocol`
that receives one observation at a time with no agent id.

:class:`AgileRLPolicyWrapper` bridges the two by:

* tracking a rotation counter through ``possible_agents`` that is reset
  at the beginning of each episode (relies on :class:`UnifiedEvaluator`
  iterating ``env.possible_agents`` in a fixed order and calling
  ``get_action`` once per agent per env step), and
* maintaining a per-agent observation cache so that every call can pass
  the full ``{agent_id: obs}`` dict required by multi-agent AgileRL
  algorithms. Since MADDPG/MATD3/IPPO use decentralized actors, the
  returned action for the current agent depends only on that agent's
  obs; stale entries for the other agents are harmless and get refreshed
  as the rotation progresses through the step.
"""

from typing import Any, Dict, Iterable, Optional

import numpy as np


class AgileRLPolicyWrapper:
    """Adapt a trained AgileRL multi-agent algorithm to :class:`PolicyProtocol`.

    Args:
        agent: Trained AgileRL algorithm (e.g. ``MADDPG``, ``MATD3``,
            ``IPPO``) exposing a ``get_action(obs_dict, training=...)``.
        agent_ids: PettingZoo ``possible_agents`` list in evaluation
            order. ``get_action`` dispatches through this list round-robin.
        algorithm_name: Optional human-readable label (used only for
            ``repr``).
    """

    def __init__(
        self,
        agent: Any,
        agent_ids: Iterable[str],
        algorithm_name: Optional[str] = None,
    ) -> None:
        self._agent = agent
        self._agent_ids = list(agent_ids)
        if not self._agent_ids:
            raise ValueError("agent_ids must be non-empty")
        self._call_index = 0
        self._obs_cache: Dict[str, np.ndarray] = {}
        self._algorithm_name = algorithm_name or type(agent).__name__

    # Policy protocol

    def get_action(self, observation: np.ndarray, **_: Any) -> Any:
        """Return a deterministic action for the next agent in rotation.

        Accepts and ignores extra keyword arguments (e.g. ``deterministic``)
        so it can be dropped into evaluators that pass such flags.
        """
        agent_id = self._agent_ids[self._call_index % len(self._agent_ids)]
        self._call_index += 1

        self._obs_cache[agent_id] = np.asarray(observation)
        obs_dict = self._build_full_obs_dict()

        try:
            result = self._agent.get_action(obs_dict, training=False)
        except TypeError:
            # On-policy algorithms may use a different kwarg name.
            result = self._agent.get_action(obs_dict)

        actions = result[0] if isinstance(result, tuple) else result
        action = np.asarray(actions[agent_id])

        # Strip leading vectorization dims of size 1.
        while action.ndim > 0 and action.shape[0] == 1:
            action = action[0]
        if action.shape == ():
            return action.item()
        return action

    def reset(self) -> None:
        """Reset the agent rotation counter and obs cache for a new episode."""
        self._call_index = 0
        self._obs_cache = {}

    # Internal helpers

    def _build_full_obs_dict(self) -> Dict[str, np.ndarray]:
        """Return ``{agent_id: obs}`` with zero-filled placeholders for any
        agent whose observation has not yet been seen this episode.

        Required because AgileRL's multi-agent algorithms expect every
        agent they were constructed with to be keyed in the input dict
        — not just the subset this wrapper dispatches for. Placeholders
        only matter for decentralized-actor algorithms in the warm-up
        phase before the first full rotation completes; thereafter
        every slot holds the most recent real observation for that
        agent.
        """
        full: Dict[str, np.ndarray] = {}
        fallback_shape = next(iter(self._obs_cache.values())).shape
        fallback_dtype = next(iter(self._obs_cache.values())).dtype
        # Use the underlying algorithm's full agent roster so subsets still satisfy MADDPG/MATD3 "every agent keyed" precondition.
        for aid in self._full_agent_roster():
            if aid in self._obs_cache:
                full[aid] = self._obs_cache[aid]
            else:
                full[aid] = self._zero_obs_for(aid, fallback_shape, fallback_dtype)
        return full

    def _full_agent_roster(self) -> list:
        """Best-effort list of every agent the underlying algorithm
        expects in its input dict. Falls back to this wrapper's own
        agent_ids if nothing else is exposed.
        """
        roster = getattr(self._agent, "agent_ids", None)
        if isinstance(roster, (list, tuple)):
            return list(roster)
        spaces = getattr(self._agent, "observation_spaces", None) or getattr(
            self._agent, "possible_observation_spaces", None
        )
        if isinstance(spaces, dict):
            return list(spaces.keys())
        return list(self._agent_ids)

    def _zero_obs_for(
        self, agent_id: str, fallback_shape, fallback_dtype
    ) -> np.ndarray:
        """Best-effort zero observation for ``agent_id``.

        Uses the wrapped algorithm's declared observation spaces when
        available (dict in older AgileRL, list aligned with ``agent_ids``
        in >=2.6); otherwise falls back to the shape/dtype of an
        already-observed agent.
        """
        spaces = getattr(self._agent, "observation_spaces", None) or getattr(
            self._agent, "possible_observation_spaces", None
        )
        space = None
        if isinstance(spaces, dict) and agent_id in spaces:
            space = spaces[agent_id]
        elif isinstance(spaces, (list, tuple)):
            roster = getattr(self._agent, "agent_ids", None)
            if roster is not None and agent_id in list(roster):
                idx = list(roster).index(agent_id)
                if idx < len(spaces):
                    space = spaces[idx]
        if space is not None:
            shape = getattr(space, "shape", fallback_shape) or fallback_shape
            dtype = getattr(space, "dtype", fallback_dtype) or fallback_dtype
            return np.zeros(shape, dtype=dtype)
        return np.zeros(fallback_shape, dtype=fallback_dtype)

    # Introspection helpers

    @property
    def agent_ids(self) -> list:
        return list(self._agent_ids)

    @property
    def underlying_agent(self) -> Any:
        """The wrapped AgileRL algorithm instance."""
        return self._agent

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AgileRLPolicyWrapper(algorithm={self._algorithm_name!r}, "
            f"agent_ids={self._agent_ids!r})"
        )
