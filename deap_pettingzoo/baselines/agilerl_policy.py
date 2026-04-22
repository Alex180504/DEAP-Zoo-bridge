"""
Policy wrapper for trained AgileRL multi-agent algorithms.

AgileRL algorithms expose :meth:`get_action` that takes a dict mapping
``agent_id -> observation`` and returns per-agent actions. The unified
baseline API, by contrast, is a stateless-looking :class:`PolicyProtocol`
that receives one observation at a time with no agent id.

:class:`AgileRLPolicyWrapper` bridges the two by tracking a rotation
counter through ``possible_agents`` that is reset at the beginning of
each episode. This works because :class:`UnifiedEvaluator` iterates
``env.possible_agents`` in a fixed order and calls ``get_action`` once
per agent per env step.
"""

from typing import Any, Iterable, Optional

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
        self._algorithm_name = algorithm_name or type(agent).__name__

    # --- PolicyProtocol ----------------------------------------------------

    def get_action(self, observation: np.ndarray) -> Any:
        """Return a deterministic action for the next agent in rotation."""
        agent_id = self._agent_ids[self._call_index % len(self._agent_ids)]
        self._call_index += 1

        obs_dict = {agent_id: np.asarray(observation)}
        try:
            result = self._agent.get_action(obs_dict, training=False)
        except TypeError:
            # On-policy algorithms may use a different kwarg name.
            result = self._agent.get_action(obs_dict)

        actions = result[0] if isinstance(result, tuple) else result
        action = np.asarray(actions[agent_id])

        # Strip a leading vectorization dimension of size 1, if present.
        if action.ndim > 0 and action.shape[0] == 1:
            action = action[0]
        return action

    def reset(self) -> None:
        """Reset the agent rotation counter for a new episode."""
        self._call_index = 0

    # --- Introspection helpers --------------------------------------------

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
