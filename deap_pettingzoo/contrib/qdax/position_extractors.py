"""Built-in position extractors for :class:`QDaxSideState` trajectory collection.

A *position extractor* is any callable with the signature::

    extractor(obs_dict: Dict[str, np.ndarray], env_wrapper: Any) -> Optional[np.ndarray]

It is called once per environment step (before :meth:`env.step`) with the
current observation dict and the :class:`PettingZooWrapper`.  It should return
an ``(N, 2)`` float32 array where ``N`` is the total number of agents
and each row is the ``(x, y)`` world position
of one agent, or ``None`` to skip this timestep.

The extracted positions are stacked across steps into a ``(T, N, 2)`` array
and passed to the QDax descriptor function.

Two built-in extractors are provided:

* :func:`obs_first_two` - reads the first two values from each agent's
  observation vector as ``(x, y)``.

* :func:`obs_indices` - returns a picklable callable reading ``obs[start:stop]``
  as ``(x, y)``.  Safe for parallel workers via ``ProcessPoolExecutor``.

For environments whose observation vectors do not encode raw positions
(e.g. marl-aquarium), supply a custom callable in
``optimizer_config["position_extractor"]``.  If the callable will be sent to
worker processes it **must** be picklable (a module-level function or a class
instance -- not a lambda or nested-function closure).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# Generic obs-based extractor


def obs_first_two(
    obs_dict: Dict[str, np.ndarray],
    env_wrapper: Any,
) -> Optional[np.ndarray]:
    """Extract ``(x, y)`` from the first two observation dimensions.

    Returns an ``(N, 2)`` array in :attr:`env_wrapper.possible_agents`
    order.  If any agent is absent from *obs_dict* (done mid-episode)
    the step is skipped (returns ``None``).

    .. note::
        For MPE environments the raw world position is typically at
        ``obs[2:4]`` (after ``vel_x, vel_y``).  If that is the case for
        a given env, use :func:`obs_indices` instead::

            from deap_pettingzoo.contrib.qdax.position_extractors import obs_indices
            extractor = obs_indices(2, 4)
    """
    agents = list(env_wrapper.possible_agents)
    rows = []
    for agent in agents:
        obs = obs_dict.get(agent)
        if obs is None:
            return None  # agent already done; skip this step
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        rows.append(obs_arr[:2] if len(obs_arr) >= 2 else np.zeros(2, dtype=np.float32))
    if not rows:
        return None
    return np.stack(rows, axis=0)


class _ObsIndicesExtractor:
    """Picklable extractor that reads ``obs[start:stop]`` as ``(x, y)``.

    Use :func:`obs_indices` to construct instances.
    """

    def __init__(self, start: int, stop: int) -> None:
        self.start = int(start)
        self.stop = int(stop)

    def __call__(
        self,
        obs_dict: Dict[str, np.ndarray],
        env_wrapper: Any,
    ) -> Optional[np.ndarray]:
        agents = list(env_wrapper.possible_agents)
        rows = []
        for agent in agents:
            obs = obs_dict.get(agent)
            if obs is None:
                return None
            obs_arr = np.asarray(obs, dtype=np.float32).ravel()
            chunk = obs_arr[self.start : self.stop]
            if len(chunk) < 2:
                chunk = np.zeros(2, dtype=np.float32)
            rows.append(chunk[:2])
        if not rows:
            return None
        return np.stack(rows, axis=0)

    def __repr__(self) -> str:
        return f"obs_indices({self.start}, {self.stop})"


def obs_indices(start: int, stop: int) -> "_ObsIndicesExtractor":
    """Return a picklable extractor that reads ``obs[start:stop]`` as ``(x, y)``.

    Example: MPE positions at indices 2-3:

        extractor = obs_indices(2, 4)
        Side(optimizer_config={"position_extractor": extractor})

    The returned callable has the standard
    ``(obs_dict, env_wrapper) -> Optional[np.ndarray]`` signature and
    is safe to ship to worker processes via ``ProcessPoolExecutor``.
    """
    return _ObsIndicesExtractor(start, stop)


__all__ = [
    "obs_first_two",
    "obs_indices",
]
