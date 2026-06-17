"""Built-in position extractors for :class:`QDaxSideState` trajectory collection.

A *position extractor* is any callable with the signature::

    extractor(obs_dict: Dict[str, np.ndarray], env_wrapper: Any) -> Optional[np.ndarray]

It is called once per environment step (before :meth:`env.step`) with the
current observation dict and the :class:`PettingZooWrapper`.  It should return
an ``(N, 2)`` float32 array where ``N`` is the total number of agents
(pursuers first, then evaders) and each row is the ``(x, y)`` world position
of one agent, or ``None`` to skip this timestep.

The extracted positions are stacked across steps into a ``(T, N, 2)`` array
and passed to the QDax descriptor function.

Three built-in extractors are provided:

* :func:`obs_first_two` - reads the first two values from each agent's
  observation vector as ``(x, y)``.

* :func:`obs_indices` - returns a picklable callable reading ``obs[start:stop]``
  as ``(x, y)``.  Safe for parallel workers via ``ProcessPoolExecutor``.

* :func:`make_aquarium_extractor` - reads raw entity positions from the
  marl-aquarium environment's internal state.  Returns a picklable callable
  safe for parallel workers.

For other environments, supply a custom callable in
``optimizer_config["position_extractor"]``.  If the callable will be sent to
worker processes it **must** be picklable (a module-level function or a class
instance -- not a lambda or nested-function closure).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Generic obs-based extractor
# ---------------------------------------------------------------------------


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
        your env, use :func:`obs_indices` instead::

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

    Example -- MPE positions at indices 2-3::

        extractor = obs_indices(2, 4)
        Side(optimizer_config={"position_extractor": extractor})

    The returned callable has the standard
    ``(obs_dict, env_wrapper) -> Optional[np.ndarray]`` signature and
    is safe to ship to worker processes via ``ProcessPoolExecutor``.
    """
    return _ObsIndicesExtractor(start, stop)


# ---------------------------------------------------------------------------
# marl-aquarium env extractor
# ---------------------------------------------------------------------------


class _AquariumExtractor:
    """Picklable position extractor for marl-aquarium environments.

    Use :func:`make_aquarium_extractor` to construct instances.
    """

    def __init__(
        self,
        arena_width: float = 2000.0,
        arena_height: float = 3000.0,
        role_order: tuple[str, str] = ("predator", "prey"),
    ) -> None:
        self._w = float(arena_width)
        self._h = float(arena_height)
        self._role_order = tuple(role_order)

    @staticmethod
    def _walk_to_raw(env_wrapper: Any) -> Any:
        """Descend wrapper chain until we find an object with 'predators'."""
        inner = env_wrapper
        for _ in range(16):
            if hasattr(inner, "predators") and hasattr(inner, "prey"):
                return inner
            found = False
            for attr in ("aec_env", "env", "unwrapped", "_env", "inner"):
                candidate = getattr(inner, attr, None)
                if candidate is not None and candidate is not inner:
                    inner = candidate
                    found = True
                    break
            if not found:
                break
        return None

    @staticmethod
    def _position_xy(pos: Any) -> Optional[tuple[float, float]]:
        """Return ``(x, y)`` for tuple/list/ndarray or aquarium Vector positions."""
        for x_attr, y_attr in (("x", "y"), ("_x", "_y")):
            if hasattr(pos, x_attr) and hasattr(pos, y_attr):
                return float(getattr(pos, x_attr)), float(getattr(pos, y_attr))
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _expected_role_counts(env_wrapper: Any) -> dict[str, int]:
        agents = list(getattr(env_wrapper, "possible_agents", []))
        return {
            "predator": sum(1 for agent in agents if str(agent).startswith("predator")),
            "prey": sum(1 for agent in agents if str(agent).startswith("prey")),
        }

    def __call__(
        self,
        obs_dict: Dict[str, np.ndarray],
        env_wrapper: Any,
    ) -> Optional[np.ndarray]:
        raw = self._walk_to_raw(env_wrapper)
        if raw is None:
            return None

        role_entities = {
            "predator": list(getattr(raw, "predators", [])),
            "prey": list(getattr(raw, "prey", [])),
        }
        predators = role_entities["predator"]
        prey = role_entities["prey"]

        if not predators and not prey:
            return None

        expected = self._expected_role_counts(env_wrapper)
        for role, count in expected.items():
            if count > 0 and len(role_entities[role]) != count:
                return None

        rows = []
        ordered_entities = []
        for role in self._role_order:
            ordered_entities.extend(role_entities.get(role, []))
        for entity in ordered_entities:
            pos = getattr(entity, "position", None)
            if pos is None:
                return None
            xy = self._position_xy(pos)
            if xy is None:
                return None
            x = xy[0] / self._w
            y = xy[1] / self._h
            rows.append([x, y])

        return np.array(rows, dtype=np.float32)

    def __repr__(self) -> str:
        return f"make_aquarium_extractor({self._w}, {self._h}, role_order={self._role_order!r})"


def make_aquarium_extractor(
    arena_width: float = 2000.0,
    arena_height: float = 3000.0,
    role_order: tuple[str, str] = ("predator", "prey"),
) -> "_AquariumExtractor":
    """Return a picklable position extractor for marl-aquarium environments.

    The extractor walks the wrapper chain to find the raw aquarium
    environment (which exposes ``predators`` and ``prey`` lists with a
    ``position`` attribute) and reads entity positions directly,
    normalising by the arena dimensions so all coordinates lie in
    ``[0, 1]``.

    By default, predators are placed first in the returned array. For
    coevolution runs where the learner side is prey, pass
    ``role_order=("prey", "predator")`` so the descriptor receives
    learner positions first and opponent positions second.

    Args:
        arena_width: Horizontal arena extent in pixels. Default ``2000``.
        arena_height: Vertical arena extent in pixels. Default ``3000``.
        role_order: Entity-role order in the returned array. Defaults to
            ``("predator", "prey")``.

    Returns:
        Picklable callable ``(obs_dict, env_wrapper) -> Optional[np.ndarray(N, 2)]``
        safe to ship to worker processes via ``ProcessPoolExecutor``.
    """
    return _AquariumExtractor(
        arena_width=arena_width,
        arena_height=arena_height,
        role_order=role_order,
    )


__all__ = [
    "obs_first_two",
    "obs_indices",
    "make_aquarium_extractor",
]
