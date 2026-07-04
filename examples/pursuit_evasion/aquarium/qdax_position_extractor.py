"""marl-aquarium position extractor for QDax trajectory collection.

This is *experiment-side* code: it reads raw entity positions from the
marl-aquarium environment's internal state and is **not** part of the generic
``deap_pettingzoo`` library. It plugs into :class:`QDaxSideState` via
``optimizer_config["position_extractor"]`` (a generic extension point of the
library) to drive behavioural descriptors from real per-step positions.

A *position extractor* is any callable with the signature::

    extractor(obs_dict: Dict[str, np.ndarray], env_wrapper: Any) -> Optional[np.ndarray]

It is called once per environment step (before ``env.step``) with the current
observation dict and the :class:`PettingZooWrapper`. It returns an ``(N, 2)``
float32 array (pursuers first, then evaders) of ``(x, y)`` world positions, or
``None`` to skip the timestep.

For environments whose observation vectors *do* encode raw positions, use the
generic extractors in
:mod:`deap_pettingzoo.contrib.qdax.position_extractors` (``obs_first_two`` /
``obs_indices``) instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


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
    "make_aquarium_extractor",
]
