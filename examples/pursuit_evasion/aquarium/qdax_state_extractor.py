"""marl-aquarium ``state_extractor`` for the QDax numpy scoring path.

Experiment-side plug-in for
:func:`deap_pettingzoo.contrib.qdax.scoring_numpy.make_numpy_scoring_function`
(``algorithm_params["state_extractor"]``).  The default extractor reads
``obs[2:4]`` as positions, which is meaningless for aquarium observations
(they encode relative bearings, not absolute coordinates) — every genotype
then receives a near-identical descriptor and the MAP-Elites archive
degenerates to a handful of cells.

:class:`AquariumStateExtractor` instead declares ``needs_env = True`` so the
scoring rollout hands it the live env wrapper; it walks down to the raw
aquarium env and reads true entity positions, normalised to ``[0, 1]²`` by
the arena dimensions (the same coordinate space as
:func:`~examples.pursuit_evasion.aquarium.qdax_position_extractor.make_aquarium_extractor`
uses for co-evolution, so one set of calibrated descriptor ranges serves
both paths).

Rows are ordered ``role_order[0]`` entities first and the row count stays
fixed for the whole episode: entities removed mid-episode (caught prey with
``keep_prey_count_constant=False``) keep their last observed position.  A
frozen catch position is exactly what the formation/engagement descriptors
should see — it marks where the interaction happened, while a fixed shape is
required for the trajectory stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .qdax_position_extractor import _AquariumExtractor


class AquariumStateExtractor:
    """Picklable ``(obs, agents, env) -> (positions, velocities)`` extractor.

    Args:
        arena_width: Horizontal arena extent in pixels.
        arena_height: Vertical arena extent in pixels.
        predator_count: Expected predator entity count (full complement).
        prey_count: Expected prey entity count.
        role_order: Role listed first in the returned rows.  Use
            ``("predator", "prey")`` when predators are the learner side
            and ``("prey", "predator")`` for evader training, matching the
            ordering the descriptor was built for.
    """

    # Signals scoring_numpy._call_state_extractor to pass the env wrapper.
    needs_env: bool = True

    def __init__(
        self,
        arena_width: float,
        arena_height: float,
        predator_count: int,
        prey_count: int,
        role_order: Tuple[str, str] = ("predator", "prey"),
    ) -> None:
        self._w = float(arena_width)
        self._h = float(arena_height)
        self._expected = {
            "predator": int(predator_count),
            "prey": int(prey_count),
        }
        self._role_order = tuple(role_order)
        # entity-id -> last normalised (x, y); rebuilt on full complement.
        self._last_xy: Dict[str, Tuple[float, float]] = {}
        self._id_order: List[str] = []

    def __call__(
        self,
        obs_dict: Dict[str, np.ndarray],
        agents: Sequence[str],
        env: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del obs_dict, agents
        raw = _AquariumExtractor._walk_to_raw(env)
        n_total = sum(self._expected.values())
        if raw is None:
            # No raw env reachable — emit a stable all-zero frame rather
            # than crashing the rollout.
            zeros = np.zeros((n_total, 2), dtype=np.float32)
            return zeros, zeros.copy()

        live: Dict[str, List[Any]] = {
            "predator": list(getattr(raw, "predators", [])),
            "prey": list(getattr(raw, "prey", [])),
        }

        # Full complement (episode start, or no deaths yet) -> rebuild the
        # row-identity order and drop any stale cache from a previous episode.
        if all(len(live[role]) == self._expected[role] for role in self._role_order):
            self._id_order = []
            self._last_xy = {}
            for role in self._role_order:
                for entity in live[role]:
                    self._id_order.append(self._entity_id(entity))

        # Update last-known positions for everything still alive.
        for role in self._role_order:
            for entity in live[role]:
                xy = _AquariumExtractor._position_xy(getattr(entity, "position", None))
                if xy is not None:
                    self._last_xy[self._entity_id(entity)] = (
                        xy[0] / self._w,
                        xy[1] / self._h,
                    )

        rows = [self._last_xy.get(eid, (0.0, 0.0)) for eid in self._id_order]
        while len(rows) < n_total:
            rows.append((0.0, 0.0))
        positions = np.asarray(rows[:n_total], dtype=np.float32)
        # Descriptors in this project only consume positions; a zero
        # velocity channel keeps the (positions, velocities) contract.
        velocities = np.zeros_like(positions)
        return positions, velocities

    @staticmethod
    def _entity_id(entity: Any) -> str:
        ident = getattr(entity, "id", None)
        if callable(ident):
            try:
                return str(ident())
            except Exception:
                pass
        return f"obj_{id(entity)}"

    def __repr__(self) -> str:
        return (
            f"AquariumStateExtractor({self._w}, {self._h}, "
            f"predators={self._expected['predator']}, prey={self._expected['prey']}, "
            f"role_order={self._role_order!r})"
        )


__all__ = ["AquariumStateExtractor"]
