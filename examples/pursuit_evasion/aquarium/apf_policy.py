"""APF (Artificial Potential Field) evader policy for ``marl_aquarium.aquarium_v0``.

The policy is a deterministic, scripted prey ("fish") controller that
flees from observed predators ("sharks").  It mirrors the MPE APF
controller in :mod:`deap_pettingzoo.contrib.pursuit_evasion.apf_policy`
but is rewritten against the aquarium observation layout, which differs
in slice shape and field semantics.

Aquarium observation layout (introspected from
``marl_aquarium/env/aquarium.py``):

* For a prey agent (``prey_K``) the observation vector has length
  ``5 + predator_observe_count * 6 + prey_observe_count * 6``::

      [ self_block (5)        ] [ predator_block (P*6) ] [ prey_block (Q*6) ]

  where:

  * **self_block** = ``[type=1, sx, sy, sdir, sspd]`` — type=1 marks prey,
    ``sx, sy`` is the observer's own position scaled to ``[0, 1]``,
    ``sdir`` the orientation in ``[0, 1]`` (raw degrees in
    ``[-180, 180]`` mapped via ``(d + 180) / 360``), ``sspd`` the speed
    in ``[0, 1]``.
  * Each **nearby entry** (``predator_block`` and ``prey_block``) is a
    flat run of ``6 * count`` floats; each entry is
    ``[type, sx, sy, sdistance, sdirection, sspeed]``, where:

    * ``type=0`` → predator (shark), ``type=1`` → prey (fish);
    * ``sx, sy`` is the target's *absolute* position scaled to ``[0, 1]``;
    * ``sdistance`` is the in-torus distance from observer to target,
      scaled by the observer's view-distance to ``[0, 1]``;
    * ``sdirection`` is the direction angle (degrees, observer→target,
      torus-aware) in ``[-180, 180]`` mapped via ``(d + 180) / 360`` to
      ``[0, 1]``.  ``0`` degrees points up, ``+90`` right, ``-90`` left,
      consistent with the action convention below.
    * ``sspeed`` is the target's speed in ``[0, 1]``.

Aquarium action mapping
(``raw_env.get_desired_velocity_from_action`` →
``utils.get_vector_from_action``)::

    angle_deg = (360 / action_count) * action
    desired_velocity = Vector(sin(rad), -cos(rad))

So action ``0`` heads ``(0, -1)`` ("up" in screen coordinates),
action ``action_count // 4`` heads ``(+1, 0)`` (right), action
``action_count // 2`` heads ``(0, +1)`` (down).  Direction angles are
measured the same way (``0°`` = up, ``+90°`` = right, ``-90°`` = left,
``±180°`` = down), so the angle reported by the env's ``sdirection``
field can be quantised by simple ``angle / (360 / action_count)``
indexing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AquariumAPFConfig:
    """Hyperparameters for :class:`AquariumAPFEvaderPolicy`.

    All gains operate on the env's normalised distance scale (``[0, 1]``),
    which is the in-torus distance divided by the prey's view distance.
    """

    k_predator: float = 1.0
    """Gain on inverse-square repulsion from each visible predator."""

    k_prey_separation: float = 0.0
    """Optional gain on inverse-square repulsion from teammate prey
    (default 0 — pure flight policy, no flock cohesion)."""

    softening: float = 1e-2
    """Constant added to scaled-distance squared to avoid singularities."""

    no_op_threshold: float = 1e-3
    """If the resulting force magnitude is below this, the policy emits
    a forward-keeping action equal to the closest match for the prey's
    *current* heading (best effort: action 0 if no self-direction info)."""


def aquarium_obs_layout(
    predator_observe_count: int,
    prey_observe_count: int,
) -> Dict[str, slice]:
    """Return per-block slices for a prey agent's observation vector.

    Args:
        predator_observe_count: Same as the env kwarg of the same name —
            how many predators each prey may observe in one frame.
        prey_observe_count: Same as the env kwarg.

    Returns:
        Dict of named slices into the prey observation vector::

            {
                "self":      slice(0, 5),
                "predators": slice(5, 5 + 6*P),
                "prey":      slice(5 + 6*P, 5 + 6*P + 6*Q),
            }
    """
    p = int(predator_observe_count)
    q = int(prey_observe_count)
    self_end = 5
    pred_end = self_end + 6 * p
    prey_end = pred_end + 6 * q
    return {
        "self": slice(0, self_end),
        "predators": slice(self_end, pred_end),
        "prey": slice(pred_end, prey_end),
    }


def _direction_to_unit_vector(scaled_direction: float) -> Tuple[float, float]:
    """Convert a ``sdirection ∈ [0, 1]`` field back to a 2D unit vector.

    Uses the same ``Vector(sin(rad), -cos(rad))`` convention as the env's
    action decoder, so the resulting vector is directly comparable to
    the action heading vectors.
    """
    angle_deg = float(scaled_direction) * 360.0 - 180.0
    rad = math.radians(angle_deg)
    return math.sin(rad), -math.cos(rad)


def _action_for_heading(target_x: float, target_y: float, action_count: int) -> int:
    """Return the discrete action whose heading-vector best matches the
    *target* direction ``(target_x, target_y)``.

    Uses the env's heading convention::

        action k → (sin(2π k / N), -cos(2π k / N))
    """
    angle_rad = math.atan2(target_x, -target_y)  # inverse of (sin, -cos)
    # Map atan2 result (-π, π] → [0, 2π) then quantise to nearest action bin.
    if angle_rad < 0:
        angle_rad += 2.0 * math.pi
    bin_size = 2.0 * math.pi / action_count
    return int(round(angle_rad / bin_size)) % int(action_count)


class AquariumAPFEvaderPolicy:
    """Scripted prey policy: flee from observed predators.

    The policy is *stateless* — :meth:`reset` is a no-op.  It is
    instantiated once per prey agent and called as ``get_action(obs)`` per
    step.

    Args:
        predator_observe_count: Env kwarg of the same name, used to size
            the predator block in the observation vector.
        prey_observe_count: Env kwarg of the same name, used to size the
            prey block.
        action_count: Env kwarg ``action_count`` (default 16).  Quantises
            the resultant force vector onto one of these evenly spaced
            heading directions.
        config: :class:`AquariumAPFConfig` instance (default values used
            if omitted).
    """

    def __init__(
        self,
        predator_observe_count: int,
        prey_observe_count: int,
        action_count: int = 16,
        config: Optional[AquariumAPFConfig] = None,
    ) -> None:
        self.predator_observe_count = int(predator_observe_count)
        self.prey_observe_count = int(prey_observe_count)
        self.action_count = int(action_count)
        self.config = config or AquariumAPFConfig()
        self._layout = aquarium_obs_layout(
            self.predator_observe_count, self.prey_observe_count
        )

    # PolicyProtocol -------------------------------------------------------

    def reset(self) -> None:
        return None

    def get_action(self, observation, deterministic: bool = True) -> int:
        del deterministic  # APF is deterministic by construction.
        obs = np.asarray(observation, dtype=np.float64).ravel()
        # Aquarium pads observation with zeros when fewer than P / Q
        # nearby entities are present (see ``raw_env.*_observations``).
        # A zero entry has type=0, distance=0 → would look like a
        # collocated predator without a guard.  We use the (type, x, y,
        # distance) tuple to discriminate "real" entries: a *real*
        # nearby entry has distance > 0 OR a non-zero position.  The
        # observer-self field (``type, sx, sy, sdir, sspd``) is always
        # populated, so we don't need the same guard there.
        force_x, force_y = self._compute_force(obs)
        magnitude = math.hypot(force_x, force_y)

        if magnitude < self.config.no_op_threshold:
            # No predators in view → keep current heading.  Fall back to
            # action 0 if no self-direction is available.
            self_block = obs[self._layout["self"]]
            if self_block.size >= 4:
                heading_x, heading_y = _direction_to_unit_vector(self_block[3])
                return _action_for_heading(heading_x, heading_y, self.action_count)
            return 0

        # Quantise the resultant *flee* force onto a discrete heading.
        return _action_for_heading(force_x, force_y, self.action_count)

    # Internals ------------------------------------------------------------

    def _compute_force(self, obs: np.ndarray) -> Tuple[float, float]:
        cfg = self.config
        fx = 0.0
        fy = 0.0

        # Predator block — repulsion (flee).
        pred_block = obs[self._layout["predators"]]
        for entry in self._iter_entries(pred_block, expected_type=0):
            if entry is None:
                continue
            sdistance, sdirection = entry  # both in [0, 1]
            ux, uy = _direction_to_unit_vector(sdirection)
            # ``sdirection`` points observer → predator.  Flee = -unit.
            inv_sq = 1.0 / (sdistance * sdistance + cfg.softening)
            fx -= cfg.k_predator * ux * inv_sq
            fy -= cfg.k_predator * uy * inv_sq

        # Optional separation from other prey (default off).
        if cfg.k_prey_separation > 0.0:
            prey_block = obs[self._layout["prey"]]
            for entry in self._iter_entries(prey_block, expected_type=1):
                if entry is None:
                    continue
                sdistance, sdirection = entry
                ux, uy = _direction_to_unit_vector(sdirection)
                inv_sq = 1.0 / (sdistance * sdistance + cfg.softening)
                fx -= cfg.k_prey_separation * ux * inv_sq
                fy -= cfg.k_prey_separation * uy * inv_sq

        return fx, fy

    @staticmethod
    def _iter_entries(block: np.ndarray, expected_type: int) -> List[Optional[Tuple[float, float]]]:
        """Yield ``(sdistance, sdirection)`` for non-padding entries in *block*.

        Entries are 6-tuples ``[type, sx, sy, sdistance, sdirection, sspeed]``.
        Padding (no entity present) is all-zeros, so ``(sx, sy, sdistance)
        == (0, 0, 0)`` is filtered out.
        """
        out: List[Optional[Tuple[float, float]]] = []
        n = block.size // 6
        for i in range(n):
            base = i * 6
            entry_type = block[base]
            sx = block[base + 1]
            sy = block[base + 2]
            sdistance = block[base + 3]
            sdirection = block[base + 4]
            # Padding heuristic: all of (sx, sy, sdistance) zero.
            if abs(sx) < 1e-9 and abs(sy) < 1e-9 and abs(sdistance) < 1e-9:
                out.append(None)
                continue
            # Type-mismatch defensive guard (should never trip in practice).
            if expected_type is not None and int(round(float(entry_type))) != int(expected_type):
                # Entry does not match expected type — skip rather than
                # silently mis-classify.
                out.append(None)
                continue
            out.append((float(sdistance), float(sdirection)))
        return out
