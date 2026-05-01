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
    ``[0, 1]``.  Encoded by ``get_angle_from_vector`` = ``−atan2(vy, vx)``:
    ``0°`` = right (east), ``+90°`` = screen-up (north), ``−90°`` =
    screen-down (south).  **This differs from the action convention** (where
    ``0°`` = screen-up).
    * ``sspeed`` is the target's speed in ``[0, 1]``.

Aquarium action mapping
(``raw_env.get_desired_velocity_from_action`` →
``utils.get_vector_from_action``)::

    angle_deg = (360 / action_count) * action
    desired_velocity = Vector(sin(rad), -cos(rad))

So action ``0`` heads ``(0, -1)`` ("up" in screen coordinates),
action ``action_count // 4`` heads ``(+1, 0)`` (right), action
``action_count // 2`` heads ``(0, +1)`` (down).

**Important**: the action angle convention (``0°`` = up) differs from the
observation ``sdirection`` convention (``0°`` = right).  Do **not**
quantise ``sdirection`` angles directly to action indices; use
:func:`_direction_to_unit_vector` followed by :func:`_action_for_heading`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AquariumAPFConfig:
    """Hyperparameters for :class:`AquariumAPFEvaderPolicy`.

    All gains operate on the env's normalised distance scale (``[0, 1]``),
    which is the in-torus distance divided by the prey's view distance.
    """

    k_predator: float = 5.0
    """Gain on inverse-square repulsion from each visible predator."""

    k_tangential: float = 3.0
    """Gain on the tangential (perpendicular-dodge) component added alongside
    the radial flee force.  Each prey picks a random turn direction at episode
    start (:meth:`AquariumAPFEvaderPolicy.reset`) and keeps it for the whole
    episode, breaking the force-cancellation symmetry that arises when
    predators are evenly distributed around the prey."""

    k_prey_separation: float = 0.0
    """Optional gain on inverse-square repulsion from teammate prey
    (default 0 — pure flight policy, no flock cohesion)."""

    separation_cap_ratio: float = 0.3
    """Maximum magnitude of the prey-separation force as a fraction of the
    predator-flee force magnitude.  Without this cap, coincident teammates
    (``sdistance ≈ 0`` saturating the inverse-square term to
    ``1/softening``) produce a separation force ≫ the predator-flee force,
    and prey end up fleeing each other instead of the predator.  With this
    cap, separation is at most a heading nudge that prevents flock collapse,
    never the dominant force."""

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

    The env encodes directions with :func:`marl_aquarium.env.utils.get_angle_from_vector`,
    which is ``angle = −atan2(vy, vx) · 180/π``.  Inverting that gives the
    unit vector ``(cos(rad), −sin(rad))`` — **not** the action-heading
    convention ``(sin(rad), −cos(rad))``, which is offset by 90°.

    Convention: ``0°`` = right (east), ``+90°`` = screen-up (north),
    ``−90°`` = screen-down (south), ``±180°`` = left (west).
    """
    angle_deg = float(scaled_direction) * 360.0 - 180.0
    rad = math.radians(angle_deg)
    return math.cos(rad), -math.sin(rad)


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
        self._turn_sign: int = 1  # randomised on each reset()

    # PolicyProtocol -------------------------------------------------------

    def reset(self) -> None:
        self._turn_sign = random.choice((-1, 1))

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
        force_x, force_y, any_visible = self._compute_force(obs)
        magnitude = math.hypot(force_x, force_y)

        if magnitude < self.config.no_op_threshold and not any_visible:
            # Genuinely no predators in view → keep current heading.  Fall
            # back to action 0 if no self-direction is available.  When a
            # predator IS visible but forces cancel, _compute_force has
            # already substituted a pure radial flee from the closest
            # threat, so we never want the heading-keep branch in that case.
            self_block = obs[self._layout["self"]]
            if self_block.size >= 4:
                heading_x, heading_y = _direction_to_unit_vector(self_block[3])
                return _action_for_heading(heading_x, heading_y, self.action_count)
            return 0

        # Quantise the resultant *flee* force onto a discrete heading.
        return _action_for_heading(force_x, force_y, self.action_count)

    # Internals ------------------------------------------------------------

    def _compute_force(self, obs: np.ndarray) -> Tuple[float, float, bool]:
        cfg = self.config
        # Accumulate predator (flee + tangential) and separation forces
        # separately so we can cap the latter relative to the former.
        pred_fx = 0.0
        pred_fy = 0.0
        any_predator_visible = False
        # Track the closest visible predator so we can fall back to a pure
        # radial flee when symmetric configurations cause the accumulated
        # force to cancel (two predators on opposite sides at equal
        # distance produce antiparallel radial AND tangential terms,
        # because ``self._turn_sign`` is constant per-episode).
        closest: Optional[Tuple[float, float, float]] = None  # (sdistance, ux, uy)

        pred_block = obs[self._layout["predators"]]
        for entry in self._iter_entries(pred_block, expected_type=0):
            if entry is None:
                continue
            any_predator_visible = True
            sdistance, sdirection = entry  # both in [0, 1]
            ux, uy = _direction_to_unit_vector(sdirection)
            if closest is None or sdistance < closest[0]:
                closest = (sdistance, ux, uy)
            # ``sdirection`` points observer → predator.  Flee = -unit.
            inv_sq = 1.0 / (sdistance * sdistance + cfg.softening)
            pred_fx -= cfg.k_predator * ux * inv_sq
            pred_fy -= cfg.k_predator * uy * inv_sq
            # Tangential (perpendicular-dodge) component: rotates the flee
            # vector ~30-45°.  Uses 1/d falloff so it remains effective at
            # longer range where the inverse-square radial term is weak.
            # self._turn_sign is fixed per episode to prevent adjacent
            # predators' tangential forces from cancelling each other.
            inv_d = 1.0 / (sdistance + math.sqrt(cfg.softening))
            pred_fx += cfg.k_tangential * (-uy) * inv_d * self._turn_sign
            pred_fy += cfg.k_tangential * ux * inv_d * self._turn_sign

        # Cancellation fallback: if any predator is visible but the net
        # force is below threshold (symmetric squeeze), override with a
        # pure radial flee from the closest predator (no tangential).
        # Without this, prey ``freeze'' between symmetric threats and the
        # heading-keep branch in ``get_action`` drives them straight
        # through the line connecting the predators.
        pred_mag = math.hypot(pred_fx, pred_fy)
        if (
            any_predator_visible
            and pred_mag < cfg.no_op_threshold * 10.0
            and closest is not None
        ):
            sdistance, ux, uy = closest
            inv_sq = 1.0 / (sdistance * sdistance + cfg.softening)
            pred_fx = -cfg.k_predator * ux * inv_sq
            pred_fy = -cfg.k_predator * uy * inv_sq

        # Optional separation from other prey.  Only applied when at least
        # one predator is visible — separation is a tie-breaker that keeps
        # the flock from collapsing onto a single safe corridor when under
        # threat, NOT a flight force on its own.  Its magnitude is capped
        # to ``separation_cap_ratio * |pred_force|`` so that a cluster of
        # coincident teammates cannot overwhelm the predator-flee force.
        if cfg.k_prey_separation > 0.0 and any_predator_visible:
            sep_fx = 0.0
            sep_fy = 0.0
            prey_block = obs[self._layout["prey"]]
            for entry in self._iter_entries(prey_block, expected_type=1):
                if entry is None:
                    continue
                sdistance, sdirection = entry
                ux, uy = _direction_to_unit_vector(sdirection)
                inv_sq = 1.0 / (sdistance * sdistance + cfg.softening)
                sep_fx -= cfg.k_prey_separation * ux * inv_sq
                sep_fy -= cfg.k_prey_separation * uy * inv_sq

            sep_mag = math.hypot(sep_fx, sep_fy)
            pred_mag_now = math.hypot(pred_fx, pred_fy)
            cap = cfg.separation_cap_ratio * pred_mag_now
            if sep_mag > cap and sep_mag > 0.0:
                scale = cap / sep_mag
                sep_fx *= scale
                sep_fy *= scale
            return pred_fx + sep_fx, pred_fy + sep_fy, any_predator_visible

        return pred_fx, pred_fy, any_predator_visible

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
