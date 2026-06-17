"""Artificial Potential Field (APF) evader policy.

Implements a deterministic reactive evader for use with
:class:`~deap_pettingzoo.environments.PursuitEvasionEnv`.  The policy
computes a resultant force vector from inverse-square repulsive
contributions of every *visible* pursuer and landmark, plus a soft
boundary push that activates near the operational edge, then quantises the
continuous force to the nearest of the five discrete MPE actions
``[no-op, left (-x), right (+x), down (-y), up (+y)]``.

The policy satisfies :class:`~deap_pettingzoo.baselines.base.PolicyProtocol`
and is therefore a drop-in for any deap-bridge evaluator.  It carries no
trainable state — it is a scripted, non-learning controller, which makes
it suitable as a stable benchmark target for RL vs EA comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from deap_pettingzoo.contrib.pursuit_evasion.env import ObsLayout


# Direction vectors for the five MPE discrete actions.
# Index order follows ``mpe2.simple_tag_v3``:
#   0 = no-op, 1 = left (-x), 2 = right (+x), 3 = down (-y), 4 = up (+y)
_ACTION_DIRECTIONS = np.array(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [+1.0, 0.0],
        [0.0, -1.0],
        [0.0, +1.0],
    ],
    dtype=np.float32,
)


@dataclass
class APFConfig:
    """Hyperparameters for :class:`APFEvaderPolicy`.

    Attributes:
        k_pursuer: Gain on inverse-square pursuer repulsion.
        k_landmark: Gain on inverse-square landmark repulsion.
        k_boundary: Gain on boundary pushback (linear in excess distance).
        boundary_threshold: Distance from origin beyond which the boundary
            pushback activates (per axis).
        no_op_threshold: Minimum force magnitude required to output any
            non-zero action; below this the policy emits the no-op.
        softening: Small constant added to squared distances to avoid
            singularities when two entities overlap.
    """

    k_pursuer: float = 1.0
    k_landmark: float = 0.3
    k_boundary: float = 2.0
    boundary_threshold: float = 0.9
    no_op_threshold: float = 1e-3
    softening: float = 1e-3


class APFEvaderPolicy:
    """Scripted evader policy for :class:`PursuitEvasionEnv`.

    The policy parses each observation vector using the supplied
    :class:`ObsLayout`, isolates visible pursuers (non-zero relative
    positions in the adversary region), isolates visible landmarks, then
    computes the resultant repulsive force::

        F = sum_p k_pursuer  * (-r_p) / (|r_p|^3 + eps)
          + sum_l k_landmark * (-r_l) / (|r_l|^3 + eps)
          + sum_axis boundary_push(self_pos_axis)

    The force is quantised by taking the ``argmax`` of ``F @ dir_i`` across
    the four non-zero MPE direction vectors; if no direction has
    sufficient projection (``< no_op_threshold``) the policy emits the
    no-op action.

    The policy is stateless: :meth:`reset` is a no-op.
    """

    def __init__(
        self,
        layout: ObsLayout,
        config: Optional[APFConfig] = None,
    ) -> None:
        """Construct the policy for a *specific* agent.

        Args:
            layout: Observation layout metadata for the agent this policy
                will be attached to.  Different agents may have different
                layouts (e.g. the number of adversaries observed).
            config: Hyperparameter configuration.  Defaults to
                :class:`APFConfig`'s field defaults.
        """
        self.layout = layout
        self.config = config or APFConfig()

    # PolicyProtocol

    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Return a discrete action index in ``{0, 1, 2, 3, 4}``.

        Accepts the ``deterministic`` keyword for interface symmetry with
        :class:`FeedforwardNeuralPolicy`; the APF policy is deterministic
        by construction and ignores the flag.
        """
        del deterministic  # unused
        force = self._compute_force(observation)
        return int(self._force_to_action(force))

    def reset(self) -> None:
        """No-op; the APF controller is memoryless."""
        return None

    # Internals

    def _compute_force(self, obs: np.ndarray) -> np.ndarray:
        """Sum repulsive force contributions from visible entities and borders."""
        layout = self.layout
        cfg = self.config

        self_pos = obs[layout.self_pos]
        force = np.zeros(2, dtype=np.float32)

        # Pursuer repulsion (visible adversaries live in adversary_rel).
        if layout.num_adv_in_other > 0:
            adv_rel = obs[layout.adversary_rel].reshape(layout.num_adv_in_other, 2)
            force += self._repulsion_sum(adv_rel, cfg.k_pursuer, cfg.softening)

        # Landmark repulsion (static obstacles).
        if layout.num_landmarks > 0:
            landmark_rel = obs[layout.landmark_rel].reshape(layout.num_landmarks, 2)
            force += self._repulsion_sum(landmark_rel, cfg.k_landmark, cfg.softening)

        # Boundary push: linear in distance past the threshold, inward.
        excess = np.maximum(0.0, np.abs(self_pos) - cfg.boundary_threshold)
        boundary = -cfg.k_boundary * np.sign(self_pos) * excess
        force += boundary.astype(force.dtype)

        return force

    @staticmethod
    def _repulsion_sum(rel_positions: np.ndarray, gain: float, softening: float) -> np.ndarray:
        """Sum ``-gain * r / (|r|^3 + softening)`` over non-zero rows.

        Zero rows are treated as absent/not-visible (per the wrapper's
        zero-padding convention for masked entities) and contribute
        nothing.
        """
        if rel_positions.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)

        norms_sq = np.sum(rel_positions ** 2, axis=1)
        visible = norms_sq > 0.0
        if not np.any(visible):
            return np.zeros(2, dtype=np.float32)

        r = rel_positions[visible]
        n = norms_sq[visible]
        denom = n * np.sqrt(n) + softening   # |r|^3 + softening
        contributions = -gain * r / denom[:, None]
        return contributions.sum(axis=0).astype(np.float32)

    def _force_to_action(self, force: np.ndarray) -> int:
        """Quantise a continuous 2-D force to a discrete MPE action index.

        Projects the force onto each of the four cardinal direction
        vectors and picks the one with the largest positive projection.
        Returns the no-op action (0) when every projection falls below
        ``no_op_threshold``.
        """
        cfg = self.config
        # Skip index 0 (no-op has zero direction vector).
        dirs = _ACTION_DIRECTIONS[1:]
        projections = dirs @ force  # shape (4,)
        best = int(np.argmax(projections))
        if projections[best] < cfg.no_op_threshold:
            return 0
        return best + 1  # shift back into [1, 4]
