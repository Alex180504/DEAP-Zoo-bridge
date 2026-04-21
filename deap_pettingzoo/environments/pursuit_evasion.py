"""Pursuit-evasion environment wrapper with partial observability.

Specialises :class:`PettingZooWrapper` for the research methodology 

* 10-vs-10 cooperative predator-prey (``simple_tag_v3``)
* Equal-ability pursuers and evaders (same ``max_speed`` and ``accel``)
* Partial observability — each agent observes only its K-nearest pursuers /
  evaders and K-nearest landmarks; non-visible entity slots are zero-padded
  so the observation vector retains a fixed size suitable for a static NN
  input layer
* Exponential boundary penalty applied to per-agent rewards, pushing agents
  back inside the operational region ``|x|, |y| <= threshold``

The layout of the underlying ``simple_tag_v3`` observation vector (used to
locate neighbour slots for masking) is, for *any* agent::

    [ self_vel (2) |
      self_pos (2) |
      landmarks_rel (2 * num_landmarks) |
      other_rel     (2 * (num_adversaries + num_good - 1)) |
      good_vel      (2 * M) ]

where ``M = num_good`` for an adversary's observation and ``M = num_good - 1``
for a good agent's observation (each good agent excludes its own velocity).
Within ``other_rel``, :mod:`pettingzoo.mpe.simple_tag_v3` orders entries as
adversaries first then good agents (skipping self), which we rely on for
type-aware masking and APF evader visibility reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pettingzoo.mpe import simple_tag_v3

from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PursuitEvasionConfig:
    """Configuration for :class:`PursuitEvasionEnv`.

    Attributes:
        num_adversaries: Number of pursuers (``adversary_*`` agents).
        num_good: Number of evaders (``agent_*`` agents).
        num_obstacles: Number of static landmarks.
        max_cycles: Maximum environment steps per episode.
        equal_max_speed: Maximum velocity applied to every agent.
        equal_accel: Acceleration limit applied to every agent.
        num_agent_neighbors: Maximum number of other agents visible to each
            agent (the closest ones are retained; the rest are zero-padded).
            ``None`` disables agent masking.
        num_landmark_neighbors: Maximum number of landmarks visible to each
            agent. ``None`` disables landmark masking.
        boundary_threshold: Absolute coordinate threshold beyond which the
            boundary penalty activates (per axis).
        boundary_scale: Multiplicative coefficient on the exponential
            penalty.
        boundary_sharpness: Exponent coefficient; higher values produce a
            sharper wall.
    """

    num_adversaries: int = 10
    num_good: int = 10
    num_obstacles: int = 3
    max_cycles: int = 200
    equal_max_speed: float = 1.0
    equal_accel: float = 3.0
    num_agent_neighbors: Optional[int] = 5
    num_landmark_neighbors: Optional[int] = 3
    boundary_threshold: float = 0.9
    boundary_scale: float = 1.0
    boundary_sharpness: float = 2.0


# ---------------------------------------------------------------------------
# Observation layout metadata
# ---------------------------------------------------------------------------


@dataclass
class ObsLayout:
    """Slice metadata for a single agent's observation vector.

    All slices index a 1-D ``np.ndarray`` representing one agent's raw
    observation.  Used for both masking (in the wrapper) and parsing (in
    downstream policies such as :class:`APFEvaderPolicy`).

    Attributes:
        self_vel: Slice for the agent's own velocity (length 2).
        self_pos: Slice for the agent's own position (length 2).
        landmark_rel: Slice covering all landmark relative positions.
        adversary_rel: Slice covering rel-positions of adversary agents
            (empty for adversary-self observations beyond the first A-1).
        good_rel: Slice covering rel-positions of good agents.
        good_vel: Slice covering velocity entries of good agents.
        is_adversary: True iff this layout describes an adversary.
    """

    self_vel: slice
    self_pos: slice
    landmark_rel: slice
    adversary_rel: slice
    good_rel: slice
    good_vel: slice
    is_adversary: bool

    # Derived counts (for convenience)
    num_landmarks: int
    num_adv_in_other: int
    num_good_in_other: int
    num_good_vel: int


def _compute_layout(
    agent_name: str,
    num_adversaries: int,
    num_good: int,
    num_landmarks: int,
) -> ObsLayout:
    """Compute the observation layout for one agent.

    See module docstring for the canonical layout.  This function returns
    slices that partition the raw observation exactly.
    """
    is_adv = agent_name.startswith("adversary")

    # Counts of "other agents" entries (self is excluded)
    if is_adv:
        n_adv_in_other = num_adversaries - 1
        n_good_in_other = num_good
        n_good_vel = num_good            # includes all good agents
    else:
        n_adv_in_other = num_adversaries
        n_good_in_other = num_good - 1
        n_good_vel = num_good - 1        # excludes self

    i = 0
    self_vel = slice(i, i + 2); i += 2
    self_pos = slice(i, i + 2); i += 2
    landmark_rel = slice(i, i + 2 * num_landmarks); i += 2 * num_landmarks
    adversary_rel = slice(i, i + 2 * n_adv_in_other); i += 2 * n_adv_in_other
    good_rel = slice(i, i + 2 * n_good_in_other); i += 2 * n_good_in_other
    good_vel = slice(i, i + 2 * n_good_vel); i += 2 * n_good_vel

    return ObsLayout(
        self_vel=self_vel,
        self_pos=self_pos,
        landmark_rel=landmark_rel,
        adversary_rel=adversary_rel,
        good_rel=good_rel,
        good_vel=good_vel,
        is_adversary=is_adv,
        num_landmarks=num_landmarks,
        num_adv_in_other=n_adv_in_other,
        num_good_in_other=n_good_in_other,
        num_good_vel=n_good_vel,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class PursuitEvasionEnv(PettingZooWrapper):
    """``simple_tag_v3`` specialised for equal-ability predator-prey research.

    Differs from :class:`PettingZooWrapper` in three ways:

    1. **Equal-ability**: after each ``reset`` the underlying world's
       ``max_speed`` and ``accel`` are overwritten for every agent to the
       values in :class:`PursuitEvasionConfig`.
    2. **Partial observability**: observations returned by ``reset`` and
       ``step`` have their neighbour slots sorted by Euclidean distance and
       zero-padded beyond the configured visibility thresholds.
    3. **Boundary penalty**: per-agent rewards returned by ``step`` are
       reduced by an exponential penalty function of the agent's position
       magnitude.

    All other wrapper features (action conversion, reward tracking, etc.)
    are inherited unchanged.
    """

    def __init__(
        self,
        config: Optional[PursuitEvasionConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Create a pursuit-evasion wrapper.

        Args:
            config: Scenario configuration. Defaults to a 10-vs-10 setup.
            render_mode: Pass-through to the underlying PettingZoo env.
        """
        self.pe_config = config or PursuitEvasionConfig()

        # Bypass the parent's SUPPORTED_ENVS check because we are constructing
        # the parallel_env directly with custom kwargs.
        self.env_name = "simple_tag_v3"
        self.env_config: Dict[str, Any] = {}
        self.render_mode = render_mode

        self.env = simple_tag_v3.parallel_env(
            num_good=self.pe_config.num_good,
            num_adversaries=self.pe_config.num_adversaries,
            num_obstacles=self.pe_config.num_obstacles,
            max_cycles=self.pe_config.max_cycles,
            continuous_actions=False,
            render_mode=render_mode,
        )
        self.possible_agents = list(self.env.possible_agents)
        self.episode_length = 0
        self.cumulative_rewards: Dict[str, float] = {}

        # Pre-compute per-agent observation layouts (static across episodes).
        self._layouts: Dict[str, ObsLayout] = {
            name: _compute_layout(
                name,
                self.pe_config.num_adversaries,
                self.pe_config.num_good,
                self.pe_config.num_obstacles,
            )
            for name in self.possible_agents
        }

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    @property
    def adversary_agents(self) -> List[str]:
        """List of adversary (pursuer) agent names."""
        return [a for a in self.possible_agents if a.startswith("adversary")]

    @property
    def good_agents(self) -> List[str]:
        """List of good (evader) agent names."""
        return [a for a in self.possible_agents if not a.startswith("adversary")]

    def get_layout(self, agent: str) -> ObsLayout:
        """Return the observation layout metadata for ``agent``."""
        return self._layouts[agent]

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset the environment, enforce equal-ability, and mask observations."""
        observations, _ = self.env.reset(seed=seed, options=options)
        self._apply_equal_ability()

        self.episode_length = 0
        self.cumulative_rewards = {agent: 0.0 for agent in observations}
        return {agent: self._mask_observation(agent, obs) for agent, obs in observations.items()}

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Step the environment, apply boundary penalty, and mask observations."""
        observations, rewards, terminations, truncations, infos = super().step(actions)

        # Apply exponential boundary penalty to each surviving agent.
        for agent in list(rewards.keys()):
            layout = self._layouts.get(agent)
            if layout is None or agent not in observations:
                continue
            raw_obs = self._raw_observations_cache.get(agent)
            if raw_obs is None:
                continue
            self_pos = raw_obs[layout.self_pos]
            penalty = self._boundary_penalty(self_pos)
            rewards[agent] -= penalty
            # Keep cumulative_rewards consistent with the modified reward.
            if agent in self.cumulative_rewards:
                self.cumulative_rewards[agent] -= penalty

        # Mask observations before returning (also refresh raw cache for next step).
        masked = {agent: self._mask_observation(agent, obs) for agent, obs in observations.items()}
        return masked, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # We need the RAW (pre-mask) observation when computing the boundary
    # penalty, because masking zeroes ``self_pos`` only if it is inside a
    # masked slot — it never is for self_pos, but keeping a cache also lets
    # policies (e.g. APF) recover exact self-position when desired.
    _raw_observations_cache: Dict[str, np.ndarray] = {}

    def _apply_equal_ability(self) -> None:
        """Overwrite every world-agent's ``max_speed`` and ``accel``."""
        world = self._get_world()
        for a in world.agents:
            a.max_speed = self.pe_config.equal_max_speed
            a.accel = self.pe_config.equal_accel

    def _get_world(self):
        """Recover the underlying MPE world (traverses the parallel wrapper)."""
        env = self.env
        if hasattr(env, "aec_env"):
            inner = env.aec_env
            if hasattr(inner, "world"):
                return inner.world
        raise RuntimeError("Could not locate MPE world on env; PettingZoo API changed?")

    def _boundary_penalty(self, self_pos: np.ndarray) -> float:
        """Exponential boundary penalty ``sum_axis scale*(exp(k*excess)-1)``.

        Zero when ``|x| <= threshold`` on both axes; grows exponentially
        outside the region.  Matches the methodology's
        ``exponentially scaling negative reward boundary function B(x)``.
        """
        cfg = self.pe_config
        excess = np.maximum(0.0, np.abs(self_pos) - cfg.boundary_threshold)
        return float(cfg.boundary_scale * np.sum(np.expm1(cfg.boundary_sharpness * excess)))

    def _mask_observation(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """Zero-pad neighbour slots beyond the visibility budget.

        Stores the pre-mask observation in ``_raw_observations_cache`` for
        this agent so the boundary penalty can read its ``self_pos``
        without having to re-derive it.
        """
        self._raw_observations_cache[agent] = obs.copy()

        layout = self._layouts[agent]
        masked = obs.copy()
        cfg = self.pe_config

        # Landmark visibility
        if cfg.num_landmark_neighbors is not None and layout.num_landmarks > 0:
            landmark_rel = masked[layout.landmark_rel].reshape(layout.num_landmarks, 2)
            keep = _select_k_nearest_indices(landmark_rel, cfg.num_landmark_neighbors)
            _apply_mask_inplace(landmark_rel, keep)
            masked[layout.landmark_rel] = landmark_rel.reshape(-1)

        # Agent visibility (applied jointly to adversary_rel, good_rel, and
        # the subset of good_vel that corresponds to visible good agents).
        if cfg.num_agent_neighbors is not None:
            total_others = layout.num_adv_in_other + layout.num_good_in_other
            if total_others > 0:
                adv_rel = masked[layout.adversary_rel].reshape(layout.num_adv_in_other, 2) \
                    if layout.num_adv_in_other > 0 else np.zeros((0, 2), dtype=masked.dtype)
                good_rel = masked[layout.good_rel].reshape(layout.num_good_in_other, 2) \
                    if layout.num_good_in_other > 0 else np.zeros((0, 2), dtype=masked.dtype)

                all_rel = np.concatenate([adv_rel, good_rel], axis=0)
                keep_idx = _select_k_nearest_indices(all_rel, cfg.num_agent_neighbors)

                # Split kept indices back into adv vs good slots.
                keep_mask = np.zeros(total_others, dtype=bool)
                keep_mask[keep_idx] = True
                adv_keep = keep_mask[: layout.num_adv_in_other]
                good_keep = keep_mask[layout.num_adv_in_other:]

                adv_rel[~adv_keep] = 0.0
                good_rel[~good_keep] = 0.0
                if layout.num_adv_in_other > 0:
                    masked[layout.adversary_rel] = adv_rel.reshape(-1)
                if layout.num_good_in_other > 0:
                    masked[layout.good_rel] = good_rel.reshape(-1)

                # Mask the trailing good_vel block for any good agent whose
                # relative position was masked out.  (Good velocities are the
                # only per-other velocity block present in simple_tag.)
                if layout.num_good_vel > 0:
                    good_vel = masked[layout.good_vel].reshape(layout.num_good_vel, 2)
                    # Layout guarantees good_vel corresponds 1:1 to the
                    # ``good_rel`` block for a good-agent observer and to a
                    # slightly larger set for an adversary observer.
                    if layout.is_adversary:
                        # For adversaries: good_vel has num_good entries,
                        # good_rel also has num_good entries (since good_rel
                        # subsumes the entire good side when observer is
                        # adversary).  They align 1:1.
                        vel_keep = good_keep
                    else:
                        # For good observers: good_vel and good_rel both have
                        # num_good - 1 entries, aligned 1:1.
                        vel_keep = good_keep
                    good_vel[~vel_keep] = 0.0
                    masked[layout.good_vel] = good_vel.reshape(-1)

        return masked


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_k_nearest_indices(rel_positions: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest non-zero rows in ``rel_positions`` (by L2 norm).

    Treats rows with norm below a tiny epsilon as "already masked / absent"
    and excludes them from selection (rather than treating them as the
    nearest).  Returns an ``np.ndarray`` of int indices, possibly shorter
    than ``k`` when fewer valid entries exist.
    """
    if rel_positions.shape[0] == 0 or k <= 0:
        return np.array([], dtype=int)

    norms = np.linalg.norm(rel_positions, axis=1)
    eps = 1e-8
    valid = norms > eps
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return valid_idx
    # Sort the valid ones by ascending distance.
    order = np.argsort(norms[valid_idx])
    return valid_idx[order][:k]


def _apply_mask_inplace(rel_positions: np.ndarray, keep_indices: np.ndarray) -> None:
    """Zero every row not in ``keep_indices`` (in-place)."""
    if rel_positions.shape[0] == 0:
        return
    mask = np.ones(rel_positions.shape[0], dtype=bool)
    mask[keep_indices] = False
    rel_positions[mask] = 0.0
