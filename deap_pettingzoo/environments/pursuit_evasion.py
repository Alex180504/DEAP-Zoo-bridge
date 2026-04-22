"""Pursuit-evasion environment wrapper with native MPE2 partial observability.

Specialises :class:`PettingZooWrapper` for the research methodology:

* Cooperative predator-prey (``mpe2.simple_tag_v3``)
* Equal-ability pursuers and evaders (same ``max_speed`` and ``accel``), opt-out
* K-nearest partial observability provided natively by MPE2
  (``num_agent_neighbors`` / ``num_landmark_neighbors``): each agent sees only
  the K nearest other agents and L nearest landmarks; non-visible slots are
  zero-padded by the environment so the observation vector has a fixed size.
* Optional prey-speed curriculum (``curriculum=True``) with three stages
  (50 / 75 / 100 % of full prey speed).  Advanced via
  :meth:`advance_curriculum` / :meth:`set_curriculum_stage`; takes effect on
  the next ``reset``.
* Optional early termination when every good agent is simultaneously caught
  (``terminate_on_success=True``).
* Exponential boundary penalty applied to per-agent rewards.

Observation layout (from MPE2)::

    [ self_vel (2) | self_pos (2) | landmark_rel (2*L) |
      other_rel (2*K) | good_vel (2*K) ]             # PO enabled
    [ self_vel (2) | self_pos (2) | landmark_rel (2*num_obstacles) |
      other_rel (2*(A+G-1)) | good_vel (2*Gvel) ]    # full observability

Under full observability ``other_rel`` orders adversaries before good
agents (self excluded) and ``Gvel = num_good`` for adversary observers,
``num_good - 1`` for good observers.  Under partial observability the
``other_rel`` block is K-nearest-regardless-of-type and agent types are
not recoverable from the observation vector alone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpe2 import simple_tag_v3

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
        equal_ability: If True (default), every world agent's ``max_speed``
            and ``accel`` are overwritten after each reset with
            ``equal_max_speed`` / ``equal_accel`` to enforce a symmetric
            pursuer/evader ability setting.  Disable when running a speed
            curriculum so stage-dependent prey speeds survive :meth:`reset`.
        equal_max_speed: Maximum velocity applied to every agent
            (only used when ``equal_ability`` is True).
        equal_accel: Acceleration limit applied to every agent
            (only used when ``equal_ability`` is True).
        num_agent_neighbors: K-nearest agents each agent observes; ``None``
            disables agent masking (full observability).  Must be a
            positive integer when set.
        num_landmark_neighbors: K-nearest landmarks each agent observes;
            ``None`` disables landmark masking.  Must be a positive
            integer when set.
        curriculum: Enable MPE2's prey-speed curriculum (stages 0/1/2 =
            50/75/100 % of full prey speed).
        terminate_on_success: Terminate the episode as soon as every good
            agent is simultaneously caught.
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
    equal_ability: bool = True
    equal_max_speed: float = 1.0
    equal_accel: float = 3.0
    num_agent_neighbors: Optional[int] = 5
    num_landmark_neighbors: Optional[int] = 3
    curriculum: bool = False
    terminate_on_success: bool = False
    boundary_threshold: float = 0.9
    boundary_scale: float = 1.0
    boundary_sharpness: float = 2.0


# ---------------------------------------------------------------------------
# Observation layout metadata
# ---------------------------------------------------------------------------


@dataclass
class ObsLayout:
    """Slice metadata for a single agent's MPE2 observation vector.

    Attributes:
        self_vel: Slice for the agent's own velocity (length 2).
        self_pos: Slice for the agent's own position (length 2).
        landmark_rel: Slice covering visible landmark relative positions.
        other_rel: Slice covering visible other-agent relative positions.
            Under full observability, adversaries are listed before good
            agents and the :attr:`adversary_rel` / :attr:`good_rel`
            sub-slices carve out each group; under K-nearest PO the
            ordering is mixed-by-distance and ``adversary_rel`` equals
            ``other_rel`` while ``good_rel`` is empty (all visible
            agents are treated uniformly by downstream policies).
        adversary_rel: Sub-slice of ``other_rel`` that is guaranteed to
            contain adversary positions.  Equal to ``other_rel`` under PO.
        good_rel: Sub-slice of ``other_rel`` that is guaranteed to contain
            good-agent positions (empty under PO).
        good_vel: Slice covering good-agent velocity entries.
        is_adversary: True iff this layout describes an adversary.
        num_landmarks: Number of landmark slots in ``landmark_rel``.
        num_others: Number of other-agent slots in ``other_rel``.
        num_adv_in_other: Number of adversary slots at the start of
            ``other_rel`` (== ``num_others`` under PO).
        num_good_in_other: Number of good-agent slots in ``other_rel``
            (0 under PO).
        num_good_vel: Number of good-agent velocity slots.
        partial_obs: True iff the environment was constructed with
            K-nearest agent masking.
    """

    self_vel: slice
    self_pos: slice
    landmark_rel: slice
    other_rel: slice
    adversary_rel: slice
    good_rel: slice
    good_vel: slice
    is_adversary: bool
    num_landmarks: int
    num_others: int
    num_adv_in_other: int
    num_good_in_other: int
    num_good_vel: int
    partial_obs: bool


def _compute_layout(
    agent_name: str,
    num_adversaries: int,
    num_good: int,
    num_landmarks_visible: int,
    num_others_visible: int,
    num_good_vel_visible: int,
    partial_obs: bool,
) -> ObsLayout:
    is_adv = agent_name.startswith("adversary")

    i = 0
    self_vel = slice(i, i + 2); i += 2
    self_pos = slice(i, i + 2); i += 2
    landmark_rel = slice(i, i + 2 * num_landmarks_visible)
    i += 2 * num_landmarks_visible
    other_rel = slice(i, i + 2 * num_others_visible)

    if partial_obs:
        adv_rel = other_rel
        good_rel = slice(other_rel.stop, other_rel.stop)  # empty
        n_adv_in_other = num_others_visible
        n_good_in_other = 0
    else:
        n_adv_in_other = num_adversaries - 1 if is_adv else num_adversaries
        n_good_in_other = num_good if is_adv else num_good - 1
        adv_rel = slice(i, i + 2 * n_adv_in_other)
        good_rel = slice(i + 2 * n_adv_in_other, other_rel.stop)

    i = other_rel.stop
    good_vel = slice(i, i + 2 * num_good_vel_visible)

    return ObsLayout(
        self_vel=self_vel,
        self_pos=self_pos,
        landmark_rel=landmark_rel,
        other_rel=other_rel,
        adversary_rel=adv_rel,
        good_rel=good_rel,
        good_vel=good_vel,
        is_adversary=is_adv,
        num_landmarks=num_landmarks_visible,
        num_others=num_others_visible,
        num_adv_in_other=n_adv_in_other,
        num_good_in_other=n_good_in_other,
        num_good_vel=num_good_vel_visible,
        partial_obs=partial_obs,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class PursuitEvasionEnv(PettingZooWrapper):
    """``mpe2.simple_tag_v3`` specialised for equal-ability predator-prey research.

    Differs from :class:`PettingZooWrapper` in:

    1. **Partial observability** — forwards ``num_agent_neighbors`` and
       ``num_landmark_neighbors`` to MPE2, which handles K-nearest
       selection and zero-padding natively.
    2. **Equal-ability** — after each ``reset`` the world's ``max_speed``
       and ``accel`` are overwritten for every agent (opt-out via
       ``equal_ability=False`` — required when using ``curriculum=True``).
    3. **Curriculum** — forwards ``curriculum`` / ``terminate_on_success``
       and exposes :meth:`advance_curriculum`,
       :meth:`set_curriculum_stage`, and :attr:`curriculum_stage`
       pass-throughs.
    4. **Boundary penalty** — per-agent rewards are reduced by an
       exponential penalty function of the agent's position magnitude.

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

        if self.pe_config.curriculum and self.pe_config.equal_ability:
            warnings.warn(
                "PursuitEvasionConfig has curriculum=True and equal_ability=True; "
                "equal-ability will overwrite the curriculum-scheduled prey speeds "
                "after every reset, nullifying the curriculum. "
                "Set equal_ability=False to let the curriculum take effect.",
                stacklevel=2,
            )

        self.env_name = "simple_tag_v3"
        self.env_config: Dict[str, Any] = {}
        self.render_mode = render_mode

        self.env = simple_tag_v3.parallel_env(
            num_good=self.pe_config.num_good,
            num_adversaries=self.pe_config.num_adversaries,
            num_obstacles=self.pe_config.num_obstacles,
            max_cycles=self.pe_config.max_cycles,
            continuous_actions=False,
            num_agent_neighbors=self.pe_config.num_agent_neighbors,
            num_landmark_neighbors=self.pe_config.num_landmark_neighbors,
            curriculum=self.pe_config.curriculum,
            terminate_on_success=self.pe_config.terminate_on_success,
            render_mode=render_mode,
        )
        self.possible_agents = list(self.env.possible_agents)
        self.episode_length = 0
        self.cumulative_rewards: Dict[str, float] = {}

        # Pre-compute per-agent observation layouts (static across episodes).
        partial_obs_agents = self.pe_config.num_agent_neighbors is not None
        partial_obs_land = self.pe_config.num_landmark_neighbors is not None
        num_land_vis = (
            self.pe_config.num_landmark_neighbors
            if partial_obs_land
            else self.pe_config.num_obstacles
        )
        self._layouts: Dict[str, ObsLayout] = {}
        for name in self.possible_agents:
            is_adv = name.startswith("adversary")
            if partial_obs_agents:
                num_others = self.pe_config.num_agent_neighbors
                num_good_vel = self.pe_config.num_agent_neighbors
            else:
                num_others = (
                    self.pe_config.num_adversaries + self.pe_config.num_good - 1
                )
                num_good_vel = (
                    self.pe_config.num_good
                    if is_adv
                    else self.pe_config.num_good - 1
                )
            self._layouts[name] = _compute_layout(
                agent_name=name,
                num_adversaries=self.pe_config.num_adversaries,
                num_good=self.pe_config.num_good,
                num_landmarks_visible=num_land_vis,
                num_others_visible=num_others,
                num_good_vel_visible=num_good_vel,
                partial_obs=partial_obs_agents,
            )

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
    # Curriculum pass-throughs
    # ------------------------------------------------------------------

    @property
    def curriculum_stage(self) -> int:
        """Current curriculum stage on the underlying MPE2 env."""
        return self.env.unwrapped.curriculum_stage

    def advance_curriculum(self) -> None:
        """Advance to the next curriculum stage (effective on next reset)."""
        self.env.unwrapped.advance_curriculum()

    def set_curriculum_stage(self, stage: int) -> None:
        """Jump to a specific curriculum stage (effective on next reset)."""
        self.env.unwrapped.set_curriculum_stage(stage)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset the environment and (optionally) enforce equal ability."""
        observations, _ = self.env.reset(seed=seed, options=options)
        if self.pe_config.equal_ability:
            self._apply_equal_ability()

        self.episode_length = 0
        self.cumulative_rewards = {agent: 0.0 for agent in observations}
        return observations

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
        """Step the environment and apply boundary penalty to rewards."""
        observations, rewards, terminations, truncations, infos = super().step(actions)

        for agent in list(rewards.keys()):
            if agent not in observations:
                continue
            layout = self._layouts.get(agent)
            if layout is None:
                continue
            self_pos = observations[agent][layout.self_pos]
            penalty = self._boundary_penalty(self_pos)
            rewards[agent] -= penalty
            if agent in self.cumulative_rewards:
                self.cumulative_rewards[agent] -= penalty

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_equal_ability(self) -> None:
        """Overwrite every world-agent's ``max_speed`` and ``accel``."""
        world = self._get_world()
        for a in world.agents:
            a.max_speed = self.pe_config.equal_max_speed
            a.accel = self.pe_config.equal_accel

    def _get_world(self):
        """Recover the underlying MPE world.

        Walks common attribute paths — ``unwrapped``, ``aec_env``, and
        ``env`` chains — until a ``world`` attribute is found.  MPE2's
        parallel wrapper exposes it directly via ``.unwrapped.world``.
        """
        candidates = []
        env = self.env
        candidates.append(env)
        for attr in ("aec_env", "unwrapped", "env"):
            sub = getattr(env, attr, None)
            if sub is not None:
                candidates.append(sub)
                deeper = getattr(sub, "unwrapped", None)
                if deeper is not None:
                    candidates.append(deeper)
        for cand in candidates:
            if hasattr(cand, "world"):
                return cand.world
        raise RuntimeError("Could not locate MPE world on env; API changed?")

    def _boundary_penalty(self, self_pos: np.ndarray) -> float:
        """Exponential boundary penalty ``sum_axis scale*(exp(k*excess)-1)``.

        Zero when ``|x| <= threshold`` on both axes; grows exponentially
        outside the region.
        """
        cfg = self.pe_config
        excess = np.maximum(0.0, np.abs(self_pos) - cfg.boundary_threshold)
        return float(cfg.boundary_scale * np.sum(np.expm1(cfg.boundary_sharpness * excess)))
