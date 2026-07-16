"""Observation featurization helpers for the scripted aquarium experiment.

The raw marl-aquarium predator observation exposes absolute positions and
wrapped scalar directions, which are harder to learn from than they need
to be: a wrapped angle scalar has a discontinuity at ``±180°``, and
absolute positions force every learner to re-derive relative geometry.
This module projects each predator observation onto a compact, local
feature set shared by every learner in the aquarium comparison
(EA/CMA/PSO/MAP-Elites/HyperNEAT/MATD3), so fitness differences reflect the
optimizer rather than observation preprocessing:

* self block -> ``[heading_x, heading_y, speed]`` (+ 4 wall-distance
  features when the env exposes them; see below)
* each nearby predator/prey entry ->
  ``[present, rel_x, rel_y, distance, speed]``

where ``rel_x`` and ``rel_y`` are the relative offset implied by the env's
scaled distance + direction fields — the wrapped ``sdirection`` scalar
decoded into a continuous XY unit vector and scaled by ``distance``.

The campaign environment is a bounded rectangle. In a bounded
arena wall proximity is tactically relevant — pursuers corner prey against
boundaries, and the scripted APF opponent already steers on it via
``k_wall`` — so when ``observable_walls=True`` the env's 4 wall-distance
features (``[left, right, top, bottom]``, 0=at wall, 1=far) are carried
through into the self block rather than discarded, giving learned agents
parity with the walls the environment and the APF proxy both model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from .apf_policy import _direction_to_unit_vector, aquarium_obs_layout


def _scaled_direction_to_unit_vector(scaled_direction: float) -> tuple[float, float]:
    """Convert aquarium's scaled direction field back to an XY unit vector.

    Delegates to :func:`apf_policy._direction_to_unit_vector` so the
    decoder cannot drift apart from the scripted-prey controller.  The
    env encodes ``sdirection`` via
    ``get_angle_from_vector(v) = -atan2(vy, vx) * 180/pi`` (0Â° = east),
    whose inverse is ``(cos(rad), -sin(rad))`` â€” **not** the action
    convention ``(sin(rad), -cos(rad))``, which is offset by 90Â°.
    """
    return _direction_to_unit_vector(scaled_direction)


def aquarium_feature_size(
    predator_observe_count: int,
    prey_observe_count: int,
    wall_features: int = 0,
) -> int:
    """Return the featurized predator observation size.

    Args:
        predator_observe_count: ``predator_observe_count`` env kwarg.
        prey_observe_count: ``prey_observe_count`` env kwarg.
        wall_features: 4 when ``observable_walls=True``, 0 otherwise —
            carried through into the self block.
    """
    return (
        3
        + int(wall_features)
        + 5 * int(predator_observe_count)
        + 5 * int(prey_observe_count)
    )


def featurize_aquarium_observation(
    observation: np.ndarray,
    predator_observe_count: int,
    prey_observe_count: int,
    wall_features: int = 0,
) -> np.ndarray:
    """Project a raw aquarium observation onto local relative features.

    Args:
        observation: Raw aquarium observation vector.
        predator_observe_count: ``predator_observe_count`` env kwarg.
        prey_observe_count: ``prey_observe_count`` env kwarg.
        wall_features: 4 when ``observable_walls=True``, 0 otherwise.  When
            non-zero, the env's wall-distance features are appended to the
            self block of the output (right after ``speed``) rather than
            dropped, so learners see the same boundary information the
            bounded-rectangle env and the scripted APF opponent both use.
    """
    obs = np.asarray(observation, dtype=np.float32).ravel()
    wall_features = int(wall_features)
    layout = aquarium_obs_layout(
        predator_observe_count, prey_observe_count,
        wall_features=wall_features,
    )

    self_block = obs[layout["self"]]
    heading_x, heading_y = _scaled_direction_to_unit_vector(self_block[3])
    features = [
        float(heading_x),
        float(heading_y),
        float(self_block[4]),
    ]
    # Wall-distance features (``[left, right, top, bottom]``, 0=at wall,
    # 1=far) live at self-block indices 5..8 when the env exposes them.
    for wall_idx in range(5, 5 + wall_features):
        features.append(float(self_block[wall_idx]))

    for block_name in ("predators", "prey"):
        block = obs[layout[block_name]]
        for idx in range(block.size // 6):
            base = idx * 6
            sx = float(block[base + 1])
            sy = float(block[base + 2])
            distance = float(block[base + 3])
            direction = float(block[base + 4])
            speed = float(block[base + 5])
            present = 1.0 if (abs(sx) > 1e-9 or abs(sy) > 1e-9 or abs(distance) > 1e-9) else 0.0
            if present == 0.0:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            rel_x, rel_y = _scaled_direction_to_unit_vector(direction)
            features.extend([
                1.0,
                float(distance * rel_x),
                float(distance * rel_y),
                distance,
                speed,
            ])

    return np.asarray(features, dtype=np.float32)


def extract_prey_caught_total(infos: Optional[Dict[str, Dict[str, Any]]]) -> int:
    """Return the highest reported cumulative catch count in an info dict."""
    best = 0
    for info in (infos or {}).values():
        if not isinstance(info, dict):
            continue
        value = info.get("prey_caught_total")
        if value is None:
            continue
        try:
            best = max(best, int(value))
        except (TypeError, ValueError):
            continue
    return best


class AquariumObservationFeatureWrapper:
    """Wrap the scripted aquarium env so all learners see featurized observations.

    When ``append_agent_id`` is True (default), a one-hot vector identifying
    the observing agent is appended to each predator's feature vector.
    With a shared homogeneous genome this breaks the structural symmetry
    between predators that all see identically-shaped local observations
    â€” the network can express role differentiation (e.g. "I am predator
    3, swing south") while still sharing parameters.  Sized to
    ``len(possible_agents)`` so prey indices coexist if learner policies
    are ever applied to prey too.
    """

    def __init__(self, inner, append_agent_id: bool = True) -> None:
        self.inner = inner
        self.predator_observe_count = int(getattr(inner, "predator_observe_count"))
        self.prey_observe_count = int(getattr(inner, "prey_observe_count"))
        self.possible_agents = list(getattr(inner, "possible_agents", []))
        self.agents = list(getattr(inner, "agents", self.possible_agents))
        self.metadata = getattr(inner, "metadata", {})
        self.last_infos: Dict[str, Dict[str, Any]] = {}

        self.append_agent_id = bool(append_agent_id)
        self._agent_index = {a: i for i, a in enumerate(self.possible_agents)}
        self.id_size = len(self.possible_agents) if self.append_agent_id else 0

        # Detect wall features from the raw env so featurization slices the
        # observation correctly and carries the wall block through.  The
        # campaign env is a bounded rectangle with ``observable_walls=True``;
        # the detection stays topology-agnostic regardless.
        _raw = getattr(inner, "_raw_env", None)
        self._wall_features = 4 if getattr(_raw, "observable_walls", False) else 0

        feat_size = aquarium_feature_size(
            self.predator_observe_count,
            self.prey_observe_count,
            wall_features=self._wall_features,
        )
        shape = (feat_size + self.id_size,)
        box = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        self.observation_spaces = {agent: box for agent in self.possible_agents}
        self.action_spaces = {
            agent: self.inner.action_space(agent) for agent in self.possible_agents
        }

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(self, seed=None, options=None):
        observations, infos = self.inner.reset(seed=seed, options=options)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))
        self.last_infos = dict(infos) if infos is not None else {}
        return self._transform_obs_dict(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.inner.step(actions)
        self.agents = list(getattr(self.inner, "agents", []))
        self.last_infos = dict(infos) if infos is not None else {}
        return (
            self._transform_obs_dict(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def _featurize_one(self, agent: str, observation: np.ndarray) -> np.ndarray:
        feats = featurize_aquarium_observation(
            observation,
            self.predator_observe_count,
            self.prey_observe_count,
            wall_features=self._wall_features,
        )
        if self.id_size == 0:
            return feats
        one_hot = np.zeros(self.id_size, dtype=np.float32)
        idx = self._agent_index.get(agent)
        if idx is not None:
            one_hot[idx] = 1.0
        return np.concatenate([feats, one_hot]).astype(np.float32, copy=False)

    def _transform_obs_dict(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            agent: self._featurize_one(agent, observation)
            for agent, observation in observations.items()
        }
