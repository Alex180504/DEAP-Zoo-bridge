"""Observation featurization helpers for the scripted aquarium experiment.

The raw marl-aquarium predator observation exposes absolute positions and
wrapped scalar directions. For pursuit this makes the input harder to
learn than it needs to be: the task is translation-invariant on a torus,
and the nearby-entry direction already contains the torus-aware relative
bearing.

This module converts each predator observation into a more local feature
set used only by the aquarium EA/CMA/MAPPO comparison:

* self block -> ``[heading_x, heading_y, speed]``
* each nearby predator/prey entry ->
  ``[present, rel_x, rel_y, distance, speed]``

where ``rel_x`` and ``rel_y`` are the torus-aware relative offset implied
by the env's scaled distance + direction fields.
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
    ``get_angle_from_vector(v) = -atan2(vy, vx) * 180/pi`` (0° = east),
    whose inverse is ``(cos(rad), -sin(rad))`` — **not** the action
    convention ``(sin(rad), -cos(rad))``, which is offset by 90°.
    """
    return _direction_to_unit_vector(scaled_direction)


def aquarium_feature_size(
    predator_observe_count: int,
    prey_observe_count: int,
) -> int:
    """Return the featurized predator observation size."""
    return 3 + 5 * int(predator_observe_count) + 5 * int(prey_observe_count)


def featurize_aquarium_observation(
    observation: np.ndarray,
    predator_observe_count: int,
    prey_observe_count: int,
) -> np.ndarray:
    """Project a raw aquarium observation onto local relative features."""
    obs = np.asarray(observation, dtype=np.float32).ravel()
    layout = aquarium_obs_layout(predator_observe_count, prey_observe_count)

    self_block = obs[layout["self"]]
    heading_x, heading_y = _scaled_direction_to_unit_vector(self_block[3])
    features = [
        float(heading_x),
        float(heading_y),
        float(self_block[4]),
    ]

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
    — the network can express role differentiation (e.g. "I am predator
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

        feat_size = aquarium_feature_size(
            self.predator_observe_count,
            self.prey_observe_count,
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


class LazyActionPenaltyWrapper:
    """Penalize agents whose action distribution collapses (entropy-based).

    Symptom this targets: with a shared homogeneous genome, EA/CMA can
    discover a "constant heading" attractor where the network ignores
    observations and emits the same action almost every step.  This
    looks like rigid linear drift in the rendered episode.

    Mechanism: per-agent action history is tracked across the episode.
    When the episode ends (terminated/truncated/horizon), the wrapper
    computes the *top-1 frequency* of each agent's action distribution
    — the share of steps spent on the single most-used action.  If
    ``top1 > top1_threshold``, a fixed negative reward is injected into
    the terminal step's rewards dict, where it flows through to
    ``MixedPolicyEvaluator`` (which sums ``rewards.values()`` each step).

    Why top-1 frequency, not Shannon entropy: with a stochastic
    softmax-sampling head (used by EA/CMA), a policy whose argmax never
    changes still samples minority actions a few percent of the time.
    That artificial sampling spread inflates entropy enough to slip a
    visually-lazy policy under any reasonable entropy threshold.  Top-1
    frequency measures the *dominant* action directly and is robust to
    that sampling noise: a policy that genuinely chases prey alternates
    headings and rarely sits above ~40–50% top-1, while a herding /
    drifting policy spends 60–100% of steps on the same heading.

    Notes
    -----
    * The penalty is applied at episode end so it only fires once per
      episode, regardless of how many env steps the trainer takes.
    * Agents that take fewer than ``min_actions`` steps are exempt
      (avoids penalising agents that died/were caught very early).
    * Set ``penalty=0.0`` to disable while keeping the wrapper in place
      (useful for ablations).
    """

    def __init__(
        self,
        inner,
        *,
        top1_threshold: float = 0.5,
        penalty: float = -25.0,
        min_actions: int = 50,
        horizon: Optional[int] = None,
    ) -> None:
        self.inner = inner
        self.possible_agents = list(getattr(inner, "possible_agents", []))
        self.agents = list(getattr(inner, "agents", self.possible_agents))
        self.metadata = getattr(inner, "metadata", {})
        self.top1_threshold = float(top1_threshold)
        self.penalty = float(penalty)
        self.min_actions = int(min_actions)
        # Step horizon for terminal-penalty detection.  The upstream
        # marl-aquarium env *does not* set truncations[agent]=True when
        # max_time_steps is reached — it just stops emitting agents on
        # the next reset.  We need our own counter so the penalty fires
        # on the final step the trainer actually executes.
        self.horizon = int(horizon) if horizon is not None else None
        # Inferred from action_space at first reset.
        self._action_n: Optional[int] = None
        self._history: Dict[str, list] = {a: [] for a in self.possible_agents}
        self._step_count: int = 0
        # Diagnostic: last episode's per-agent top-1 frequencies and
        # whether each was penalised.  Useful for logging / debugging.
        self.last_top1: Dict[str, float] = {}
        self.last_penalised: Dict[str, bool] = {}

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def _infer_action_n(self) -> int:
        if self._action_n is None and self.possible_agents:
            space = self.inner.action_space(self.possible_agents[0])
            self._action_n = int(getattr(space, "n", 0)) or 1
        return int(self._action_n or 1)

    def reset(self, seed=None, options=None):
        out = self.inner.reset(seed=seed, options=options)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))
        self._history = {a: [] for a in self.possible_agents}
        self._step_count = 0
        self.last_top1 = {}
        self.last_penalised = {}
        return out

    def step(self, actions):
        for agent, act in actions.items():
            self._history.setdefault(agent, []).append(int(act))

        observations, rewards, terminations, truncations, infos = self.inner.step(actions)
        self.agents = list(getattr(self.inner, "agents", []))
        self._step_count += 1

        episode_done = (
            (terminations and all(terminations.values()))
            or (truncations and all(truncations.values()))
            or (not self.agents)
            or (self.horizon is not None and self._step_count >= self.horizon)
        )
        if episode_done and self.penalty != 0.0:
            self._apply_terminal_penalty(rewards, infos)

        return observations, rewards, terminations, truncations, infos

    def _apply_terminal_penalty(self, rewards: Dict[str, float], infos: Dict[str, Any]) -> None:
        action_n = self._infer_action_n()
        for agent, history in self._history.items():
            if len(history) < self.min_actions:
                continue
            counts = np.bincount(np.asarray(history, dtype=np.int64), minlength=action_n)
            total = int(counts.sum())
            if total <= 0:
                continue
            top1 = float(counts.max()) / float(total)
            self.last_top1[agent] = top1
            if top1 > self.top1_threshold:
                rewards[agent] = float(rewards.get(agent, 0.0)) + self.penalty
                self.last_penalised[agent] = True
                if isinstance(infos, dict):
                    info = infos.setdefault(agent, {})
                    if isinstance(info, dict):
                        info["lazy_penalty_applied"] = True
                        info["lazy_action_top1"] = top1
            else:
                self.last_penalised[agent] = False