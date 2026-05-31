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
    """Penalize agents whose headings collapse into a narrow cone.

    Symptom this targets: with a shared homogeneous genome, EA/CMA can
    discover a "constant heading" attractor where the network ignores
    observations and emits the same action almost every step.  This
    looks like rigid linear drift in the rendered episode.

    Mechanism: per-agent action history is tracked across the episode.
    When the episode ends (terminated/truncated/horizon), the wrapper
    converts those discrete actions into equally spaced unit heading
    vectors on the aquarium action circle and computes the circular
    *heading concentration* — the norm of the mean heading vector.
    If ``concentration > concentration_threshold``, a fixed negative
    reward is injected into the terminal step's rewards dict, where it flows through to
    ``MixedPolicyEvaluator`` (which sums ``rewards.values()`` each step).

    Why heading concentration instead of top-1 action frequency: a
    visually straight trajectory can dither between *adjacent* headings
    without repeating any single action often enough to trip a top-1
    threshold.  Circular concentration measures whether the agent keeps
    moving in nearly the same direction even when it jitters between
    neighboring bins, which matches the actual failure mode better.

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
        concentration_threshold: float = 0.7,
        penalty: float = 0.0,
        min_actions: int = 50,
        horizon: Optional[int] = None,
        mode: str = "terminal",
        window_size: int = 50,
    ) -> None:
        if mode not in ("terminal", "per_step"):
            raise ValueError(
                f"LazyActionPenaltyWrapper.mode must be 'terminal' or 'per_step', got {mode!r}"
            )
        self.inner = inner
        self.possible_agents = list(getattr(inner, "possible_agents", []))
        self.agents = list(getattr(inner, "agents", self.possible_agents))
        self.metadata = getattr(inner, "metadata", {})
        self.concentration_threshold = float(concentration_threshold)
        self.penalty = float(penalty)
        self.min_actions = int(min_actions)
        # Step horizon for terminal-penalty detection.  The upstream
        # marl-aquarium env *does not* set truncations[agent]=True when
        # max_time_steps is reached — it just stops emitting agents on
        # the next reset.  We need our own counter so the penalty fires
        # on the final step the trainer actually executes.
        self.horizon = int(horizon) if horizon is not None else None
        # ``mode='terminal'`` (default): apply the full ``penalty`` once
        # at episode end, keyed on full-episode heading concentration.  Cheap
        # and works fine for EA/CMA which only see episode return, but
        # gives off-policy TD learning a brutal credit-assignment cliff.
        #
        # ``mode='per_step'``: at every step after the rolling window is
        # filled, compute concentration over the last ``window_size``
        # actions and inject ``penalty / eligible_steps`` into that step's
        # reward whenever the agent is currently in lazy mode.
        # ``eligible_steps`` = ``horizon - window_size + 1`` so a
        # fully-lazy episode sums to ~``penalty`` total — matching the
        # terminal-mode signal that EA/CMA learnt to avoid, while giving
        # RL a dense per-step gradient pointing away from the lazy
        # attractor.
        self.mode = str(mode)
        self.window_size = int(window_size)
        # Inferred from action_space at first reset.
        self._action_n: Optional[int] = None
        self._history: Dict[str, list] = {a: [] for a in self.possible_agents}
        self._step_count: int = 0
        # Diagnostic: last episode's per-agent heading concentrations and
        # whether each was penalised.  Useful for logging / debugging.
        self.last_heading_concentration: Dict[str, float] = {}
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

    def _heading_concentration(self, history: list[int]) -> float:
        action_n = self._infer_action_n()
        if action_n <= 0 or not history:
            return 0.0
        actions = np.mod(np.asarray(history, dtype=np.int64), action_n)
        angles = 2.0 * np.pi * np.arange(action_n, dtype=np.float64) / float(action_n)
        headings = np.stack([np.sin(angles), -np.cos(angles)], axis=1)
        mean_heading = headings[actions].mean(axis=0)
        return float(np.linalg.norm(mean_heading))

    def reset(self, seed=None, options=None):
        out = self.inner.reset(seed=seed, options=options)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))
        self._history = {a: [] for a in self.possible_agents}
        self._step_count = 0
        self.last_heading_concentration = {}
        self.last_penalised = {}
        return out

    def step(self, actions):
        for agent, act in actions.items():
            self._history.setdefault(agent, []).append(int(act))

        observations, rewards, terminations, truncations, infos = self.inner.step(actions)
        self.agents = list(getattr(self.inner, "agents", []))
        self._step_count += 1

        if self.mode == "per_step" and self.penalty != 0.0:
            self._apply_per_step_penalty(rewards, infos)

        episode_done = (
            (terminations and all(terminations.values()))
            or (truncations and all(truncations.values()))
            or (not self.agents)
            or (self.horizon is not None and self._step_count >= self.horizon)
        )
        if episode_done and self.mode == "terminal" and self.penalty != 0.0:
            self._apply_terminal_penalty(rewards, infos)

        return observations, rewards, terminations, truncations, infos

    def _per_step_amount(self) -> float:
        """Per-step penalty magnitude in 'per_step' mode.

        Scaled so a fully-lazy agent (above threshold every step after
        warmup) accumulates ~``self.penalty`` over the episode — i.e.
        EA/CMA see the same episode-return signal as in terminal mode.
        """
        horizon = int(self.horizon) if self.horizon is not None else 1
        eligible = max(1, horizon - self.window_size + 1)
        return self.penalty / float(eligible)

    def _apply_per_step_penalty(
        self, rewards: Dict[str, float], infos: Dict[str, Any]
    ) -> None:
        per_step = self._per_step_amount()
        for agent, history in self._history.items():
            if len(history) < self.window_size:
                continue
            window = history[-self.window_size:]
            concentration = self._heading_concentration(window)
            self.last_heading_concentration[agent] = concentration
            if concentration > self.concentration_threshold:
                rewards[agent] = float(rewards.get(agent, 0.0)) + per_step
                self.last_penalised[agent] = True
                if isinstance(infos, dict):
                    info = infos.setdefault(agent, {})
                    if isinstance(info, dict):
                        info["lazy_penalty_applied"] = True
                        info["lazy_heading_concentration"] = concentration
            else:
                self.last_penalised[agent] = False

    def _apply_terminal_penalty(self, rewards: Dict[str, float], infos: Dict[str, Any]) -> None:
        for agent, history in self._history.items():
            if len(history) < self.min_actions:
                continue
            concentration = self._heading_concentration(history)
            self.last_heading_concentration[agent] = concentration
            if concentration > self.concentration_threshold:
                rewards[agent] = float(rewards.get(agent, 0.0)) + self.penalty
                self.last_penalised[agent] = True
                if isinstance(infos, dict):
                    info = infos.setdefault(agent, {})
                    if isinstance(info, dict):
                        info["lazy_penalty_applied"] = True
                        info["lazy_heading_concentration"] = concentration
            else:
                self.last_penalised[agent] = False