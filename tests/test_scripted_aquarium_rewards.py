from __future__ import annotations

import numpy as np
import pytest

from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    ScriptedOpponentEnvWrapper,
)


class _Position:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


class _Entity:
    def __init__(self, name: str, x: float, y: float, *, recently_died: bool = False) -> None:
        self._name = name
        self.position = _Position(x, y)
        self.recently_died = recently_died

    def id(self) -> str:
        return self._name


class _Torus:
    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)

    def get_distance_in_torus(self, position1: _Position, position2: _Position) -> float:
        dx = min(abs(position1.x - position2.x), self.width - abs(position1.x - position2.x))
        dy = min(abs(position1.y - position2.y), self.height - abs(position1.y - position2.y))
        return float((dx * dx + dy * dy) ** 0.5)


class _FakeAquariumEnv:
    def __init__(
        self,
        predator_positions: list[tuple[float, float]],
        prey_positions: list[tuple[float, float]],
        *,
        width: float = 100.0,
        height: float = 100.0,
        rewards: dict[str, float] | None = None,
        recently_died: bool = False,
    ) -> None:
        self.predators = [
            _Entity(f"predator_{idx}", x, y)
            for idx, (x, y) in enumerate(predator_positions)
        ]
        self.prey = [
            _Entity(f"prey_{idx}", x, y, recently_died=(recently_died and idx == 0))
            for idx, (x, y) in enumerate(prey_positions)
        ]
        self.possible_agents = [entity.id() for entity in self.predators + self.prey]
        self.agents = list(self.possible_agents)
        self.torus = _Torus(width, height)
        self._rewards = dict(rewards or {})
        self.last_actions: dict[str, int] | None = None

    def observation_space(self, agent: str):
        raise NotImplementedError(agent)

    def action_space(self, agent: str):
        raise NotImplementedError(agent)

    def reset(self, seed=None, options=None):
        del seed, options
        self.agents = list(self.possible_agents)
        observations = {
            agent: np.zeros(17, dtype=np.float64) for agent in self.possible_agents
        }
        infos = {agent: {} for agent in self.possible_agents}
        return observations, infos

    def step(self, actions):
        self.last_actions = dict(actions)
        observations = {
            agent: np.zeros(17, dtype=np.float64) for agent in self.possible_agents
        }
        rewards = {
            agent: float(self._rewards.get(agent, 0.0)) for agent in self.possible_agents
        }
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None


def test_recently_died_bonus_preserves_team_total() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(10.0, 10.0)] * 5,
        prey_positions=[(20.0, 20.0)],
        recently_died=True,
    )
    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        predator_reward=10,
        reward_shape_alpha=0.0,
    )

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert set(inner.last_actions or {}) == set(inner.possible_agents)
    assert sum(rewards.values()) == pytest.approx(10.0)
    assert all(reward == pytest.approx(2.0) for reward in rewards.values())
    assert wrapper.prey_caught == 1
    assert wrapper.last_team_reward == pytest.approx(10.0)


def test_reward_shaping_uses_torus_distance() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(95.0, 50.0)],
        prey_positions=[(5.0, 50.0)],
    )
    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        reward_shape_alpha=1.0,
    )

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(0.0)


def test_reward_shaping_rewards_closing_distance() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(95.0, 50.0)],
        prey_positions=[(5.0, 50.0)],
    )
    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        reward_shape_alpha=1.0,
    )

    observations, _ = wrapper.reset(seed=0)
    # Simulate the predator moving 4 px closer across the torus before the step reward is shaped.
    inner.predators[0].position.x = 99.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    expected = (10.0 - 6.0) / 200.0
    assert rewards["predator_0"] == pytest.approx(expected)


def test_fov_shape_alpha_rewards_visible_prey() -> None:
    """fov_shape_alpha adds alpha * n_visible_prey per step.

    When fov_enabled is absent (defaults to False in _count_fov_visible_prey),
    all prey count as visible.  With alpha=1.0 and 2 prey the bonus is 2.0.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0), (70.0, 50.0)],
    )
    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        reward_shape_alpha=0.0,
        fov_shape_alpha=1.0,
    )

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    # 2 prey visible (fov_enabled absent -> all visible), alpha=1.0 -> bonus = 2.0
    assert rewards["predator_0"] == pytest.approx(2.0)


def test_fov_shape_alpha_zero_means_no_bonus() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
    )
    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        reward_shape_alpha=0.0,
        fov_shape_alpha=0.0,
    )

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(0.0)


def test_proximity_shaping_skipped_when_prey_outside_fov() -> None:
    """No proximity shaping when the only prey is outside the predator's FOV.

    We attach ``fov_enabled=True``, ``predator_fov=90``, and a torus
    ``check_if_entity_is_in_view_in_torus`` that always returns False
    (simulating prey behind the FOV cone).  The expected reward is 0.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
    )
    # Attach FOV attributes and a check function that always says "not in view".
    inner.fov_enabled = True
    inner.predator_view_distance = 500
    inner.predator_fov = 90
    inner.torus.check_if_entity_is_in_view_in_torus = lambda obs, tgt, vd, fov: False

    wrapper = ScriptedOpponentEnvWrapper(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        reward_shape_alpha=1.0,
        fov_shape_alpha=0.0,
    )

    observations, _ = wrapper.reset(seed=0)
    # Move predator closer so there would normally be a proximity bonus.
    inner.predators[0].position.x = 54.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    # Prey is outside FOV -> no proximity shaping -> reward == 0.0
    assert rewards["predator_0"] == pytest.approx(0.0)