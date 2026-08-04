"""Reward contract of the vs-scripted-APF aquarium wrappers.

Both wrappers shape on the shared potential
(:mod:`examples.pursuit_evasion.aquarium.reward_shaping`), so the tests here
pin the *properties* that make the potential valid — telescoping, no FOV
gating, a bounded target-switch — rather than the arithmetic of any single
step.  Pinning the divisor instead, as an earlier version did, passed happily
while the episode total was unbounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.pursuit_evasion.aquarium.reward_shaping import POTENTIAL_SCALE
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    AquariumScriptedPredatorWrapper,
    AquariumScriptedPreyWrapper,
)


class _Position:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


class _Entity:
    def __init__(self, name: str, x: float, y: float) -> None:
        self._name = name
        self.position = _Position(x, y)

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
        predator_max_velocity: float = 12.0,
        prey_max_velocity: float = 12.0,
    ) -> None:
        self.predators = [
            _Entity(f"predator_{idx}", x, y)
            for idx, (x, y) in enumerate(predator_positions)
        ]
        self.prey = [
            _Entity(f"prey_{idx}", x, y)
            for idx, (x, y) in enumerate(prey_positions)
        ]
        self.possible_agents = [entity.id() for entity in self.predators + self.prey]
        self.agents = list(self.possible_agents)
        # The raw env carries the arena extent alongside the torus; the
        # close-range penalty normalises distances by it.
        self.width = float(width)
        self.height = float(height)
        # Defaults match the shipped symmetric config, so the shaping cap the
        # wrappers derive here is the one the experiments actually run with.
        self.predator_max_velocity = float(predator_max_velocity)
        self.prey_max_velocity = float(prey_max_velocity)
        self.torus = _Torus(width, height)
        self._rewards = dict(rewards or {})
        self.last_actions: dict[str, int] | None = None

    def observation_space(self, agent: str):
        raise NotImplementedError(agent)

    def action_space(self, agent: str):
        raise NotImplementedError(agent)

    def kill_prey(self, name: str) -> None:
        """Remove a prey the way aquarium does with keep_prey_count_constant=False.

        The entity leaves ``self.prey`` and ``self.agents`` without any
        termination flag — the behaviour that makes positional indexing of the
        surviving agents silently wrong.
        """
        self.prey = [p for p in self.prey if p.id() != name]
        self.agents = [a for a in self.agents if a != name]

    def reset(self, seed=None, options=None):
        del seed, options
        observations = {
            agent: np.zeros(17, dtype=np.float64) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.last_actions = dict(actions)
        observations = {
            agent: np.zeros(17, dtype=np.float64) for agent in self.agents
        }
        rewards = {
            agent: float(self._rewards.get(agent, 0.0)) for agent in self.agents
        }
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None


def _pursuer_wrapper(inner, **kwargs) -> AquariumScriptedPreyWrapper:
    kwargs.setdefault("reward_shape_alpha", 1.0)
    return AquariumScriptedPreyWrapper(
        inner, predator_observe_count=1, prey_observe_count=1, **kwargs
    )


def _evader_wrapper(inner, **kwargs) -> AquariumScriptedPredatorWrapper:
    kwargs.setdefault("reward_shape_alpha", 1.0)
    return AquariumScriptedPredatorWrapper(
        inner, predator_observe_count=1, prey_observe_count=1, **kwargs
    )


# Pursuer side (predators learn, prey scripted)


def test_reward_shaping_uses_torus_distance() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(95.0, 50.0)],
        prey_positions=[(5.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner)

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(0.0)


def test_reward_shaping_rewards_closing_distance() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(95.0, 50.0)],
        prey_positions=[(5.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner)

    observations, _ = wrapper.reset(seed=0)
    # Close 4 units across the torus seam before the step reward is shaped.
    inner.predators[0].position.x = 99.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx((10.0 - 6.0) / POTENTIAL_SCALE)


def test_episode_shaping_telescopes_to_endpoint_difference() -> None:
    """The property the potential exists for: the path cannot be farmed.

    Any wandering route from ``d_0`` to ``d_T`` must pay out exactly
    ``(d_0 - d_T) / scale`` in total, so circling a prey at constant distance
    earns nothing however long the episode runs.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(10.0, 50.0)],
        prey_positions=[(50.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner)
    observations, _ = wrapper.reset(seed=0)
    start = inner.torus.get_distance_in_torus(
        inner.predators[0].position, inner.prey[0].position
    )

    total = 0.0
    # A deliberately non-monotone route: forward, back, forward again.
    for dx in (4.0, 7.0, -5.0, 9.0, -3.0, 6.0, 2.0, -8.0, 5.0):
        inner.predators[0].position.x += dx
        _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})
        total += rewards["predator_0"]

    end = inner.torus.get_distance_in_torus(
        inner.predators[0].position, inner.prey[0].position
    )
    assert total == pytest.approx((start - end) / POTENTIAL_SCALE)
    assert wrapper.shaping_total == pytest.approx(total)


def test_shaping_applies_outside_the_view_cone() -> None:
    """No FOV gating — the potential must be defined on every step.

    Gating left the potential undefined on most steps of the shipped config
    (400-unit view distance on a 1500-unit arena), which is what stopped the
    episode total from telescoping.  This asserts the inverse of the old
    behaviour on purpose.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
    )
    inner.fov_enabled = True
    inner.predator_view_distance = 500
    inner.predator_fov = 90
    inner.torus.check_if_entity_is_in_view_in_torus = lambda obs, tgt, vd, fov: False

    wrapper = _pursuer_wrapper(inner, fov_shape_alpha=0.0)
    observations, _ = wrapper.reset(seed=0)
    inner.predators[0].position.x = 54.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(4.0 / POTENTIAL_SCALE)


def test_shaping_delta_is_capped() -> None:
    """A target switch cannot inject an unbounded one-step reward.

    Killing the nearest prey moves the potential by a whole target switch
    rather than one step of motion; the symmetric clip bounds it.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(52.0, 50.0), (99.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner)
    observations, _ = wrapper.reset(seed=0)
    # Nearest prey (distance 2) is caught; the next nearest sits 49 away, a
    # 47-unit jump against the 36-unit cap.
    inner.kill_prey("prey_0")
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    cap = wrapper._shape_progress_cap
    assert cap == pytest.approx(1.5 * (12.0 + 12.0))
    assert rewards["predator_0"] == pytest.approx(-cap / POTENTIAL_SCALE)


def test_shaping_requires_the_raw_env() -> None:
    """Silently training on the unshaped reward is worse than failing."""

    class _Opaque:
        possible_agents = ["predator_0", "prey_0"]
        agents = ["predator_0", "prey_0"]

    with pytest.raises(RuntimeError, match="raw aquarium env"):
        _pursuer_wrapper(_Opaque(), reward_shape_alpha=1.0)


def test_zero_alpha_leaves_rewards_untouched() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
        rewards={"predator_0": 10.0},
    )
    wrapper = _pursuer_wrapper(inner, reward_shape_alpha=0.0)
    observations, _ = wrapper.reset(seed=0)
    inner.predators[0].position.x = 55.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(10.0)
    assert wrapper.shaping_total == pytest.approx(0.0)


# FOV-presence bonus — a separate, deliberately non-potential mechanism


def test_fov_shape_alpha_rewards_visible_prey() -> None:
    """fov_shape_alpha adds alpha * n_visible_prey per step.

    When fov_enabled is absent (defaults to False in _count_fov_visible_prey),
    all prey count as visible.  With alpha=1.0 and 2 prey the bonus is 2.0.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0), (70.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner, reward_shape_alpha=0.0, fov_shape_alpha=1.0)

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(2.0)


def test_fov_shape_alpha_zero_means_no_bonus() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
    )
    wrapper = _pursuer_wrapper(inner, reward_shape_alpha=0.0, fov_shape_alpha=0.0)

    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["predator_0"] == pytest.approx(0.0)


# Evader side (prey learn, predators scripted)


def test_evader_shaping_rewards_opening_distance() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(60.0, 50.0)],
    )
    wrapper = _evader_wrapper(inner)
    observations, _ = wrapper.reset(seed=0)
    inner.prey[0].position.x = 66.0
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["prey_0"] == pytest.approx(6.0 / POTENTIAL_SCALE)


def test_evader_episode_shaping_telescopes() -> None:
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(55.0, 50.0)],
    )
    wrapper = _evader_wrapper(inner)
    observations, _ = wrapper.reset(seed=0)
    start = inner.torus.get_distance_in_torus(
        inner.prey[0].position, inner.predators[0].position
    )

    total = 0.0
    for dx in (3.0, -6.0, 8.0, 4.0, -2.0, 7.0):
        inner.prey[0].position.x += dx
        _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})
        total += rewards["prey_0"]

    end = inner.torus.get_distance_in_torus(
        inner.prey[0].position, inner.predators[0].position
    )
    assert total == pytest.approx((end - start) / POTENTIAL_SCALE)


def test_evader_shaping_skips_dead_prey() -> None:
    """A caught prey is absent from the potential, not credited at distance 0.

    Entities are looked up by their own ``id()``; a prey removed from the raw
    env simply drops out, so no surviving agent inherits its distance and the
    dead one earns nothing.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(20.0, 50.0), (80.0, 50.0)],
    )
    wrapper = _evader_wrapper(inner)
    observations, _ = wrapper.reset(seed=0)

    inner.kill_prey("prey_0")
    inner.prey[0].position.x = 84.0          # the survivor opens by 4
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    assert rewards["prey_0"] == pytest.approx(0.0)
    assert rewards["prey_1"] == pytest.approx(4.0 / POTENTIAL_SCALE)


def test_close_range_penalty_is_independent_of_shape_alpha() -> None:
    """The danger penalty depends on the current state only.

    It used to sit inside the ``reward_shape_alpha > 0`` branch, so turning
    shaping off silently disabled it too.
    """
    inner = _FakeAquariumEnv(
        predator_positions=[(50.0, 50.0)],
        prey_positions=[(50.5, 50.0)],
    )
    wrapper = _evader_wrapper(
        inner, reward_shape_alpha=0.0, close_range_penalty=5.0, close_range_sigma2=0.01
    )
    observations, _ = wrapper.reset(seed=0)
    _, rewards, _, _, _ = wrapper.step({agent: 0 for agent in observations})

    d_norm = 0.5 / 100.0
    expected = -5.0 * float(np.exp(-(d_norm * d_norm) / 0.01))
    assert rewards["prey_0"] == pytest.approx(expected)
