"""Potential-based shaping in the coevolution both-sides aquarium adapter.

The properties asserted here are the ones the previous per-step-delta shaping
lacked: the episode total telescopes, the potential is defined on every step
(no FOV gate), and entities are keyed by identity so removing a caught prey
cannot re-pair the survivors.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from examples.pursuit_evasion.aquarium_coevolution import _CoevBothSidesAdapter


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

    def move_to(self, x: float, y: float) -> None:
        self.position = _Position(x, y)


class _Torus:
    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)

    def get_distance_in_torus(self, position1: _Position, position2: _Position) -> float:
        dx = min(abs(position1.x - position2.x), self.width - abs(position1.x - position2.x))
        dy = min(abs(position1.y - position2.y), self.height - abs(position1.y - position2.y))
        return float((dx * dx + dy * dy) ** 0.5)


class _FakeAquariumEnv:
    """Minimal stand-in for the raw marl-aquarium parallel env."""

    def __init__(
        self,
        predator_positions: List[tuple],
        prey_positions: List[tuple],
        *,
        width: float = 1000.0,
        height: float = 1000.0,
    ) -> None:
        self.predators = [
            _Entity(f"predator_{idx}", x, y)
            for idx, (x, y) in enumerate(predator_positions)
        ]
        self.prey = [
            _Entity(f"prey_{idx}", x, y) for idx, (x, y) in enumerate(prey_positions)
        ]
        self.possible_agents = [e.id() for e in self.predators + self.prey]
        self.agents = list(self.possible_agents)
        self.torus = _Torus(width, height)
        self.predator_max_velocity = 12.0
        self.prey_max_velocity = 12.0

    def observation_space(self, agent: str):
        raise NotImplementedError(agent)

    def action_space(self, agent: str):
        raise NotImplementedError(agent)

    def reset(self, seed=None, options=None):
        del seed, options
        self.agents = list(self.possible_agents)
        return (
            {a: np.zeros(4, dtype=np.float64) for a in self.agents},
            {a: {} for a in self.agents},
        )

    def step(self, actions):
        del actions
        # Only live agents appear in the step outputs, as in the real env with
        # keep_prey_count_constant=False.
        obs = {a: np.zeros(4, dtype=np.float64) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        flags = {a: False for a in self.agents}
        return obs, rewards, dict(flags), dict(flags), {a: {} for a in self.agents}

    def kill_prey(self, name: str) -> None:
        """Remove a caught prey, as the real env does on a catch."""
        self.prey = [p for p in self.prey if p.id() != name]
        self.agents = [a for a in self.agents if a != name]


def _adapter(inner, *, pred_alpha: float = 0.0, prey_alpha: float = 0.0):
    return _CoevBothSidesAdapter(
        inner,
        predator_observe_count=1,
        prey_observe_count=1,
        pred_shape_alpha=pred_alpha,
        prey_shape_alpha=prey_alpha,
    )


def _step(adapter) -> Dict[str, float]:
    _, rewards, _, _, _ = adapter.step({a: 0 for a in adapter.agents})
    return rewards


def test_closed_loop_shaping_telescopes_to_zero() -> None:
    """A predator returning to its start earns ~0 net shaping.

    This is the defining property of a potential: the total depends only on
    the endpoints, so circling prey at a constant distance cannot be farmed.
    """
    inner = _FakeAquariumEnv([(100.0, 100.0)], [(500.0, 100.0)])
    adapter = _adapter(inner, pred_alpha=1.0)
    adapter.reset(seed=0)

    total = 0.0
    for x, y in [(110.0, 100.0), (110.0, 110.0), (100.0, 110.0), (100.0, 100.0)]:
        inner.predators[0].move_to(x, y)
        total += _step(adapter)["predator_0"]

    assert total == pytest.approx(0.0, abs=1e-9)
    assert adapter.shaping_totals["predator"] == pytest.approx(0.0, abs=1e-9)


def test_shaping_equals_potential_difference() -> None:
    """One step of closing pays exactly ``alpha * Δd / 200``."""
    inner = _FakeAquariumEnv([(100.0, 100.0)], [(500.0, 100.0)])
    adapter = _adapter(inner, pred_alpha=0.5)
    adapter.reset(seed=0)

    inner.predators[0].move_to(110.0, 100.0)
    assert _step(adapter)["predator_0"] == pytest.approx(0.5 * (10.0 / 200.0))


def test_shaping_ignores_the_fov_gate() -> None:
    """The potential covers every prey, visible or not.

    Gating on the view cone leaves Φ undefined on most steps under the shipped
    config (400-unit view distance on a 1500x1500 arena), which is what broke
    telescoping in the campaign.
    """
    inner = _FakeAquariumEnv([(100.0, 100.0)], [(500.0, 100.0)])
    inner.fov_enabled = True
    inner.predator_fov = 90
    inner.predator_view_distance = 50
    inner.torus.check_if_entity_is_in_view_in_torus = lambda *args: False

    adapter = _adapter(inner, pred_alpha=1.0)
    adapter.reset(seed=0)

    inner.predators[0].move_to(120.0, 100.0)
    assert _step(adapter)["predator_0"] == pytest.approx(20.0 / 200.0)


def test_prey_shaping_rewards_opening_distance() -> None:
    inner = _FakeAquariumEnv([(500.0, 500.0)], [(300.0, 500.0)])
    adapter = _adapter(inner, prey_alpha=1.0)
    adapter.reset(seed=0)

    inner.prey[0].move_to(280.0, 500.0)
    assert _step(adapter)["prey_0"] == pytest.approx(20.0 / 200.0)


def test_surviving_prey_keeps_its_own_potential_after_a_catch() -> None:
    """Entities are keyed by identity, not by roster position.

    With ``keep_prey_count_constant=False`` a caught prey is removed from the
    raw env's list.  Index keying would hand ``prey_0``'s slot to the entity
    that used to be ``prey_1`` — every survivor's potential would jump, and the
    dead name would keep drawing shaping.
    """
    inner = _FakeAquariumEnv([(500.0, 500.0)], [(100.0, 100.0), (300.0, 500.0)])
    adapter = _adapter(inner, prey_alpha=1.0)
    adapter.reset(seed=0)

    inner.kill_prey("prey_0")
    inner.prey[0].move_to(280.0, 500.0)   # prey_1 opens 20 units
    rewards = _step(adapter)

    assert rewards["prey_1"] == pytest.approx(20.0 / 200.0)
    assert "prey_0" not in rewards   # dead names draw no shaping


def test_no_shaping_once_every_target_is_gone() -> None:
    """With no prey left the potential is undefined and shaping stops."""
    inner = _FakeAquariumEnv([(100.0, 100.0)], [(500.0, 100.0)])
    adapter = _adapter(inner, pred_alpha=1.0)
    adapter.reset(seed=0)

    inner.kill_prey("prey_0")
    rewards = _step(adapter)

    assert rewards["predator_0"] == pytest.approx(0.0)
    assert adapter.shaping_totals["predator"] == pytest.approx(0.0)


def test_target_switch_delta_is_clipped() -> None:
    """The nearest-prey switch on a catch cannot inject an unbounded reward."""
    inner = _FakeAquariumEnv([(500.0, 500.0)], [(490.0, 500.0), (100.0, 500.0)])
    adapter = _adapter(inner, pred_alpha=1.0)
    adapter.reset(seed=0)

    # The near prey is caught: the nearest distance jumps 10 -> 400.
    inner.kill_prey("prey_0")
    reward = _step(adapter)["predator_0"]

    cap = 1.5 * (inner.predator_max_velocity + inner.prey_max_velocity)
    assert reward == pytest.approx(-cap / 200.0)


def test_shaping_totals_reset_per_episode() -> None:
    inner = _FakeAquariumEnv([(100.0, 100.0)], [(500.0, 100.0)])
    adapter = _adapter(inner, pred_alpha=1.0)
    adapter.reset(seed=0)

    inner.predators[0].move_to(110.0, 100.0)
    _step(adapter)
    assert adapter.shaping_totals["predator"] != 0.0

    adapter.reset(seed=1)
    assert adapter.shaping_totals == {"predator": 0.0, "prey": 0.0}


def test_missing_raw_env_raises_instead_of_shaping_silently() -> None:
    """A shaping request that cannot be honoured must fail loudly."""

    class _Bare:
        possible_agents = ["predator_0", "prey_0"]
        agents = ["predator_0", "prey_0"]

    with pytest.raises(RuntimeError, match="raw aquarium env"):
        _adapter(_Bare(), pred_alpha=1.0)
