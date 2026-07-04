"""Tests for QDax trajectory hook and position extractors.

These tests exercise the ``trajectory_hook`` plumbing in
``_QDaxPolicyEvaluator`` and the built-in position extractors without
requiring JAX or qdax (to keep the test suite fast on machines that lack
those optional deps).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal PettingZooWrapper stub for testing position extractors."""

    possible_agents = ["pursuer_0", "pursuer_1", "prey_0"]

    def action_space(self, agent):  # noqa: D401
        return MagicMock(n=4)


class _FakeEnvWithEntities(_FakeEnv):
    """Stub that exposes predator/prey entity lists (aquarium-style)."""

    def __init__(self):
        class _Entity:
            def __init__(self, x, y):
                self.position = (x, y)

        self.predators = [_Entity(100, 200), _Entity(300, 400)]
        self.prey = [_Entity(1000, 1500)]


class _FakeAquariumVector:
    """Minimal stand-in for marl-aquarium's non-subscriptable Vector."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeEnvWithVectorEntities(_FakeEnv):
    """Aquarium-style entity lists whose positions expose x/y attributes."""

    def __init__(self):
        class _Entity:
            def __init__(self, x, y):
                self.position = _FakeAquariumVector(x, y)

        self.predators = [_Entity(100, 200), _Entity(300, 400)]
        self.prey = [_Entity(1000, 1500)]


# ---------------------------------------------------------------------------
# Position extractor tests
# ---------------------------------------------------------------------------


def test_obs_first_two_extracts_positions() -> None:
    from deap_pettingzoo.contrib.qdax.position_extractors import obs_first_two

    env = _FakeEnv()
    obs_dict = {
        "pursuer_0": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "pursuer_1": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "prey_0": np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
    }
    result = obs_first_two(obs_dict, env)
    assert result is not None
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[1], [5.0, 6.0])
    np.testing.assert_allclose(result[2], [9.0, 10.0])


def test_obs_first_two_returns_none_when_agent_missing() -> None:
    from deap_pettingzoo.contrib.qdax.position_extractors import obs_first_two

    env = _FakeEnv()
    obs_dict = {
        "pursuer_0": np.array([1.0, 2.0], dtype=np.float32),
        # pursuer_1 absent (done)
        "prey_0": np.array([9.0, 10.0], dtype=np.float32),
    }
    assert obs_first_two(obs_dict, env) is None


def test_obs_indices_extractor() -> None:
    from deap_pettingzoo.contrib.qdax.position_extractors import obs_indices

    env = _FakeEnv()
    obs_dict = {
        "pursuer_0": np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "pursuer_1": np.array([0.0, 0.0, 5.0, 6.0], dtype=np.float32),
        "prey_0": np.array([0.0, 0.0, 9.0, 10.0], dtype=np.float32),
    }
    extractor = obs_indices(2, 4)
    result = extractor(obs_dict, env)
    assert result is not None
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[2], [9.0, 10.0])


def test_aquarium_extractor_reads_entity_positions() -> None:
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    env = _FakeEnvWithEntities()
    extractor = make_aquarium_extractor(arena_width=2000.0, arena_height=3000.0)
    result = extractor({}, env)
    assert result is not None
    assert result.shape == (3, 2)
    # predator_0 at (100, 200) → (0.05, 0.0667)
    np.testing.assert_allclose(result[0, 0], 100 / 2000, atol=1e-5)
    np.testing.assert_allclose(result[0, 1], 200 / 3000, atol=1e-5)
    # prey_0 at (1000, 1500) → (0.5, 0.5)
    np.testing.assert_allclose(result[2, 0], 0.5, atol=1e-5)
    np.testing.assert_allclose(result[2, 1], 0.5, atol=1e-5)


def test_aquarium_extractor_reads_vector_positions() -> None:
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    env = _FakeEnvWithVectorEntities()
    extractor = make_aquarium_extractor(arena_width=2000.0, arena_height=3000.0)
    result = extractor({}, env)
    assert result is not None
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [0.05, 200 / 3000], atol=1e-5)
    np.testing.assert_allclose(result[2], [0.5, 0.5], atol=1e-5)


def test_aquarium_extractor_can_put_prey_first() -> None:
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    env = _FakeEnvWithEntities()
    extractor = make_aquarium_extractor(
        arena_width=2000.0,
        arena_height=3000.0,
        role_order=("prey", "predator"),
    )
    result = extractor({}, env)
    assert result is not None
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [0.5, 0.5], atol=1e-5)
    np.testing.assert_allclose(result[1], [0.05, 200 / 3000], atol=1e-5)


def test_aquarium_extractor_skips_frames_after_entity_removal() -> None:
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    env = _FakeEnvWithEntities()
    env.possible_agents = ["predator_0", "predator_1", "prey_0"]
    env.prey = []
    extractor = make_aquarium_extractor(arena_width=2000.0, arena_height=3000.0)
    assert extractor({}, env) is None


def test_qdax_descriptor_skips_variable_shape_frames() -> None:
    from deap_pettingzoo.contrib.qdax.coevolution import _compute_qdax_descriptor

    class _Descriptor:
        min_bd = (0.0, 0.0)
        max_bd = (1.0, 1.0)

        def __call__(self, trajectory):
            positions = trajectory["positions"]
            return np.array([positions.shape[0] / 10.0, positions.shape[1] / 10.0])

    frames = [
        np.zeros((3, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
    ]
    result = _compute_qdax_descriptor(frames, _Descriptor())
    np.testing.assert_allclose(result, [0.2, 0.3])


@pytest.mark.skipif(
    os.name != "nt",
    reason="_mask_optional_pyarrow_imports only masks pyarrow on Windows; it is a "
    "no-op elsewhere, so the ImportError it guards against cannot be raised.",
)
def test_mask_optional_pyarrow_imports_blocks_pyarrow_only() -> None:
    from deap_pettingzoo.contrib.qdax.coevolution import _mask_optional_pyarrow_imports

    with _mask_optional_pyarrow_imports():
        with pytest.raises(ImportError):
            __import__("pyarrow")

        math_mod = __import__("math")
        assert math_mod.sqrt(9) == 3.0


def test_qdax_bootstrap_replaces_temporary_repertoire() -> None:
    jax = pytest.importorskip("jax")

    from deap_pettingzoo.contrib.qdax.coevolution import QDaxSideState

    state = object.__new__(QDaxSideState)
    state._key = jax.random.PRNGKey(0)
    state._init_genotypes = object()
    state._centroids = object()
    state._emitter_state = object()

    class _MapElites:
        def init(self, genotypes, centroids, key):
            assert genotypes is state._init_genotypes
            assert centroids is state._centroids
            state._last_batch_fitness = np.array([7.0, 11.0])
            return SimpleNamespace(fitnesses=np.array([7.0, 11.0])), "emitter", {}

    class _Pool:
        def __init__(self):
            self.updated_with = None

        def update(self, repertoire):
            self.updated_with = repertoire

    pool = _Pool()
    state._map_elites = _MapElites()
    state._pool = pool

    state.evaluate_population()

    np.testing.assert_allclose(state._repertoire.fitnesses, [7.0, 11.0])
    assert state._emitter_state == "emitter"
    assert pool.updated_with is state._repertoire


def test_qdax_generation_stats_use_current_batch_not_archive() -> None:
    jax = pytest.importorskip("jax")

    from deap_pettingzoo.contrib.qdax.coevolution import QDaxSideState

    state = object.__new__(QDaxSideState)
    state._key = jax.random.PRNGKey(0)
    state._emitter_state = object()
    state._repertoire = SimpleNamespace(fitnesses=np.array([0.0, 100.0]))
    state._last_batch_fitness = np.array([1.0])
    state.generation = 0
    state.side = SimpleNamespace(name="predators")
    state.best_fitness = -np.inf
    state.history = []

    class _MapElites:
        def update(self, repertoire, emitter_state, key):
            state._last_batch_fitness = np.array([10.0, 20.0, 30.0])
            return SimpleNamespace(fitnesses=np.array([0.0, 100.0])), emitter_state, {}

    class _Pool:
        def __init__(self):
            self.updated_with = None

        def update(self, repertoire):
            self.updated_with = repertoire

    pool = _Pool()
    state._map_elites = _MapElites()
    state._pool = pool

    stats = state.step_generation(lambda _state: object())

    assert stats.generation == 1
    assert stats.side_name == "predators"
    assert stats.mean == pytest.approx(20.0)
    assert stats.min == pytest.approx(10.0)
    assert stats.max == pytest.approx(30.0)
    assert stats.best_fitness == pytest.approx(30.0)
    assert state.best_fitness == pytest.approx(100.0)
    assert pool.updated_with is state._repertoire


def test_aquarium_extractor_walks_wrapper_chain() -> None:
    """Extractor should find entities through a one-level wrapper."""
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    class _Wrapper:
        def __init__(self):
            self.env = _FakeEnvWithEntities()

    extractor = make_aquarium_extractor()
    result = extractor({}, _Wrapper())
    assert result is not None
    assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# trajectory_hook plumbing in _QDaxPolicyEvaluator
# ---------------------------------------------------------------------------


def _build_qdax_policy_evaluator():
    """Build a _QDaxPolicyEvaluator with a stub env and one opponent side."""
    import sys
    import types

    # Provide a minimal fake trainer module so the import works without the
    # full trainer loading all optional deps.
    from deap_pettingzoo.baselines.coevolution.trainer import _QDaxPolicyEvaluator
    from deap_pettingzoo.baselines.coevolution.pools import FrozenPolicyPool

    class _RandomPolicy:
        def __init__(self):
            self.calls = 0

        def get_action(self, obs, action_mask=None):
            return 0

        def reset(self):
            self.calls = 0

    class _StubEnv:
        possible_agents = ["learner_0", "opponent_0"]

        def reset(self, seed=None):
            return {
                "learner_0": np.zeros(4, dtype=np.float32),
                "opponent_0": np.zeros(4, dtype=np.float32),
            }

        def step(self, actions):
            obs = {}
            rew = {"learner_0": 1.0, "opponent_0": -1.0}
            terms = {"learner_0": True, "opponent_0": True}
            truncs = {"learner_0": False, "opponent_0": False}
            return obs, rew, terms, truncs, {}

        def action_space(self, agent):
            return MagicMock(n=4, sample=lambda: 0)

        last_infos = {}

    policy = _RandomPolicy()
    pool = FrozenPolicyPool([_RandomPolicy()])
    evaluator = _QDaxPolicyEvaluator(
        env_wrapper=_StubEnv(),
        learner_role_prefixes=("learner",),
        opponent_sides=[("opp", ("opponent",), pool)],
        k_opponents=1,
        aggregation="mean",
        max_steps=5,
        rng_factory=lambda: np.random.default_rng(0),
    )
    return evaluator, policy


def test_trajectory_hook_is_called() -> None:
    """Hook must be called with (obs_dict, env_wrapper) on each step."""
    pytest.importorskip("pettingzoo")
    evaluator, policy = _build_qdax_policy_evaluator()

    calls: List[tuple] = []

    def _hook(obs_dict, env_wrapper):
        calls.append((dict(obs_dict), env_wrapper))

    evaluator.trajectory_hook = _hook
    evaluator.evaluate(policy)

    assert len(calls) >= 1, "hook should be called at least once per episode"
    obs_arg, _ = calls[0]
    assert isinstance(obs_arg, dict)


def test_trajectory_hook_none_by_default() -> None:
    """trajectory_hook must default to None (no hook fires by default)."""
    pytest.importorskip("pettingzoo")
    evaluator, _ = _build_qdax_policy_evaluator()
    assert evaluator.trajectory_hook is None


def test_trajectory_hook_exception_does_not_crash_episode() -> None:
    """A hook that raises must not abort the episode."""
    pytest.importorskip("pettingzoo")
    evaluator, policy = _build_qdax_policy_evaluator()

    def _bad_hook(obs_dict, env_wrapper):
        raise RuntimeError("deliberate test error")

    evaluator.trajectory_hook = _bad_hook
    fitness = evaluator.evaluate(policy)  # must not raise
    assert isinstance(fitness, float)
