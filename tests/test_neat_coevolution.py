"""Tests for NeatSideState in co-evolution."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.coevolution.sides import (
    Side,
    GenerationStats,
    make_side_state,
)
from deap_pettingzoo.baselines.coevolution.pools import OpponentSnapshot
from deap_pettingzoo.contrib.neat.coevolution import NeatSideState


class TestNeatSideState:
    """Tests for NeatSideState."""

    @pytest.fixture
    def side(self):
        return Side(
            name="neat_agents",
            role_prefixes=("agent_",),
            optimizer="neat",
            optimizer_config={
                "pop_size": 10,
                "seed": 42,
                "eval_episodes_per_genome": 1,
                "n_generations_per_step": 1,
            },
        )

    @pytest.fixture
    def env(self):
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )
        return PettingZooWrapper(
            "simple_spread_v3",
            env_config={"max_cycles": 25, "continuous_actions": False},
        )

    def test_factory_dispatches(self, side):
        state = make_side_state(side)
        assert isinstance(state, NeatSideState)

    def test_initialise(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        assert state._population_obj is not None
        assert state._neat_config is not None
        assert len(state._genomes_by_id) == 10  # pop_size

    def test_best_policy(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        policy = state.best_policy(env)
        assert policy is not None
        obs = np.random.randn(18).astype(np.float32)  # simple_spread obs dim
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5

    def test_current_pool(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        pool = state.current_pool()
        assert len(pool) == 10

        # Sample opponents.
        rng = np.random.default_rng(42)
        opp = pool.sample_opponent(rng)
        assert opp is not None
        obs = np.random.randn(18).astype(np.float32)  # simple_spread obs dim
        action = opp.get_action(obs)
        assert isinstance(action, (int, np.integer))

    def test_current_pool_snapshots(self, side, env):
        """Snapshot opponents should produce kind='neat' payloads."""
        state = make_side_state(side)
        state.initialise(env)
        pool = state.current_pool()

        rng = np.random.default_rng(42)
        snapshots = pool.snapshot_opponents(rng, k=3)
        assert len(snapshots) == 3
        for snap in snapshots:
            assert snap.kind == "neat"
            assert "genome" in snap.payload
            assert "config_input_size" in snap.payload
            assert "config_output_size" in snap.payload

    def test_generation_state_tracking(self, side, env):
        """After initialise, generation should be 0 before step."""
        state = make_side_state(side)
        state.initialise(env)
        assert state.generation == 0

    def test_best_snapshot(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        snapshot = state.best_snapshot()
        assert snapshot is not None
        # Should be a neat genome.
        assert hasattr(snapshot, "fitness")
        assert hasattr(snapshot, "key")
