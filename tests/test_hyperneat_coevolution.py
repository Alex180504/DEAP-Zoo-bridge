"""Tests for HyperNEAT / ES-HyperNEAT co-evolution side states."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.coevolution.sides import (
    Side,
    make_side_state,
)
from deap_pettingzoo.contrib.neat.hyperneat_coevolution import (
    HyperNeatSideState,
    ESHyperNeatSideState,
)


class TestHyperNeatSideState:
    @pytest.fixture
    def side(self):
        return Side(
            name="hyperneat_agents",
            role_prefixes=("agent_",),
            optimizer="hyperneat",
            optimizer_config={
                "pop_size": 6,
                "seed": 42,
                "eval_episodes_per_genome": 1,
                "n_generations_per_step": 1,
                "substrate_hidden_layers": [4],
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
        assert isinstance(state, HyperNeatSideState)

    def test_initialise(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        assert state._population_obj is not None
        assert state._neat_config is not None
        assert len(state._genomes_by_id) == 6

    def test_best_policy(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        policy = state.best_policy(env)
        assert policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))

    def test_current_pool(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        pool = state.current_pool()
        assert len(pool) == 6
        rng = np.random.default_rng(42)
        opp = pool.sample_opponent(rng)
        obs = np.random.randn(18).astype(np.float32)
        action = opp.get_action(obs)
        assert isinstance(action, (int, np.integer))


class TestESHyperNeatSideState:
    @pytest.fixture
    def side(self):
        return Side(
            name="es_hyperneat_agents",
            role_prefixes=("agent_",),
            optimizer="es_hyperneat",
            optimizer_config={
                "pop_size": 6,
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
        assert isinstance(state, ESHyperNeatSideState)

    def test_initialise(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        assert state._population_obj is not None
        assert len(state._genomes_by_id) == 6

    def test_best_policy(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        policy = state.best_policy(env)
        assert policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
