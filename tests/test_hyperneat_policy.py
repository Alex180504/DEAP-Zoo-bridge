"""Tests for HyperNeatPolicy and ESHyperNeatPolicy."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")
import gymnasium.spaces as spaces

from deap_pettingzoo.contrib.neat.config import build_cppn_config
from deap_pettingzoo.contrib.neat.substrate import make_uniform_substrate
from deap_pettingzoo.contrib.neat.hyperneat import (
    HyperNeatPolicy,
    ESHyperNeatPolicy,
)


class TestHyperNeatPolicy:
    @pytest.fixture
    def cppn_config(self):
        return build_cppn_config(input_size=5, output_size=1, pop_size=10)

    @pytest.fixture
    def genome(self, cppn_config):
        pop = neat.Population(cppn_config)
        return next(iter(pop.population.values()))

    @pytest.fixture
    def substrate(self):
        return make_uniform_substrate(8, 5, hidden_layers=[4, 4])

    def test_construct_discrete(self, genome, cppn_config, substrate):
        policy = HyperNeatPolicy(
            genome, cppn_config, substrate,
            action_space=spaces.Discrete(5),
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5

    def test_construct_continuous(self, genome, cppn_config, substrate):
        action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        policy = HyperNeatPolicy(
            genome, cppn_config, substrate,
            action_space=action_space,
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)

    def test_reset_noop(self, genome, cppn_config, substrate):
        policy = HyperNeatPolicy(genome, cppn_config, substrate)
        policy.reset()

    def test_properties(self, genome, cppn_config, substrate):
        policy = HyperNeatPolicy(genome, cppn_config, substrate)
        assert policy.genome is genome
        assert policy.substrate is substrate


class TestESHyperNeatPolicy:
    @pytest.fixture
    def cppn_config(self):
        return build_cppn_config(input_size=5, output_size=1, pop_size=10)

    @pytest.fixture
    def genome(self, cppn_config):
        pop = neat.Population(cppn_config)
        return next(iter(pop.population.values()))

    @pytest.fixture
    def substrate(self):
        # ES-HyperNEAT: no hidden nodes needed.
        return make_uniform_substrate(8, 5)

    def test_construct_discrete(self, genome, cppn_config, substrate):
        policy = ESHyperNeatPolicy(
            genome, cppn_config, substrate,
            action_space=spaces.Discrete(5),
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5

    def test_construct_continuous(self, genome, cppn_config, substrate):
        action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        policy = ESHyperNeatPolicy(
            genome, cppn_config, substrate,
            action_space=action_space,
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)

    def test_reset_noop(self, genome, cppn_config, substrate):
        policy = ESHyperNeatPolicy(genome, cppn_config, substrate)
        policy.reset()

    def test_properties(self, genome, cppn_config, substrate):
        policy = ESHyperNeatPolicy(genome, cppn_config, substrate)
        assert policy.genome is genome
        assert policy.substrate is substrate

    def test_custom_es_params(self, genome, cppn_config, substrate):
        policy = ESHyperNeatPolicy(
            genome, cppn_config, substrate,
            action_space=spaces.Discrete(5),
            es_params={"initial_depth": 1, "max_depth": 2,
                       "variance_threshold": 0.03, "band_threshold": 0.3,
                       "iteration_level": 1, "division_threshold": 0.5,
                       "max_weight": 5.0, "activation": "sigmoid"},
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
