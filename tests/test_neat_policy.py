"""Tests for NeatPolicy."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")
import gymnasium.spaces as spaces

from deap_pettingzoo.contrib.neat.config import build_neat_config
from deap_pettingzoo.contrib.neat.policy import NeatPolicy


class TestNeatPolicy:
    """Tests for NeatPolicy."""

    @pytest.fixture
    def neat_config(self):
        return build_neat_config(input_size=8, output_size=5, pop_size=10)

    @pytest.fixture
    def genome(self, neat_config):
        pop = neat.Population(neat_config)
        # Grab a random genome from the initial population.
        return next(iter(pop.population.values()))

    def test_construct_discrete(self, genome, neat_config):
        action_space = spaces.Discrete(5)
        policy = NeatPolicy(
            genome, neat_config, action_space=action_space
        )
        assert policy._head is not None
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5

    def test_construct_continuous(self, genome, neat_config):
        action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        policy = NeatPolicy(
            genome, neat_config, action_space=action_space
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_construct_raw(self, genome, neat_config):
        """No action_space -> raw network output."""
        policy = NeatPolicy(genome, neat_config)
        obs = np.random.randn(8).astype(np.float32)
        output = policy.get_action(obs)
        assert isinstance(output, np.ndarray)
        assert output.shape == (5,)

    def test_masked_discrete(self, genome, neat_config):
        action_space = spaces.Discrete(5)
        policy = NeatPolicy(
            genome, neat_config, action_space=action_space, masked=True
        )
        obs = np.random.randn(8).astype(np.float32)
        # Mask out action 0 only; the action should be 1–4.
        mask = np.array([False, True, True, True, True])
        action = policy.get_action(obs, action_mask=mask)
        assert isinstance(action, (int, np.integer))
        assert 1 <= action < 5

    def test_reset_is_noop(self, genome, neat_config):
        policy = NeatPolicy(genome, neat_config)
        # Should not raise.
        policy.reset()
        # get_action still works.
        obs = np.random.randn(8).astype(np.float32)
        assert policy.get_action(obs) is not None

    def test_convenience_properties(self, genome, neat_config):
        policy = NeatPolicy(genome, neat_config)
        assert policy.genome is genome
        assert policy.config is neat_config

    def test_from_snapshot_payload(self, genome, neat_config):
        policy = NeatPolicy.from_snapshot_payload(
            genome, neat_config, action_space=spaces.Discrete(5)
        )
        obs = np.random.randn(8).astype(np.float32)
        assert isinstance(policy.get_action(obs), (int, np.integer))
