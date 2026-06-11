"""Tests for NEAT genome pickling (via OpponentSnapshot kind="neat")."""

import pickle

import numpy as np
import pytest

neat = pytest.importorskip("neat")
import gymnasium.spaces as spaces

from deap_pettingzoo.contrib.neat.config import build_neat_config
from deap_pettingzoo.baselines.coevolution.pools import (
    OpponentSnapshot,
    rebuild_opponent_policy,
)


class TestNeatGenomePickle:
    """Genome picklability and round-trip via OpponentSnapshot."""

    def test_genome_is_picklable(self):
        cfg = build_neat_config(input_size=4, output_size=2, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))
        data = pickle.dumps(genome)
        genome2 = pickle.loads(data)
        assert genome2.key == genome.key

    def test_snapshot_round_trip_discrete(self):
        cfg = build_neat_config(input_size=4, output_size=3, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))

        snapshot = OpponentSnapshot(
            kind="neat",
            payload={
                "genome": genome,
                "config_input_size": 4,
                "config_output_size": 3,
            },
        )
        # Verify snapshot is picklable.
        data = pickle.dumps(snapshot)
        snapshot2 = pickle.loads(data)

        # Rebuild policy via rebuild_opponent_policy.
        action_space = spaces.Discrete(3)
        policy = rebuild_opponent_policy(
            snapshot2, action_space=action_space
        )
        from deap_pettingzoo.contrib.neat.policy import NeatPolicy
        assert isinstance(policy, NeatPolicy)

        obs = np.random.randn(4).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3

    def test_snapshot_round_trip_continuous(self):
        cfg = build_neat_config(input_size=4, output_size=2, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))

        snapshot = OpponentSnapshot(
            kind="neat",
            payload={
                "genome": genome,
                "config_input_size": 4,
                "config_output_size": 2,
            },
        )

        action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        policy = rebuild_opponent_policy(
            snapshot, action_space=action_space
        )
        obs = np.random.randn(4).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
