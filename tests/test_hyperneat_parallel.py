"""Tests for HyperNEAT / ES-HyperNEAT parallel snapshot roundtrips."""

import pickle

import numpy as np
import pytest

neat = pytest.importorskip("neat")
import gymnasium.spaces as spaces

from deap_pettingzoo.baselines.coevolution.pools import (
    OpponentSnapshot,
    rebuild_opponent_policy,
)
from deap_pettingzoo.contrib.neat.config import build_cppn_config
from deap_pettingzoo.contrib.neat.substrate import make_uniform_substrate


class TestHyperNeatParallel:
    def test_hyperneat_snapshot_roundtrip(self):
        cfg = build_cppn_config(input_size=5, output_size=1, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))

        snapshot = OpponentSnapshot(
            kind="hyperneat",
            payload={
                "genome": genome,
                "env_obs_size": 8,
                "env_act_size": 5,
            },
        )

        # Pickle round-trip.
        data = pickle.dumps(snapshot)
        snapshot2 = pickle.loads(data)

        action_space = spaces.Discrete(5)
        policy = rebuild_opponent_policy(
            snapshot2, action_space=action_space
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5

    def test_es_hyperneat_snapshot_roundtrip(self):
        cfg = build_cppn_config(input_size=5, output_size=1, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))

        snapshot = OpponentSnapshot(
            kind="es_hyperneat",
            payload={
                "genome": genome,
                "env_obs_size": 8,
                "env_act_size": 5,
            },
        )

        data = pickle.dumps(snapshot)
        snapshot2 = pickle.loads(data)

        action_space = spaces.Discrete(5)
        policy = rebuild_opponent_policy(
            snapshot2, action_space=action_space
        )
        obs = np.random.randn(8).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
