"""Tests for NEAT parallel evaluation support."""

import pickle

import numpy as np
import pytest

neat = pytest.importorskip("neat")
import gymnasium.spaces as spaces

from deap_pettingzoo.contrib.neat.config import build_neat_config
from deap_pettingzoo.contrib.neat.policy import NeatPolicy
from deap_pettingzoo.baselines.coevolution.pools import (
    OpponentSnapshot,
    rebuild_opponent_policy,
)


class TestNeatParallel:
    """Parallel evaluation: snapshot -> worker rebuild -> policy evaluation."""

    def test_snapshot_roundtrip_many_genomes(self):
        """Multiple genomes should survive the round-trip."""
        cfg = build_neat_config(input_size=6, output_size=3, pop_size=20)
        pop = neat.Population(cfg)
        genomes = list(pop.population.values())

        for genome in genomes[:5]:  # test first 5
            snapshot = OpponentSnapshot(
                kind="neat",
                payload={
                    "genome": genome,
                    "config_input_size": 6,
                    "config_output_size": 3,
                },
            )
            # Pickle round-trip
            data = pickle.dumps(snapshot)
            snapshot2 = pickle.loads(data)

            action_space = spaces.Discrete(3)
            policy = rebuild_opponent_policy(
                snapshot2, action_space=action_space
            )
            obs = np.random.randn(6).astype(np.float32)
            action = policy.get_action(obs)
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < 3

    def test_policy_from_payload_equivalence(self):
        """NeatPolicy.from_snapshot_payload and rebuild_opponent_policy
        should both produce working policies."""
        cfg = build_neat_config(input_size=4, output_size=2, pop_size=10)
        pop = neat.Population(cfg)
        genome = next(iter(pop.population.values()))

        # Via from_snapshot_payload
        p1 = NeatPolicy.from_snapshot_payload(
            genome, cfg, action_space=spaces.Discrete(2)
        )

        # Via rebuild_opponent_policy
        snapshot = OpponentSnapshot(
            kind="neat",
            payload={
                "genome": genome,
                "config_input_size": 4,
                "config_output_size": 2,
            },
        )
        p2 = rebuild_opponent_policy(
            snapshot, action_space=spaces.Discrete(2)
        )

        obs = np.random.randn(4).astype(np.float32)
        a1 = p1.get_action(obs)
        a2 = p2.get_action(obs)
        # Same genome, same obs -> same action (deterministic).
        assert a1 == a2
