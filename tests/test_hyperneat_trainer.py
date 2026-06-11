"""Tests for HyperNEATTrainer and ESHyperNEATTrainer."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.contrib.neat.hyperneat_trainer import (
    HyperNEATTrainer,
    ESHyperNEATTrainer,
)


class TestHyperNEATTrainer:
    @pytest.fixture
    def trainer(self):
        return HyperNEATTrainer()

    @pytest.fixture
    def minimal_config(self, tmp_path):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=60,
            num_seeds=1,
            algorithm_params={
                "pop_size": 6,
                "eval_episodes_per_genome": 1,
                "seed": 42,
                "substrate_hidden_layers": [4],
            },
            env_config={"max_cycles": 25, "continuous_actions": False},
            output_dir=tmp_path,
        )

    def test_capabilities(self, trainer):
        caps = trainer.get_capabilities()
        assert caps["algorithm"] == "HyperNEAT"
        assert caps["discrete_action"] is True

    def test_supports_budget_mode(self, trainer):
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES)

    def test_train_smoke(self, trainer, minimal_config):
        result = trainer.train(minimal_config)
        assert result.algorithm == "HyperNEAT"
        assert result.policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = result.policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5


class TestESHyperNEATTrainer:
    @pytest.fixture
    def trainer(self):
        return ESHyperNEATTrainer()

    @pytest.fixture
    def minimal_config(self, tmp_path):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=60,
            num_seeds=1,
            algorithm_params={
                "pop_size": 6,
                "eval_episodes_per_genome": 1,
                "seed": 42,
            },
            env_config={"max_cycles": 25, "continuous_actions": False},
            output_dir=tmp_path,
        )

    def test_capabilities(self, trainer):
        caps = trainer.get_capabilities()
        assert caps["algorithm"] == "ES-HyperNEAT"

    def test_train_smoke(self, trainer, minimal_config):
        result = trainer.train(minimal_config)
        assert result.algorithm == "ES-HyperNEAT"
        assert result.policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = result.policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
