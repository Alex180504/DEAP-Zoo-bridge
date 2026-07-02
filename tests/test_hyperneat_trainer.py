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

    def test_train_parallel_updates_fitness_history_and_generation(self, tmp_path):
        """TrainingConfig.num_workers must actually
        route through parallel evaluation, even when the trainer was
        constructed with the (serial) default of 1."""
        n_generations = 3
        cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=1,
            num_seeds=1,
            algorithm_params={
                "pop_size": 4,
                "eval_episodes_per_genome": 1,
                "substrate_hidden_layers": [4],
                "n_generations": n_generations,
            },
            env_config={"max_cycles": 10, "continuous_actions": False},
            seed=0,
            num_workers=2,
            output_dir=tmp_path,
        )
        trainer = HyperNEATTrainer(num_workers=2)
        result = trainer.train(cfg)
        assert result.metrics["final_generation"] == n_generations
        assert len(result.metrics["fitness_history"]) == n_generations


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
