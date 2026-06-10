"""Tests for NEATTrainer."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.contrib.neat.trainer import NEATTrainer


class TestNEATTrainer:
    """Smoke tests for NEATTrainer."""

    @pytest.fixture
    def trainer(self):
        return NEATTrainer()

    @pytest.fixture
    def minimal_config(self, tmp_path):
        return TrainingConfig(
            env_name="simple_spread_v3",  # homogeneous agents for single-policy NEAT
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=100,  # small budget for fast test
            num_seeds=1,
            algorithm_params={
                "pop_size": 10,
                "eval_episodes_per_genome": 1,
                "seed": 42,
            },
            env_config={"max_cycles": 25, "continuous_actions": False},
            output_dir=tmp_path,
        )

    def test_capabilities(self, trainer):
        caps = trainer.get_capabilities()
        assert caps["algorithm"] == "NEAT"
        assert caps["discrete_action"] is True
        assert caps["continuous_action"] is True
        assert caps["population_based"] is True

    def test_supports_budget_mode(self, trainer):
        assert trainer.supports_budget_mode(BudgetMode.ENV_STEPS) is True
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES) is True
        # WALL_CLOCK not supported.
        assert trainer.supports_budget_mode(BudgetMode.WALL_CLOCK) is False

    def test_train_smoke(self, trainer, minimal_config):
        """End-to-end smoke test on simple_spread_v3."""
        result = trainer.train(minimal_config)
        assert result.algorithm == "NEAT"
        assert result.policy is not None
        assert result.metrics is not None
        assert "fitness_history" in result.metrics
        assert result.training_time is not None
        assert result.training_time > 0

        # Policy should produce valid actions.
        # simple_spread has 18-dim observations and Discrete(5) action space.
        obs = np.random.randn(18).astype(np.float32)
        action = result.policy.get_action(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
