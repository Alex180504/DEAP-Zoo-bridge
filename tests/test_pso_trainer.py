"""Tests for PSOTrainer (Phase 1.4)."""
import pytest
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer


class TestPSOTrainerInit:
    def test_defaults(self):
        t = PSOTrainer()
        assert t.hidden_size == 64
        assert t.population_size == 50
        assert t.phi1 == 2.0
        assert t.phi2 == 2.0
        assert t.weight == 0.7298

    def test_negative_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            PSOTrainer(num_workers=-1)


class TestPSOTrainerSmoke:
    def _tiny_config(self, seed=0):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=12,
            num_seeds=1,
            algorithm_params={"seed": seed, "max_steps_per_episode": 10},
            env_config={"max_cycles": 10},
        )

    def test_smoke_train(self):
        trainer = PSOTrainer(hidden_size=8, population_size=6)
        result = trainer.train(self._tiny_config(seed=0))
        assert result.algorithm == "pso"
        assert result.policy is not None
        history = result.metrics["fitness_history"]
        assert len(history) >= 1

    def test_capabilities(self):
        caps = PSOTrainer().get_capabilities()
        assert caps["discrete_action"] is True
        assert caps["continuous_action"] is True
        assert caps["on_policy"] is False

    def test_budget_modes(self):
        t = PSOTrainer()
        assert t.supports_budget_mode(BudgetMode.ENV_STEPS)
        assert t.supports_budget_mode(BudgetMode.EVAL_EPISODES)
