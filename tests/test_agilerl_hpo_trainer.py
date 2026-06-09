"""Surface tests for AgileRL HPO trainer (Phase 2.1).

Does NOT import agilerl — tests the constructor, config parsing,
and capability surface using the abstract trainer infrastructure.
"""
import pytest

from deap_pettingzoo.baselines.agilerl_hpo_trainer import (
    EvolveHPOTrainer,
    MADDPGHPOTrainer,
    MATD3HPOTrainer,
)
from deap_pettingzoo.baselines.base import BudgetMode


class TestEvolveHPOTrainerInit:
    def test_defaults(self):
        t = EvolveHPOTrainer()
        assert t._hpo_population_size >= 2
        assert t._evo_every_n_steps > 0

    def test_population_size_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="population_size"):
            EvolveHPOTrainer(population_size=1)

    def test_mutation_probs_defaults(self):
        t = EvolveHPOTrainer()
        assert "no_mutation" in t._mutation_probs
        assert "architecture" in t._mutation_probs
        assert "rl_hp" in t._mutation_probs

    def test_custom_mutation_probs(self):
        t = EvolveHPOTrainer(mutation_probs={"no_mutation": 0.5, "rl_hp": 0.5})
        assert t._mutation_probs["no_mutation"] == 0.5
        assert t._mutation_probs["rl_hp"] == 0.5
        assert "architecture" in t._mutation_probs  # filled by default


class TestConcreteTrainers:
    def test_maddpg_hpo_name(self):
        t = MADDPGHPOTrainer()
        assert t.ALGORITHM_NAME == "maddpg"
        assert not t.ON_POLICY

    def test_matd3_hpo_name(self):
        t = MATD3HPOTrainer()
        assert t.ALGORITHM_NAME == "matd3"
        assert not t.ON_POLICY

    def test_budget_support(self):
        for cls in (MADDPGHPOTrainer, MATD3HPOTrainer):
            t = cls()
            assert t.supports_budget_mode(BudgetMode.ENV_STEPS)
            assert t.supports_budget_mode(BudgetMode.EVAL_EPISODES)

    def test_capabilities(self):
        for cls in (MADDPGHPOTrainer, MATD3HPOTrainer):
            t = cls()
            caps = t.get_capabilities()
            assert caps["discrete_action"] is True
            assert caps["continuous_action"] is True


class TestBuildHPConfig:
    def test_empty_spec_returns_empty_config(self):
        from agilerl.algorithms.core.registry import (
            HyperparameterConfig,
            RLParameter,
        )
        result = EvolveHPOTrainer._build_hp_config(
            {}, HyperparameterConfig, RLParameter,
        )
        assert not result  # bool(HyperparameterConfig()) is False

    def test_simple_range_spec(self):
        from agilerl.algorithms.core.registry import (
            HyperparameterConfig,
            RLParameter,
        )
        spec = {"lr": [1e-4, 1e-2]}
        result = EvolveHPOTrainer._build_hp_config(
            spec, HyperparameterConfig, RLParameter,
        )
        assert result  # non-empty
        assert "lr" in result
