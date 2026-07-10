"""Tests for :class:`CMAESTrainer` and its registration."""

import pytest

from deap_pettingzoo.baselines import (
    BudgetMode,
    CMAESTrainer,
    TrainingConfig,
    UnifiedEvaluator,
)
from deap_pettingzoo.baselines.comparison import get_trainer


class TestCMAESTrainerBasics:
    def test_initialization(self):
        trainer = CMAESTrainer(hidden_size=32, population_size=10, sigma0=0.2)
        assert trainer.hidden_size == 32
        assert trainer.population_size == 10
        assert trainer.sigma0 == 0.2

    def test_capabilities(self):
        caps = CMAESTrainer().get_capabilities()
        assert caps["discrete_action"] is True
        assert caps["continuous_action"] is True
        assert caps["needs_global_state"] is False
        assert caps["on_policy"] is False

    def test_budget_support(self):
        trainer = CMAESTrainer()
        assert trainer.supports_budget_mode(BudgetMode.ENV_STEPS) is True
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES) is True
        assert trainer.supports_budget_mode(BudgetMode.GENERATIONS) is True

    def test_budget_translation_env_steps(self):
        trainer = CMAESTrainer(population_size=20)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=10000,
            max_steps_per_episode=200,
        )
        params = trainer._translate_budget(config)
        # 10000 / (20 * 200) = 2.5 -> 2 generations, matches EATrainer math.
        assert params["num_generations"] == 2

    def test_budget_translation_eval_episodes(self):
        trainer = CMAESTrainer(population_size=20)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=100,
        )
        params = trainer._translate_budget(config)
        assert params["num_generations"] == 5

    def test_budget_translation_minimum_one_generation(self):
        trainer = CMAESTrainer(population_size=100)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=100,
            max_steps_per_episode=200,
        )
        params = trainer._translate_budget(config)
        assert params["num_generations"] == 1


class TestCMAESTrainerRegistry:
    def test_registered_as_cma_es(self):
        trainer = get_trainer("cma-es")
        assert isinstance(trainer, CMAESTrainer)


class TestCMAESTrainerTraining:
    @pytest.mark.slow
    def test_training_smoke(self):
        """
        Tiny end-to-end run: verifies train() produces a usable policy.

        Notes:
            The budget is intentionally tiny so the test runs quickly in CI;
            use ``>=`` rather than ``>`` when comparing fitness values because such a small budget
            may cause the optimiser to plateau naturally without it being a bug.
        """
        trainer = CMAESTrainer(hidden_size=16, population_size=8, sigma0=0.3)

        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=6000,
            max_steps_per_episode=100, seed=0,
        )

        result = trainer.train(config)

        assert result.algorithm == "cma-es"
        assert result.policy is not None
        assert "fitness_history" in result.metrics
        assert result.metrics["num_generations"] >= 1
        assert result.metrics["dimension"] > 0
        assert result.training_time is not None and result.training_time > 0

        # Every recorded generation has the canonical stat keys.
        for record in result.metrics["fitness_history"]:
            assert {"generation", "mean", "std", "min", "max"} <= record.keys()

        history = result.metrics["fitness_history"]
        assert len(history) >= 2, (
            "need at least two generations to assess improvement"
        )
        # Compare gen 0 against the best of *later* generations only. A small
        # tolerance (scaled to gen 0's own population spread) absorbs sampling
        # noise and cross-platform floating-point drift in the non-elitist
        # search without masking a genuinely broken optimiser.
        best_ever_end = max(rec["max"] for rec in history[1:])
        tolerance = 0.5 * history[0]["std"]
        assert best_ever_end >= history[0]["max"] - tolerance, (
            f"best-ever fitness regressed beyond noise tolerance: "
            f"gen0_max={history[0]['max']}, best_ever={best_ever_end}, "
            f"tolerance={tolerance}"
        )

    @pytest.mark.slow
    def test_trained_policy_is_usable_by_unified_evaluator(self):
        trainer = CMAESTrainer(hidden_size=16, population_size=6, sigma0=0.3)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=3000,
            max_steps_per_episode=100, seed=0,
        )
        training_result = trainer.train(config)

        evaluator = UnifiedEvaluator(max_steps=100)
        eval_result = evaluator.evaluate(
            policy=training_result.policy,
            env_name="simple_spread_v3",
            num_episodes=2,
            seed=42,
        )
        assert eval_result.num_episodes == 2
        assert len(eval_result.episode_rewards) == 2
        evaluator.close()
