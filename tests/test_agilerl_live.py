"""
Live training tests for AgileRL trainers.

These tests actually run AgileRL training for a handful of steps and
therefore require the optional ``rl-baselines`` extra to be installed::

    pip install 'deap-pettingzoo[rl-baselines]'

Tests are gated behind the ``slow`` marker so they are opt-in::

    pytest tests/test_agilerl_live.py -m slow
"""

from __future__ import annotations

import pytest
from pathlib import Path

from deap_pettingzoo.baselines import (
    AgileRLPolicyWrapper,
    BudgetMode,
    IPPOTrainer,
    MADDPGTrainer,
    MATD3Trainer,
    TrainingConfig,
    TrainingResult,
    UnifiedEvaluator,
)


agilerl = pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)


pytestmark = [pytest.mark.slow, pytest.mark.smoke]


@pytest.fixture()
def tiny_env_config():
    """Small training budget so tests finish quickly."""
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=500,
        algorithm_params={
            "seed": 0,
            "batch_size": 32,
            "memory_size": 1000,
            "learning_delay": 100,
        },
    )


def _vectorized_off_policy_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=64,
        algorithm_params={
            "seed": seed,
            "num_envs": 2,
            "batch_size": 16,
            "memory_size": 256,
            "learning_delay": 10,
            "learn_step": 8,
        },
        output_dir=Path("results/_pytest_agilerl_vec_smoke"),
    )


class TestMADDPGLive:
    def test_trains_and_returns_policy(self, tiny_env_config):
        result = MADDPGTrainer().train(tiny_env_config)
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "maddpg"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metadata["env_name"] == "simple_spread_v3"
        assert result.metadata["on_policy"] is False
        assert result.metrics["env_steps"] == 500

    def test_policy_is_evaluable(self, tiny_env_config):
        result = MADDPGTrainer().train(tiny_env_config)
        evaluator = UnifiedEvaluator(max_steps=25)
        eval_result = evaluator.evaluate(
            result.policy,
            env_name="simple_spread_v3",
            num_episodes=2,
            seed=42,
        )
        assert eval_result.num_episodes == 2
        assert len(eval_result.episode_rewards) == 2

    def test_trains_with_vectorized_envs(self):
        result = MADDPGTrainer().train(_vectorized_off_policy_config())
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "maddpg"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metadata["on_policy"] is False
        assert result.metrics["env_steps"] == 64
        assert result.metrics["num_episodes"] >= 1


class TestMATD3Live:
    def test_trains_and_returns_policy(self, tiny_env_config):
        result = MATD3Trainer().train(tiny_env_config)
        assert result.algorithm == "matd3"
        assert isinstance(result.policy, AgileRLPolicyWrapper)

    def test_trains_with_vectorized_envs(self):
        result = MATD3Trainer().train(_vectorized_off_policy_config(seed=1))
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "matd3"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metadata["on_policy"] is False
        assert result.metrics["env_steps"] == 64
        assert result.metrics["num_episodes"] >= 1


class TestIPPOLive:
    def test_trains_and_returns_policy(self, tiny_env_config):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=500,
            algorithm_params={"seed": 0, "rollout_steps": 128},
        )
        result = IPPOTrainer().train(config)
        assert result.algorithm == "ippo"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metadata["on_policy"] is True

    def test_trains_with_vectorized_envs(self):
        """Smoke test that IPPO runs end-to-end with ``num_envs > 1``.

        Validates the AsyncPettingZooVecEnv plumbing in
        :func:`agilerl_adapter.build_vec_parallel_env` and the
        vectorized rollout branch in
        :meth:`AgileRLTrainer._run_on_policy`.
        """
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=256,
            algorithm_params={
                "seed": 0,
                "learn_step": 64,
                "num_envs": 2,
            },
        )
        result = IPPOTrainer().train(config)
        assert result.algorithm == "ippo"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metrics["env_steps"] >= 256
        # Per-sub-env episode tracking should produce at least one entry
        # in a 256-step run with 25-step MPE episodes across 2 envs.
        assert result.metrics["num_episodes"] >= 1
