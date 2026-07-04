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


pytestmark = pytest.mark.slow


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
            seed=0, algorithm_params={"rollout_steps": 128},
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


class TestIPPOMultiDiscreteLive:
    """IPPO end-to-end on a synthetic homogeneous MultiDiscrete env."""

    @pytest.fixture(autouse=True)
    def _md_env(self):
        from deap_pettingzoo.environments.registry import (
            is_registered,
            unregister_env,
        )
        from tests.multidiscrete_env import MD_ENV_NAME, register_md_env

        register_md_env()
        yield MD_ENV_NAME
        if is_registered(MD_ENV_NAME):
            unregister_env(MD_ENV_NAME)

    def _config(self):
        from tests.multidiscrete_env import MD_ENV_NAME

        return TrainingConfig(
            env_name=MD_ENV_NAME,
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=128,
            seed=0, algorithm_params={"rollout_steps": 64},
        )

    def test_trains_and_produces_valid_multidiscrete_actions(self):
        import numpy as np

        from deap_pettingzoo.environments import PettingZooWrapper
        from tests.multidiscrete_env import MD_ENV_NAME

        result = IPPOTrainer().train(self._config())
        assert result.algorithm == "ippo"
        assert isinstance(result.policy, AgileRLPolicyWrapper)

        env = PettingZooWrapper(MD_ENV_NAME)
        try:
            obs = env.reset(seed=1)
            agent = env.possible_agents[0]
            action = np.asarray(result.policy.get_action(obs[agent]))
            assert action.shape == (2,)
            assert 0 <= action[0] < 3 and 0 <= action[1] < 4
        finally:
            env.close()

    def test_frozen_snapshot_roundtrip_multidiscrete(self):
        """IPPO frozen CPU snapshots now build via state_dict reconstruction
        (``StochasticActor`` is no longer ``deepcopy``'d) and decode
        MultiDiscrete actions correctly through a pickle round-trip."""
        import pickle

        import numpy as np

        from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper
        from deap_pettingzoo.environments import PettingZooWrapper
        from tests.multidiscrete_env import MD_ENV_NAME

        result = IPPOTrainer().train(self._config())
        assert isinstance(result.policy, AgileRLPolicyWrapper)

        frozen = result.policy.make_frozen_cpu_snapshot()
        frozen = pickle.loads(pickle.dumps(frozen))
        assert isinstance(frozen, FrozenRLPolicyWrapper)

        env = PettingZooWrapper(MD_ENV_NAME)
        try:
            obs = env.reset(seed=2)
            agent = env.possible_agents[0]
            action = np.asarray(frozen.get_action(obs[agent]))
            assert action.shape == (2,)
            assert 0 <= action[0] < 3 and 0 <= action[1] < 4
        finally:
            env.close()
