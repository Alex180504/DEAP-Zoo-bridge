"""
Unit tests for AgileRL-backed trainers (mocked; no AgileRL install required).

These tests exercise the unified-API surface of the AgileRL trainers
without actually running AgileRL. Live-training behavior is covered by
``tests/test_agilerl_live.py``.
"""

from __future__ import annotations

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deap_pettingzoo.baselines import (
    AgileRLPolicyWrapper,
    AgileRLTrainer,
    BudgetMode,
    IPPOTrainer,
    MADDPGTrainer,
    MATD3Trainer,
    TrainingConfig,
    get_trainer,
    register_trainer,
)
from deap_pettingzoo.baselines.agilerl_adapter import (
    build_parallel_env,
    build_vec_parallel_env,
    extract_space_metadata,
    translate_budget_to_env_steps,
)


# ---------------------------------------------------------------------------
# Capabilities + registry
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_maddpg_capabilities(self):
        caps = MADDPGTrainer().get_capabilities()
        assert caps["discrete_action"] is True
        assert caps["continuous_action"] is True
        assert caps["on_policy"] is False
        assert caps["needs_global_state"] is False

    def test_matd3_capabilities(self):
        caps = MATD3Trainer().get_capabilities()
        assert caps["on_policy"] is False
        assert caps["needs_global_state"] is False

    def test_ippo_capabilities(self):
        caps = IPPOTrainer().get_capabilities()
        assert caps["on_policy"] is True
        assert caps["needs_global_state"] is False

    def test_supports_env_steps_and_eval_episodes(self):
        trainer = MADDPGTrainer()
        assert trainer.supports_budget_mode(BudgetMode.ENV_STEPS) is True
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES) is True
        assert trainer.supports_budget_mode(BudgetMode.WALL_CLOCK) is False


class TestRegistry:
    @pytest.mark.parametrize(
        "name, cls",
        [
            ("maddpg", MADDPGTrainer),
            ("matd3", MATD3Trainer),
            ("ippo", IPPOTrainer),
        ],
    )
    def test_builtin_agilerl_registered(self, name, cls):
        trainer = get_trainer(name)
        assert isinstance(trainer, cls)


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_build_parallel_env_returns_pettingzoo_env(self):
        env = build_parallel_env("simple_spread_v3")
        try:
            assert hasattr(env, "possible_agents")
            assert hasattr(env, "reset")
            assert hasattr(env, "step")
            assert len(env.possible_agents) > 0
        finally:
            env.close()

    def test_build_parallel_env_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_parallel_env("nonexistent_env_v99")

    def test_build_vec_parallel_env_rejects_zero(self):
        with pytest.raises(ValueError, match="num_envs must be"):
            build_vec_parallel_env("simple_spread_v3", num_envs=0)

    @pytest.mark.parametrize("trainer_cls", [MADDPGTrainer, MATD3Trainer])
    def test_off_policy_train_uses_vec_env_when_num_envs_gt_1(self, trainer_cls):
        pytest.importorskip(
            "agilerl",
            reason="AgileRL required for vec-env construction.",
        )

        trainer = trainer_cls(verbose=False)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=4,
            algorithm_params={"seed": 0, "num_envs": 2},
        )

        captured = {}

        class DummyLogger:
            def row(self, *_args, **_kwargs):
                return None

        def fake_run_off_policy(env, *_args, **_kwargs):
            captured["num_envs"] = getattr(env, "num_envs", 1)
            return {"episode_rewards": [], "num_episodes": 0, "env_steps": 4}

        with patch.object(trainer, "_import_algorithm", return_value=object()), \
             patch.object(trainer, "_build_agent", return_value=SimpleNamespace()), \
             patch.object(trainer, "_run_off_policy", side_effect=fake_run_off_policy), \
             patch("deap_pettingzoo.baselines._logging_helpers.build_logger", return_value=nullcontext(DummyLogger())), \
             patch("deap_pettingzoo.baselines._logging_helpers.write_summary"), \
             patch("deap_pettingzoo.utils.logging.save_agilerl_agent"):
            result = trainer.train(config)

        assert captured["num_envs"] == 2
        assert result.metrics["env_steps"] == 4

    def test_extract_space_metadata_shapes(self):
        env = build_parallel_env("simple_spread_v3")
        try:
            obs_spaces, act_spaces, agent_ids = extract_space_metadata(env)
            assert len(obs_spaces) == len(act_spaces) == len(agent_ids)
            assert agent_ids == list(env.possible_agents)
        finally:
            env.close()

    def test_translate_budget_env_steps(self):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=1000,
        )
        assert translate_budget_to_env_steps(config) == 1000

    def test_translate_budget_eval_episodes(self):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=10,
        )
        # Default MPE episode length is 25.
        assert translate_budget_to_env_steps(config) == 250

    def test_translate_budget_wallclock_rejected(self):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.WALL_CLOCK,
            budget_value=60,
        )
        with pytest.raises(ValueError, match="wall-clock"):
            translate_budget_to_env_steps(config)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class TestAgileRLPolicyWrapper:
    def _make_agent(self, agent_ids):
        """Fake AgileRL agent whose get_action returns deterministic values."""

        def get_action(obs_dict, training=False):
            actions = {}
            for aid, obs in obs_dict.items():
                # Return action dependent on agent id so we can check routing.
                actions[aid] = np.array([[float(agent_ids.index(aid))]])
            return actions, actions

        mock = MagicMock()
        mock.get_action.side_effect = get_action
        return mock

    def test_rotates_through_agents_in_order(self):
        agent_ids = ["agent_0", "agent_1", "agent_2"]
        wrapper = AgileRLPolicyWrapper(self._make_agent(agent_ids), agent_ids)

        obs = np.zeros(4, dtype=np.float32)
        a0 = wrapper.get_action(obs)
        a1 = wrapper.get_action(obs)
        a2 = wrapper.get_action(obs)
        assert float(a0) == 0.0
        assert float(a1) == 1.0
        assert float(a2) == 2.0

    def test_reset_restores_rotation(self):
        agent_ids = ["agent_0", "agent_1"]
        wrapper = AgileRLPolicyWrapper(self._make_agent(agent_ids), agent_ids)

        wrapper.get_action(np.zeros(4))
        wrapper.get_action(np.zeros(4))
        wrapper.reset()
        # After reset, next call should target agent_0 again.
        assert float(wrapper.get_action(np.zeros(4))) == 0.0

    def test_handles_non_tuple_return(self):
        agent_ids = ["a"]
        agent = MagicMock()
        agent.get_action.return_value = {"a": np.array([[0.42]])}
        wrapper = AgileRLPolicyWrapper(agent, agent_ids)
        assert float(wrapper.get_action(np.zeros(4))) == pytest.approx(0.42)

    def test_falls_back_without_training_kwarg(self):
        agent_ids = ["a"]

        def get_action_no_training(obs_dict):
            return {"a": np.array([[1.0]])}

        agent = MagicMock()
        agent.get_action.side_effect = get_action_no_training
        wrapper = AgileRLPolicyWrapper(agent, agent_ids)
        # First call should raise TypeError with training=False, then retry.
        agent.get_action.side_effect = [
            TypeError("unexpected training"),
            {"a": np.array([[1.0]])},
        ]
        result = wrapper.get_action(np.zeros(4))
        assert float(result) == 1.0

    def test_rejects_empty_agent_ids(self):
        with pytest.raises(ValueError):
            AgileRLPolicyWrapper(MagicMock(), [])


# ---------------------------------------------------------------------------
# Lazy-import / install-guard behaviour
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_importerror_if_agilerl_missing(self, monkeypatch):
        # Simulate missing AgileRL: patch importlib to raise for the
        # trainer's target module.
        import importlib

        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name.startswith("agilerl"):
                raise ImportError("simulated missing agilerl")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)

        trainer = MADDPGTrainer()
        with pytest.raises(ImportError, match="rl-baselines"):
            trainer._import_algorithm()
