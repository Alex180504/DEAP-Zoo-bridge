"""Tests for parallel NEAT evaluation."""

import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.contrib.neat.trainer import NEATTrainer


class TestNeatParallelEval:
    """Smoke tests for parallel NEAT evaluation."""

    def test_num_workers_stored(self):
        t = NEATTrainer(num_workers=4)
        assert t._num_workers == 4

    def test_num_workers_default(self):
        t = NEATTrainer()
        assert t._num_workers == 1

    def test_num_workers_invalid(self):
        with pytest.raises(ValueError, match="num_workers"):
            NEATTrainer(num_workers=0)

    def test_capabilities_parallel(self):
        t = NEATTrainer(num_workers=4)
        caps = t.get_capabilities()
        assert caps["parallel_training"] is True

    def test_capabilities_serial(self):
        t = NEATTrainer()
        caps = t.get_capabilities()
        assert caps["parallel_training"] is False

    def test_train_smoke_serial(self, tmp_path):
        """Serial path still works."""
        cfg = TrainingConfig(
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
        trainer = NEATTrainer(num_workers=1)
        result = trainer.train(cfg)
        assert result.algorithm == "NEAT"
        assert result.policy is not None
