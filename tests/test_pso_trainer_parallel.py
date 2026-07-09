"""Parallel PSO trainer test."""
import numpy as np
import pytest

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer


class TestPSOTrainerParallel:
    def _config(self, seed=0, **kw):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=10,
            num_seeds=1,
            seed=seed, max_steps_per_episode=10,
            env_config={"max_cycles": 10},
            **kw,
        )

    def test_parallel_smoke(self):
        # num_workers is now a first-class TrainingConfig field.
        trainer = PSOTrainer(hidden_size=8, population_size=6)
        result = trainer.train(self._config(seed=0, num_workers=2))
        assert result.algorithm == "pso"
        assert result.policy is not None

    def test_dask_backend_completes(self):
        pytest.importorskip("dask.distributed")
        trainer = PSOTrainer(hidden_size=8, population_size=6)
        result = trainer.train(
            self._config(seed=0, num_workers=2, parallel_backend="dask")
        )
        assert result.algorithm == "pso"
        assert np.isfinite(result.metrics["final_fitness"])
