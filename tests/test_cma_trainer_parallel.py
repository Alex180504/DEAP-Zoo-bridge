"""Parallel-evaluation tests for :class:`CMAESTrainer`."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer


def _make_config() -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,
        algorithm_params={"seed": 0},
    )


class TestCMATrainerParallelArguments:
    def test_default_metadata_serial(self):
        t = CMAESTrainer(hidden_size=8, population_size=4)
        result = t.train(_make_config())
        assert result.metadata["num_workers"] == 1
        assert result.metadata["parallel_backend"] == "process"

    def test_negative_num_workers_rejected(self):
        with pytest.raises(ValueError, match="num_workers"):
            CMAESTrainer(hidden_size=8, population_size=4, num_workers=-2)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="parallel_backend"):
            CMAESTrainer(
                hidden_size=8,
                population_size=4,
                num_workers=2,
                parallel_backend="bogus",
            )


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs at least 2 CPUs"
)
class TestCMATrainerProcessPool:
    def test_parallel_run_completes(self):
        t = CMAESTrainer(
            hidden_size=8, population_size=4, num_workers=2
        )
        result = t.train(_make_config())
        assert result.metadata["num_workers"] == 2
        assert np.isfinite(result.metrics["final_fitness"])
