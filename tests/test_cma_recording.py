"""End-to-end frame recording smoke tests for :class:`CMAESTrainer`."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("imageio")

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    RecordingConfig,
    TrainingConfig,
)
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer


def _count_frames(run_root: Path) -> int:
    total = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        for ext in ("mp4", "gif"):
            total += len(list((run_dir / "frames").glob(f"train_episode_*.{ext}")))
    return total


@pytest.mark.slow
def test_cma_records_per_generation(tmp_path: Path):
    trainer = CMAESTrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 4 * 50,  # population * generations * steps
        algorithm_params={"max_steps_per_episode": 50, "seed": 0},
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=50),
    )
    trainer.train(config)
    # 4 generations × cadence-1 = 4 train artefacts.
    assert _count_frames(tmp_path) == 4


def test_cma_recording_disabled_writes_nothing(tmp_path: Path):
    trainer = CMAESTrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 2 * 25,
        algorithm_params={"max_steps_per_episode": 25, "seed": 0},
        output_dir=tmp_path,
    )
    trainer.train(config)
    assert _count_frames(tmp_path) == 0
