"""End-to-end frame recording smoke tests for :class:`EATrainer`.

Each test trains a tiny EA for a handful of generations with
``RecordingConfig(enabled=True)`` and asserts the expected
``frames/train_episode_*.{mp4,gif}`` artefacts appear on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("imageio")

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    RecordingConfig,
    RecordingMode,
    TrainingConfig,
)
from deap_pettingzoo.baselines.ea_trainer import EATrainer


def _count_frames(run_root: Path) -> int:
    """Sum frame artefacts across every run directory under ``run_root``."""
    total = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        for ext in ("mp4", "gif"):
            total += len(list((run_dir / "frames").glob(f"train_episode_*.{ext}")))
    return total


@pytest.mark.slow
def test_ea_records_per_generation(tmp_path: Path):
    """``simple`` EA produces one artefact per generation when enabled."""
    trainer = EATrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 5 * 50,  # population * generations * steps
        algorithm_params={"max_steps_per_episode": 50, "seed": 0},
        output_dir=tmp_path,
        recording=RecordingConfig(
            enabled=True, mode=RecordingMode.EVERY, fps=10,
            max_frames_per_episode=50,
        ),
    )
    trainer.train(config)
    # mode=EVERY → one train artefact per generation (5).
    assert _count_frames(tmp_path) == 5


def test_ea_recording_disabled_writes_nothing(tmp_path: Path):
    """Default RecordingConfig produces no frames/ artefacts."""
    trainer = EATrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 2 * 25,
        algorithm_params={"max_steps_per_episode": 25, "seed": 0},
        output_dir=tmp_path,
    )
    trainer.train(config)
    assert _count_frames(tmp_path) == 0


@pytest.mark.slow
def test_ea_count_mode_caps_recordings(tmp_path: Path):
    """``mode=COUNT, max_recordings=3`` caps to 3 evenly-spread artefacts."""
    trainer = EATrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 5 * 50,
        algorithm_params={"max_steps_per_episode": 50, "seed": 0},
        output_dir=tmp_path,
        recording=RecordingConfig(
            enabled=True,
            mode=RecordingMode.COUNT,
            max_recordings=3,
            max_frames_per_episode=50,
        ),
    )
    trainer.train(config)
    assert _count_frames(tmp_path) == 3
