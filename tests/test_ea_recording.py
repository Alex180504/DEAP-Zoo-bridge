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
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=50),
    )
    trainer.train(config)
    # 5 generations × cadence-1 = 5 train artefacts.
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


def test_ea_default_cadence_is_one():
    """``EATrainer._default_recording_every_n`` is 1 (record every gen)."""
    trainer = EATrainer()
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=1000,
        recording=RecordingConfig(enabled=True),
    )
    assert trainer._resolve_recording_every_n(cfg) == 1


def test_ea_explicit_cadence_overrides_default():
    """User-supplied ``every_n_generations`` always wins."""
    trainer = EATrainer()
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=1000,
        recording=RecordingConfig(enabled=True, every_n_generations=3),
    )
    assert trainer._resolve_recording_every_n(cfg) == 3


@pytest.mark.slow
def test_ea_explicit_cadence_writes_subset(tmp_path: Path):
    """``every_n_generations=2`` records on gens 2, 4 (out of 5)."""
    trainer = EATrainer(hidden_size=8, population_size=4)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=4 * 5 * 50,
        algorithm_params={"max_steps_per_episode": 50, "seed": 0},
        output_dir=tmp_path,
        recording=RecordingConfig(
            enabled=True,
            every_n_generations=2,
            max_frames_per_episode=50,
        ),
    )
    trainer.train(config)
    # Gens 2 and 4 hit the cadence (gen 1, 3, 5 don't).
    assert _count_frames(tmp_path) == 2
