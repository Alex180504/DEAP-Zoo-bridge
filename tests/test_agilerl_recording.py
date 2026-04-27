"""End-to-end frame recording smoke test for AgileRL trainers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("imageio")
pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)

from deap_pettingzoo.baselines import (
    BudgetMode,
    MADDPGTrainer,
    RecordingConfig,
    TrainingConfig,
)

pytestmark = pytest.mark.slow


def _count_frames(run_root: Path) -> int:
    total = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        for ext in ("mp4", "gif"):
            total += len(list((run_dir / "frames").glob(f"train_episode_*.{ext}")))
    return total


def test_agilerl_records_one_rollout_at_end(tmp_path: Path):
    """AgileRL records exactly one champion rollout at end-of-training."""
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=200,
        algorithm_params={
            "seed": 0,
            "batch_size": 32,
            "memory_size": 500,
            "learning_delay": 100,
            "max_steps_per_episode": 25,
        },
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=25),
    )
    MADDPGTrainer().train(config)
    assert _count_frames(tmp_path) == 1


def test_agilerl_default_cadence_is_ten():
    """``AgileRLTrainer._default_recording_every_n`` is 10."""
    trainer = MADDPGTrainer()
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=100,
        recording=RecordingConfig(enabled=True),
    )
    assert trainer._resolve_recording_every_n(cfg) == 10
