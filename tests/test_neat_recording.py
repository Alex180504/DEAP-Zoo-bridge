"""End-to-end frame recording smoke tests for NEAT-family trainers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("imageio")
pytest.importorskip("neat")

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    RecordingConfig,
    TrainingConfig,
)
from deap_pettingzoo.contrib.neat import NEATTrainer, HyperNEATTrainer, ESHyperNEATTrainer


def _count_frames(run_root: Path) -> int:
    total = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        frames_dir = run_dir / "frames"
        if frames_dir.is_dir():
            for ext in ("mp4", "gif"):
                total += len(list(frames_dir.glob(f"train_episode_*.{ext}")))
    return total


# --- NEAT ---------------------------------------------------------------


@pytest.mark.slow
def test_neat_records_per_generation(tmp_path: Path):
    """NEAT produces one artefact per generation when recording is enabled."""
    trainer = NEATTrainer()
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=3 * 5,  # pop=3, gens=5 → 15 eval episodes
        algorithm_params={
            "pop_size": 3,
            "eval_episodes_per_genome": 1,
            "seed": 42,
        },
        env_config={"max_cycles": 25, "continuous_actions": False},
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=50),
    )
    trainer.train(config)
    # 5 generations, but gen 0 is skipped (no prior eval) → 4 artefacts.
    assert _count_frames(tmp_path) == 4


def test_neat_recording_disabled_writes_nothing(tmp_path: Path):
    """Default RecordingConfig produces no frames/ artefacts."""
    trainer = NEATTrainer()
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=3 * 2,  # 6 eval episodes
        algorithm_params={
            "pop_size": 3,
            "eval_episodes_per_genome": 1,
            "seed": 42,
        },
        env_config={"max_cycles": 25, "continuous_actions": False},
        output_dir=tmp_path,
    )
    trainer.train(config)
    assert _count_frames(tmp_path) == 0


def test_neat_explicit_cadence_overrides_default():
    trainer = NEATTrainer()
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=200,
        recording=RecordingConfig(enabled=True, every_n_generations=3),
    )
    assert trainer._resolve_recording_every_n(cfg) == 3


def test_neat_default_cadence_is_one():
    trainer = NEATTrainer()
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=200,
        recording=RecordingConfig(enabled=True),
    )
    assert trainer._resolve_recording_every_n(cfg) == 1


# --- HyperNEAT -----------------------------------------------------------


@pytest.mark.slow
def test_hyperneat_records_per_generation(tmp_path: Path):
    """HyperNEAT recording works."""
    trainer = HyperNEATTrainer()
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=3 * 3,
        algorithm_params={
            "pop_size": 3,
            "eval_episodes_per_genome": 1,
            "seed": 42,
        },
        env_config={"max_cycles": 25, "continuous_actions": False},
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=50),
    )
    trainer.train(config)
    # 3 generations, gen 0 skipped → 2 artefacts.
    assert _count_frames(tmp_path) == 2


# --- ES-HyperNEAT --------------------------------------------------------


@pytest.mark.slow
def test_es_hyperneat_records_per_generation(tmp_path: Path):
    """ES-HyperNEAT recording works."""
    trainer = ESHyperNEATTrainer()
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=3 * 3,
        algorithm_params={
            "pop_size": 3,
            "eval_episodes_per_genome": 1,
            "seed": 42,
        },
        env_config={"max_cycles": 25, "continuous_actions": False},
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=50),
    )
    trainer.train(config)
    # 3 generations, gen 0 skipped → 2 artefacts.
    assert _count_frames(tmp_path) == 2
