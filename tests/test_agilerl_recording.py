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


def test_agilerl_recording_and_metrics_share_run_dir(tmp_path: Path):
    """Frames, metrics.csv and summary.json all live under the same run_dir.

    Regression test for a bug in which the trainer built two separate
    ``ExperimentLogger`` instances (one for recording, one for metrics).
    Each call to ``ExperimentLogger.from_config`` generates a fresh
    timestamp+uuid run_id, so artefacts ended up split across two sibling
    directories. The fix: build a single logger and reuse it for both.
    """
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

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    # Exactly one run_dir should be created per train() call.
    assert len(run_dirs) == 1, (
        f"Expected a single run_dir under {tmp_path}, found {len(run_dirs)}: "
        f"{[p.name for p in run_dirs]}"
    )
    run_dir = run_dirs[0]

    # All artefacts must live under that single run_dir.
    frames = list((run_dir / "frames").glob("train_episode_*.mp4")) + list(
        (run_dir / "frames").glob("train_episode_*.gif")
    )
    assert len(frames) == 1, f"Expected 1 frame file under {run_dir}, found {frames}"
    assert (run_dir / "summary.json").is_file(), (
        f"summary.json missing under {run_dir} (artefacts split across run_dirs?)"
    )


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


def test_agilerl_records_mid_training_at_every_n_steps(tmp_path: Path):
    """``RecordingConfig.every_n_steps`` triggers extra mid-training rollouts.

    With a 600-step budget and ``every_n_steps=200`` the off-policy loop
    should fire the recording callback at steps 200, 400, 600, plus the
    end-of-training rollout, for a total of 4 frame files. The exact
    count is treated as ``>= 2`` to keep the test robust to the
    quantisation logic in the on-policy variant; the goal is to prove
    that mid-training recording happens at all.
    """
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=600,
        algorithm_params={
            "seed": 0,
            "batch_size": 32,
            "memory_size": 500,
            "learning_delay": 100,
            "max_steps_per_episode": 25,
        },
        output_dir=tmp_path,
        recording=RecordingConfig(
            enabled=True,
            fps=10,
            max_frames_per_episode=25,
            every_n_steps=200,
        ),
    )
    MADDPGTrainer().train(config)

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    frames = list((run_dir / "frames").glob("train_episode_*.mp4")) + list(
        (run_dir / "frames").glob("train_episode_*.gif")
    )
    assert len(frames) >= 2, (
        f"Expected mid-training + end-of-training rollouts (>=2 files) "
        f"under {run_dir / 'frames'}, found {len(frames)}: {frames}"
    )


def test_recording_config_validates_every_n_steps_and_max_dim():
    """``RecordingConfig`` rejects non-positive ``every_n_steps`` / ``max_dim``."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="every_n_steps"):
        RecordingConfig(enabled=True, every_n_steps=0)
    with _pytest.raises(ValueError, match="max_dim"):
        RecordingConfig(enabled=True, max_dim=-1)
    # Sane values accepted.
    cfg = RecordingConfig(enabled=True, every_n_steps=100, max_dim=512)
    assert cfg.every_n_steps == 100
    assert cfg.max_dim == 512
