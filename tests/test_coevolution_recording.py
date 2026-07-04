"""End-to-end frame recording smoke tests for :class:`CoEvolutionTrainer`."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("deap")
pytest.importorskip("imageio")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    RecordingConfig,
    TrainingConfig,
)
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)


def _count_frames(run_root: Path) -> int:
    total = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        for ext in ("mp4", "gif"):
            total += len(list((run_dir / "frames").glob(f"train_episode_*.{ext}")))
    return total


def _tiny_ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _make_coev_cfg() -> CoEvolutionConfig:
    return CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
            Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
        ],
        k_opponents=1,
        max_steps_per_episode=25,
        seed=7,
    )


@pytest.mark.slow
def test_coevolution_records_per_generation(tmp_path: Path):
    cfg = _make_coev_cfg()
    config = TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=20,  # ~3 generations with population_size=4 + k=1
        coevolution_config=cfg, algorithm_params={"max_steps_per_episode": 25},
        output_dir=tmp_path,
        recording=RecordingConfig(enabled=True, fps=10, max_frames_per_episode=25),
    )
    CoEvolutionTrainer().train(config)
    # At least one artefact must be produced (synchronous schedule fires
    # the callback once per generation; budget is small but >0).
    assert _count_frames(tmp_path) >= 1


def test_coevolution_recording_disabled_writes_nothing(tmp_path: Path):
    cfg = _make_coev_cfg()
    config = TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=10,
        coevolution_config=cfg, algorithm_params={"max_steps_per_episode": 25},
        output_dir=tmp_path,
    )
    CoEvolutionTrainer().train(config)
    assert _count_frames(tmp_path) == 0
