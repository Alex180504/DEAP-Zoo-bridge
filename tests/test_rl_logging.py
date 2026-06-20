"""Structured logging smoke tests for RL trainers and co-evolution RL sides.

Verifies that:
- ``AgileRLTrainer`` (MADDPG) emits per-interval rows to ``metrics.csv``
  during training, in addition to the existing ``episodes.csv`` and
  ``summary.json``.
- ``EvolveHPOTrainer`` (MADDPG-HPO) emits per-generation rows and persists
  a ``summary.json`` and agent checkpoint.
- A co-evolution run with an RL side writes ``metrics.csv`` rows for that
  side with non-zero ``env_steps``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("agilerl")
pytest.importorskip("torch")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines import BudgetMode, MADDPGTrainer, TrainingConfig
from deap_pettingzoo.baselines.agilerl_hpo_trainer import MADDPGHPOTrainer


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_dirs(output_dir: Path) -> List[Path]:
    return [p for p in output_dir.iterdir() if p.is_dir()]


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _maddpg_cfg(tmp_path: Path, budget: int = 250) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=budget,
        algorithm_params={
            "seed": 0,
            "batch_size": 32,
            "memory_size": 500,
            "learning_delay": 50,
            "max_steps_per_episode": 25,
        },
        output_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Standalone AgileRL trainer
# ---------------------------------------------------------------------------


def test_agilerl_emits_metrics_csv_rows(tmp_path: Path) -> None:
    """After training, metrics.csv must exist and contain ≥1 row with a
    non-null mean and non-zero env_steps."""
    MADDPGTrainer().train(_maddpg_cfg(tmp_path))

    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1
    run_dir = dirs[0]

    metrics_path = run_dir / "metrics.csv"
    assert metrics_path.is_file(), f"metrics.csv missing under {run_dir}"

    rows = _read_csv(metrics_path)
    assert rows, "metrics.csv is empty"

    # At least one row must have env_steps > 0.
    env_steps_vals = [int(r["env_steps"]) for r in rows if r.get("env_steps")]
    assert any(v > 0 for v in env_steps_vals), (
        f"All env_steps are 0 in metrics.csv: {rows}"
    )

    # At least one row must have a numeric mean (not None / empty).
    means = [r.get("mean", "") for r in rows]
    assert any(m not in ("", "None") for m in means), (
        f"No numeric 'mean' in metrics.csv rows: {rows}"
    )


def test_agilerl_summary_and_agent_present(tmp_path: Path) -> None:
    """summary.json and the agent checkpoint must both land in the run_dir."""
    MADDPGTrainer().train(_maddpg_cfg(tmp_path))

    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1
    run_dir = dirs[0]

    assert (run_dir / "summary.json").is_file()
    # Agent checkpoint: at least one file in the checkpoints sub-dir.
    checkpoints = list((run_dir / "checkpoints").iterdir()) if (run_dir / "checkpoints").is_dir() else []
    assert checkpoints, f"No checkpoint files under {run_dir / 'checkpoints'}"


# ---------------------------------------------------------------------------
# HPO trainer
# ---------------------------------------------------------------------------


def test_hpo_emits_metrics_and_summary(tmp_path: Path) -> None:
    """EvolveHPOTrainer must write per-generation metrics.csv rows and
    a summary.json — neither existed before this wiring."""
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=300,
        algorithm_params={
            "seed": 0,
            "population_size": 2,
            "evo_every_n_steps": 150,
            "max_steps_per_episode": 25,
        },
        output_dir=tmp_path,
    )
    MADDPGHPOTrainer(population_size=2, evo_every_n_steps=150).train(cfg)

    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1
    run_dir = dirs[0]

    metrics_path = run_dir / "metrics.csv"
    assert metrics_path.is_file(), f"metrics.csv missing under {run_dir}"
    rows = _read_csv(metrics_path)
    assert rows, "metrics.csv is empty for HPO run"
    assert any(int(r.get("env_steps", 0)) > 0 for r in rows)

    assert (run_dir / "summary.json").is_file(), f"summary.json missing under {run_dir}"


# ---------------------------------------------------------------------------
# Co-evolution RL side env_steps
# ---------------------------------------------------------------------------


def test_coev_rl_side_emits_nonzero_env_steps(tmp_path: Path) -> None:
    """In a co-evolution run with an RL side, metrics.csv rows for that side
    must carry env_steps > 0 (previously hardcoded to 0)."""
    from deap_pettingzoo.baselines.coevolution import (
        CoEvolutionConfig,
        CoEvolutionTrainer,
        Side,
    )

    ea_cfg = {"population_size": 4, "hidden_size": 8, "seed": 0}
    rl_cfg = {
        "algorithm": "maddpg",
        "rollout_episodes_per_gen": 2,
        "batch_size": 8,
        "memory_size": 256,
        "learning_delay": 0,
        "device": "cpu",
        "seed": 0,
    }
    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side("pursuers", ("adversary",), "ea", ea_cfg),
            Side("evaders", ("agent",), "rl", rl_cfg),
        ],
        k_opponents=1,
        max_steps_per_episode=10,
        num_workers=1,
        seed=42,
    )
    training_cfg = TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True},
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        algorithm_params={"coevolution": coev_cfg},
        output_dir=tmp_path,
    )
    CoEvolutionTrainer().train(training_cfg)

    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1
    run_dir = dirs[0]

    metrics_path = run_dir / "metrics.csv"
    assert metrics_path.is_file(), f"metrics.csv missing under {run_dir}"
    rows = _read_csv(metrics_path)

    # Rows keyed by the RL side name ("evaders") should have env_steps > 0.
    rl_rows = [r for r in rows if r.get("side") == "evaders"]
    assert rl_rows, "No metrics.csv rows for the RL side"
    assert any(int(r.get("env_steps", 0)) > 0 for r in rl_rows), (
        f"All RL-side env_steps are 0: {rl_rows}"
    )
