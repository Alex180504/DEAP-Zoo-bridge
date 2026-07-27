"""Tests for the held-out champion probe (ProbeConfig / ProbeController).

Covers:
- ``ProbeConfig`` validation.
- ``UnifiedEvaluator`` per-episode info-metric tracking (``info_keys``).
- ``ProbeController`` cadence on the env-step axis, probe.csv row schema,
  finalize dedup, buffered mode + ``flush_to``, and NoOpLogger disabling.
- ``load_run`` merging ``probe.csv`` rows into the metrics frame (the path
  that makes probe curves selectable in the Compare page / rliable bridge).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    ProbeConfig,
    ProgressEvent,
    TrainingConfig,
)
from deap_pettingzoo.baselines.probing import ProbeController
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.utils.logging import NoOpLogger


class _StubEnv:
    """Minimal PettingZooWrapper-shaped stub with a cumulative info counter."""

    def __init__(self, ep_len: int = 4) -> None:
        self.possible_agents = ["a0", "a1"]
        self.ep_len = int(ep_len)
        self.last_infos: Dict[str, Dict[str, Any]] = {}
        self._step = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        self._step = 0
        return {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

    def action_space(self, agent: str):  # pragma: no cover - unused
        raise NotImplementedError

    def step(self, actions: Dict[str, Any]):
        self._step += 1
        done = self._step >= self.ep_len
        obs = (
            {}
            if done
            else {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        )
        rewards = {a: 1.0 for a in self.possible_agents}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        infos = {a: {"prey_lost_total": self._step} for a in self.possible_agents}
        return obs, rewards, terms, truncs, infos

    def render(self) -> np.ndarray:  # pragma: no cover - unused
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _StubPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_action(self, observation, *, action_mask=None):
        return 0


class _StubLogger:
    """Captures ``row`` calls; shaped like ExperimentLogger for the probe."""

    def __init__(self) -> None:
        self.rows: List[tuple] = []

    def row(self, table: str, row: Dict[str, Any]) -> None:
        self.rows.append((table, dict(row)))


def _inject_stub_env(evaluator: UnifiedEvaluator, env: _StubEnv, name: str) -> None:
    """Force the cache so ``evaluate()`` skips real env construction."""
    for render_mode in (None, "rgb_array", "human"):
        evaluator._env_cache[(name, None, render_mode)] = env  # type: ignore[assignment]


def _probe_training_config(**probe_kwargs) -> TrainingConfig:
    return TrainingConfig(
        env_name="stub_env",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=1_000,
        probe=ProbeConfig(**probe_kwargs),
    )


def _event(current: int, env_steps: int, total: int = 10) -> ProgressEvent:
    return ProgressEvent(
        unit="generation", current=current, total=total, env_steps=env_steps
    )


def test_probe_config_validation():
    with pytest.raises(ValueError):
        ProbeConfig(enabled=True)  # missing every_n_steps
    with pytest.raises(ValueError):
        ProbeConfig(enabled=True, every_n_steps=0)
    with pytest.raises(ValueError):
        ProbeConfig(n_episodes=0)
    cfg = ProbeConfig(enabled=True, every_n_steps=100, info_keys=["k"])
    assert cfg.info_keys == ("k",)


def test_unified_evaluator_tracks_info_keys():
    evaluator = UnifiedEvaluator(max_steps=10, info_keys=("prey_lost_total",))
    _inject_stub_env(evaluator, _StubEnv(ep_len=4), "stub_env")
    result = evaluator.evaluate(_StubPolicy(), "stub_env", num_episodes=3)
    # Episode-max of the cumulative counter == episode length.
    assert result.info_metrics == {"prey_lost_total": [4.0, 4.0, 4.0]}
    assert "info_metrics" in result.to_dict()


def test_probe_controller_cadence_and_rows():
    config = _probe_training_config(
        enabled=True, every_n_steps=100, n_episodes=2,
        info_keys=("prey_lost_total",), seed=0,
    )
    log = _StubLogger()
    controller = ProbeController(config, log, max_steps=10, seed=0)
    assert controller.enabled
    _inject_stub_env(controller.evaluator, _StubEnv(), "stub_env")

    # First event always probes (baseline point).
    controller.observe(_event(1, env_steps=10), _StubPolicy)
    # Below the advanced threshold (110): no probe.
    controller.observe(_event(2, env_steps=50), _StubPolicy)
    # Crossing the threshold probes once even over multiple intervals.
    controller.observe(_event(3, env_steps=350), _StubPolicy)
    controller.observe(_event(4, env_steps=360), _StubPolicy)

    assert [t for t, _ in log.rows] == ["probe", "probe"]
    row = log.rows[0][1]
    assert row["iteration"] == 1
    assert row["env_steps"] == 10
    assert row["probe_episodes"] == 2
    assert row["probe_reward_mean"] == pytest.approx(8.0)  # 2 agents * 4 steps
    assert row["probe_prey_lost_total_mean"] == pytest.approx(4.0)
    assert set(row) == set(log.rows[1][1])  # stable schema across rows

    # finalize dedups against an already-probed (iteration, env_steps) key...
    controller.finalize(_StubPolicy, iteration=3, env_steps=350)
    assert len(log.rows) == 2
    # ...and probes when the end-of-run point is new.
    controller.finalize(_StubPolicy, iteration=9, env_steps=900)
    assert len(log.rows) == 3
    controller.close()


def test_probe_controller_buffers_until_flush():
    config = _probe_training_config(enabled=True, every_n_steps=100, n_episodes=1)
    controller = ProbeController(config, None, max_steps=10, seed=0)
    assert controller.enabled
    _inject_stub_env(controller.evaluator, _StubEnv(), "stub_env")

    controller.observe(_event(1, env_steps=10), _StubPolicy)
    log = _StubLogger()
    assert not log.rows
    controller.flush_to(log)
    assert [t for t, _ in log.rows] == ["probe"]
    # Post-flush probes write directly.
    controller.observe(_event(2, env_steps=250), _StubPolicy)
    assert len(log.rows) == 2
    controller.close()


def test_probe_controller_disabled_paths():
    # Disabled config: no-op.
    config = _probe_training_config(enabled=False)
    controller = ProbeController(config, _StubLogger(), max_steps=10)
    assert not controller.enabled
    controller.observe(_event(1, env_steps=10), _StubPolicy)  # no crash
    controller.finalize(_StubPolicy, iteration=1, env_steps=10)
    controller.close()

    # NoOpLogger (output_dir=None runs): probing is wasted work, disabled.
    config = _probe_training_config(enabled=True, every_n_steps=100)
    controller = ProbeController(
        config, NoOpLogger.instance(), max_steps=10
    )
    assert not controller.enabled


def test_probe_failure_is_isolated():
    config = _probe_training_config(enabled=True, every_n_steps=100)
    log = _StubLogger()
    controller = ProbeController(config, log, max_steps=10, seed=0)
    _inject_stub_env(controller.evaluator, _StubEnv(), "stub_env")

    def _broken_champion():
        raise RuntimeError("boom")

    with pytest.warns(RuntimeWarning):
        controller.observe(_event(1, env_steps=10), _broken_champion)
    assert not log.rows
    # The controller keeps working for later, healthy probes.
    controller.observe(_event(2, env_steps=250), _StubPolicy)
    assert len(log.rows) == 1
    controller.close()


def _write_run_dir(tmp_path: Path, *, with_metrics: bool = True) -> Path:
    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run0", "algorithm": "ea", "env_name": "e", "seed": 0}),
        encoding="utf-8",
    )
    if with_metrics:
        (run_dir / "metrics.csv").write_text(
            "iteration,env_steps,best\n1,100,0.5\n2,200,0.7\n",
            encoding="utf-8",
        )
    (run_dir / "probe.csv").write_text(
        "iteration,env_steps,probe_reward_mean,probe_reward_std,probe_episodes\n"
        "1,100,3.0,0.5,2\n2,200,4.0,0.4,2\n",
        encoding="utf-8",
    )
    return run_dir


def test_load_run_merges_probe_into_metrics(tmp_path):
    from deap_pettingzoo.analysis.loaders import load_run, metrics_aligned_to_x

    record = load_run(_write_run_dir(tmp_path))
    assert record.metrics is not None
    cols = set(record.metrics.columns)
    assert {"best", "probe_reward_mean", "env_steps"} <= cols
    # Trainer rows + probe rows are both present...
    assert len(record.metrics) == 4
    # ...and each column pair aligns to its own rows (dropna path).
    aligned = metrics_aligned_to_x([record], x_col="env_steps", y_col="probe_reward_mean")
    assert len(aligned["run0"].dropna()) == 2


def test_load_run_probe_only(tmp_path):
    from deap_pettingzoo.analysis.loaders import load_run

    record = load_run(_write_run_dir(tmp_path, with_metrics=False))
    assert record.metrics is not None
    assert list(record.metrics["probe_reward_mean"]) == [3.0, 4.0]
