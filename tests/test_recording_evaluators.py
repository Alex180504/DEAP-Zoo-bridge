"""Tests for FrameRecorder integration in :class:`UnifiedEvaluator`.

Covers the recording feature:
- ``frame_recorder`` is wired through ``__init__`` and exercised per
  episode (start/add/end lifecycle).
- ``record_episode_offset`` advances across calls so successive
  evaluations don't overwrite each other.
- An ``rgb_array`` render mode is selected automatically when an
  enabled recorder is present.
- ``frame_recorder`` defaulting to ``None`` keeps the legacy code path
  unchanged (no rgb_array mode, no extra render() calls beyond
  ``render=True``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from deap_pettingzoo.baselines.base import RecordingConfig
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.utils.logging.replay import FrameRecorder


class _StubFrameRecorder:
    """In-memory stand-in capturing lifecycle calls without imageio."""

    def __init__(self) -> None:
        self.enabled = True
        self.episode_calls: List[int] = []
        self.meta_calls: List[Dict[str, Any]] = []
        self.frames_per_episode: List[int] = []
        self.end_calls: int = 0
        self._current_count: Optional[int] = None

    def start_episode(
        self, idx: int, *, iteration=None, env_steps=None, extra=None
    ) -> None:
        self.episode_calls.append(int(idx))
        self.meta_calls.append(
            {"iteration": iteration, "env_steps": env_steps, "extra": extra}
        )
        self._current_count = 0

    def add_frame(self, frame: Any) -> None:
        if frame is None or self._current_count is None:
            return
        self._current_count += 1

    def end_episode(self) -> None:
        if self._current_count is not None:
            self.frames_per_episode.append(self._current_count)
        self.end_calls += 1
        self._current_count = None


class _StubEnv:
    """Minimal :class:`PettingZooWrapper`-shaped stub yielding rgb frames."""

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
        obs = {} if done else {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        rewards = {a: 1.0 for a in self.possible_agents}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        return obs, rewards, terms, truncs, {}

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:  # pragma: no cover - cache cleanup
        return None


class _StubPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_action(self, observation, *, action_mask=None):
        return 0


def _inject_stub_env(evaluator: UnifiedEvaluator, env: _StubEnv) -> None:
    """Force the cache so ``evaluate()`` skips real env construction."""
    # The cache key shape mirrors ``UnifiedEvaluator.evaluate``.
    for render_mode in (None, "rgb_array", "human"):
        evaluator._env_cache[("stub_env", None, render_mode)] = env  # type: ignore[assignment]


def test_recording_config_validation():
    """RecordingConfig rejects invalid arguments."""
    with pytest.raises(ValueError):
        RecordingConfig(every_n_generations=0)
    with pytest.raises(ValueError):
        RecordingConfig(fps=0)
    with pytest.raises(ValueError):
        RecordingConfig(fmt="webm")
    with pytest.raises(ValueError):
        RecordingConfig(max_frames_per_episode=-1)
    # Defaults are valid.
    rc = RecordingConfig()
    assert rc.enabled is False
    assert rc.every_n_generations is None
    assert rc.episode_prefix == ""


def test_unified_evaluator_invokes_recorder_lifecycle():
    """When a recorder is supplied, evaluate() drives start/add/end per ep."""
    recorder = _StubFrameRecorder()
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=4)
    _inject_stub_env(evaluator, env)

    result = evaluator.evaluate(
        policy=_StubPolicy(),
        env_name="stub_env",
        num_episodes=3,
        seed=0,
    )

    assert result.num_episodes == 3
    # Three episodes recorded with 0, 1, 2 indices.
    assert recorder.episode_calls == [0, 1, 2]
    # Every episode captures a reset frame + one frame per step (4 steps).
    assert recorder.frames_per_episode == [5, 5, 5]
    assert recorder.end_calls == 3
    # Offset advances so the next evaluate() picks up at index 3.
    assert evaluator.record_episode_offset == 3


def test_unified_evaluator_offset_persists_across_calls():
    """Successive evaluate() calls keep numbering distinct artefacts."""
    recorder = _StubFrameRecorder()
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=2)
    _inject_stub_env(evaluator, env)

    evaluator.evaluate(_StubPolicy(), env_name="stub_env", num_episodes=2, seed=1)
    evaluator.evaluate(_StubPolicy(), env_name="stub_env", num_episodes=2, seed=2)
    assert recorder.episode_calls == [0, 1, 2, 3]


def test_unified_evaluator_disabled_recorder_skipped():
    """A recorder with ``enabled=False`` is treated as absent."""
    recorder = _StubFrameRecorder()
    recorder.enabled = False
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=2)
    _inject_stub_env(evaluator, env)

    evaluator.evaluate(_StubPolicy(), env_name="stub_env", num_episodes=1, seed=0)
    assert recorder.episode_calls == []
    assert recorder.end_calls == 0


def test_unified_evaluator_no_recorder_unchanged():
    """The legacy ``frame_recorder=None`` path is untouched."""
    evaluator = UnifiedEvaluator(max_steps=10)
    env = _StubEnv(ep_len=2)
    _inject_stub_env(evaluator, env)
    result = evaluator.evaluate(
        _StubPolicy(), env_name="stub_env", num_episodes=2, seed=0
    )
    assert result.num_episodes == 2
    assert evaluator.record_episode_offset == 0


def test_real_frame_recorder_writes_artifact(tmp_path: Path):
    """End-to-end: a real FrameRecorder produces a prefixed artefact."""
    pytest.importorskip("imageio")
    recorder = FrameRecorder(run_dir=tmp_path, episode_prefix="train_")
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=3)
    _inject_stub_env(evaluator, env)

    evaluator.evaluate(_StubPolicy(), env_name="stub_env", num_episodes=1, seed=0)
    frames_dir = tmp_path / "frames"
    artefacts = list(frames_dir.glob("train_episode_*"))
    assert len(artefacts) == 1
    assert artefacts[0].suffix in {".mp4", ".gif"}
    assert artefacts[0].name.startswith("train_episode_0000")


def test_unified_evaluator_passes_iteration_metadata():
    """evaluate(iteration=..., env_steps=...) reaches the recorder."""
    recorder = _StubFrameRecorder()
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=2)
    _inject_stub_env(evaluator, env)

    evaluator.evaluate(
        _StubPolicy(),
        env_name="stub_env",
        num_episodes=1,
        seed=0,
        iteration=7,
        env_steps=12345,
        record_extra={"side": "predators"},
    )
    assert recorder.meta_calls == [
        {"iteration": 7, "env_steps": 12345, "extra": {"side": "predators"}}
    ]


def test_frame_recorder_writes_recordings_sidecar(tmp_path: Path):
    """A real FrameRecorder records iteration metadata in recordings.json."""
    import json

    pytest.importorskip("imageio")
    recorder = FrameRecorder(run_dir=tmp_path, episode_prefix="train_")
    evaluator = UnifiedEvaluator(max_steps=10, frame_recorder=recorder)
    env = _StubEnv(ep_len=3)
    _inject_stub_env(evaluator, env)

    # Two generations recorded with distinct iteration metadata.
    evaluator.evaluate(
        _StubPolicy(), env_name="stub_env", num_episodes=1, seed=0,
        iteration=2, env_steps=200,
    )
    evaluator.evaluate(
        _StubPolicy(), env_name="stub_env", num_episodes=1, seed=1,
        iteration=5, env_steps=500,
    )

    sidecar = tmp_path / "frames" / "recordings.json"
    assert sidecar.is_file()
    entries = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(entries) == 2
    by_iter = {e["iteration"]: e for e in entries}
    assert by_iter[2]["env_steps"] == 200
    assert by_iter[2]["file"].startswith("train_episode_0000")
    assert by_iter[5]["env_steps"] == 500
    assert by_iter[5]["file"].startswith("train_episode_0001")
    # Filenames recorded in the sidecar exist on disk.
    for e in entries:
        assert (tmp_path / "frames" / e["file"]).is_file()
