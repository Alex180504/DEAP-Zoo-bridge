"""Tests for the experiment-logging stack (deap_pettingzoo.utils.logging).

These tests exercise the public surface only: ExperimentLogger, the
sinks, RunManifest, and FrameRecorder. They run with no optional deps
beyond numpy + pyarrow (pyarrow is required for the parquet sink test
and is skipped if missing).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from deap_pettingzoo.utils.logging import (
    ExperimentLogger,
    NoOpLogger,
    RunManifest,
)
from deap_pettingzoo.utils.logging.sinks import (
    CsvScalarSink,
    JsonManifestSink,
    MultiSink,
    NoOpSink,
)


def _read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------


def test_run_manifest_round_trip(tmp_path: Path):
    m = RunManifest(
        run_id="r0",
        algorithm="ea",
        env_name="simple_spread_v3",
        env_config={},
        seed=1,
        budget_mode="GENERATIONS",
        budget_value=2,
        hyperparams={"pop": 4},
    )
    p = tmp_path / "manifest.json"
    m.write(p)
    assert p.is_file()
    data = json.loads(p.read_text())
    assert data["algorithm"] == "ea"
    assert data["seed"] == 1
    assert data["schema_version"] >= 1


def test_run_manifest_capture_environment_keys(tmp_path: Path):
    from deap_pettingzoo.utils.logging.manifest import capture_environment_metadata

    env = capture_environment_metadata()
    # Always-present keys; values are best-effort.
    assert "python" in env
    assert "platform" in env
    assert "package_versions" in env


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_csv_sink_appends_rows(tmp_path: Path):
    sink = CsvScalarSink()
    sink.open(tmp_path)
    sink.log_row("metrics", {"iteration": 0, "env_steps": 0, "wall_time": 0.0, "best": 1.0})
    sink.log_row("metrics", {"iteration": 1, "env_steps": 10, "wall_time": 0.1, "best": 1.5})
    sink.close()
    rows = _read_csv_rows(tmp_path / "metrics.csv")
    assert len(rows) == 2
    assert rows[0]["iteration"] == "0"
    assert rows[1]["best"] == "1.5"


def test_csv_sink_rejects_unknown_columns(tmp_path: Path):
    sink = CsvScalarSink()
    sink.open(tmp_path)
    sink.log_row("metrics", {"iteration": 0, "env_steps": 0, "wall_time": 0.0, "x": 1})
    with pytest.raises(ValueError):
        sink.log_row("metrics", {"iteration": 1, "env_steps": 0, "wall_time": 0.0, "y": 2})
    sink.close()


def test_json_manifest_sink_only_writes_whitelist(tmp_path: Path):
    sink = JsonManifestSink(tables=("summary",))
    sink.open(tmp_path)
    sink.log_row("metrics", {"iteration": 0, "best": 1.0})  # ignored
    sink.log_row("summary", {"final": 42.0})
    sink.close()
    assert (tmp_path / "summary.json").is_file()
    assert not (tmp_path / "metrics.json").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["final"] == 42.0


def test_multi_sink_fans_out(tmp_path: Path):
    csv_sink = CsvScalarSink()
    json_sink = JsonManifestSink(tables=("summary",))
    multi = MultiSink([csv_sink, json_sink, NoOpSink()])
    multi.open(tmp_path)
    multi.log_row("metrics", {"iteration": 0, "env_steps": 0, "wall_time": 0.0, "best": 9})
    multi.log_row("summary", {"final": 9})
    multi.close()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "summary.json").is_file()


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


def test_no_op_logger_when_output_dir_is_none():
    log = ExperimentLogger.create(
        output_dir=None, algorithm="ea", env_name="env", seed=0,
        budget_mode="GENERATIONS", budget_value=1, hyperparams={},
    )
    assert isinstance(log, NoOpLogger)
    # All public methods accept calls and return without raising.
    with log:
        log.row("metrics", {"iteration": 0})
        log.scalar("x", 1.0, step=0)
        log.summary({"a": 1})


def test_experiment_logger_writes_run_dir(tmp_path: Path):
    with ExperimentLogger.create(
        output_dir=tmp_path,
        algorithm="ea",
        env_name="env",
        seed=0,
        budget_mode="GENERATIONS",
        budget_value=2,
        hyperparams={"pop": 4},
    ) as log:
        log.row("metrics", {"iteration": 0, "env_steps": 0, "wall_time": 0.0, "best": 1.0})
        log.row("metrics", {"iteration": 1, "env_steps": 10, "wall_time": 0.05, "best": 1.5})
        log.summary({"final_fitness": 1.5})

    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    rd = runs[0]
    assert (rd / "manifest.json").is_file()
    assert (rd / "metrics.csv").is_file()
    assert (rd / "summary.json").is_file()

    rows = _read_csv_rows(rd / "metrics.csv")
    assert len(rows) == 2

    manifest = json.loads((rd / "manifest.json").read_text())
    assert manifest["algorithm"] == "ea"
    assert manifest["final_summary"]["final_fitness"] == 1.5
    # wall_time_s should be set (close() captures it)
    assert manifest["wall_time_s"] is not None


def test_experiment_logger_run_id_is_unique(tmp_path: Path):
    log_a = ExperimentLogger.create(
        output_dir=tmp_path, algorithm="ea", env_name="env", seed=0,
        budget_mode="GENERATIONS", budget_value=1, hyperparams={},
    )
    log_b = ExperimentLogger.create(
        output_dir=tmp_path, algorithm="ea", env_name="env", seed=0,
        budget_mode="GENERATIONS", budget_value=1, hyperparams={},
    )
    assert log_a.manifest.run_id != log_b.manifest.run_id
    assert log_a.run_dir != log_b.run_dir
    log_a.close()
    log_b.close()


def test_experiment_logger_from_config_no_output_dir():
    class Cfg:
        output_dir = None
        env_name = "env"
        env_config = {}
        budget_mode = type("M", (), {"name": "GENERATIONS"})()
        budget_value = 1
        seeds = [0]
        algorithm_params = {}
    log = ExperimentLogger.from_config(Cfg(), algorithm="ea")
    assert isinstance(log, NoOpLogger)


# ---------------------------------------------------------------------------
# FrameRecorder
# ---------------------------------------------------------------------------


def test_frame_recorder_skipped_when_imageio_missing(tmp_path: Path):
    pytest.importorskip("imageio")
    from deap_pettingzoo.utils.logging import FrameRecorder

    rec = FrameRecorder(run_dir=tmp_path)
    rec.start_episode(0)
    frame = (np.random.default_rng(0).integers(0, 255, size=(8, 8, 3))).astype(np.uint8)
    rec.add_frame(frame)
    rec.add_frame(frame)
    out = rec.end_episode()
    if out is not None:
        # mp4 or gif fallback
        assert out.is_file()
        assert out.suffix in {".mp4", ".gif"}


def test_frame_recorder_episode_prefix(tmp_path: Path):
    """``episode_prefix`` is woven into the artefact filename."""
    pytest.importorskip("imageio")
    from deap_pettingzoo.utils.logging import FrameRecorder

    rec = FrameRecorder(run_dir=tmp_path, episode_prefix="train_")
    rec.start_episode(7)
    frame = (np.random.default_rng(0).integers(0, 255, size=(8, 8, 3))).astype(np.uint8)
    rec.add_frame(frame)
    rec.add_frame(frame)
    out = rec.end_episode()
    if out is not None:
        assert out.is_file()
        assert out.name.startswith("train_episode_0007")
        assert out.suffix in {".mp4", ".gif"}
