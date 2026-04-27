"""Tests for ``deap_pettingzoo.ui.services`` modules.

These intentionally avoid importing ``streamlit`` so the suite stays
runnable without the ``[ui]`` extra installed. Page modules
(``deap_pettingzoo/ui/pages/*.py``) do import ``streamlit`` and are
covered separately by Phase 6 smoke tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deap_pettingzoo.ui.services.config import (
    DEFAULT_ROOT_ENV_VAR,
    UISettings,
    load_settings,
)
from deap_pettingzoo.ui.services.data import (
    RunSummary,
    discover_runs,
    load_experiment_cached,
)
from deap_pettingzoo.ui.services.state import UIState
from deap_pettingzoo.utils.logging import ExperimentLogger


def _write_run(tmp_path: Path, *, algorithm: str = "ea", seed: int = 0) -> Path:
    pre_existing = {p for p in tmp_path.iterdir() if p.is_dir()} if tmp_path.is_dir() else set()
    with ExperimentLogger.create(
        output_dir=tmp_path,
        algorithm=algorithm,
        env_name="env",
        seed=seed,
        budget_mode="GENERATIONS",
        budget_value=1,
        hyperparams={"pop": 4},
    ) as log:
        log.row(
            "metrics",
            {"iteration": 0, "env_steps": 0, "wall_time": 0.0, "best": 1.0},
        )
        log.summary({"final_fitness": 1.0})
    new_dirs = [
        p for p in tmp_path.iterdir() if p.is_dir() and p not in pre_existing
    ]
    assert len(new_dirs) == 1, f"expected exactly one new run dir, got {new_dirs}"
    return new_dirs[0]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_load_settings_explicit_override(tmp_path: Path):
    settings = load_settings(results_root=str(tmp_path))
    assert settings.results_root == tmp_path.resolve()


def test_load_settings_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(DEFAULT_ROOT_ENV_VAR, str(tmp_path))
    settings = load_settings()
    assert settings.results_root == tmp_path.resolve()


def test_load_settings_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(DEFAULT_ROOT_ENV_VAR, raising=False)
    settings = load_settings()
    assert settings.results_root == (Path("results")).expanduser().resolve()


# ---------------------------------------------------------------------------
# data — discover_runs / load_experiment_cached
# ---------------------------------------------------------------------------


def test_discover_runs_flat(tmp_path: Path):
    rd = _write_run(tmp_path, algorithm="ea", seed=0)
    summaries = discover_runs(tmp_path)
    assert len(summaries) == 1
    s = summaries[0]
    assert isinstance(s, RunSummary)
    assert s.run_dir == rd
    assert s.algorithm == "ea"
    assert s.seed == 0


def test_discover_runs_nested_one_level(tmp_path: Path):
    nested = tmp_path / "experiment_a"
    nested.mkdir()
    rd = _write_run(nested, algorithm="cma-es")
    summaries = discover_runs(tmp_path)
    assert {s.run_dir for s in summaries} == {rd}
    assert summaries[0].algorithm == "cma-es"


def test_discover_runs_nested_deep(tmp_path: Path):
    # results/<exp>/<algo>/seed_<n>/<run>/manifest.json — 4 levels deep.
    deep = tmp_path / "ea_vs_maddpg" / "ea" / "seed_0"
    deep.mkdir(parents=True)
    rd = _write_run(deep, algorithm="ea", seed=0)
    summaries = discover_runs(tmp_path)
    assert {s.run_dir for s in summaries} == {rd}


def test_discover_runs_skips_non_run_dirs(tmp_path: Path):
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "not_a_run" / "random.txt").write_text("x")
    _write_run(tmp_path, algorithm="ea")
    summaries = discover_runs(tmp_path)
    assert len(summaries) == 1


def test_discover_runs_returns_empty_for_missing_root(tmp_path: Path):
    assert discover_runs(tmp_path / "does-not-exist") == []


def test_discover_runs_sorts_newest_first(tmp_path: Path):
    rd_a = _write_run(tmp_path, algorithm="ea", seed=0)
    rd_b = _write_run(tmp_path, algorithm="ea", seed=1)
    # Force a clear mtime gap regardless of OS clock granularity.
    mfa = rd_a / "manifest.json"
    mfb = rd_b / "manifest.json"
    os.utime(mfa, (mfa.stat().st_atime, mfa.stat().st_mtime - 1000))
    os.utime(mfb, (mfb.stat().st_atime, mfb.stat().st_mtime + 1000))
    summaries = discover_runs(tmp_path)
    assert summaries[0].run_dir == rd_b
    assert summaries[-1].run_dir == rd_a


def test_load_experiment_cached_round_trip(tmp_path: Path):
    rd = _write_run(tmp_path, algorithm="ea")
    exp = load_experiment_cached([rd])
    assert len(exp) == 1
    assert exp[0].algorithm == "ea"


def test_load_experiment_cached_skips_invalid_paths(tmp_path: Path):
    rd = _write_run(tmp_path)
    exp = load_experiment_cached([rd, tmp_path / "missing"])
    assert len(exp) == 1


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def test_uistate_round_trips_through_mapping():
    backing: dict = {}
    s1 = UIState.from_mapping(backing)
    s1.metric = "best"
    s1.statistic = "iqm"
    s1.ci_level = 0.9
    s1.n_resamples = 500
    s1.update_selected(["r1", "r2", "r1"])
    s1.results_root = Path("/tmp/out")
    s1.persist()

    s2 = UIState.from_mapping(backing)
    assert s2.metric == "best"
    assert s2.statistic == "iqm"
    assert s2.ci_level == 0.9
    assert s2.n_resamples == 500
    assert s2.selected_run_ids == ["r1", "r2"]
    assert s2.results_root == Path("/tmp/out")


def test_uistate_defaults_when_backing_empty():
    s = UIState.from_mapping({})
    assert s.metric == "best_fitness"
    assert s.statistic == "iqm"
    assert s.selected_run_ids == []


def test_uistate_persist_without_backing_is_noop():
    s = UIState()
    # Should not raise.
    s.persist()


def test_uistate_update_selected_dedupes_preserve_order():
    s = UIState()
    s.update_selected(["a", "b", "a", "c", "b"])
    assert s.selected_run_ids == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# UISettings dataclass
# ---------------------------------------------------------------------------


def test_uisettings_results_root_exists_flag(tmp_path: Path):
    s = UISettings(results_root=tmp_path)
    assert s.results_root_exists is True
    s2 = UISettings(results_root=tmp_path / "missing")
    assert s2.results_root_exists is False


# ---------------------------------------------------------------------------
# RunSummary — started_at and display_name fields
# ---------------------------------------------------------------------------


def test_run_summary_started_at_populated(tmp_path: Path):
    """discover_runs() populates started_at from the manifest ISO timestamp."""
    from datetime import datetime

    rd = _write_run(tmp_path, algorithm="ea", seed=0)
    # ExperimentLogger writes started_at automatically; just verify it round-trips.
    summaries = discover_runs(tmp_path)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.started_at is not None
    assert isinstance(s.started_at, datetime)


def test_run_summary_started_at_none_if_missing(tmp_path: Path):
    """If manifest.json has no started_at key, field is None (not an error)."""
    import json

    rd = _write_run(tmp_path, algorithm="ea", seed=1)
    mf = rd / "manifest.json"
    data = json.loads(mf.read_text())
    data.pop("started_at", None)
    mf.write_text(json.dumps(data))

    summaries = discover_runs(tmp_path)
    assert summaries[0].started_at is None


def test_run_summary_display_name_populated(tmp_path: Path):
    """discover_runs() populates display_name when present in manifest."""
    import json

    rd = _write_run(tmp_path, algorithm="ea", seed=2)
    mf = rd / "manifest.json"
    data = json.loads(mf.read_text())
    data["display_name"] = "My cool run"
    mf.write_text(json.dumps(data))

    summaries = discover_runs(tmp_path)
    assert summaries[0].display_name == "My cool run"


def test_run_summary_display_name_none_if_absent(tmp_path: Path):
    """If manifest.json has no display_name, field is None."""
    rd = _write_run(tmp_path, algorithm="ea", seed=3)
    summaries = discover_runs(tmp_path)
    assert summaries[0].display_name is None
