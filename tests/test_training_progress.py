"""Tests for Phase 1 trainer-facing additions: progress, aggregation, recording.

Covers the base.py data models (:class:`ProgressEvent`, :class:`FitnessConfig`,
:class:`RecordingConfig.max_recordings`) and their wiring into the EA / CMA-ES /
PSO trainers (progress callbacks, configurable aggregation, ``run_dir`` in
result metadata).
"""

from __future__ import annotations

import pytest

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    FitnessConfig,
    ProgressEvent,
    RecordingConfig,
    TrainingConfig,
)
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer

ENV = "simple_spread_v3"


def _tiny_config(tmp_path=None, **kwargs):
    return TrainingConfig(
        env_name=ENV,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,
        seed=0, max_steps_per_episode=10,
        output_dir=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ProgressEvent
# ---------------------------------------------------------------------------


class TestProgressEvent:
    def test_fraction(self):
        assert ProgressEvent("generation", 0, 10).fraction == 0.0
        assert ProgressEvent("generation", 5, 10).fraction == 0.5
        assert ProgressEvent("generation", 20, 10).fraction == 1.0
        assert ProgressEvent("generation", 1, 0).fraction == 0.0

    def test_to_dict_serialisable(self):
        ev = ProgressEvent("env_step", 3, 9, best_fitness=1.5, mean_fitness=0.5)
        d = ev.to_dict()
        assert d["unit"] == "env_step"
        assert d["current"] == 3
        assert d["fraction"] == pytest.approx(1 / 3)
        assert d["best_fitness"] == 1.5


# ---------------------------------------------------------------------------
# FitnessConfig
# ---------------------------------------------------------------------------


class TestFitnessConfig:
    def test_defaults_to_sum(self):
        assert FitnessConfig().aggregation_method == "sum"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError):
            FitnessConfig(aggregation_method="median")

    def test_team_requires_mapping(self):
        with pytest.raises(ValueError):
            FitnessConfig(aggregation_method="team")
        # With a mapping it is accepted.
        FitnessConfig(aggregation_method="team", team_mapping={"agent_0": "a"})


# ---------------------------------------------------------------------------
# RecordingConfig mode + ProgressEvent.env_steps
# ---------------------------------------------------------------------------


class TestRecordingConfigMode:
    def test_default_mode_is_count(self):
        from deap_pettingzoo.baselines.base import RecordingMode

        assert RecordingConfig().mode is RecordingMode.COUNT

    def test_str_mode_coerces_to_enum(self):
        from deap_pettingzoo.baselines.base import RecordingMode

        assert RecordingConfig(mode="every").mode is RecordingMode.EVERY

    def test_interval_mode_requires_every_n_steps(self):
        from deap_pettingzoo.baselines.base import RecordingMode

        with pytest.raises(ValueError):
            RecordingConfig(enabled=True, mode=RecordingMode.INTERVAL)

    def test_invalid_max_recordings_rejected(self):
        with pytest.raises(ValueError):
            RecordingConfig(max_recordings=0)

    def test_progress_event_carries_env_steps(self):
        ev = ProgressEvent(unit="generation", current=2, total=10, env_steps=500)
        assert ev.env_steps == 500
        assert ev.to_dict()["env_steps"] == 500


# ---------------------------------------------------------------------------
# Progress callbacks fire from the trainers
# ---------------------------------------------------------------------------


def _assert_progress(events, total):
    assert events, "no progress events were emitted"
    assert all(e.unit == "generation" for e in events)
    assert all(e.total == total for e in events)
    currents = [e.current for e in events]
    assert currents == sorted(currents), "progress current must be monotonic"
    assert max(currents) == total
    assert all(e.wall_time_s >= 0.0 for e in events)


class TestTrainerProgress:
    def test_ea_simple_progress(self):
        events = []
        trainer = EATrainer(hidden_size=8, population_size=4)
        trainer.train(_tiny_config(), progress_callback=events.append)
        # EA emits gen 0 plus one per generation (budget 8 / pop 4 = 2 gens).
        _assert_progress(events, total=2)
        assert events[0].current == 0

    def test_cma_progress(self):
        events = []
        trainer = CMAESTrainer(hidden_size=8, population_size=4)
        trainer.train(_tiny_config(), progress_callback=events.append)
        _assert_progress(events, total=2)

    def test_pso_progress(self):
        events = []
        trainer = PSOTrainer(hidden_size=8, population_size=4)
        trainer.train(_tiny_config(), progress_callback=events.append)
        _assert_progress(events, total=2)

    def test_progress_fires_without_output_dir(self):
        """NoOpLogger path (output_dir=None) must still emit progress."""
        events = []
        trainer = CMAESTrainer(hidden_size=8, population_size=4)
        result = trainer.train(_tiny_config(tmp_path=None), progress_callback=events.append)
        assert events
        assert result.metadata["run_dir"] is None

    def test_callback_error_is_isolated(self):
        """A raising callback must not crash training (only warns)."""
        def boom(_event):
            raise RuntimeError("boom")

        trainer = EATrainer(hidden_size=8, population_size=4)
        with pytest.warns(RuntimeWarning):
            result = trainer.train(_tiny_config(), progress_callback=boom)
        assert result.metrics["num_generations"] == 2


# ---------------------------------------------------------------------------
# run_dir + aggregation plumbing
# ---------------------------------------------------------------------------


class TestRunDirAndAggregation:
    def test_run_dir_recorded(self, tmp_path):
        trainer = EATrainer(hidden_size=8, population_size=4)
        result = trainer.train(_tiny_config(tmp_path=tmp_path))
        run_dir = result.metadata["run_dir"]
        assert run_dir is not None
        from pathlib import Path

        assert Path(run_dir).exists()

    def test_aggregation_is_reachable(self):
        """'mean' and 'sum' aggregation must now be selectable and differ.

        Previously every trainer hard-coded 'sum'; this is the gap Phase 1
        closes. With identical seeds the two runs should not coincide
        (sum aggregates across all agents, mean divides by agent count).
        """
        cfg_sum = _tiny_config(fitness=FitnessConfig(aggregation_method="sum"))
        cfg_mean = _tiny_config(fitness=FitnessConfig(aggregation_method="mean"))
        fit_sum = EATrainer(hidden_size=8, population_size=4).train(cfg_sum)
        fit_mean = EATrainer(hidden_size=8, population_size=4).train(cfg_mean)
        assert fit_sum.metrics["final_fitness"] != fit_mean.metrics["final_fitness"]
