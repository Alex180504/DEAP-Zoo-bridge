"""Tests for the analysis library (deap_pettingzoo.analysis)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from deap_pettingzoo.analysis import (
    Experiment,
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    iqm,
    learning_curve_with_ci,
    load_experiment,
    load_run,
    mann_whitney,
    mean_with_ci,
    pairwise_table,
    welch_t,
)
from deap_pettingzoo.analysis.aggregate import aggregate_summary, iqm_with_ci
from deap_pettingzoo.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    output_dir: Path,
    *,
    algorithm: str,
    seed: int,
    final_fitness: float,
    series: List[float],
) -> Path:
    """Create a logged run on disk and return its run directory."""
    with ExperimentLogger.create(
        output_dir=output_dir,
        algorithm=algorithm,
        env_name="dummy_env",
        seed=seed,
        budget_mode="GENERATIONS",
        budget_value=len(series),
        hyperparams={},
    ) as log:
        for i, v in enumerate(series):
            log.row(
                "metrics",
                {
                    "iteration": i,
                    "env_steps": i * 10,
                    "wall_time": 0.0,
                    "best": float(v),
                    "mean": float(v),
                },
            )
        log.summary({"final_fitness": float(final_fitness), "algorithm": algorithm})
    # Logger created exactly one new run dir under output_dir.
    return log.run_dir  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Statistics correctness
# ---------------------------------------------------------------------------


def test_iqm_matches_rliable_when_available():
    rliable_metrics = pytest.importorskip("rliable.metrics")
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ours = iqm(arr)
    theirs = float(rliable_metrics.aggregate_iqm(arr.reshape(-1, 1)))
    assert abs(ours - theirs) < 1e-9


def test_iqm_handles_small_sample():
    # < 4 samples -> falls back to mean
    assert abs(iqm([1.0, 2.0, 3.0]) - 2.0) < 1e-12


def test_iqm_tie_heavy_count_trims_not_quantile_masks():
    # Test trim IQM quantile mask returns 1.75
    arr = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(iqm(arr) - 1.75) < 1e-12


def test_iqm_matches_scipy_trim_mean_on_ties():
    stats = pytest.importorskip("scipy.stats")
    arr = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(iqm(arr) - float(stats.trim_mean(arr, 0.25))) < 1e-12


def test_iqm_empty_returns_nan():
    assert np.isnan(iqm([]))


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=10.0, scale=1.0, size=200).tolist()
    lo, hi = bootstrap_ci(sample, n_bootstrap=500, rng=rng)
    assert lo < 10.0 < hi


def test_mean_with_ci_returns_three_floats():
    m, lo, hi = mean_with_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_bootstrap=200)
    assert lo <= m <= hi


def test_iqm_with_ci_brackets_iqm():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, size=100).tolist()
    pt, lo, hi = iqm_with_ci(sample, n_bootstrap=200, rng=rng)
    assert lo <= pt <= hi


def test_cohens_d_sign_and_magnitude():
    # Two distinct means, equal variance: Cohen's d ≈ (m_a - m_b)/sigma
    a = np.full(20, 1.0) + 1e-3 * np.arange(20)
    b = np.full(20, 0.0) + 1e-3 * np.arange(20)
    d = cohens_d(a, b)
    assert d > 100  # essentially identical std → very large effect


def test_cliffs_delta_bounds():
    a = [10, 11, 12]
    b = [1, 2, 3]
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0
    assert cliffs_delta(a, a) == 0.0


def test_mann_whitney_returns_expected_keys():
    res = mann_whitney([1, 2, 3, 4], [10, 11, 12, 13])
    for k in ("statistic", "p_value", "n_x", "n_y", "effect_cliffs_delta"):
        assert k in res
    assert res["p_value"] < 0.1
    assert res["effect_cliffs_delta"] == -1.0


def test_welch_t_returns_expected_keys():
    res = welch_t([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])
    for k in ("t", "p_value", "df", "n_x", "n_y", "effect_cohens_d"):
        assert k in res
    assert res["p_value"] < 0.05


# ---------------------------------------------------------------------------
# Loaders + Experiment
# ---------------------------------------------------------------------------


def test_load_run_reads_csv_and_summary(tmp_path: Path):
    run_dir = _make_run(
        tmp_path,
        algorithm="ea",
        seed=0,
        final_fitness=1.5,
        series=[0.5, 1.0, 1.5],
    )
    rec = load_run(run_dir)
    assert rec.algorithm == "ea"
    assert rec.env_name == "dummy_env"
    assert rec.seed == 0
    assert rec.final_fitness == 1.5
    assert rec.metrics is not None
    assert len(rec.metrics) == 3


def test_load_run_missing_manifest_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_load_experiment_finds_all_runs(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed, final_fitness=float(seed),
                  series=[0.0, float(seed)])
        _make_run(tmp_path, algorithm="cma-es", seed=seed, final_fitness=10.0 + seed,
                  series=[0.0, 10.0 + seed])
    exp = load_experiment(tmp_path)
    assert len(exp) == 6
    assert sorted(exp.by_algorithm()) == ["cma-es", "ea"]


def test_experiment_filter_and_final_scores(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed, final_fitness=1.0 + seed,
                  series=[0.0, 1.0 + seed])
        _make_run(tmp_path, algorithm="cma-es", seed=seed, final_fitness=2.0 + seed,
                  series=[0.0, 2.0 + seed])
    exp = load_experiment(tmp_path)
    only_ea = exp.filter(algorithm="ea")
    assert len(only_ea) == 3
    assert all(r.algorithm == "ea" for r in only_ea)

    scores = exp.final_scores()
    assert sorted(scores) == ["cma-es", "ea"]
    assert sorted(scores["ea"]) == [1.0, 2.0, 3.0]


def test_experiment_summary_table(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    for seed in range(2):
        _make_run(tmp_path, algorithm="ea", seed=seed, final_fitness=float(seed),
                  series=[0.0, float(seed)])
    exp = load_experiment(tmp_path)
    df = exp.summary_table()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert {"run_id", "algorithm", "env_name", "seed", "final_fitness"} <= set(df.columns)


# ---------------------------------------------------------------------------
# Curves & pairwise tables
# ---------------------------------------------------------------------------


def test_learning_curve_with_ci_aligns_runs(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed, final_fitness=float(seed),
                  series=[0.0, 0.5 + seed * 0.1, 1.0 + seed * 0.2])
    exp = load_experiment(tmp_path)
    runs = list(exp.by_algorithm()["ea"])
    df = learning_curve_with_ci(runs, x_col="env_steps", y_col="best")
    assert len(df) == 3
    assert {"env_steps", "mean", "lo", "hi", "n_runs"} <= set(df.columns)
    # CI should bracket the point estimate.
    for _, row in df.iterrows():
        assert row["lo"] <= row["mean"] <= row["hi"]


def test_learning_curve_raises_when_columns_missing(tmp_path: Path):
    _make_run(tmp_path, algorithm="ea", seed=0, final_fitness=0.0, series=[0.0])
    exp = load_experiment(tmp_path)
    runs = list(exp.by_algorithm()["ea"])
    with pytest.raises(ValueError):
        learning_curve_with_ci(runs, x_col="env_steps", y_col="nonexistent_col")


def _make_mean_only_run(
    output_dir: Path, *, algorithm: str, seed: int, series: List[float]
) -> Path:
    """Create a product-path-shaped run: metrics carry ``mean`` but no ``best``."""
    with ExperimentLogger.create(
        output_dir=output_dir,
        algorithm=algorithm,
        env_name="dummy_env",
        seed=seed,
        budget_mode="GENERATIONS",
        budget_value=len(series),
        hyperparams={},
    ) as log:
        for i, v in enumerate(series):
            log.row(
                "metrics",
                {
                    "iteration": i,
                    "env_steps": i * 10,
                    "wall_time": 0.0,
                    "mean": float(v),
                    "min": float(v) - 1.0,
                    "max": float(v) + 1.0,
                },
            )
        log.summary({"final_fitness": float(series[-1]), "algorithm": algorithm})
    return log.run_dir  # type: ignore[attr-defined]


def test_learning_curve_auto_detects_mean_when_best_absent(tmp_path: Path):
    # The UI path writes mean/min/max but
    # never "best", so the old default y_col="best" produced an empty
    # curve. Auto-detection (best->mean->max) must yield a non-empty curve.
    for seed in range(3):
        _make_mean_only_run(
            tmp_path, algorithm="ea", seed=seed, series=[0.0, 1.0 + seed, 2.0 + seed]
        )
    exp = load_experiment(tmp_path)
    runs = list(exp.by_algorithm()["ea"])
    df = learning_curve_with_ci(runs)  # no y_col -> auto-detect
    assert len(df) > 0
    assert {"env_steps", "mean", "lo", "hi", "n_runs"} <= set(df.columns)


def test_aggregate_summary_warns_on_unequal_n(tmp_path: Path):
    # Pooling unequal seed counts must warn and still surface each algorithm's n.
    for seed in range(4):
        _make_run(tmp_path, algorithm="ea", seed=seed,
                  final_fitness=1.0 + 0.1 * seed, series=[0.0, 1.0])
    for seed in range(2):
        _make_run(tmp_path, algorithm="cma-es", seed=seed,
                  final_fitness=2.0 + 0.1 * seed, series=[0.0, 2.0])
    exp = load_experiment(tmp_path)
    with pytest.warns(UserWarning, match="unequal run counts"):
        agg = aggregate_summary(exp.final_scores())
    assert dict(agg["n_seeds"]) == {"ea": 4, "cma-es": 2}


def test_aggregate_summary_require_equal_n_raises(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed, final_fitness=float(seed),
                  series=[0.0, 1.0])
    _make_run(tmp_path, algorithm="cma-es", seed=0, final_fitness=2.0,
              series=[0.0, 2.0])
    exp = load_experiment(tmp_path)
    with pytest.raises(ValueError, match="unequal run counts"):
        aggregate_summary(exp.final_scores(), require_equal_n=True)


def test_per_agent_table_one_row_per_agent():
    # A per-agent decomposition sourced from a
    # run's trajectories frame. The frame is built in memory (the exact shape
    # trajectories.parquet hydrates into via load_run) to avoid a
    # known Windows pyarrow native crash in the test suite.
    pd = pytest.importorskip("pandas")
    from deap_pettingzoo.analysis.loaders import RunRecord, per_agent_table

    traj = pd.DataFrame(
        {
            "episode_idx": [0, 0, 1, 1],
            "step": [0, 0, 0, 0],
            "agent": ["agent_0", "agent_1", "agent_0", "agent_1"],
            "reward": [1.0, 2.0, 3.0, 4.0],
            "done": [True, True, True, True],
        }
    )
    rec = RunRecord(
        run_dir=Path("run_0"),
        manifest={"algorithm": "ea", "env_name": "dummy_env", "seed": 0,
                  "run_id": "run_0"},
        trajectories=traj,
    )
    table = per_agent_table([rec])
    assert sorted(table["agent"]) == ["agent_0", "agent_1"]
    by_agent = dict(zip(table["agent"], table["total_reward"]))
    assert by_agent["agent_0"] == 4.0  # 1 + 3
    assert by_agent["agent_1"] == 6.0  # 2 + 4
    # mean over the 2 episodes
    mean_by_agent = dict(zip(table["agent"], table["mean_episode_reward"]))
    assert mean_by_agent["agent_0"] == 2.0
    assert (table["source"] == "trajectories").all()
    # Also reachable through the Experiment facade.
    exp_table = Experiment(root=Path("."), runs=[rec]).per_agent_table()
    assert len(exp_table) == 2


def test_per_agent_table_falls_back_to_held_out_eval():
    # When no trajectories.parquet exists,
    # the per-agent view must still surface the held-out per-agent means that
    # the runner persists to summary['eval_agent_rewards'].
    pytest.importorskip("pandas")
    from deap_pettingzoo.analysis.loaders import RunRecord, per_agent_table

    rec = RunRecord(
        run_dir=Path("run_0"),
        manifest={"algorithm": "maddpg", "env_name": "simple_tag", "seed": 1,
                  "run_id": "run_0"},
        trajectories=None,
        summary={"eval_agent_rewards": {"adversary_0": -1.5, "agent_0": 1.5}},
    )
    table = per_agent_table([rec])
    assert sorted(table["agent"]) == ["adversary_0", "agent_0"]
    assert (table["source"] == "held_out_eval").all()
    means = dict(zip(table["agent"], table["mean_episode_reward"]))
    assert means["adversary_0"] == -1.5
    assert means["agent_0"] == 1.5


def test_pairwise_table_two_algos(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed,
                  final_fitness=1.0 + seed * 0.01, series=[0.0, 1.0])
        _make_run(tmp_path, algorithm="cma-es", seed=seed,
                  final_fitness=10.0 + seed * 0.01, series=[0.0, 10.0])
    exp = load_experiment(tmp_path)
    pt = pairwise_table(exp)
    assert len(pt) == 1
    row = pt.iloc[0]
    assert {row["algo_x"], row["algo_y"]} == {"ea", "cma-es"}
    assert row["n_x"] == 3 and row["n_y"] == 3
    assert abs(abs(row["effect"]) - 1.0) < 1e-12  # perfectly separated


def test_aggregate_summary_returns_per_algo_rows(tmp_path: Path):
    for seed in range(4):
        _make_run(tmp_path, algorithm="ea", seed=seed,
                  final_fitness=1.0 + 0.1 * seed, series=[0.0, 1.0])
        _make_run(tmp_path, algorithm="cma-es", seed=seed,
                  final_fitness=2.0 + 0.1 * seed, series=[0.0, 2.0])
    exp = load_experiment(tmp_path)
    agg = aggregate_summary(exp.final_scores())
    assert sorted(agg.index) == ["cma-es", "ea"]
    for col in ("mean", "iqm", "median", "n_seeds"):
        assert col in agg.columns


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_build_markdown_report(tmp_path: Path):
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed,
                  final_fitness=1.0 + seed, series=[0.0, 1.0])
        _make_run(tmp_path, algorithm="cma-es", seed=seed,
                  final_fitness=2.0 + seed, series=[0.0, 2.0])
    exp = load_experiment(tmp_path)

    from deap_pettingzoo.analysis.report import build_report

    out_dir = tmp_path / "report"
    path = build_report(exp, out_dir, fmt="md", include_figures=False)
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "Experiment report" in body
    assert "ea" in body and "cma-es" in body


# ---------------------------------------------------------------------------
# rliable bridge (skip when rliable is not installed)
# ---------------------------------------------------------------------------


def test_rliable_aggregate_metrics_table(tmp_path: Path):
    pytest.importorskip("rliable")
    for seed in range(3):
        _make_run(tmp_path, algorithm="ea", seed=seed,
                  final_fitness=1.0 + seed * 0.1, series=[0.0, 1.0])
        _make_run(tmp_path, algorithm="cma-es", seed=seed,
                  final_fitness=2.0 + seed * 0.1, series=[0.0, 2.0])
    exp = load_experiment(tmp_path)
    from deap_pettingzoo.analysis.rliable_bridge import (
        aggregate_metrics_table,
        performance_profile,
    )

    tbl = aggregate_metrics_table(exp.final_scores(), reps=100)
    assert sorted(tbl.index) == ["cma-es", "ea"]
    assert {"mean", "median", "iqm"} <= set(tbl.columns)
    # Optimality_gap is meaningless on raw single-task scores, so
    # it is dropped from that path unless explicitly requested.
    assert "optimality_gap" not in tbl.columns
    forced = aggregate_metrics_table(
        exp.final_scores(), reps=100, include_optimality_gap=True
    )
    assert "optimality_gap" in forced.columns

    taus, points, intervals = performance_profile(exp.final_scores(), n_tau=5, reps=100)
    assert taus.size == 5
    assert sorted(points) == ["cma-es", "ea"]
