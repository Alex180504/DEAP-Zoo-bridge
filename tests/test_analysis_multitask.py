"""Tests for multi-task aggregation pipeline (Phase A–D)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deap_pettingzoo.analysis import (
    NormalizationSpec,
    aggregate_metrics_forest,
    collect_score_matrices,
    final_score_table,
    load_experiment,
    normalize_score_matrices,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_run(
    root: Path, *, algorithm: str, env: str, seed: int, score: float
) -> None:
    run_dir = root / f"{algorithm}_{env}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_dir.name,
        "algorithm": algorithm,
        "env_name": env,
        "seed": seed,
        "final_summary": {"final_fitness": float(score)},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def multi_task_experiment(tmp_path: Path):
    """3 algos x 3 envs x 5 seeds, deterministic scores."""
    rng = np.random.default_rng(0)
    centers = {"ea": 0.5, "cma": 0.65, "rl": 0.4}
    for algo, c in centers.items():
        for env in ("envA", "envB", "envC"):
            for seed in range(5):
                _write_run(
                    tmp_path,
                    algorithm=algo,
                    env=env,
                    seed=seed,
                    score=float(c + rng.normal(0, 0.05)),
                )
    return load_experiment(tmp_path)


# ---------------------------------------------------------------------------
# Phase A — collect_score_matrices / final_score_table
# ---------------------------------------------------------------------------


def test_collect_score_matrices_shape_and_order(multi_task_experiment):
    matrices, envs = collect_score_matrices(multi_task_experiment)
    assert set(matrices.keys()) == {"ea", "cma", "rl"}
    assert set(envs) == {"envA", "envB", "envC"}
    for mat in matrices.values():
        assert mat.shape == (5, 3)
        assert np.all(np.isfinite(mat))


def test_collect_score_matrices_explicit_orders(multi_task_experiment):
    matrices, envs = collect_score_matrices(
        multi_task_experiment,
        algorithms=["rl", "ea"],
        envs=["envC", "envA"],
    )
    assert list(matrices.keys()) == ["rl", "ea"]
    assert envs == ["envC", "envA"]
    for mat in matrices.values():
        assert mat.shape == (5, 2)


def test_collect_score_matrices_nan_fill(tmp_path: Path):
    # 'ea' has 3 seeds in envA but only 1 in envB -> column padded with NaN
    _write_run(tmp_path, algorithm="ea", env="envA", seed=0, score=1.0)
    _write_run(tmp_path, algorithm="ea", env="envA", seed=1, score=2.0)
    _write_run(tmp_path, algorithm="ea", env="envA", seed=2, score=3.0)
    _write_run(tmp_path, algorithm="ea", env="envB", seed=0, score=10.0)
    exp = load_experiment(tmp_path)
    matrices, envs = collect_score_matrices(exp, fill="nan")
    assert envs == ["envA", "envB"] or envs == ["envB", "envA"]
    mat = matrices["ea"]
    assert mat.shape == (3, 2)
    finite_per_col = np.sum(np.isfinite(mat), axis=0)
    assert sorted(finite_per_col.tolist()) == [1, 3]


def test_collect_score_matrices_truncate(tmp_path: Path):
    _write_run(tmp_path, algorithm="ea", env="envA", seed=0, score=1.0)
    _write_run(tmp_path, algorithm="ea", env="envA", seed=1, score=2.0)
    _write_run(tmp_path, algorithm="ea", env="envB", seed=0, score=10.0)
    exp = load_experiment(tmp_path)
    matrices, _ = collect_score_matrices(exp, fill="truncate")
    assert matrices["ea"].shape == (1, 2)
    assert np.all(np.isfinite(matrices["ea"]))


def test_collect_score_matrices_min_seeds_filter(tmp_path: Path):
    _write_run(tmp_path, algorithm="ea", env="envA", seed=0, score=1.0)
    _write_run(tmp_path, algorithm="weak", env="envA", seed=0, score=0.0)
    exp = load_experiment(tmp_path)
    matrices, _ = collect_score_matrices(exp, min_seeds=2)
    assert matrices == {}


def test_final_score_table_long_form(multi_task_experiment):
    df = final_score_table(multi_task_experiment)
    assert set(df.columns) == {"algorithm", "env_name", "seed", "run_id", "score"}
    assert len(df) == 3 * 3 * 5


# ---------------------------------------------------------------------------
# Phase A — NormalizationSpec / normalize_score_matrices
# ---------------------------------------------------------------------------


def test_normspec_from_minmax_rescales_to_unit_range():
    matrices = {
        "a": np.array([[0.0, 10.0], [1.0, 20.0]]),
        "b": np.array([[0.5, 15.0]]),
    }
    spec = NormalizationSpec.from_minmax(matrices, ["e1", "e2"])
    assert spec.low["e1"] == 0.0 and spec.high["e1"] == 1.0
    assert spec.low["e2"] == 10.0 and spec.high["e2"] == 20.0
    normed = normalize_score_matrices(matrices, ["e1", "e2"], spec)
    np.testing.assert_allclose(normed["a"], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(normed["b"], [[0.5, 0.5]])


def test_normspec_from_dict_passes_unknown_envs_through():
    matrices = {"a": np.array([[5.0, 3.0]])}
    spec = NormalizationSpec.from_dict({"e1": 0.0}, {"e1": 10.0})
    normed = normalize_score_matrices(matrices, ["e1", "e2"], spec)
    # e1 normalised to 0.5, e2 left unchanged at 3.0
    np.testing.assert_allclose(normed["a"], [[0.5, 3.0]])


def test_normspec_from_reference_algorithm():
    matrices = {
        "ref": np.array([[2.0], [4.0]]),  # mean = 3.0
        "other": np.array([[1.5]]),
    }
    spec = NormalizationSpec.from_reference_algorithm(
        matrices, ["e"], algorithm="ref"
    )
    assert spec.low["e"] == 0.0
    assert spec.high["e"] == 3.0
    normed = normalize_score_matrices(matrices, ["e"], spec)
    np.testing.assert_allclose(normed["other"], [[0.5]])


def test_normspec_skips_constant_envs():
    matrices = {"a": np.array([[5.0, 1.0], [5.0, 2.0]])}
    spec = NormalizationSpec.from_minmax(matrices, ["constant", "varying"])
    assert "constant" not in spec.low
    assert "varying" in spec.low


def test_normalize_preserves_nans():
    matrices = {"a": np.array([[0.0, np.nan], [10.0, 5.0]])}
    spec = NormalizationSpec.from_minmax(matrices, ["e1", "e2"])
    normed = normalize_score_matrices(matrices, ["e1", "e2"], spec)
    assert np.isnan(normed["a"][0, 1])


# ---------------------------------------------------------------------------
# Phase B — rliable bridge multi-task
# ---------------------------------------------------------------------------


def _rliable_or_skip():
    pytest.importorskip("rliable")


def test_aggregate_metrics_forest_shape():
    _rliable_or_skip()
    rng = np.random.default_rng(0)
    matrices = {
        "a": rng.normal(0.5, 0.1, size=(6, 3)),
        "b": rng.normal(0.7, 0.1, size=(6, 3)),
    }
    forest = aggregate_metrics_forest(matrices, reps=200)
    assert set(forest.columns) == {"algorithm", "metric", "point", "lo", "hi"}
    assert len(forest) == 2 * 4  # 2 algos x 4 metrics
    assert (forest["lo"] <= forest["point"]).all()
    assert (forest["point"] <= forest["hi"]).all()


def test_rliable_score_matrices_kwarg_paths():
    _rliable_or_skip()
    from deap_pettingzoo.analysis.rliable_bridge import (
        aggregate_metrics_table,
        performance_profile,
        probability_of_improvement,
    )

    rng = np.random.default_rng(1)
    matrices = {
        "a": rng.normal(0.5, 0.1, size=(5, 2)),
        "b": rng.normal(0.7, 0.1, size=(5, 2)),
    }
    table = aggregate_metrics_table(score_matrices=matrices, reps=200)
    assert set(table.index) == {"a", "b"}
    taus, point, _ = performance_profile(
        score_matrices=matrices, reps=200, n_tau=8
    )
    assert taus.shape == (8,)
    assert point["a"].shape[-1] == 8
    poi = probability_of_improvement(score_matrices=matrices, reps=200)
    assert {"algo_x", "algo_y", "probability"}.issubset(poi.columns)


def test_rliable_rejects_both_kwargs():
    _rliable_or_skip()
    from deap_pettingzoo.analysis.rliable_bridge import aggregate_metrics_table
    with pytest.raises(ValueError):
        aggregate_metrics_table(
            {"a": [0.1, 0.2]}, score_matrices={"a": np.array([[0.1], [0.2]])}
        )


def test_rliable_single_task_back_compat():
    _rliable_or_skip()
    from deap_pettingzoo.analysis.rliable_bridge import aggregate_metrics_table
    table = aggregate_metrics_table({"a": [0.1, 0.2, 0.3]}, reps=200)
    assert "iqm" in table.columns


# ---------------------------------------------------------------------------
# Phase C — plots smoke
# ---------------------------------------------------------------------------


def test_plot_aggregate_metrics_forest_smoke():
    _rliable_or_skip()
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    matrices = {
        "a": rng.normal(0.5, 0.1, size=(5, 2)),
        "b": rng.normal(0.7, 0.1, size=(5, 2)),
    }
    forest = aggregate_metrics_forest(matrices, reps=200)
    from deap_pettingzoo.analysis.plots import plot_aggregate_metrics_forest
    fig = plot_aggregate_metrics_forest(forest)
    assert len(fig.axes) >= 1


def test_plot_per_env_eval_smoke():
    pytest.importorskip("matplotlib")
    matrices = {
        "a": np.array([[1.0, 2.0], [1.5, 2.5]]),
        "b": np.array([[0.5, 1.0], [0.7, 1.5]]),
    }
    from deap_pettingzoo.analysis.plots import plot_per_env_eval
    fig = plot_per_env_eval(matrices, ["e1", "e2"])
    assert len(fig.axes) == 2


def test_plot_normalized_performance_profile_smoke():
    pytest.importorskip("matplotlib")
    taus = np.linspace(0, 1.5, 10)
    point = {"a": np.linspace(1.0, 0.0, 10)}
    from deap_pettingzoo.analysis.plots import plot_normalized_performance_profile
    fig = plot_normalized_performance_profile(taus, point)
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 1.5)


# ---------------------------------------------------------------------------
# Phase D — report end-to-end
# ---------------------------------------------------------------------------


def test_report_includes_multitask_section(multi_task_experiment, tmp_path: Path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("rliable")
    from deap_pettingzoo.analysis.report import build_report

    out = tmp_path / "report"
    md_path = build_report(multi_task_experiment, out, fmt="md")
    text = md_path.read_text(encoding="utf-8")
    assert "Multi-task aggregate" in text
    assert "envA" in text and "envB" in text and "envC" in text
    html_path = build_report(multi_task_experiment, out, fmt="html")
    html = html_path.read_text(encoding="utf-8")
    assert "Multi-task aggregate" in html
