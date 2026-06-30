"""Tests for reproducible experiment export."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from deap_pettingzoo.analysis import load_experiment
from deap_pettingzoo.utils.logging import ExperimentLogger


def _make_run(
    output_dir: Path,
    *,
    algorithm: str,
    seed: int,
    final_fitness: float,
    series: List[float],
    hyperparams: dict,
) -> Path:
    with ExperimentLogger.create(
        output_dir=output_dir,
        algorithm=algorithm,
        env_name="dummy_env",
        seed=seed,
        budget_mode="GENERATIONS",
        budget_value=len(series),
        hyperparams=hyperparams,
    ) as log:
        for i, v in enumerate(series):
            log.row(
                "metrics",
                {
                    "iteration": i,
                    "env_steps": i * 10,
                    "wall_time": 0.0,
                    "mean": float(v),
                },
            )
        log.summary({"final_fitness": float(final_fitness), "algorithm": algorithm})
    return log.run_dir  # type: ignore[attr-defined]


def _seed_experiment(tmp_path: Path):
    for seed in range(3):
        _make_run(
            tmp_path,
            algorithm="ea",
            seed=seed,
            final_fitness=1.0 + seed,
            series=[0.0, 1.0 + seed],
            hyperparams={"population_size": 10, "hidden_size": 32},
        )
        _make_run(
            tmp_path,
            algorithm="cma-es",
            seed=seed,
            final_fitness=2.0 + seed,
            series=[0.0, 2.0 + seed],
            hyperparams={"sigma": 0.5},
        )
    return load_experiment(tmp_path)


def test_to_frame_emits_per_algo_env_seed_rows(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    exp = _seed_experiment(tmp_path)
    df = exp.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert {"run_id", "algorithm", "env_name", "seed", "score"} <= set(df.columns)
    # one (algorithm, seed) per row, both algorithms present
    assert sorted(df["algorithm"].unique()) == ["cma-es", "ea"]
    ea = df[df["algorithm"] == "ea"].sort_values("seed")
    assert list(ea["score"]) == [1.0, 2.0, 3.0]


def test_to_frame_include_config_carries_provenance(tmp_path: Path):
    pytest.importorskip("pandas")
    exp = _seed_experiment(tmp_path)
    df = exp.to_frame(include_config=True)
    for col in ("seed", "budget_mode", "budget_value", "git_sha"):
        assert col in df.columns
    assert (df["budget_mode"] == "GENERATIONS").all()
    # hyperparameters are flattened with an hp. prefix
    assert "hp.population_size" in df.columns
    ea = df[df["algorithm"] == "ea"]
    assert (ea["hp.population_size"] == 10).all()


def test_export_csv_roundtrip(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    exp = _seed_experiment(tmp_path)
    out = tmp_path / "campaign.csv"
    returned = exp.export(out)
    assert returned == out
    assert out.is_file()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 6
    assert {"algorithm", "seed", "score", "budget_mode"} <= set(reloaded.columns)


def test_export_infers_format_and_rejects_unknown(tmp_path: Path):
    pytest.importorskip("pandas")
    exp = _seed_experiment(tmp_path)
    json_path = exp.export(tmp_path / "out.json")
    assert json_path.is_file()
    with pytest.raises(ValueError, match="fmt must be"):
        exp.export(tmp_path / "out.xyz")
