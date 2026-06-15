"""Tests for the ExperimentSpec / build_configs / validate_spec layer (Phase 1)."""

from __future__ import annotations

import pytest

import deap_pettingzoo.environments  # noqa: F401 (register envs)
from deap_pettingzoo.baselines.base import BudgetMode
from deap_pettingzoo.baselines.comparison import get_trainer
from deap_pettingzoo.baselines.experiment_spec import (
    BudgetSpec,
    CoevolutionSettings,
    CoevolutionSideSpec,
    ExperimentSpec,
    FitnessSettings,
    ParallelismSettings,
    RecordingSettings,
    build_configs,
    deep_merge,
    validate_spec,
)


def _spec(**kwargs) -> ExperimentSpec:
    base = dict(
        environment="simple_spread_v3",
        algorithms=["ea"],
        budget=BudgetSpec(mode="eval_episodes", value=16, num_seeds=1),
    )
    base.update(kwargs)
    return ExperimentSpec(**base)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_overlay_wins_and_nested_merges(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}}
        overlay = {"b": {"y": 3, "z": 4}, "c": 5}
        assert deep_merge(base, overlay) == {
            "a": 1,
            "b": {"x": 1, "y": 3, "z": 4},
            "c": 5,
        }

    def test_inputs_not_mutated(self):
        base = {"b": {"x": 1}}
        deep_merge(base, {"b": {"x": 2}})
        assert base == {"b": {"x": 1}}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_round_trip(self):
        spec = ExperimentSpec(
            environment="simple_tag_v3",
            algorithms=["ea", "cma-es"],
            env_config={"num_good": 2},
            hyperparameters={"ea": {"population_size": 10}},
            budget=BudgetSpec(mode="env_steps", value=1000, num_seeds=3),
            fitness=FitnessSettings(aggregation_method="mean"),
            recording=RecordingSettings(enabled=True, max_recordings=2),
            parallelism=ParallelismSettings(num_workers=2),
            raw_config={"environment": {"env_config": {"max_cycles": 40}}},
            label="my run",
        )
        restored = ExperimentSpec.from_dict(spec.to_dict())
        assert restored.to_dict() == spec.to_dict()

    def test_from_dict_requires_environment(self):
        with pytest.raises(ValueError):
            ExperimentSpec.from_dict({"algorithms": ["ea"]})


# ---------------------------------------------------------------------------
# build_configs
# ---------------------------------------------------------------------------


class TestBuildConfigs:
    def test_standard_single_algorithm(self):
        plan = build_configs(_spec(budget=BudgetSpec("eval_episodes", 16, 2)))
        assert plan.mode == "standard"
        assert len(plan.sub_runs) == 2  # 1 algo x 2 seeds
        assert [s.seed for s in plan.sub_runs] == [0, 1]
        assert all(s.config.budget_mode is BudgetMode.EVAL_EPISODES for s in plan.sub_runs)

    def test_multi_algorithm_runs(self):
        plan = build_configs(
            _spec(algorithms=["ea", "pso"], budget=BudgetSpec("eval_episodes", 16, 2))
        )
        algos = {s.algorithm for s in plan.sub_runs}
        assert algos == {"ea", "pso"}
        assert len(plan.sub_runs) == 4

    def test_hyperparameter_routing(self):
        spec = _spec(hyperparameters={"ea": {"population_size": 12, "max_steps_per_episode": 33}})
        sub = build_configs(spec).sub_runs[0]
        # population_size -> trainer constructor; max_steps_per_episode -> algorithm_params.
        assert sub.trainer_params["population_size"] == 12
        assert sub.config.algorithm_params["max_steps_per_episode"] == 33
        assert sub.config.algorithm_params["seed"] == 0

    def test_output_dir_layout(self, tmp_path):
        plan = build_configs(_spec(algorithms=["ea"]), output_root=tmp_path)
        sub = plan.sub_runs[0]
        assert sub.config.output_dir == tmp_path / "ea" / "seed_0"

    def test_alias_canonicalised(self):
        plan = build_configs(_spec(algorithms=["cma"]))  # alias of cma-es
        assert plan.sub_runs[0].algorithm == "cma-es"

    def test_three_layer_merge_precedence(self):
        spec = _spec(
            hyperparameters={"ea": {"population_size": 99}},
            raw_config={
                "algorithms": {
                    "ea": {"hyperparameters": {"population_size": 7, "tournament_size": 5}}
                }
            },
        )
        tp = build_configs(spec).sub_runs[0].trainer_params
        assert tp["population_size"] == 99  # explicit form beats raw
        assert tp["tournament_size"] == 5  # raw beats catalog default

    def test_env_config_merge(self):
        spec = _spec(
            env_config={"N": 5},
            raw_config={"environment": {"env_config": {"max_cycles": 40}}},
        )
        cfg = build_configs(spec).sub_runs[0].config
        assert cfg.env_config["N"] == 5
        assert cfg.env_config["max_cycles"] == 40


# ---------------------------------------------------------------------------
# validate_spec — table-driven edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_spec_passes(self):
        assert validate_spec(_spec()).ok

    def test_unknown_environment(self):
        report = validate_spec(_spec(environment="does_not_exist_v0"))
        assert not report.ok
        assert any(i.code == "unknown_env" for i in report.errors)

    def test_unknown_algorithm(self):
        report = validate_spec(_spec(algorithms=["supervised-dl"]))
        assert any(i.code == "unknown_algorithm" for i in report.errors)

    def test_no_algorithms(self):
        report = validate_spec(_spec(algorithms=[]))
        assert any(i.code == "no_algorithms" for i in report.errors)

    def test_wall_clock_rejected(self):
        report = validate_spec(_spec(budget=BudgetSpec(mode="wall_clock", value=10)))
        assert any(i.code == "invalid_budget_mode" for i in report.errors)

    def test_negative_budget_and_seeds(self):
        report = validate_spec(_spec(budget=BudgetSpec("eval_episodes", -1, 0)))
        codes = {i.code for i in report.errors}
        assert "out_of_range" in codes

    def test_out_of_range_hyperparameter(self):
        report = validate_spec(_spec(hyperparameters={"ea": {"population_size": 1}}))
        assert any(i.code == "out_of_range" for i in report.errors)

    def test_wrong_type_hyperparameter(self):
        report = validate_spec(_spec(hyperparameters={"ea": {"population_size": "lots"}}))
        assert any(i.code == "type_error" for i in report.errors)

    def test_invalid_enum_hyperparameter(self):
        report = validate_spec(_spec(hyperparameters={"ea": {"strategy": "quantum"}}))
        assert any(i.code == "type_error" for i in report.errors)

    def test_unknown_hyperparameter_warns(self):
        report = validate_spec(_spec(hyperparameters={"ea": {"made_up": 1}}))
        assert report.ok  # warning only
        assert any(i.code == "unknown_param" for i in report.warnings)

    def test_unknown_env_param_warns(self):
        report = validate_spec(_spec(env_config={"made_up": 1}))
        assert any(i.code == "unknown_param" for i in report.warnings)

    def test_out_of_range_env_param(self):
        report = validate_spec(_spec(env_config={"max_cycles": 0}))
        assert any(i.code == "out_of_range" for i in report.errors)

    def test_invalid_aggregation_for_population(self):
        report = validate_spec(
            _spec(fitness=FitnessSettings(aggregation_method="median"))
        )
        assert any(i.code == "invalid_aggregation" for i in report.errors)

    def test_team_aggregation_needs_mapping(self):
        report = validate_spec(
            _spec(fitness=FitnessSettings(aggregation_method="team"))
        )
        assert any(i.code == "missing_team_mapping" for i in report.errors)

    def test_invalid_parallel_backend(self):
        report = validate_spec(
            _spec(parallelism=ParallelismSettings(parallel_backend="quantum"))
        )
        assert any(i.code == "invalid_backend" for i in report.errors)

    def test_invalid_recording(self):
        spec = _spec(recording=RecordingSettings(enabled=True, fps=0))
        report = validate_spec(spec)
        assert any(i.code == "out_of_range" for i in report.errors)


class TestCoevolutionValidation:
    def _coev_spec(self, **coev_kwargs):
        defaults = dict(
            enabled=True,
            mode="competitive_2pop",
            schedule="synchronous",
            sides=[
                CoevolutionSideSpec("pred", "ea", ("adversary",)),
                CoevolutionSideSpec("prey", "cma-es", ("agent",)),
            ],
        )
        defaults.update(coev_kwargs)
        return _spec(
            environment="simple_tag_v3",
            algorithms=[],
            coevolution=CoevolutionSettings(**defaults),
        )

    def test_valid_coevolution(self):
        assert validate_spec(self._coev_spec()).ok

    def test_too_few_sides(self):
        report = validate_spec(
            self._coev_spec(sides=[CoevolutionSideSpec("solo", "ea", ("adversary",))])
        )
        assert any(i.code == "too_few_sides" for i in report.errors)

    def test_invalid_mode(self):
        report = validate_spec(self._coev_spec(mode="bogus"))
        assert any(i.code == "invalid_mode" for i in report.errors)

    def test_uncovered_team(self):
        report = validate_spec(
            self._coev_spec(
                sides=[
                    CoevolutionSideSpec("pred", "ea", ("adversary",)),
                    CoevolutionSideSpec("pred2", "cma-es", ("adversary",)),
                ]
            )
        )
        assert any(i.code == "uncovered_team" for i in report.errors)

    def test_ineligible_optimizer(self):
        report = validate_spec(
            self._coev_spec(
                sides=[
                    CoevolutionSideSpec("pred", "ippo", ("adversary",)),
                    CoevolutionSideSpec("prey", "ea", ("agent",)),
                ]
            )
        )
        assert any(i.code == "ineligible_optimizer" for i in report.errors)

    def test_build_coevolution_plan(self):
        plan = build_configs(self._coev_spec(), output_root=None)
        assert plan.mode == "coevolution"
        assert len(plan.sub_runs) == 1
        sub = plan.sub_runs[0]
        assert sub.algorithm == "coevolution"
        coev = sub.config.algorithm_params["coevolution"]
        assert [s.optimizer for s in coev.sides] == ["ea", "cma"]


# ---------------------------------------------------------------------------
# Integration: build -> train (tiny EA) with progress + run_dir
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_build_and_train_tiny(self, tmp_path):
        spec = _spec(
            algorithms=["ea"],
            hyperparameters={"ea": {"population_size": 4, "hidden_size": 8}},
            budget=BudgetSpec(mode="eval_episodes", value=8, num_seeds=1),
        )
        report = validate_spec(spec)
        assert report.ok, [i.message for i in report.errors]

        plan = build_configs(spec, output_root=tmp_path)
        sub = plan.sub_runs[0]
        events = []
        trainer = get_trainer(sub.algorithm, **sub.trainer_params)
        result = trainer.train(sub.config, progress_callback=events.append)

        assert events, "expected progress events"
        assert events[-1].current == events[-1].total
        run_dir = result.metadata["run_dir"]
        assert run_dir is not None
        from deap_pettingzoo.analysis.loaders import load_run

        record = load_run(run_dir)
        assert record.algorithm == "ea"
