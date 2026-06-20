"""Tests for the unified catalog (Phase 0 of the UI overhaul).

Covers the dependency-free schema primitives, hybrid hyperparameter discovery,
algorithm / environment / co-evolution enumeration, availability degradation,
compatibility sourcing, JSON serialisation, and drift guards that keep the
catalog in sync with the trainers and co-evolution constants it describes.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

import deap_pettingzoo.environments  # noqa: F401  (triggers env registration)
from deap_pettingzoo import catalog
from deap_pettingzoo.catalog import (
    algorithms as cat_algorithms,
    availability,
    coevolution as cat_coevolution,
    environments as cat_environments,
    introspection,
)
from deap_pettingzoo.catalog.schema import (
    AlgorithmSpec,
    ParamSpec,
    ParamTarget,
    ParamType,
    TeamSpec,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# ParamSpec coercion / validation
# ---------------------------------------------------------------------------


class TestParamSpecCoercion:
    def test_int_coercion_from_string_and_float(self):
        spec = ParamSpec("n", ParamType.INT, 1)
        assert spec.coerce("50") == 50
        assert spec.coerce(50.0) == 50
        assert spec.coerce(7) == 7

    def test_int_rejects_non_integer_float_and_bool(self):
        spec = ParamSpec("n", ParamType.INT, 1)
        with pytest.raises(ValueError):
            spec.coerce(2.5)
        with pytest.raises(ValueError):
            spec.coerce(True)

    def test_float_coercion(self):
        spec = ParamSpec("x", ParamType.FLOAT, 0.0)
        assert spec.coerce("0.5") == 0.5
        assert spec.coerce(3) == 3.0
        with pytest.raises(ValueError):
            spec.coerce(True)

    def test_bool_coercion_variants(self):
        spec = ParamSpec("b", ParamType.BOOL, False)
        assert spec.coerce("true") is True
        assert spec.coerce("0") is False
        assert spec.coerce(1) is True
        assert spec.coerce(False) is False
        with pytest.raises(ValueError):
            spec.coerce("maybe")

    def test_enum_coercion_and_rejection(self):
        spec = ParamSpec(
            "mode", ParamType.ENUM, "a", choices=("a", "b", "c")
        )
        assert spec.coerce("b") == "b"
        with pytest.raises(ValueError):
            spec.coerce("z")

    def test_list_coercion_with_item_type(self):
        spec = ParamSpec(
            "grid", ParamType.LIST, None, item_type=ParamType.INT, nullable=True
        )
        assert spec.coerce([1, "2", 3.0]) == [1, 2, 3]
        with pytest.raises(ValueError):
            spec.coerce("not-a-list")

    def test_nullable_accepts_none(self):
        spec = ParamSpec("seed", ParamType.INT, None, nullable=True)
        assert spec.coerce(None) is None
        non_nullable = ParamSpec("seed", ParamType.INT, 0)
        with pytest.raises(ValueError):
            non_nullable.coerce(None)


class TestParamSpecValidation:
    def test_range_checks(self):
        spec = ParamSpec("x", ParamType.FLOAT, 0.5, minimum=0.0, maximum=1.0)
        assert spec.validate(0.5) == []
        assert spec.validate(-0.1)[0].code == "out_of_range"
        assert spec.validate(1.5)[0].code == "out_of_range"

    def test_type_error_reported(self):
        spec = ParamSpec("n", ParamType.INT, 1)
        issues = spec.validate("abc")
        assert len(issues) == 1
        assert issues[0].code == "type_error"

    def test_nullable_skips_validation(self):
        spec = ParamSpec("seed", ParamType.INT, None, nullable=True, minimum=0)
        assert spec.validate(None) == []

    def test_to_dict_shape(self):
        spec = ParamSpec(
            "population_size",
            ParamType.INT,
            50,
            minimum=2,
            target=ParamTarget.TRAINER,
            group="Population",
        )
        d = spec.to_dict()
        assert d["name"] == "population_size"
        assert d["type"] == "int"
        assert d["target"] == "trainer"
        assert d["label"] == "Population Size"  # humanised fallback
        assert d["minimum"] == 2


class TestValidationReport:
    def test_ok_and_partitioning(self):
        report = ValidationReport()
        assert report.ok
        report.add("a", "x", "an error")
        report.add("b", "y", "a warning", severity="warning")
        assert not report.ok
        assert len(report.errors) == 1
        assert len(report.warnings) == 1

    def test_raise_if_invalid(self):
        report = ValidationReport()
        report.add("a", "x", "bad")
        with pytest.raises(ValueError):
            report.raise_if_invalid()


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_signature_defaults(self):
        def fn(a, b=2, *args, c=3, **kwargs):
            return a

        assert introspection.signature_defaults(fn) == {"b": 2, "c": 3}

    def test_collect_defaults_merges_default_params(self):
        def ctor(self, device="cpu"):
            return None

        merged = introspection.collect_defaults(ctor, {"batch_size": 64})
        assert merged == {"device": "cpu", "batch_size": 64}

    def test_refresh_defaults_updates_matching(self):
        specs = (ParamSpec("hidden_size", ParamType.INT, 1),)
        refreshed = introspection.refresh_defaults(specs, {"hidden_size": 64})
        assert refreshed[0].default == 64

    def test_refresh_defaults_ignores_uncoercible(self):
        specs = (ParamSpec("mode", ParamType.ENUM, "a", choices=("a", "b")),)
        refreshed = introspection.refresh_defaults(specs, {"mode": object()})
        assert refreshed[0].default == "a"


# ---------------------------------------------------------------------------
# Algorithm catalog
# ---------------------------------------------------------------------------


_EXPECTED_ALGORITHMS = {
    "ea",
    "cma-es",
    "pso",
    "maddpg",
    "matd3",
    "ippo",
    "ippo-grouped",
}


class TestAlgorithmCatalog:
    def test_all_algorithms_present_including_pso(self):
        names = set(cat_algorithms.list_algorithm_names())
        assert names == _EXPECTED_ALGORITHMS

    def test_alias_reconciliation(self):
        assert catalog.canonical_algorithm_name("cma") == "cma-es"
        assert catalog.canonical_algorithm_name("cmaes") == "cma-es"
        assert catalog.canonical_algorithm_name("ea") == "ea"
        assert catalog.canonical_algorithm_name("does-not-exist") is None

    def test_get_algorithm_via_alias(self):
        spec = catalog.get_algorithm("cma")
        assert spec.name == "cma-es"
        assert isinstance(spec, AlgorithmSpec)

    def test_pso_registered_in_comparison_registry(self):
        from deap_pettingzoo.baselines.comparison import _TRAINER_REGISTRY

        assert "pso" in _TRAINER_REGISTRY

    def test_every_algorithm_has_hyperparameters(self):
        for spec in catalog.list_algorithms():
            assert spec.hyperparameters, f"{spec.name} has no hyperparameters"
            # Every spec must have a seed knob routed to algorithm_params.
            seed = spec.hyperparameter("seed")
            assert seed is not None
            assert seed.target is ParamTarget.ALGORITHM_PARAMS

    def test_coevolution_optimizer_mapping(self):
        assert catalog.get_algorithm("ea").coevolution_optimizer == "ea"
        assert catalog.get_algorithm("cma-es").coevolution_optimizer == "cma"
        assert catalog.get_algorithm("pso").coevolution_optimizer == "pso"
        assert catalog.get_algorithm("maddpg").coevolution_optimizer == "rl"
        # On-policy IPPO / GroupedIPPO are now valid co-evolution RL sides.
        assert catalog.get_algorithm("ippo").coevolution_optimizer == "rl"
        assert catalog.get_algorithm("ippo-grouped").coevolution_optimizer == "rl"

    def test_marl_algorithms_require_extras(self):
        maddpg = catalog.get_algorithm("maddpg")
        assert "agilerl" in maddpg.required_modules
        assert "rl-baselines" in maddpg.required_extras
        # Population trainers need only core deps.
        assert catalog.get_algorithm("ea").required_modules == ("deap",)

    @pytest.mark.parametrize("name", sorted(_EXPECTED_ALGORITHMS))
    def test_catalog_capabilities_match_trainer(self, name):
        """Drift guard: declared capabilities == trainer.get_capabilities()."""
        from deap_pettingzoo.baselines.comparison import get_trainer

        spec = catalog.get_algorithm(name)
        try:
            trainer = get_trainer(name)
        except Exception:  # pragma: no cover - missing optional extra
            pytest.skip(f"{name} trainer not instantiable in this environment")
        assert trainer.get_capabilities() == spec.capabilities

    @pytest.mark.parametrize("name", ["ea", "cma-es", "pso"])
    def test_overlay_defaults_match_constructor(self, name):
        """Drift guard: overlay defaults track the constructor defaults."""
        from deap_pettingzoo.baselines.comparison import _TRAINER_REGISTRY

        spec = catalog.get_algorithm(name)
        trainer_cls = _TRAINER_REGISTRY[name]
        discovered = introspection.collect_defaults(
            trainer_cls.__init__, getattr(trainer_cls, "DEFAULT_PARAMS", None)
        )
        assert introspection.drift_report(spec.hyperparameters, discovered) == []


class TestAvailabilityDegradation:
    def test_missing_extra_marks_marl_unavailable(self, monkeypatch):
        availability.clear_cache()
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name in ("agilerl", "torch"):
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        availability.clear_cache()
        try:
            assert cat_algorithms.is_algorithm_available("maddpg") is False
            assert cat_algorithms.is_algorithm_available("ea") is True
            # The unavailable algorithm is still listed (so the UI can grey it).
            names = cat_algorithms.list_algorithm_names()
            assert "maddpg" in names
            available_only = cat_algorithms.list_algorithms(only_available=True)
            assert "maddpg" not in {s.name for s in available_only}
        finally:
            availability.clear_cache()


# ---------------------------------------------------------------------------
# Environment catalog
# ---------------------------------------------------------------------------


class TestEnvironmentCatalog:
    def test_known_mpe_envs_present(self):
        names = {e.name for e in catalog.list_environments()}
        assert "simple_spread_v3" in names
        assert "simple_tag_v3" in names

    def test_env_config_params_typed(self):
        env = catalog.get_environment("simple_spread_v3")
        param_names = {p.name for p in env.env_config}
        assert {"max_cycles", "continuous_actions", "N"} <= param_names
        max_cycles = env.env_param("max_cycles")
        assert max_cycles.type is ParamType.INT
        assert max_cycles.target is ParamTarget.ENV_CONFIG
        assert max_cycles.minimum == 1

    def test_multi_team_detection(self):
        tag = catalog.get_environment("simple_tag_v3")
        assert tag.multi_team
        assert {t.name for t in tag.teams} == {"predators", "prey"}
        predators = next(t for t in tag.teams if t.name == "predators")
        assert predators.agent_prefixes == ("adversary",)

        spread = catalog.get_environment("simple_spread_v3")
        assert not spread.multi_team

    def test_env_spec_serialises(self):
        d = catalog.get_environment("simple_tag_v3").to_dict(available=True)
        assert d["multi_team"] is True
        assert d["available"] is True
        assert isinstance(d["env_config"], list)
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Co-evolution options
# ---------------------------------------------------------------------------


class TestCoevolutionCatalog:
    def test_modes_match_trainer_constants(self):
        from deap_pettingzoo.baselines.coevolution.trainer import (
            VALID_MODES,
            VALID_SCHEDULES,
        )

        assert tuple(cat_coevolution.COEV_MODES) == tuple(VALID_MODES)
        assert tuple(cat_coevolution.COEV_SCHEDULES) == tuple(VALID_SCHEDULES)

    def test_aggregations_match_evaluator_constant(self):
        from deap_pettingzoo.baselines.coevolution.evaluator import (
            VALID_AGGREGATIONS,
        )

        assert tuple(cat_coevolution.COEV_AGGREGATIONS) == tuple(VALID_AGGREGATIONS)

    def test_options_payload(self):
        opts = catalog.get_coevolution_options()
        assert {m["name"] for m in opts["modes"]} == set(cat_coevolution.COEV_MODES)
        optimizers = {entry["optimizer"] for entry in opts["side_optimizers"]}
        assert {"ea", "cma", "pso", "rl", "frozen"} <= optimizers


# ---------------------------------------------------------------------------
# Compatibility + full snapshot
# ---------------------------------------------------------------------------


class TestCompatibilityAndSnapshot:
    def test_compatible_algorithms_for_parallel_env(self):
        compatible = catalog.compatible_algorithms("simple_spread_v3")
        # Population trainers are always compatible with a parallel MPE env.
        assert {"ea", "cma-es", "pso"} <= set(compatible)

    def test_compatibility_matrix_shape(self):
        matrix = catalog.compatibility_matrix("simple_spread_v3", ["ea", "maddpg"])
        assert matrix["env_name"] == "simple_spread_v3"
        assert "ea" in matrix["results"]
        assert "available" in matrix["results"]["ea"]

    def test_to_json_is_serialisable_and_versioned(self):
        snapshot = catalog.to_json()
        assert snapshot["catalog_version"]
        assert {a["name"] for a in snapshot["algorithms"]} == _EXPECTED_ALGORITHMS
        for algo in snapshot["algorithms"]:
            assert "available" in algo
        json.dumps(snapshot)  # must not raise
