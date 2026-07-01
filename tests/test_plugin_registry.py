"""Tests for the algorithm / co-evolution-side plugin seam.

The catalog algorithm table and the co-evolution side factory are the two
authoritative *open* registries: a single :func:`register_algorithm` (or
:func:`register_side_state`) call makes a third-party algorithm resolvable
through :func:`catalog.get_algorithm`, :func:`list_algorithm_names`,
:func:`catalog.to_json`, the self-healing :func:`comparison.get_trainer`, and,
when a ``side_state`` is supplied — :func:`make_side_state` + :class:`Side`
validation. Entry-point plugins are discovered once at catalog import and a
broken plugin degrades to a :class:`RuntimeWarning` without breaking the
catalog.

Every test runs under the :func:`clean_registry` fixture, which snapshots and
restores all four pieces of mutable registry state so a stub registration never
leaks into the exact-match assertions in ``test_catalog.py``.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from deap_pettingzoo import catalog
from deap_pettingzoo.baselines import comparison
from deap_pettingzoo.baselines.coevolution import (
    Side,
    make_side_state,
    register_side_state,
)
from deap_pettingzoo.baselines.coevolution import sides as coev_sides
from deap_pettingzoo.baselines.coevolution.sides import SideState
from deap_pettingzoo.catalog import algorithms as cat_algorithms

_STUB_CAPS = {
    "discrete_action": True,
    "continuous_action": True,
    "multidiscrete_action": False,
    "needs_global_state": False,
    "on_policy": False,
    "supports_turn_based": True,
}

# Module-level so they are importable by dotted path (worker-safe re-import
# and the trainer-registry self-heal both rely on this).
_EA_TRAINER_BINDING = "deap_pettingzoo.baselines.ea_trainer:EATrainer"


class StubTrainer:
    """Minimal constructible trainer for the plugin-seam tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def describe_hyperparameters(cls):
        return ()

    def get_capabilities(self):
        return dict(_STUB_CAPS)


class StubSideState(SideState):
    """Minimal :class:`SideState` subclass — only needs to be instantiable."""


class Outer:
    """Namespace holding a *nested* trainer (``__qualname__ == 'Outer.Inner'``)."""

    class Inner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def describe_hyperparameters(cls):
            return ()

        def get_capabilities(self):
            return dict(_STUB_CAPS)


@pytest.fixture
def clean_registry():
    """Snapshot + restore every mutable registry the plugin seam touches."""
    algo_reg = dict(cat_algorithms._REGISTRY)
    side_reg = dict(coev_sides._SIDE_STATE_FACTORIES)
    trainer_reg = dict(comparison._TRAINER_REGISTRY)
    discovered = cat_algorithms._plugins_discovered
    try:
        yield
    finally:
        cat_algorithms._REGISTRY.clear()
        cat_algorithms._REGISTRY.update(algo_reg)
        cat_algorithms._rebuild_lookups()
        cat_algorithms.clear_cache()
        coev_sides._SIDE_STATE_FACTORIES.clear()
        coev_sides._SIDE_STATE_FACTORIES.update(side_reg)
        comparison._TRAINER_REGISTRY.clear()
        comparison._TRAINER_REGISTRY.update(trainer_reg)
        cat_algorithms._plugins_discovered = discovered


def _register_stub(name="stub_algo", **overrides):
    kwargs = dict(
        label="Stub Algorithm",
        family="evolutionary",
        trainer=StubTrainer,
        capabilities=_STUB_CAPS,
        aliases=("stubby",),
        required_modules=(),
        description="stub",
    )
    kwargs.update(overrides)
    cat_algorithms.register_algorithm(name, **kwargs)


class TestRegisterAlgorithm:
    def test_appears_across_catalog_surfaces(self, clean_registry):
        _register_stub()

        assert "stub_algo" in cat_algorithms.list_algorithm_names()
        spec = catalog.get_algorithm("stub_algo")
        assert spec.name == "stub_algo"
        assert spec.label == "Stub Algorithm"
        assert "stub_algo" in {a["name"] for a in catalog.to_json()["algorithms"]}

    def test_alias_resolves(self, clean_registry):
        _register_stub()
        assert catalog.canonical_algorithm_name("stubby") == "stub_algo"
        assert catalog.get_algorithm("stubby").name == "stub_algo"

    def test_buildable_via_get_trainer_self_heal(self, clean_registry):
        # The trainer registry does not yet know the stub — it must self-heal
        # from the catalog on the miss.
        assert "stub_algo" not in comparison._TRAINER_REGISTRY
        _register_stub()
        trainer = comparison.get_trainer("stub_algo")
        assert isinstance(trainer, StubTrainer)

    def test_dotted_string_trainer_binding(self, clean_registry):
        _register_stub("stub_from_string", trainer=_EA_TRAINER_BINDING, aliases=())
        trainer = comparison.get_trainer("stub_from_string")
        assert type(trainer).__name__ == "EATrainer"

    def test_root_export(self, clean_registry):
        import deap_pettingzoo

        deap_pettingzoo.register_algorithm(
            "stub_root",
            label="Root",
            family="evolutionary",
            trainer=StubTrainer,
            capabilities=_STUB_CAPS,
        )
        assert "stub_root" in cat_algorithms.list_algorithm_names()

    def test_nested_qualname_trainer_resolves(self, clean_registry):
        # A trainer with a nested ``__qualname__`` ("Outer.Inner") must resolve
        # through both get_algorithm (hyperparameter load) and get_trainer
        # (self-heal), not AttributeError on a naive getattr.
        _register_stub("nested_stub", trainer=Outer.Inner, aliases=())
        # get_algorithm builds the spec (walks the qualname to load the trainer).
        assert catalog.get_algorithm("nested_stub").name == "nested_stub"
        trainer = comparison.get_trainer("nested_stub")
        assert isinstance(trainer, Outer.Inner)

    def test_bad_binding_does_not_poison_other_algorithms(self, clean_registry):
        # A malformed trainer binding (class that does not exist) must be skipped
        # by the self-heal, never aborting resolution of the *other* algorithms.
        _register_stub(
            "bad_bind",
            trainer="deap_pettingzoo.baselines.ea_trainer:NoSuchTrainer",
            aliases=(),
        )
        _register_stub("good_bind", trainer=_EA_TRAINER_BINDING, aliases=())
        # The good one still resolves despite the bad sibling.
        assert type(comparison.get_trainer("good_bind")).__name__ == "EATrainer"
        # The bad one is simply absent (skipped), surfacing as a clean ValueError.
        with pytest.raises(ValueError, match="Unknown algorithm"):
            comparison.get_trainer("bad_bind")


class TestCacheInvalidation:
    def test_clear_cache_reflected_after_prior_get(self, clean_registry):
        # Prime the lru_cache with a built-in and the stub.
        catalog.get_algorithm("ea")
        _register_stub()
        assert catalog.get_algorithm("stub_algo").label == "Stub Algorithm"
        # Overwrite must invalidate the cached spec.
        _register_stub(label="Stub v2", overwrite=True)
        assert catalog.get_algorithm("stub_algo").label == "Stub v2"


class TestCollisions:
    def test_collision_without_overwrite_raises(self, clean_registry):
        _register_stub()
        with pytest.raises(ValueError, match="already registered"):
            _register_stub()

    def test_alias_collision_with_builtin_raises(self, clean_registry):
        with pytest.raises(ValueError, match="already resolve"):
            _register_stub("brand_new", aliases=("ea",))

    def test_overwrite_replaces(self, clean_registry):
        _register_stub()
        _register_stub(label="Replaced", overwrite=True)
        assert catalog.get_algorithm("stub_algo").label == "Replaced"


class TestSideStateRegistration:
    def test_register_side_state_class(self, clean_registry):
        register_side_state("stub_side", StubSideState)
        side = Side("s", ("agent",), "stub_side", {})  # validates
        state = make_side_state(side)
        assert isinstance(state, StubSideState)

    def test_register_algorithm_with_side_state(self, clean_registry):
        _register_stub(
            "stub_coev",
            coevolution_optimizer="stub_opt",
            side_state=StubSideState,
        )
        assert "stub_opt" in coev_sides._SIDE_STATE_FACTORIES
        side = Side("s", ("agent",), "stub_opt", {})
        assert isinstance(make_side_state(side), StubSideState)

    def test_side_state_without_optimizer_token_raises(self, clean_registry):
        with pytest.raises(ValueError, match="coevolution_optimizer"):
            _register_stub("bad_coev", side_state=StubSideState)

    def test_side_token_collision_is_atomic(self, clean_registry):
        # A rejected side registration (token already taken by a built-in) must
        # NOT leave the algorithm half-registered in the catalog.
        with pytest.raises(ValueError, match="already registered"):
            _register_stub(
                "atomic_stub",
                coevolution_optimizer="ea",  # already a built-in side token
                side_state=StubSideState,
            )
        assert "atomic_stub" not in cat_algorithms.list_algorithm_names()
        assert "atomic_stub" not in {a["name"] for a in catalog.to_json()["algorithms"]}

    def test_register_side_state_collision(self, clean_registry):
        register_side_state("stub_side", StubSideState)
        with pytest.raises(ValueError, match="already registered"):
            register_side_state("stub_side", StubSideState)
        # overwrite=True replaces without raising.
        register_side_state("stub_side", StubSideState, overwrite=True)


class _FakeEntryPoint:
    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader


def _good_plugin_register():
    cat_algorithms.register_algorithm(
        "ep_algo",
        label="Entry Point Algo",
        family="evolutionary",
        trainer=StubTrainer,
        capabilities=_STUB_CAPS,
    )


def _broken_plugin_loader():  # pragma: no cover - invoked to raise
    raise RuntimeError("deliberately broken plugin")


class TestEntryPointDiscovery:
    def test_discovery_registers_good_and_warns_broken(
        self, clean_registry, monkeypatch
    ):
        def fake_entry_points(*, group):
            if group == cat_algorithms._ALGORITHM_ENTRY_POINT_GROUP:
                return [
                    _FakeEntryPoint("ep_algo", _good_plugin_register),
                    _FakeEntryPoint("ep_broken", _broken_plugin_loader),
                ]
            return []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
        cat_algorithms._plugins_discovered = False

        with pytest.warns(RuntimeWarning, match="ep_broken"):
            cat_algorithms._discover_entry_point_plugins()

        # Good plugin registered; catalog remains fully usable.
        assert "ep_algo" in cat_algorithms.list_algorithm_names()
        assert "ea" in cat_algorithms.list_algorithm_names()

    def test_discovery_is_load_once(self, clean_registry, monkeypatch):
        calls = {"n": 0}

        def fake_entry_points(*, group):
            calls["n"] += 1
            return []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
        cat_algorithms._plugins_discovered = False
        cat_algorithms._discover_entry_point_plugins()
        first = calls["n"]
        cat_algorithms._discover_entry_point_plugins()  # guarded no-op
        assert calls["n"] == first
