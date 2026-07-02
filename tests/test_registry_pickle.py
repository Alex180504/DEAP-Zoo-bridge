"""Regression tests for worker-snapshot pickling of env factories.

Covers:

* :func:`get_registry_snapshot` warns (naming the env) and drops
  a custom env whose factory is a lambda / closure / locally-defined callable,
  instead of silently omitting it (the old cryptic "unregistered env" under
  ``num_workers>1``). Module-object factories (the built-in MPE2/aquarium
  pattern, re-registered in every worker at import) stay silent.
* The JaxMARL / OpenSpiel contrib factories are module-level
  picklable classes (no closures), so an :class:`EnvSpec` built from one
  survives a ``pickle.dumps`` round trip and reaches the parallel workers.
"""

from __future__ import annotations

import pickle
import types
import warnings

import pytest

from deap_pettingzoo.environments import registry
from deap_pettingzoo.environments.registry import (
    get_registry_snapshot,
    is_registered,
    register_env,
    unregister_env,
)


class _PicklableFactory:
    """Module-level, picklable callable factory (captures only a tag)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __call__(self, render_mode=None, **cfg):  # pragma: no cover - never built
        raise NotImplementedError


def _module_level_parallel_env(render_mode=None, **cfg):  # pragma: no cover
    raise NotImplementedError


@pytest.fixture
def temp_env():
    """Register/cleanup helper yielding a registrar that tracks its names."""
    added = []

    def _register(name, factory, **kwargs):
        register_env(name, factory, overwrite=True, **kwargs)
        added.append(name)
        return name

    yield _register

    for name in added:
        if is_registered(name):
            unregister_env(name)


class TestSnapshotWarnsOnUnpicklableFactory:
    def test_closure_factory_warns_and_is_dropped(self, temp_env):
        name = "_pickle_closure_env_v0"

        def _make():
            def _local_factory(render_mode=None, **cfg):  # pragma: no cover
                return object()

            return _local_factory

        temp_env(name, _make())
        with pytest.warns(RuntimeWarning, match=name):
            snapshot = get_registry_snapshot()
        assert name not in {n for n, _ in snapshot}

    def test_lambda_factory_warns_and_is_dropped(self, temp_env):
        name = "_pickle_lambda_env_v0"
        temp_env(name, lambda render_mode=None, **cfg: object())
        with pytest.warns(RuntimeWarning, match=name):
            snapshot = get_registry_snapshot()
        assert name not in {n for n, _ in snapshot}

    def test_unpicklable_wrapper_is_attributed_to_metadata(self, temp_env):
        # Picklable factory, but a lambda in the wrapper chain. The warning must
        # blame the metadata, not the (fine) factory.
        name = "_pickle_wrapper_env_v0"
        temp_env(name, _PicklableFactory("x"), wrappers=(lambda e: e,))
        with pytest.warns(RuntimeWarning, match="metadata"):
            snapshot = get_registry_snapshot()
        assert name not in {n for n, _ in snapshot}

    def test_module_factory_is_silently_omitted(self, temp_env):
        """The built-in pattern (module object) drops without a warning."""
        name = "_pickle_module_env_v0"
        fake_module = types.ModuleType("_fake_mpe_like_module")
        fake_module.parallel_env = _module_level_parallel_env
        temp_env(name, fake_module)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            snapshot = get_registry_snapshot()

        assert name not in {n for n, _ in snapshot}
        assert not [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning) and name in str(w.message)
        ]


class TestSnapshotWarnsOnMainModuleFactory:
    """Factories defined in the driver script (``__main__``) pickle fine by
    reference but never resolve on remote Dask workers — the snapshot must
    warn (and keep the entry) when the target executor is remote."""

    def _main_module_factory(self):
        # An instance attribute shadows the class's real __module__, mimicking
        # a factory defined in the driver script while staying picklable.
        factory = _PicklableFactory("x")
        factory.__module__ = "__main__"
        return factory

    def test_warns_and_keeps_entry_when_target_remote(self, temp_env):
        name = "_pickle_main_env_v0"
        temp_env(name, self._main_module_factory())
        with pytest.warns(RuntimeWarning, match="__main__"):
            snapshot = get_registry_snapshot(target_remote=True)
        # Warned, not dropped: local-cluster runs still resolve it.
        assert name in {n for n, _ in snapshot}

    def test_silent_when_target_is_local(self, temp_env):
        name = "_pickle_main_env_v1"
        temp_env(name, self._main_module_factory())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            snapshot = get_registry_snapshot()
        assert name in {n for n, _ in snapshot}
        assert not [w for w in caught if name in str(w.message)]


class TestSnapshotKeepsPicklableFactory:
    def test_module_level_callable_survives(self, temp_env):
        name = "_pickle_ok_env_v0"
        temp_env(name, _PicklableFactory("x"), default_episode_len=42)
        snapshot = get_registry_snapshot()
        recovered = dict(snapshot)
        assert name in recovered
        # The whole spec round-trips (this is what a worker replays).
        spec = pickle.loads(pickle.dumps(recovered[name]))
        assert spec.default_episode_len == 42
        assert isinstance(spec.factory, _PicklableFactory)


class TestContribFactoriesPickle:
    def test_jaxmarl_factory_round_trips(self):
        # Constructed directly so the test needs neither jaxmarl nor jax.
        from deap_pettingzoo.contrib.jaxmarl import JaxmarlEnvFactory

        factory = JaxmarlEnvFactory("MPE_simple_tag_v3", {"num_agents": 3})
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.env_id == "MPE_simple_tag_v3"
        assert restored.jaxmarl_kwargs == {"num_agents": 3}

    def test_openspiel_factory_round_trips(self):
        from deap_pettingzoo.contrib.openspiel import OpenSpielGameFactory

        factory = OpenSpielGameFactory("connect_four")
        restored = pickle.loads(pickle.dumps(factory))
        assert restored.game_name == "connect_four"

    def test_registered_jaxmarl_spec_survives_snapshot(self, temp_env):
        """A JaxMARL-style registration reaches the worker snapshot."""
        from deap_pettingzoo.contrib.jaxmarl import JaxmarlEnvFactory

        name = "_pickle_jaxmarl_like_v0"
        temp_env(name, JaxmarlEnvFactory("MPE_simple_tag_v3", {}), api="jax_parallel")
        snapshot = dict(get_registry_snapshot())
        assert name in snapshot
        pickle.loads(pickle.dumps(snapshot[name]))  # must not raise
