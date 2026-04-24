"""Tests for :mod:`deap_pettingzoo.utils.parallel`."""

from __future__ import annotations

import os

import pytest

from deap_pettingzoo.utils.parallel import (
    SUPPORTED_BACKENDS,
    _SerialExecutor,
    make_executor,
    parallel_map,
    resolve_num_workers,
)


def _square(x: int) -> int:
    return x * x


class TestResolveNumWorkers:
    def test_none_yields_one(self):
        assert resolve_num_workers(None) == 1

    def test_zero_yields_one(self):
        assert resolve_num_workers(0) == 1

    def test_one_yields_one(self):
        assert resolve_num_workers(1) == 1

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            resolve_num_workers(-1)

    def test_caps_to_cpu_count(self):
        cpu = os.cpu_count() or 1
        with pytest.warns(UserWarning, match="capping"):
            assert resolve_num_workers(cpu * 1000) == cpu

    def test_passes_through_within_cpu(self):
        cpu = os.cpu_count() or 1
        if cpu >= 2:
            assert resolve_num_workers(2) == 2


class TestMakeExecutor:
    def test_serial_for_single_worker(self):
        with make_executor(1) as ex:
            assert isinstance(ex, _SerialExecutor)
            results = list(ex.map(_square, [1, 2, 3]))
            assert results == [1, 4, 9]

    def test_serial_for_none(self):
        with make_executor(None) as ex:
            assert isinstance(ex, _SerialExecutor)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="parallel_backend"):
            make_executor(2, backend="bogus")

    def test_supported_backends_list(self):
        assert "process" in SUPPORTED_BACKENDS
        assert "scoop" in SUPPORTED_BACKENDS
        assert "dask" in SUPPORTED_BACKENDS

    @pytest.mark.skipif(
        (os.cpu_count() or 1) < 2,
        reason="needs at least 2 CPUs for a real process pool",
    )
    def test_process_backend_runs_and_preserves_order(self):
        # Real ProcessPoolExecutor — ordered map.
        with make_executor(2, backend="process") as ex:
            results = list(ex.map(_square, [3, 1, 4, 1, 5, 9, 2, 6]))
        assert results == [9, 1, 16, 1, 25, 81, 4, 36]


class TestParallelMap:
    def test_serial_path(self):
        with make_executor(1) as ex:
            results = parallel_map(_square, [1, 2, 3, 4], executor=ex)
        assert results == [1, 4, 9, 16]

    @pytest.mark.skipif(
        (os.cpu_count() or 1) < 2,
        reason="needs at least 2 CPUs",
    )
    def test_process_path(self):
        with make_executor(2, backend="process") as ex:
            results = parallel_map(_square, list(range(8)), executor=ex)
        assert results == [i * i for i in range(8)]


# --------------------------------------------------------------------------
# Phase 7: SCOOP + Dask backend dispatch tests.
#
# SCOOP is not a listed dev dep; installing it requires a launcher
# (``python -m scoop``). We cover the dispatch path by stubbing
# ``scoop.futures`` in ``sys.modules``. Dask is ``importorskip`` ed;
# when present we stand up a threaded Client so the test is
# self-contained and ResourceWarning-clean.
# --------------------------------------------------------------------------


class TestScoopBackend:
    def test_scoop_import_error_when_missing(self, monkeypatch):
        # Ensure scoop is hidden from importers.
        import sys

        monkeypatch.setitem(sys.modules, "scoop", None)
        with pytest.raises(ImportError, match="scoop"):
            make_executor(2, backend="scoop")

    def test_scoop_dispatch_via_stub(self, monkeypatch):
        """Round-trip map through a stubbed ``scoop.futures``."""
        import sys
        import types

        scoop_pkg = types.ModuleType("scoop")
        futures_mod = types.ModuleType("scoop.futures")

        def _stub_map(fn, *iterables):
            # SCOOP returns a generator in submission order.
            return (fn(*args) for args in zip(*iterables))

        futures_mod.map = _stub_map
        scoop_pkg.futures = futures_mod
        monkeypatch.setitem(sys.modules, "scoop", scoop_pkg)
        monkeypatch.setitem(sys.modules, "scoop.futures", futures_mod)

        with make_executor(2, backend="scoop") as ex:
            results = list(ex.map(_square, [1, 2, 3, 4]))

        assert results == [1, 4, 9, 16]


class TestDaskBackend:
    def test_dask_dispatch_preserves_order(self):
        pytest.importorskip("dask.distributed")
        from dask.distributed import Client

        # Threaded client — no child processes, no dashboard.
        client = Client(
            processes=False,
            n_workers=2,
            threads_per_worker=1,
            dashboard_address=None,
            set_as_default=False,
        )
        try:
            from deap_pettingzoo.utils.parallel import _DaskExecutor

            with _DaskExecutor(2, client=client) as ex:
                results = list(ex.map(_square, [3, 1, 4, 1, 5, 9, 2, 6]))
            assert results == [9, 1, 16, 1, 25, 81, 4, 36]
        finally:
            client.close()

    def test_dask_import_error_when_missing(self, monkeypatch):
        import sys

        # Hide dask.distributed regardless of whether it's installed.
        monkeypatch.setitem(sys.modules, "dask.distributed", None)
        with pytest.raises(ImportError, match="dask"):
            make_executor(2, backend="dask")


# --------------------------------------------------------------------------
# Phase 7: pickle regression guard.
#
# SCOOP and Dask use strict ``pickle`` — no ``cloudpickle`` fallback is
# guaranteed. Every payload the trainers ship across the process
# boundary must round-trip through ``pickle.dumps`` / ``pickle.loads``.
# This test is a structural regression check so that accidental
# reintroduction of closures or unpicklable state is caught at
# unit-test time rather than at deploy time.
# --------------------------------------------------------------------------


class TestPayloadPickleability:
    def test_genome_payload_roundtrip(self):
        import pickle

        import numpy as np

        from deap_pettingzoo.integration.deap_integration import (
            _GenomePayload,
        )

        p = _GenomePayload(
            weights=np.arange(6, dtype=np.float64),
            layer_sizes=(2, 3, 2),
            activation="tanh",
            weight_range=(-1.0, 1.0),
            seed=42,
        )
        p2 = pickle.loads(pickle.dumps(p))
        assert p2.layer_sizes == (2, 3, 2)
        assert p2.activation == "tanh"
        assert p2.seed == 42
        assert np.array_equal(p2.weights, p.weights)

    def test_eval_worker_spec_roundtrip(self):
        import pickle

        from deap_pettingzoo.integration.deap_integration import (
            _EvalWorkerSpec,
        )

        spec = _EvalWorkerSpec(
            env_name="simple_spread_v3",
            env_config={"N": 3, "max_cycles": 25},
            max_steps=25,
            aggregation_method="mean",
            team_mapping=None,
        )
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.env_name == "simple_spread_v3"
        assert spec2.env_config == {"N": 3, "max_cycles": 25}
        assert spec2.aggregation_method == "mean"

    def test_rollout_spec_roundtrip_with_descriptor(self):
        """Regression: closures in descriptors broke pickling pre-Phase 3."""
        pytest.importorskip("flax")
        import pickle

        from deap_pettingzoo.contrib.qdax.descriptors.pursuit_evasion import (
            _FormationSpreadAndMeanDistance,
            formation_spread_and_mean_distance,
        )
        from deap_pettingzoo.contrib.qdax.scoring_numpy import _RolloutSpec

        descriptor = formation_spread_and_mean_distance(
            num_pursuers=2, num_evaders=1
        )
        assert isinstance(descriptor, _FormationSpreadAndMeanDistance)

        spec = _RolloutSpec(
            env_name="simple_tag_v3",
            env_config=None,
            layer_sizes=(14, 8, 5),
            activation="tanh",
            episode_length=25,
            descriptor_fn=descriptor,
            action_type="discrete",
            extract_state_fn=None,
        )
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.env_name == "simple_tag_v3"
        assert spec2.layer_sizes == (14, 8, 5)
        assert isinstance(spec2.descriptor_fn, _FormationSpreadAndMeanDistance)
