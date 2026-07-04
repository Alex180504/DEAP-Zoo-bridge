"""Tests for :mod:`deap_pettingzoo.utils.parallel`."""

from __future__ import annotations

import os
import warnings

import pytest

from deap_pettingzoo.utils.parallel import (
    DASK_CONNECT_TIMEOUT_ENV_VAR,
    DASK_SCHEDULER_ENV_VAR,
    SUPPORTED_BACKENDS,
    _dask_address_kwargs,
    _DaskExecutor,
    _SerialExecutor,
    make_executor,
    parallel_map,
    resolve_num_workers,
    resolve_scheduler_address,
)


def _square(x: int) -> int:
    return x * x


# Per-key first-call failure marker for the retry test. In-process dask
# workers (processes=False) share this module global with the driver.
_FLAKY_SEEN: dict = {}


def _flaky_square(x: int) -> int:
    """Raises on the first call per input, succeeds when retried."""
    if x not in _FLAKY_SEEN:
        _FLAKY_SEEN[x] = True
        raise RuntimeError(f"transient-{x}")
    return x * x


def _always_boom(x: int) -> int:
    raise RuntimeError("boom-deterministic")


@pytest.fixture(scope="module")
def local_cluster():
    """In-process 2-worker Dask cluster acting as the 'remote' scheduler."""
    pytest.importorskip("dask.distributed")
    from dask.distributed import LocalCluster

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    yield cluster
    cluster.close()


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

    def test_no_cap_when_cap_to_cpu_false(self):
        cpu = os.cpu_count() or 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_num_workers(cpu * 1000, cap_to_cpu=False) == cpu * 1000


class TestResolveSchedulerAddress:
    def test_explicit_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(DASK_SCHEDULER_ENV_VAR, "tcp://env:8786")
        assert resolve_scheduler_address("tcp://arg:8786") == "tcp://arg:8786"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv(DASK_SCHEDULER_ENV_VAR, "tcp://env:8786")
        assert resolve_scheduler_address(None) == "tcp://env:8786"

    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        assert resolve_scheduler_address(None) is None

    def test_blank_values_count_as_unset(self, monkeypatch):
        monkeypatch.setenv(DASK_SCHEDULER_ENV_VAR, "   ")
        assert resolve_scheduler_address("") is None


class TestDaskAddressHeuristic:
    def test_url_connects_directly(self):
        assert _dask_address_kwargs("tcp://10.0.0.1:8786") == {
            "address": "tcp://10.0.0.1:8786"
        }
        assert _dask_address_kwargs("tls://head-node:8786") == {
            "address": "tls://head-node:8786"
        }

    def test_plain_value_is_a_scheduler_file(self):
        assert _dask_address_kwargs("/shared/scheduler.json") == {
            "scheduler_file": "/shared/scheduler.json"
        }
        assert _dask_address_kwargs(r"C:\shared\scheduler.json") == {
            "scheduler_file": r"C:\shared\scheduler.json"
        }


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
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        with pytest.raises(ImportError, match="dask"):
            make_executor(2, backend="dask")


class TestMakeExecutorRemoteGates:
    def test_scheduler_address_requires_dask_backend(self):
        with pytest.raises(ValueError, match="scheduler_address"):
            make_executor(2, backend="process", scheduler_address="tcp://x:8786")

    def test_env_var_ignored_for_process_backend(self, monkeypatch):
        monkeypatch.setenv(DASK_SCHEDULER_ENV_VAR, "tcp://nowhere:8786")
        with make_executor(1, backend="process") as ex:
            assert isinstance(ex, _SerialExecutor)

    def test_injected_client_bypasses_serial_gate(self):
        pytest.importorskip("dask.distributed")
        from dask.distributed import Client

        client = Client(
            processes=False,
            n_workers=1,
            threads_per_worker=1,
            dashboard_address=None,
            set_as_default=False,
        )
        try:
            # Explicit remote intent wins over the num_workers <= 1 fast-path.
            with make_executor(1, backend="dask", dask_client=client) as ex:
                assert isinstance(ex, _DaskExecutor)
                assert list(ex.map(_square, [3, 1, 4])) == [9, 1, 16]
        finally:
            client.close()


class TestRemoteDaskExecutor:
    def test_env_var_reaches_cluster_and_bypasses_serial_gate(
        self, monkeypatch, local_cluster
    ):
        monkeypatch.setenv(DASK_SCHEDULER_ENV_VAR, local_cluster.scheduler_address)
        # num_workers=1 would previously short-circuit to _SerialExecutor.
        with make_executor(1, backend="dask") as ex:
            assert isinstance(ex, _DaskExecutor)
            assert ex._remote is True
            results = list(ex.map(_square, [3, 1, 4, 1, 5, 9]))
        assert results == [9, 1, 16, 1, 25, 81]

    def test_explicit_address_and_worker_info(self, monkeypatch, local_cluster):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        with make_executor(
            2, backend="dask", scheduler_address=local_cluster.scheduler_address
        ) as ex:
            info = ex.worker_info()
        assert len(info) == 2
        for worker in info:
            assert {"address", "host", "nthreads"} <= set(worker)
            assert worker["nthreads"] == 1

    def test_remote_skips_cpu_cap(self, monkeypatch, local_cluster):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        cpu = os.cpu_count() or 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with make_executor(
                cpu * 1000,
                backend="dask",
                scheduler_address=local_cluster.scheduler_address,
            ) as ex:
                assert isinstance(ex, _DaskExecutor)

    def test_retries_resubmit_transient_failures(self, monkeypatch, local_cluster):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        _FLAKY_SEEN.clear()
        with make_executor(
            2, backend="dask", scheduler_address=local_cluster.scheduler_address
        ) as ex:
            results = list(ex.map(_flaky_square, [1, 2, 3]))
        assert results == [1, 4, 9]

    def test_deterministic_failure_propagates(self, monkeypatch, local_cluster):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        with make_executor(
            2, backend="dask", scheduler_address=local_cluster.scheduler_address
        ) as ex:
            with pytest.raises(RuntimeError, match="boom-deterministic"):
                list(ex.map(_always_boom, [1]))

    def test_main_module_fn_rejected(self, monkeypatch, local_cluster):
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)

        def fn(x):  # pragma: no cover - never dispatched
            return x

        fn.__module__ = "__main__"
        with make_executor(
            2, backend="dask", scheduler_address=local_cluster.scheduler_address
        ) as ex:
            with pytest.raises(ValueError, match="__main__"):
                ex.map(fn, [1])

    def test_warns_on_multithreaded_workers(self, monkeypatch):
        pytest.importorskip("dask.distributed")
        from dask.distributed import LocalCluster

        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=2,
            processes=False,
            dashboard_address=None,
        )
        try:
            with pytest.warns(RuntimeWarning, match="nthreads"):
                ex = make_executor(
                    2,
                    backend="dask",
                    scheduler_address=cluster.scheduler_address,
                )
            with ex:
                pass
        finally:
            cluster.close()

    def test_unreachable_scheduler_raises_clear_error(self, monkeypatch):
        pytest.importorskip("dask.distributed")
        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        monkeypatch.setenv(DASK_CONNECT_TIMEOUT_ENV_VAR, "2")
        with pytest.raises(RuntimeError, match="scheduler"):
            # Port 1 is never a Dask scheduler; fails fast with the 2s timeout.
            make_executor(2, backend="dask", scheduler_address="tcp://127.0.0.1:1")


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
            output_activation="linear",
            weight_range=(-1.0, 1.0),
            seed=42,
        )
        p2 = pickle.loads(pickle.dumps(p))
        assert p2.layer_sizes == (2, 3, 2)
        assert p2.activation == "tanh"
        assert p2.output_activation == "linear"
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
        pytest.importorskip("flax")
        import pickle

        from examples.qdax.descriptors import (
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
            eval_episodes=1,
            descriptor_fn=descriptor,
            action_type="discrete",
            state_extractor=None,
        )
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.env_name == "simple_tag_v3"
        assert spec2.layer_sizes == (14, 8, 5)
        assert isinstance(spec2.descriptor_fn, _FormationSpreadAndMeanDistance)
