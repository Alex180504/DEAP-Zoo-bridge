"""Custom-env propagation across the parallel backends.

These tests cover the registry-snapshot-on-worker-spec mechanism that lets
SCOOP / Dask workers (which have no per-worker ``initializer`` hook) and the
initializer-less process path re-register **runtime-registered** custom envs
before building them.

The synthetic env in :mod:`tests.multidiscrete_env` is ideal: its factory is a
module-level (picklable) callable, but the env is registered only at runtime
(via :func:`register_md_env`), so a freshly-spawned worker does not know it
unless the snapshot is replayed.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from deap import creator

from deap_pettingzoo.environments.registry import (
    _REPLAYED_SNAPSHOT_KEYS,
    get_registry_snapshot,
    is_registered,
    replay_registry_snapshot_once,
    unregister_env,
)
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    _EvalWorkerSpec,
    evaluate_population_parallel,
    setup_deap_creators,
)
from deap_pettingzoo.utils.parallel import make_executor

from tests.multidiscrete_env import (
    MD_ENV_NAME,
    MD_NVEC,
    MD_OBS_DIM,
    register_md_env,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def md_env():
    """Register the synthetic custom env for the test, cleaning up after."""
    register_md_env()
    try:
        yield MD_ENV_NAME
    finally:
        if is_registered(MD_ENV_NAME):
            unregister_env(MD_ENV_NAME)


def _make_inds(n: int) -> list:
    setup_deap_creators()
    out_size = int(sum(MD_NVEC))
    inds = []
    for k in range(n):
        genome = FixedTopologyGenome(
            layer_sizes=[MD_OBS_DIM, 8, out_size],
            activation="tanh",
            weight_range=(-1.0, 1.0),
            random_seed=k,
        )
        ind = creator.Individual()
        ind.genome = genome
        inds.append(ind)
    return inds


def _md_spec(snapshot) -> _EvalWorkerSpec:
    return _EvalWorkerSpec(
        env_name=MD_ENV_NAME,
        env_config=None,
        max_steps=10,
        aggregation_method="sum",
        team_mapping=None,
        registry_snapshot=snapshot,
    )


class TestReplaySnapshotOnce:
    """Unit tests for the per-process replay guard."""

    def test_none_snapshot_is_noop(self):
        # Must not raise on the serial / unpopulated path.
        replay_registry_snapshot_once(None)
        replay_registry_snapshot_once([])

    def test_replays_into_fresh_registry(self, md_env):
        snapshot = get_registry_snapshot()
        assert any(name == MD_ENV_NAME for name, _ in snapshot)

        unregister_env(MD_ENV_NAME)
        assert not is_registered(MD_ENV_NAME)

        _REPLAYED_SNAPSHOT_KEYS.clear()
        replay_registry_snapshot_once(snapshot)
        assert is_registered(MD_ENV_NAME)

    def test_guard_skips_second_call(self, md_env):
        snapshot = get_registry_snapshot()
        _REPLAYED_SNAPSHOT_KEYS.clear()

        replay_registry_snapshot_once(snapshot)
        assert is_registered(MD_ENV_NAME)

        # Drop the env, then replay the SAME snapshot: the guard should skip,
        # so the env stays unregistered.
        unregister_env(MD_ENV_NAME)
        replay_registry_snapshot_once(snapshot)
        assert not is_registered(MD_ENV_NAME)

        # Clearing the guard makes a fresh replay register it again.
        _REPLAYED_SNAPSHOT_KEYS.clear()
        replay_registry_snapshot_once(snapshot)
        assert is_registered(MD_ENV_NAME)


class TestSpecPickle:
    """A populated ``registry_snapshot`` must survive pickling to workers."""

    def test_eval_worker_spec_roundtrip(self, md_env):
        spec = _md_spec(get_registry_snapshot())
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.env_name == MD_ENV_NAME
        names = {name for name, _ in (spec2.registry_snapshot or [])}
        assert MD_ENV_NAME in names

    def test_neat_worker_spec_roundtrip(self, md_env):
        from deap_pettingzoo.contrib.neat.parallel import _NeatWorkerSpec

        spec = _NeatWorkerSpec(
            env_name=MD_ENV_NAME,
            env_config={},
            input_size=MD_OBS_DIM,
            output_size=int(sum(MD_NVEC)),
            action_space_type="multidiscrete",
            action_space_args={"nvec": list(MD_NVEC)},
            registry_snapshot=get_registry_snapshot(),
        )
        spec2 = pickle.loads(pickle.dumps(spec))
        names = {name for name, _ in (spec2.registry_snapshot or [])}
        assert MD_ENV_NAME in names

    def test_coev_episode_payload_roundtrip(self, md_env):
        from deap_pettingzoo.baselines.coevolution.evaluator import _EpisodePayload

        payload = _EpisodePayload(
            env_name=MD_ENV_NAME,
            env_config=None,
            learner_genome=None,
            learner_role_prefixes=("agent_0",),
            learner_masked=False,
            opponent_snapshots={},
            side_role_prefixes={},
            episode_seed=0,
            max_steps=10,
            registry_snapshot=get_registry_snapshot(),
        )
        payload2 = pickle.loads(pickle.dumps(payload))
        names = {name for name, _ in (payload2.registry_snapshot or [])}
        assert MD_ENV_NAME in names


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs at least 2 CPUs for a process pool"
)
class TestProcessSpecPropagation:
    """The snapshot on the spec carries the env even with no initializer."""

    def test_missing_snapshot_fails_in_worker(self, md_env):
        # No initializer (mirrors SCOOP/Dask) AND no snapshot on the spec:
        # fresh workers cannot resolve the runtime-registered env.
        inds = _make_inds(2)
        spec = _md_spec(None)
        with make_executor(2, backend="process") as executor:
            with pytest.raises(Exception):
                evaluate_population_parallel(inds, executor=executor, spec=spec)

    def test_snapshot_propagates(self, md_env):
        inds = _make_inds(2)
        spec = _md_spec(get_registry_snapshot())
        with make_executor(2, backend="process") as executor:
            results = evaluate_population_parallel(
                inds, executor=executor, spec=spec
            )
        assert len(results) == 2
        assert all(np.isfinite(r[0]) for r in results)


class TestDaskSpecPropagation:
    """Genuine cross-process Dask propagation (no initializer hook)."""

    def test_snapshot_propagates_via_dask(self, md_env):
        pytest.importorskip("dask.distributed")
        from dask.distributed import Client

        from deap_pettingzoo.utils.parallel import _DaskExecutor

        inds = _make_inds(2)
        spec = _md_spec(get_registry_snapshot())
        client = Client(
            processes=True,
            n_workers=2,
            threads_per_worker=1,
            dashboard_address=None,
            set_as_default=False,
        )
        try:
            with _DaskExecutor(2, client=client) as executor:
                results = evaluate_population_parallel(
                    inds, executor=executor, spec=spec
                )
        finally:
            client.close()
        assert len(results) == 2
        assert all(np.isfinite(r[0]) for r in results)


@pytest.mark.scoop
def test_scoop_custom_env_subprocess():
    """A real ``python -m scoop`` run propagates a runtime-registered env."""
    pytest.importorskip("scoop")
    script = os.path.join(os.path.dirname(__file__), "_scoop_custom_env_script.py")
    proc = subprocess.run(
        [sys.executable, "-m", "scoop", "-n", "2", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined
    assert "SCOOP_CUSTOM_ENV_OK" in proc.stdout, combined
