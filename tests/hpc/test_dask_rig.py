"""End-to-end validation of the multi-container Dask HPC rig.

Opt-in (see ``conftest.py``): ``DEAP_HPC_TESTS=1 pytest tests/hpc/test_dask_rig.py``.
"""

from __future__ import annotations

import csv
import io
import math
import uuid

import pytest

pytestmark = pytest.mark.hpc


def _unique(name: str) -> str:
    return f"/shared/{name}-{uuid.uuid4().hex[:8]}"


class TestDaskRigBasics:
    def test_run_completes_with_multiple_workers_and_finite_fitness(
        self, run_driver, read_driver_file
    ):
        summary = run_driver(
            [
                "--backend", "dask",
                "--num-workers", "2",
                "--population", "6",
                "--generations", "2",
                "--seed", "1",
                "--output-dir", _unique("basic"),
            ]
        )
        assert math.isfinite(summary["final_fitness"])
        assert summary["parallel_backend"] == "dask"

        dask_workers = summary["dask_workers"]
        assert dask_workers, "expected worker hosts to be recorded in metadata"
        assert len(set(dask_workers)) >= 2, (
            f"expected >=2 distinct worker hosts, got {dask_workers}"
        )

        csv_text = read_driver_file(f"{summary['run_dir']}/metrics.csv")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert rows, "metrics.csv has no rows"
        for row in rows:
            for key in ("mean", "min", "max"):
                assert math.isfinite(float(row[key])), f"non-finite {key} in row {row}"

    def test_custom_env_propagates_to_remote_workers(self, run_driver):
        summary = run_driver(
            [
                "--backend", "dask",
                "--num-workers", "2",
                "--population", "6",
                "--generations", "2",
                "--seed", "2",
                "--output-dir", _unique("customenv"),
                "--register-module", "tests.hpc._custom_env_driver",
            ]
        )
        assert math.isfinite(summary["final_fitness"])


class TestDaskRigReproducibility:
    def test_serial_vs_distributed_identical(self, run_driver):
        """Same seed, same image -> bit-identical fitness histories.

        Both invocations run inside the *same* driver container (the rig's
        ``driver`` service is long-lived); only the executor differs.
        """
        seed = "3"
        common = [
            "--population", "6",
            "--generations", "2",
            "--seed", seed,
        ]
        serial = run_driver(
            ["--backend", "process", "--num-workers", "1", *common,
             "--output-dir", _unique("serial")]
        )
        distributed = run_driver(
            ["--backend", "dask", "--num-workers", "2", *common,
             "--output-dir", _unique("distributed")]
        )
        assert serial["fitness_history"] == distributed["fitness_history"]
        assert serial["final_fitness"] == distributed["final_fitness"]


class TestDaskRigDebuggability:
    def test_broken_worker_env_surfaces_real_traceback(self, run_driver):
        """A worker-side exception must surface the real error, not a hang
        or an opaque ``KilledWorker``."""
        result = run_driver(
            [
                "--backend", "dask",
                "--num-workers", "2",
                "--population", "4",
                "--generations", "1",
                "--seed", "4",
                "--output-dir", _unique("broken"),
                "--register-module", "tests.hpc._broken_env_driver",
            ],
            check=False,
        )
        assert result.returncode != 0, "expected the broken env to fail the run"
        combined = result.stdout + result.stderr
        assert "HPC_RIG_INTENTIONAL_FAILURE" in combined, (
            "expected the real remote traceback to surface on the driver:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "KilledWorker" not in combined, (
            "the deterministic worker failure should surface as the real "
            "error after retries, not an opaque KilledWorker:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
