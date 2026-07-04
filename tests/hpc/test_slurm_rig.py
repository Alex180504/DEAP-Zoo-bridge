"""End-to-end validation of the Docker Slurm simulation rig.

Submits the *real* sbatch templates from ``examples/hpc/slurm/`` (baked into
the image at ``/app``) with ``sbatch`` on the ``slurmctld`` container: a real
allocation, real ``srun``-launched Dask scheduler/workers spanning both
"nodes" (c1, c2), and the driver on the batch step. That the exact same
files run here is the trust mechanism for real-cluster day — the deltas are
only the commented-out partition/account/module headers and node counts.

Opt-in (see ``conftest.py``):
``DEAP_HPC_TESTS=1 pytest tests/hpc/test_slurm_rig.py``; add
``DEAP_HPC_GPU=1`` on a machine with an NVIDIA GPU to also validate the
AgileRL GPU template against the ``g1`` node.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
import uuid

import pytest

pytestmark = pytest.mark.hpc

#: CPU compute nodes defined in docker/slurm-cluster/slurm.conf.
_CPU_NODES = ("c1", "c2")
#: Ceiling for one submitted job: queue + srun-launched Dask spin-up + a
#: tiny training run. Also guards against a job hanging outright.
_JOB_TIMEOUT_S = 420
_POLL_INTERVAL_S = 5
#: States Slurm considers terminal for a batch job.
_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}

#: Training arguments shared by the distributed sbatch run and the serial
#: reference so their fitness histories are comparable.
_TRAIN_ARGS = ("--population", "6", "--generations", "2", "--seed", "7")


def _submit(slurm_exec, template: str, script_args=(), sbatch_opts=()) -> str:
    """sbatch a template from /app; the job runs on the shared volume."""
    result = slurm_exec(
        "sbatch", "--chdir=/shared", *sbatch_opts,
        f"examples/hpc/slurm/{template}", *script_args,
        # Forwarded into the job by sbatch's default --export=ALL: the
        # scheduler-file handshake and all results must live on the one
        # filesystem every "node" shares.
        env={"DEAP_HPC_SHARED_DIR": "/shared"},
        workdir="/app",
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    assert match, (
        f"could not parse job id from sbatch output:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return match.group(1)


def _wait_for_job(slurm_exec, job_id: str) -> str:
    """Poll scontrol until the job reaches a terminal state (or time out)."""
    deadline = time.monotonic() + _JOB_TIMEOUT_S
    state = "<never observed>"
    while time.monotonic() < deadline:
        result = slurm_exec("scontrol", "show", "job", job_id, check=False)
        match = re.search(r"JobState=(\w+)", result.stdout)
        if match:
            state = match.group(1)
            if state in _TERMINAL_STATES:
                return state
        time.sleep(_POLL_INTERVAL_S)
    return f"<timed out after {_JOB_TIMEOUT_S:.0f}s; last state {state}>"


def _job_output(slurm_exec, job_name: str, job_id: str) -> str:
    """Read the job's stdout file (--output=%x-%j.out, relative to --chdir)."""
    result = slurm_exec("cat", f"/shared/{job_name}-{job_id}.out", check=False)
    if result.returncode != 0:
        return f"<no output file /shared/{job_name}-{job_id}.out>\n{result.stderr}"
    return result.stdout


def _parse_summary(text: str, key: str = "final_fitness") -> dict:
    """Find the driver's one-line JSON summary (identified by ``key``) among
    the interleaved scheduler/worker logs sharing the job's stdout."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and key in data:
            return data
    pytest.fail(f"no JSON summary line with {key!r} found in job output:\n{text}")


def _run_job(slurm_exec, template: str, job_name: str, script_args=(),
             sbatch_opts=(), summary_key: str = "final_fitness") -> dict:
    job_id = _submit(slurm_exec, template, script_args, sbatch_opts)
    state = _wait_for_job(slurm_exec, job_id)
    output = _job_output(slurm_exec, job_name, job_id)
    assert state == "COMPLETED", (
        f"job {job_id} ({template}) finished in state {state}, expected "
        f"COMPLETED.\njob output:\n{output}"
    )
    return _parse_summary(output, key=summary_key)


@pytest.fixture(scope="module")
def dask_job_summary(slurm_exec):
    """Submit the multi-node Dask template once; every test shares the result."""
    return _run_job(
        slurm_exec, "train_dask.sbatch", "deap-hpc-dask", script_args=_TRAIN_ARGS
    )


class TestSlurmRigMultiNodeDask:
    def test_training_completed_via_scheduler_file(self, dask_job_summary):
        assert dask_job_summary["parallel_backend"] == "dask"
        assert math.isfinite(dask_job_summary["final_fitness"])
        # The driver must have found the cluster through DEAP_DASK_SCHEDULER
        # pointing at the scheduler *file* on the shared FS -- the canonical
        # dask-on-Slurm handshake, not a hardcoded tcp:// address.
        assert str(dask_job_summary["scheduler_address"]).endswith("scheduler.json")

    def test_results_land_on_the_shared_filesystem(self, dask_job_summary, slurm_exec):
        run_dir = dask_job_summary["run_dir"]
        assert run_dir and run_dir.startswith("/shared/"), (
            f"expected results under /shared, got {run_dir!r}"
        )
        csv_text = slurm_exec("cat", f"{run_dir}/metrics.csv").stdout
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert rows, "metrics.csv has no rows"
        for row in rows:
            for key in ("mean", "min", "max"):
                assert math.isfinite(float(row[key])), f"non-finite {key} in {row}"

    def test_workers_span_both_nodes(self, dask_job_summary, slurm_exec):
        """The srun-launched workers must actually land on c1 AND c2 (dask
        records worker hosts by IP; resolve the node names to compare)."""
        node_ips = {
            node: slurm_exec("getent", "hosts", node).stdout.split()[0]
            for node in _CPU_NODES
        }
        observed = set(dask_job_summary["dask_workers"] or [])
        missing = {n: ip for n, ip in node_ips.items() if ip not in observed}
        assert not missing, (
            f"no Dask worker from node(s) {missing} in observed worker "
            f"hosts {sorted(observed)}"
        )

    def test_fixed_seed_matches_serial_reference(self, dask_job_summary, slurm_exec):
        """Same seed + same image -> the distributed sbatch run and a plain
        serial run produce bit-identical fitness histories."""
        result = slurm_exec(
            "python3", "examples/hpc/train_ea_dask.py",
            "--backend", "process", "--num-workers", "1",
            *_TRAIN_ARGS,
            "--output-dir", f"/shared/serial-ref-{uuid.uuid4().hex[:8]}",
            workdir="/app",
        )
        serial = _parse_summary(result.stdout)
        assert serial["fitness_history"] == dask_job_summary["fitness_history"]
        assert serial["final_fitness"] == dask_job_summary["final_fitness"]


class TestSlurmRigSingleNode:
    def test_singlenode_template_completes(self, slurm_exec):
        summary = _run_job(
            slurm_exec,
            "train_singlenode.sbatch",
            "deap-hpc-singlenode",
            script_args=("--population", "4", "--generations", "1", "--seed", "11"),
            # The template asks for 4 CPUs; the rig's nodes have 2. CLI
            # options override #SBATCH directives, so shrink it to fit.
            sbatch_opts=("--cpus-per-task=2",),
        )
        assert summary["parallel_backend"] == "process"
        assert math.isfinite(summary["final_fitness"])


@pytest.mark.skipif(
    os.environ.get("DEAP_HPC_GPU") != "1",
    reason="GPU node is opt-in: set DEAP_HPC_GPU=1 on a machine with an "
    "NVIDIA GPU passed through to Docker.",
)
class TestSlurmRigGpuNode:
    def test_gpu_template_trains_agilerl_on_cuda(self, slurm_exec):
        """--partition=gpu --gres=gpu:1 job lands on g1 and MADDPG really
        trains on CUDA (the template passes --require-cuda, so a silent CPU
        fallback fails the job instead of passing vacuously)."""
        summary = _run_job(
            slurm_exec,
            "train_rl_gpu.sbatch",
            "deap-hpc-gpu",
            script_args=("--steps", "400", "--seed", "3"),
            summary_key="cuda_available",
        )
        assert summary["hostname"] == "g1"
        assert summary["cuda_available"] is True
        assert str(summary["resolved_device"]).startswith("cuda")
        assert summary["algorithm"] == "maddpg"
        assert summary["run_dir"] and summary["run_dir"].startswith("/shared/")

    def test_coevolution_hetjob_spans_gpu_driver_and_cpu_workers(self, slurm_exec):
        """The heterogeneous-job template: EA-vs-MADDPG co-evolution with the
        driver (and CUDA training) on g1 while the EA generation evaluates on
        Dask workers spanning both CPU nodes; both sides report finite
        per-generation fitness."""
        summary = _run_job(
            slurm_exec,
            "train_coevolution_ea_vs_rl.sbatch",
            "deap-hpc-coev",
            script_args=("--population", "4", "--episodes", "24", "--seed", "5"),
            summary_key="cuda_available",
        )
        assert summary["hostname"] == "g1"
        assert summary["cuda_available"] is True
        assert summary["parallel_backend"] == "dask"
        assert str(summary["scheduler_address"]).endswith("scheduler.json")
        node_ips = {
            node: slurm_exec("getent", "hosts", node).stdout.split()[0]
            for node in _CPU_NODES
        }
        observed = set(summary["dask_workers"] or [])
        missing = {n: ip for n, ip in node_ips.items() if ip not in observed}
        assert not missing, (
            f"no Dask evaluation worker from node(s) {missing} in observed "
            f"worker hosts {sorted(observed)}"
        )
        assert summary["num_generations"] >= 1
        per_side = summary["per_side_best"]
        assert set(per_side) == {"pursuers", "evaders"}, (
            f"unexpected sides: {sorted(per_side)}"
        )
        for side, history in per_side.items():
            assert history, f"side {side} recorded no generations"
            assert all(math.isfinite(v) for v in history), (
                f"non-finite fitness for side {side}: {history}"
            )
