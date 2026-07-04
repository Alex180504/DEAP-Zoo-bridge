"""Session-scoped Docker Compose lifecycles for the HPC validation rigs.

Two independent rigs share this conftest, and each one only starts when a
test that needs it runs (so ``pytest tests/hpc/test_slurm_rig.py`` never
boots the Dask stack, and vice versa):

- **Dask rig** (``docker/dask-cluster``): scheduler + 2 workers + an idle
  driver container for fast iteration on the remote-Dask trainer API.
- **Slurm rig** (``docker/slurm-cluster``): slurmctld + 2 slurmd "nodes"
  (c1, c2) running real Slurm that executes the actual sbatch templates
  from ``examples/hpc/slurm/``. ``DEAP_HPC_GPU=1`` additionally starts the
  ``g1`` GPU node (compose profile ``gpu``; needs an NVIDIA GPU passed
  through by Docker Desktop) so the AgileRL GPU template can run.

Opt-in: this whole package skips at collection time unless
``DEAP_HPC_TESTS=1`` is set, so a default ``pytest`` run (CI, local) never
touches Docker. Run explicitly via ``scripts/hpc_validate.ps1`` / ``.sh``, or
``DEAP_HPC_TESTS=1 pytest tests/hpc``.

Set ``DEAP_HPC_KEEP=1`` to leave the rigs running after the session (e.g. to
poke at the Dask dashboard on http://localhost:8787 or ``docker compose
exec`` in by hand).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

if os.environ.get("DEAP_HPC_TESTS") != "1":
    pytest.skip(
        "Docker HPC rig tests are opt-in; set DEAP_HPC_TESTS=1 to run them "
        "(requires Docker Desktop with Compose v2 running).",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
HPC_BASE_DOCKERFILE = REPO_ROOT / "docker" / "hpc-base" / "Dockerfile"
DASK_COMPOSE_FILE = REPO_ROOT / "docker" / "dask-cluster" / "docker-compose.yml"
SLURM_COMPOSE_FILE = REPO_ROOT / "docker" / "slurm-cluster" / "docker-compose.yml"
DASK_PROJECT_NAME = "deap-hpc-dask"
SLURM_PROJECT_NAME = "deap-hpc-slurm"

#: Container builds + cluster startup can take a while on a cold cache (the
#: GPU node image downloads a multi-GB CUDA torch wheel on first build).
_UP_TIMEOUT_S = 1800
#: Per-invocation ceiling for a driver run; also guards against the exact
#: failure mode the debuggability test checks for (a hang instead of an error).
_EXEC_TIMEOUT_S = 240
#: slurmd registration is asynchronous w.r.t. slurmctld's healthcheck.
_NODES_READY_TIMEOUT_S = 180


def _gpu_enabled() -> bool:
    return os.environ.get("DEAP_HPC_GPU") == "1"


def _run_compose(
    compose_file: Path,
    project: str,
    *args: str,
    profiles: tuple = (),
    timeout: float = _EXEC_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    profile_args: List[str] = []
    for profile in profiles:
        profile_args += ["--profile", profile]
    try:
        return subprocess.run(
            ["docker", "compose", *profile_args, "-f", str(compose_file),
             "-p", project, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            # Docker/BuildKit can emit UTF-8 progress-bar glyphs that don't
            # decode under Windows' default cp1252 console encoding.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"docker compose {' '.join(args)} timed out after {timeout:.0f}s "
            f"(this itself is a rig failure -- a driver run must never hang):\n"
            f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )


def _dask_compose(*args: str, timeout: float = _EXEC_TIMEOUT_S):
    return _run_compose(DASK_COMPOSE_FILE, DASK_PROJECT_NAME, *args, timeout=timeout)


def _slurm_compose(*args: str, timeout: float = _EXEC_TIMEOUT_S):
    profiles = ("gpu",) if _gpu_enabled() else ()
    return _run_compose(
        SLURM_COMPOSE_FILE, SLURM_PROJECT_NAME, *args,
        profiles=profiles, timeout=timeout,
    )


def _build_base_image() -> None:
    """Build ``deap-hpc-base:latest`` — the FROM of the Slurm node image.

    Compose can't order FROM-chained builds across services, so images that
    other images build on are produced explicitly (and cheaply, due to
    the build cache) before ``compose up --build``.
    """
    result = subprocess.run(
        ["docker", "build", "-t", "deap-hpc-base:latest",
         "-f", str(HPC_BASE_DOCKERFILE), str(REPO_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_UP_TIMEOUT_S,
    )
    if result.returncode != 0:
        pytest.fail(
            "docker build of deap-hpc-base:latest failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.fixture(scope="session")
def dask_rig():
    """Bring the Dask rig (scheduler + 2 workers + an idle driver) up once."""
    result = _dask_compose("up", "-d", "--build", "--wait", timeout=_UP_TIMEOUT_S)
    if result.returncode != 0:
        pytest.fail(
            "docker compose up failed for the Dask rig:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        yield
    finally:
        if os.environ.get("DEAP_HPC_KEEP") != "1":
            _dask_compose("down", "-v", timeout=_UP_TIMEOUT_S)


@pytest.fixture(scope="session")
def slurm_rig():
    """Bring the Slurm rig up once and wait until every node registers idle."""
    _build_base_image()
    if _gpu_enabled():
        # Dockerfile.gpu is FROM deap-slurm-node:latest; build that first
        # (any non-profile service produces it), then the GPU node's image.
        for target in (("build", "slurmctld"), ("build", "g1")):
            result = _slurm_compose(*target, timeout=_UP_TIMEOUT_S)
            if result.returncode != 0:
                pytest.fail(
                    f"docker compose {' '.join(target)} failed for the Slurm "
                    f"rig:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
    result = _slurm_compose("up", "-d", "--build", "--wait", timeout=_UP_TIMEOUT_S)
    if result.returncode != 0:
        pytest.fail(
            "docker compose up failed for the Slurm rig:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        _wait_for_idle_nodes(("c1", "c2") + (("g1",) if _gpu_enabled() else ()))
        yield
    finally:
        if os.environ.get("DEAP_HPC_KEEP") != "1":
            _slurm_compose("down", "-v", timeout=_UP_TIMEOUT_S)


def _wait_for_idle_nodes(nodes, timeout_s: float = _NODES_READY_TIMEOUT_S) -> None:
    """Poll sinfo until every expected node has registered and gone idle."""
    deadline = time.monotonic() + timeout_s
    last = "<no sinfo output>"
    while time.monotonic() < deadline:
        result = _slurm_compose(
            "exec", "-T", "slurmctld", "sinfo", "-h", "-N", "-o", "%N %t"
        )
        last = result.stdout + result.stderr
        if result.returncode == 0:
            idle = {
                fields[0]
                for fields in (line.split() for line in result.stdout.splitlines())
                if len(fields) == 2 and fields[1] == "idle"
            }
            if idle.issuperset(nodes):
                return
        time.sleep(2)
    pytest.fail(
        f"Slurm rig: nodes {tuple(nodes)} did not all reach 'idle' within "
        f"{timeout_s:.0f}s; last sinfo:\n{last}"
    )


@pytest.fixture(scope="session")
def slurm_exec(slurm_rig):
    """Callable: run a command inside the ``slurmctld`` container.

    ``env`` entries are visible to the exec'd process (and to the submitted jobs
    via ``sbatch``'s default ``--export=ALL``).
    """

    def _exec(
        *cmd: str,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        check: bool = True,
        timeout: float = _EXEC_TIMEOUT_S,
    ) -> subprocess.CompletedProcess:
        args: List[str] = ["exec", "-T"]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        if workdir is not None:
            args += ["-w", workdir]
        args += ["slurmctld", *cmd]
        result = _slurm_compose(*args, timeout=timeout)
        if check:
            assert result.returncode == 0, (
                f"command {' '.join(cmd)!r} failed in slurmctld "
                f"(exit {result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    return _exec


@pytest.fixture()
def run_driver(dask_rig):
    """Callable: run ``examples/hpc/train_ea_dask.py`` inside the driver container.

    On success (or ``check=False``) returns the parsed JSON summary line
    (``check=True``, default) or the raw ``CompletedProcess`` (``check=False``,
    for tests that expect and inspect a failure).
    """

    def _run(
        args: List[str], *, check: bool = True
    ) -> Union[Dict[str, Any], subprocess.CompletedProcess]:
        result = _dask_compose(
            "exec", "-T", "driver", "python3", "examples/hpc/train_ea_dask.py", *args
        )
        if not check:
            return result
        assert result.returncode == 0, (
            f"driver run failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        assert stdout_lines, f"driver run produced no stdout:\n{result.stderr}"
        return json.loads(stdout_lines[-1])

    return _run


@pytest.fixture()
def read_driver_file(dask_rig):
    """Callable: read a text file from inside the driver container."""

    def _read(path: str) -> str:
        result = _dask_compose("exec", "-T", "driver", "cat", path)
        assert result.returncode == 0, f"cat {path} failed:\n{result.stderr}"
        return result.stdout

    return _read
