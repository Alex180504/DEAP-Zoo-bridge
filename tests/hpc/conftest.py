"""Session-scoped Docker Compose lifecycle for the multi-container Dask rig.

Opt-in: this whole package skips at collection time unless
``DEAP_HPC_TESTS=1`` is set, so a default ``pytest`` run (CI, local) never
touches Docker. Run explicitly via ``scripts/hpc_validate.ps1`` / ``.sh``, or
``DEAP_HPC_TESTS=1 pytest tests/hpc``.

Set ``DEAP_HPC_KEEP=1`` to leave the rig running after the session (e.g. to
poke at the dashboard on http://localhost:8787 or ``docker compose exec`` in
by hand).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest

if os.environ.get("DEAP_HPC_TESTS") != "1":
    pytest.skip(
        "Docker HPC rig tests are opt-in; set DEAP_HPC_TESTS=1 to run them "
        "(requires Docker Desktop with Compose v2 running).",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DASK_COMPOSE_FILE = REPO_ROOT / "docker" / "dask-cluster" / "docker-compose.yml"
PROJECT_NAME = "deap-hpc-dask"

#: Container builds + cluster startup can take a while on a cold cache.
_UP_TIMEOUT_S = 900
#: Per-invocation ceiling for a driver run -- also guards against the exact
#: failure mode the debuggability test checks for (a hang instead of an error).
_EXEC_TIMEOUT_S = 240


def _compose(*args: str, timeout: float = _EXEC_TIMEOUT_S) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", "compose", "-f", str(DASK_COMPOSE_FILE), "-p", PROJECT_NAME, *args],
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


@pytest.fixture(scope="session", autouse=True)
def dask_rig():
    """Bring the rig (scheduler + 2 workers + an idle driver) up once."""
    result = _compose("up", "-d", "--build", "--wait", timeout=_UP_TIMEOUT_S)
    if result.returncode != 0:
        pytest.fail(
            "docker compose up failed for the Dask rig:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        yield
    finally:
        if os.environ.get("DEAP_HPC_KEEP") != "1":
            _compose("down", "-v", timeout=_UP_TIMEOUT_S)


@pytest.fixture()
def run_driver():
    """Callable: run ``examples/hpc/train_ea_dask.py`` inside the driver container.

    On success (or ``check=False``) returns the parsed JSON summary line
    (``check=True``, default) or the raw ``CompletedProcess`` (``check=False``,
    for tests that expect and inspect a failure).
    """

    def _run(
        args: List[str], *, check: bool = True
    ) -> Union[Dict[str, Any], subprocess.CompletedProcess]:
        result = _compose(
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
def read_driver_file():
    """Callable: read a text file from inside the driver container."""

    def _read(path: str) -> str:
        result = _compose("exec", "-T", "driver", "cat", path)
        assert result.returncode == 0, f"cat {path} failed:\n{result.stderr}"
        return result.stdout

    return _read
