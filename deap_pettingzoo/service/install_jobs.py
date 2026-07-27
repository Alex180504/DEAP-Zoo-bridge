"""Dependency-install job manager for the advanced UI.

Backs ``POST /env/install`` and ``GET /env/install/{job_id}``.  Lets an advanced
user install a third-party environment package (e.g. ``marl-aquarium``) into the
backend's Python environment so it can then be used by a custom experiment
script.

Safety model (this is gated behind ``DEAP_ENABLE_ADVANCED`` — see
:mod:`deap_pettingzoo.service.advanced`):

* **Allowlist** — only ``pip install``, ``pip uninstall`` and ``git clone`` are
  accepted; anything else raises :class:`InstallValidationError` (HTTP 400).
* **No shell** — commands are spawned via ``create_subprocess_exec`` with an
  explicit argument list, so user args are never interpreted by a shell.  A
  light metacharacter check rejects newline / NUL / command-chaining characters
  as defense in depth.
* **Confined clones** — ``git clone`` always writes under
  ``<DEAP_RESULTS_ROOT>/_custom_envs/`` regardless of any user-supplied path.

Jobs run detached; combined stdout/stderr stream to
``<DEAP_RESULTS_ROOT>/_installs/<job_id>/install.log`` and are tailed by the
status endpoint via :func:`deap_pettingzoo.service.logtail.read_tail`.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set

from deap_pettingzoo.service.logtail import read_tail

# Allowed (manager, action) pairs.
_ALLOWED = {
    ("pip", "install"),
    ("pip", "uninstall"),
    ("git", "clone"),
}

# Characters never legitimately needed in a package spec or git URL and which
# would matter if the command were ever routed through a shell. We use exec
# (no shell) so this is belt-and-suspenders, but reject them early for clarity.
_FORBIDDEN = re.compile(r"[;&|`\n\r\x00<>]")


class InstallValidationError(ValueError):
    """Raised when an install request fails the allowlist / arg checks."""


# job_id -> record dict
_JOBS: Dict[str, Dict] = {}

# Strong references to the in-flight completion watchers. The event loop keeps
# only a weak reference to a task, so a watcher whose sole reference was the
# discarded return value of create_task() may be garbage-collected while still
# awaiting the subprocess -- leaving its job stuck at "running" forever, since
# install jobs have no database row and no startup reconciliation to recover
# them. Entries remove themselves on completion via the done callback below.
_BACKGROUND_TASKS: Set["asyncio.Task[None]"] = set()


def _installs_root() -> Path:
    return Path(os.environ.get("DEAP_RESULTS_ROOT", "results")) / "_installs"


def _custom_envs_root() -> Path:
    return Path(os.environ.get("DEAP_RESULTS_ROOT", "results")) / "_custom_envs"


def _check_args(args: List[str]) -> None:
    for a in args:
        if not isinstance(a, str) or not a.strip():
            raise InstallValidationError("Empty or non-string argument.")
        if _FORBIDDEN.search(a):
            raise InstallValidationError(
                f"Argument {a!r} contains a forbidden character."
            )


def _clone_dest(url: str) -> Path:
    """Derive a confined destination directory for ``git clone``."""
    name = url.rstrip("/").split("/")[-1]
    name = re.sub(r"\.git$", "", name)
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name) or "repo"
    return _custom_envs_root() / name


def build_command(manager: str, action: str, args: List[str]) -> List[str]:
    """Validate + build the argv for an install request.

    Raises :class:`InstallValidationError` on a disallowed or malformed request.
    """
    manager = (manager or "").strip().lower()
    action = (action or "").strip().lower()
    if (manager, action) not in _ALLOWED:
        allowed = ", ".join(f"{m} {a}" for m, a in sorted(_ALLOWED))
        raise InstallValidationError(
            f"Operation '{manager} {action}' is not allowed. Allowed: {allowed}."
        )
    _check_args(args)
    if not args:
        raise InstallValidationError("No arguments provided.")

    if manager == "pip":
        cmd = [sys.executable, "-m", "pip", action]
        if action == "uninstall":
            cmd.append("-y")  # never prompt — runs non-interactively
        cmd.extend(args)
        return cmd

    # manager == "git", action == "clone"
    url = args[0]
    dest = _clone_dest(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return ["git", "clone", "--depth", "1", url, str(dest)]


async def start_install(manager: str, action: str, args: List[str]) -> str:
    """Validate, spawn the install subprocess, and return a job id.

    The subprocess runs detached; a background task awaits it and records the
    terminal status. Output streams to the job's ``install.log``.
    """
    cmd = build_command(manager, action, args)

    job_id = uuid.uuid4().hex[:16]
    job_dir = _installs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "install.log"

    log_fh = open(str(log_path), "wb")
    # Record the command for display (the python exe is noisy — show "pip ...").
    display = (
        " ".join(["pip" if c == sys.executable else c for c in cmd[2:]])
        if manager == "pip"
        else " ".join(cmd)
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # pragma: no cover - spawn failure
        log_fh.close()
        _JOBS[job_id] = {
            "job_id": job_id,
            "command": display,
            "log_path": log_path,
            "status": "failed",
            "returncode": None,
            "error": str(exc),
        }
        return job_id

    _JOBS[job_id] = {
        "job_id": job_id,
        "command": display,
        "log_path": log_path,
        "status": "running",
        "returncode": None,
    }

    async def _await_proc() -> None:
        try:
            rc = await proc.wait()
        finally:
            log_fh.close()
        rec = _JOBS.get(job_id)
        if rec is not None:
            rec["returncode"] = rc
            rec["status"] = "completed" if rc == 0 else "failed"

    task = asyncio.create_task(_await_proc())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return job_id


def get_status(
    job_id: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    lines: Optional[int] = None,
) -> Optional[Dict]:
    """Return the job's status + a log tail, or ``None`` if unknown."""
    rec = _JOBS.get(job_id)
    if rec is None:
        return None
    text, next_offset, eof = read_tail(rec["log_path"], offset, limit, lines)
    return {
        "job_id": job_id,
        "status": rec["status"],
        "returncode": rec.get("returncode"),
        "command": rec.get("command", ""),
        "text": text,
        "next_offset": next_offset,
        "eof": eof,
    }
