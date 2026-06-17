"""Asyncio experiment dispatcher — manages the queue lifecycle.

The dispatcher runs as a background task in the FastAPI lifespan. Its loop:

1. Crash-recovery: marks any ``running`` experiment as ``failed(interrupted)``.
2. While the queue is not paused and no experiment is running and the queue is
   non-empty: dequeues the next ``queued`` experiment (FIFO by ``enqueue_seq``),
   writes the spec JSON to a temp file, and spawns the runner subprocess with
   stdout/stderr redirected to ``<experiment-dir>/_stdout.log``.
3. Awaits the subprocess. On exit code 0 the experiment is marked
   ``completed`` with its discovered run dirs. On non-zero it is marked
   ``failed`` with the captured error from ``_result.json``.
4. Cancel support: a ``queued`` experiment is set to ``cancelled``; a
   ``running`` experiment has its subprocess terminated (+kill timeout).

The ``worker_limit`` (currently 1) is the concurrency seam: when it becomes
 >1 the dispatcher will spawn up to N subprocesses and track them in a set,
awaiting ``asyncio.wait(FIRST_COMPLETED)``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from deap_pettingzoo.service import db


class Dispatcher:
    """Manages the experiment queue in an asyncio background task.

    Args:
        db_path: SQLite database path (optional).
        experiments_dir: Root directory for experiment working dirs.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        experiments_dir: Optional[Path] = None,
    ) -> None:
        self._db_path = db_path
        self._task: Optional[asyncio.Task] = None
        self._running_proc: Optional[asyncio.subprocess.Process] = None
        self._running_id: Optional[str] = None
        self._experiments_dir = experiments_dir or Path(
            os.environ.get("DEAP_RESULTS_ROOT", "results")
        ) / "_experiments"

    # -- lifecycle (called from FastAPI lifespan) --------------------------

    async def start(self) -> None:
        """Boot the dispatcher: crash-recovery, then begin the main loop."""
        db_path = self._db_path
        conn = None
        try:
            conn = db._connect(db_path)  # type: ignore[attr-defined]
            db.create_tables(conn)
        finally:
            if conn:
                conn.close()
        await db.set_interrupted_to_failed(path=db_path)
        self._task = asyncio.create_task(self._loop())
        print("[dispatcher] started", flush=True)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel the loop, terminate the running process."""
        print("[dispatcher] shutting down...", flush=True)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._running_proc and self._running_proc.returncode is None:
            self._running_proc.terminate()
            try:
                await asyncio.wait_for(self._running_proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._running_proc.kill()
        print("[dispatcher] stopped", flush=True)

    # -- main loop ---------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)

            # Wait while paused.
            qs = await db.get_queue_state(path=self._db_path)
            if qs["paused"]:
                continue
            if self._running_proc is not None:
                continue

            # Dequeue
            pair = await db.dequeue_next(path=self._db_path)
            if pair is None:
                continue

            exp_id, spec_json = pair
            self._running_id = exp_id
            await self._run_one(exp_id, spec_json)

    async def _run_one(self, exp_id: str, spec_json: str) -> None:
        """Execute a single experiment: write spec, spawn runner, wait, report."""
        experiment_dir = self._experiments_dir / exp_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Write spec to a temp file inside the experiment dir.
        spec_file = experiment_dir / "_spec.json"
        spec_file.write_text(spec_json, encoding="utf-8")

        # Spawn the runner.
        log_path = experiment_dir / "_stdout.log"
        log_fh = open(str(log_path), "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "deap_pettingzoo.service.runner",
                "--spec-file",
                str(spec_file),
                "--experiment-dir",
                str(experiment_dir),
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:  # pragma: no cover
            log_fh.close()
            await db.set_failed(
                exp_id,
                f"Failed to spawn runner: {exc}",
                path=self._db_path,
            )
            self._running_id = None
            return

        self._running_proc = proc
        await db.set_running(exp_id, proc.pid, path=self._db_path)

        try:
            await proc.wait()
        except asyncio.CancelledError:  # pragma: no cover
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
            raise
        finally:
            log_fh.close()

        self._running_proc = None
        self._running_id = None

        # If cancel() already marked this experiment cancelled while we were
        # waiting for the process to die, skip the result-file handling so we
        # don't overwrite the terminal status with "failed".
        current = await db.get_experiment(exp_id, path=self._db_path)
        if current and current["status"] == "cancelled":
            return

        # Read result.
        result_path = experiment_dir / "_result.json"
        run_dirs: list = []
        error: Optional[str] = None
        if result_path.exists():
            try:
                result_data = json.loads(result_path.read_text(encoding="utf-8"))
                run_dirs = result_data.get("run_dirs", [])
                status = result_data.get("status", "failed")
                if status == "completed":
                    await db.set_completed(exp_id, run_dirs, path=self._db_path)
                else:
                    error = result_data.get("error", "Experiment failed (unknown)")
                    await db.set_failed(exp_id, error, run_dirs, path=self._db_path)
            except (json.JSONDecodeError, OSError):
                await db.set_failed(
                    exp_id,
                    "Could not read _result.json",
                    path=self._db_path,
                )
        else:
            await db.set_failed(
                exp_id,
                f"Runner exited with code {proc.returncode} but no _result.json found",
                path=self._db_path,
            )

    # -- external control --------------------------------------------------

    async def cancel(self, exp_id: str) -> Optional[str]:
        """Cancel ``exp_id``. Returns ``'terminated'`` if running,
        ``'cancelled'`` if queued, or ``None`` if already terminal."""
        record = await db.get_experiment(exp_id, path=self._db_path)
        if record is None:
            return None
        status = record["status"]
        if status == "running":
            if self._running_id == exp_id and self._running_proc is not None:
                self._running_proc.terminate()
                try:
                    await asyncio.wait_for(self._running_proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._running_proc.kill()
                finally:
                    self._running_proc = None
                    self._running_id = None
            await db.set_cancelled(exp_id, path=self._db_path)
            return "terminated"
        elif status == "queued":
            await db.set_cancelled(exp_id, path=self._db_path)
            return "cancelled"
        return None  # already terminal
