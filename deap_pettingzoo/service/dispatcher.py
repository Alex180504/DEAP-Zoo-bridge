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
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from deap_pettingzoo.service import db

logger = logging.getLogger(__name__)


def _script_env() -> dict:
    """Environment for a custom-script subprocess.

    Prepends the project root (``DEAP_PROJECT_ROOT``, default ``/app`` in the
    container image) to ``PYTHONPATH`` so a pasted script can
    ``import deap_pettingzoo`` and ``import examples...`` regardless of the
    experiment working directory it runs in.
    """
    env = dict(os.environ)
    project_root = os.environ.get("DEAP_PROJECT_ROOT") or os.getcwd()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        project_root + os.pathsep + existing if existing else project_root
    )
    return env


def _synthesise_script_result(experiment_dir: Path, returncode: Optional[int]) -> None:
    """Write ``_result.json`` for a custom-script run.

    Scripts don't produce a result file themselves, so we derive status from the
    exit code and discover run dirs by globbing for ``manifest.json`` under the
    experiment dir (scripts run with ``cwd`` set there, so relative ``results/``
    outputs land inside it and are linked to this experiment).
    """
    run_dirs: list = []
    seen = set()
    try:
        for manifest in experiment_dir.rglob("manifest.json"):
            d = str(manifest.parent)
            if d not in seen:
                seen.add(d)
                run_dirs.append(d)
    except OSError:
        pass

    status = "completed" if returncode == 0 else "failed"
    payload = {
        "status": status,
        "run_dirs": run_dirs,
        "finished_at": time.time(),
    }
    if status == "failed":
        payload["error"] = f"Script exited with code {returncode}"
    result_path = experiment_dir / "_result.json"
    tmp_path = experiment_dir / "_result.tmp"
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(str(tmp_path), str(result_path))
    except OSError:  # pragma: no cover
        pass


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
        logger.info("Dispatcher started (experiments_dir=%s)", self._experiments_dir)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel the loop, terminate the running process."""
        logger.info("Dispatcher shutting down")
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
        logger.info("Dispatcher stopped")

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
        """Execute a single experiment: write spec, spawn runner, wait, report.

        Two execution kinds are supported, discriminated by the ``"kind"`` field
        of the stored spec JSON:

        * ``"spec"`` (default): the catalog-driven path — spawn
          ``python -m deap_pettingzoo.service.runner`` against ``_spec.json``.
        * ``"script"``: an advanced user-pasted script — write it to
          ``user_script.py`` and spawn ``python user_script.py`` in the
          experiment dir, then synthesise ``_result.json`` from the exit code.
        """
        logger.info("Experiment %s dequeued, starting", exp_id)
        experiment_dir = self._experiments_dir / exp_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        try:
            spec = json.loads(spec_json)
        except json.JSONDecodeError:
            spec = {}
        kind = spec.get("kind", "spec")

        # Write spec to a file inside the experiment dir (kept for both kinds).
        spec_file = experiment_dir / "_spec.json"
        spec_file.write_text(spec_json, encoding="utf-8")

        # Build the subprocess command for this kind.
        log_path = experiment_dir / "_stdout.log"
        log_fh = open(str(log_path), "wb")
        if kind == "script":
            script_path = experiment_dir / "user_script.py"
            script_path.write_text(spec.get("script", ""), encoding="utf-8")
            cmd = [sys.executable, str(script_path)]
            popen_kwargs = {"cwd": str(experiment_dir), "env": _script_env()}
        else:
            cmd = [
                sys.executable,
                "-m",
                "deap_pettingzoo.service.runner",
                "--spec-file",
                str(spec_file),
                "--experiment-dir",
                str(experiment_dir),
            ]
            popen_kwargs = {}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
                **popen_kwargs,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to spawn runner for experiment %s", exp_id)
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
        logger.info(
            "Experiment %s running (kind=%s, pid=%s, log=%s)",
            exp_id, kind, proc.pid, log_path,
        )

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
            logger.info("Experiment %s cancelled", exp_id)
            return

        # Custom scripts don't write _result.json; synthesise one from the exit
        # code and the run dirs discovered under the experiment dir so the
        # existing _record_result path (and Browse/Compare discovery) work.
        if kind == "script":
            _synthesise_script_result(experiment_dir, proc.returncode)

        await self._record_result(exp_id, experiment_dir, proc.returncode)

    async def _record_result(
        self, exp_id: str, experiment_dir: Path, returncode: Optional[int]
    ) -> None:
        """Read ``_result.json`` and transition the experiment to its terminal
        DB status (completed / failed), logging the outcome."""
        result_path = experiment_dir / "_result.json"
        if not result_path.exists():
            logger.error(
                "Experiment %s: runner exited with code %s but no _result.json",
                exp_id, returncode,
            )
            await db.set_failed(
                exp_id,
                f"Runner exited with code {returncode} but no _result.json found",
                path=self._db_path,
            )
            return

        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.error("Experiment %s: could not read _result.json", exp_id)
            await db.set_failed(
                exp_id, "Could not read _result.json", path=self._db_path
            )
            return

        run_dirs = result_data.get("run_dirs", [])
        if result_data.get("status", "failed") == "completed":
            await db.set_completed(exp_id, run_dirs, path=self._db_path)
            logger.info(
                "Experiment %s completed (%d run dir(s))", exp_id, len(run_dirs)
            )
        else:
            error = result_data.get("error", "Experiment failed (unknown)")
            await db.set_failed(exp_id, error, run_dirs, path=self._db_path)
            summary = (error or "").strip().splitlines()
            logger.error(
                "Experiment %s failed: %s", exp_id, summary[0] if summary else error
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
