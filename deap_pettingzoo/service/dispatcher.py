"""Asyncio experiment dispatcher that manages the queue lifecycle.

The dispatcher runs as a background task in the FastAPI lifespan. Its loop:

1. Crash-recovery: re-queues any ``running`` experiment found at startup (the
   *server* died mid-run, not the experiment — see
   :func:`~deap_pettingzoo.service.db.requeue_interrupted`).
2. While the queue is not paused and no experiment is running and the queue is
   non-empty: atomically claims the next ``queued`` experiment (FIFO by
   ``enqueue_seq``; :func:`~deap_pettingzoo.service.db.dequeue_next` flips it
   straight to ``running``), writes the spec JSON to a file, and spawns the
   runner subprocess — in its own process group/session — with stdout/stderr
   redirected to ``<experiment-dir>/_stdout.log``.
3. Awaits the subprocess with a per-experiment timeout derived from its
   budget (see :func:`_compute_timeout_seconds`). On exit code 0 the
   experiment is marked ``completed`` with its discovered run dirs; on
   non-zero, ``failed`` with the captured error from ``_result.json``; on
   timeout, the whole process group is killed and the experiment is marked
   ``failed("...timed out...")``.
4. Cancel support: a ``queued`` experiment is set to ``cancelled``; a
   ``running`` experiment has its whole process group terminated
   (+kill-after-grace).

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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from deap_pettingzoo.service import db

logger = logging.getLogger(__name__)

# Per-experiment subprocess timeout knobs. The estimate is a
# heuristic: ``budget.value`` doesn't map to wall-clock time uniformly
# across algorithms/envs/hardware, therefore it isclamped to [floor, ceiling] so it can
# never be absurdly short or run forever. All three are overridable per
# deployment via env vars.
_DEFAULT_TIMEOUT_FLOOR_S = 1  # 1 second
_DEFAULT_TIMEOUT_CEILING_S = 12 * 60 * 60  # 12 hours
_DEFAULT_TIMEOUT_PER_UNIT_S = 0.25  # seconds per (budget unit * seed * sub-run)


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on unset/empty/bad
    values instead of raising — a misconfigured (e.g. blank) timeout knob
    must degrade to the safe default, not fail every experiment in the
    queue with the same parse error."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s", name, raw, default)
        return default


def _compute_timeout_seconds(spec: Dict[str, Any]) -> float:
    """Derive a per-experiment subprocess timeout from the spec's budget.

    ``script``-kind specs carry no budget and always get the ceiling. For
    catalog-driven specs the estimate is
    ``budget.value * num_seeds * num_algorithms * per_unit_seconds``, clamped
    to ``[floor, ceiling]``.
    """
    floor_s = _env_float("DEAP_EXPERIMENT_TIMEOUT_FLOOR_S", _DEFAULT_TIMEOUT_FLOOR_S)
    ceiling_s = _env_float(
        "DEAP_EXPERIMENT_TIMEOUT_CEILING_S", _DEFAULT_TIMEOUT_CEILING_S
    )
    if spec.get("kind") == "script":
        return ceiling_s
    per_unit_s = _env_float(
        "DEAP_EXPERIMENT_TIMEOUT_PER_UNIT_S", _DEFAULT_TIMEOUT_PER_UNIT_S
    )
    budget = spec.get("budget") or {}
    value = float(budget.get("value", 50_000) or 50_000)
    num_seeds = max(1, int(budget.get("num_seeds", 1) or 1))
    num_sub_runs = max(1, len(spec.get("algorithms") or []))
    estimate = value * num_seeds * num_sub_runs * per_unit_s
    return min(ceiling_s, max(floor_s, estimate))


def _process_group_kwargs() -> Dict[str, Any]:
    """``Popen`` kwargs that put the runner subprocess in its own process
    group/session, so cancel/timeout can kill the whole tree —
    the runner may itself spawn worker subprocesses for parallel evaluation."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _terminate_group(
    proc: "asyncio.subprocess.Process", *, force: bool = False
) -> None:
    """Kill ``proc`` and its whole process group/session.

    On Windows, ``Process.terminate()`` only kills the immediate process, not
    children it spawned, so we shell out to ``taskkill /T /F`` (kill tree) —
    run via :func:`asyncio.to_thread` so the short blocking call doesn't stall
    the event loop. ``taskkill`` has no reliable graceful-stop equivalent for
    an arbitrary console process tree, so ``force`` is a no-op on Windows
    (every kill there is immediate); only POSIX gets a soft SIGTERM-first
    escalation. On POSIX, ``proc`` was started with ``start_new_session=True``
    so its pid is also its process-group id; signal the whole group.
    """
    if proc.returncode is not None:
        return
    if os.name == "nt":
        await asyncio.to_thread(
            subprocess.run,
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
        )
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, OSError):
        pass


async def _wait_or_kill(proc: "asyncio.subprocess.Process", grace: float = 5.0) -> None:
    """Wait up to ``grace`` seconds for ``proc`` to exit; force-kill its
    process group if it doesn't."""
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:
        await _terminate_group(proc, force=True)
        await proc.wait()


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
        self._running_exp_id: Optional[str] = None
        self._experiments_dir = (
            experiments_dir
            or Path(os.environ.get("DEAP_RESULTS_ROOT", "results")) / "_experiments"
        )

    # Lifecycle (called from FastAPI lifespan)

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
        await db.requeue_interrupted(path=db_path)
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
            await _terminate_group(self._running_proc)
            await _wait_or_kill(self._running_proc)
        logger.info("Dispatcher stopped")

    # Main loop

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(1.0)

                # Wait while paused.
                qs = await db.get_queue_state(path=self._db_path)
                if qs["paused"]:
                    continue
                if self._running_proc is not None:
                    continue

                # Atomically claim the next queued experiment.
                pair = await db.dequeue_next(path=self._db_path)
                if pair is None:
                    continue

                exp_id, spec_json = pair
                await self._run_one(exp_id, spec_json)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Dispatcher loop iteration failed unexpectedly; continuing"
                )

    async def _run_one(self, exp_id: str, spec_json: str) -> None:
        """Execute a single experiment: write spec, spawn runner, wait, report.

        Two execution kinds are supported, discriminated by the ``"kind"`` field
        of the stored spec JSON:

        * ``"spec"`` (default): the catalog-driven path — spawn
          ``python -m deap_pettingzoo.service.runner`` against ``_spec.json``.
        * ``"script"``: an advanced user-pasted script — write it to
          ``user_script.py`` and spawn ``python user_script.py`` in the
          experiment dir, then synthesise ``_result.json`` from the exit code.

        ``self._running_proc``/``self._running_exp_id`` are set together right
        after spawn and only cleared here, at the very end (success, failure,
        timeout, or cancellation) — never by :meth:`cancel`, which only reads
        them. Keeping a single writer and clearing late (rather than the
        instant ``proc.wait()`` resolves) closes two races a stale clear
        otherwise opens: a concurrent ``cancel()`` finding ``None`` and
        silently no-op'ing while still reporting success, or finding a
        *different*, just-claimed experiment's process and killing that one
        instead.
        """
        logger.info("Experiment %s dequeued, starting", exp_id)
        experiment_dir = self._experiments_dir / exp_id

        try:
            spec = json.loads(spec_json)
        except json.JSONDecodeError:
            spec = {}
        kind = spec.get("kind", "spec")

        try:
            timeout_s = _compute_timeout_seconds(spec)

            # All pre-spawn filesystem work (mkdir / write spec / open the
            # stdout log) plus the actual spawn live in one try/except: a
            # filesystem error here must
            # mark the experiment failed and let the loop move on, not
            # propagate and kill the dispatcher task.
            log_fh = None
            try:
                experiment_dir.mkdir(parents=True, exist_ok=True)
                spec_file = experiment_dir / "_spec.json"
                spec_file.write_text(spec_json, encoding="utf-8")

                log_path = experiment_dir / "_stdout.log"
                log_fh = open(str(log_path), "wb")
                if kind == "script":
                    script_path = experiment_dir / "user_script.py"
                    script_path.write_text(spec.get("script", ""), encoding="utf-8")
                    cmd = [sys.executable, str(script_path)]
                    popen_kwargs: Dict[str, Any] = {
                        "cwd": str(experiment_dir),
                        "env": _script_env(),
                    }
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
                popen_kwargs.update(_process_group_kwargs())

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_fh,
                    stderr=asyncio.subprocess.STDOUT,
                    **popen_kwargs,
                )
            except Exception as exc:
                logger.exception("Failed to start experiment %s", exp_id)
                if log_fh is not None:
                    log_fh.close()
                await db.set_failed(
                    exp_id,
                    f"Failed to start experiment: {exc}",
                    path=self._db_path,
                )
                return

            self._running_proc = proc
            self._running_exp_id = exp_id
            await db.set_pid(exp_id, proc.pid, path=self._db_path)
            logger.info(
                "Experiment %s running (kind=%s, pid=%s, log=%s, timeout=%.0fs)",
                exp_id,
                kind,
                proc.pid,
                log_path,
                timeout_s,
            )

            timed_out = False
            try:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    timed_out = True
                except asyncio.CancelledError:
                    await _terminate_group(proc)
                    await _wait_or_kill(proc)
                    raise
            finally:
                log_fh.close()

            if timed_out:
                logger.error(
                    "Experiment %s exceeded timeout of %.0fs; killing process group",
                    exp_id,
                    timeout_s,
                )
                await _terminate_group(proc)
                await _wait_or_kill(proc)
                # No-ops if cancel() already won the race and set 'cancelled'
                # (set_failed only transitions out of a non-terminal status).
                await db.set_failed(
                    exp_id,
                    f"Experiment exceeded its {timeout_s:.0f}s timeout and was terminated",
                    path=self._db_path,
                )
                return

            # If cancel() already marked this experiment cancelled while we
            # were waiting for the process to die, skip the result-file
            # handling so we don't overwrite the terminal status with "failed"
            # (belt-and-suspenders: set_completed/set_failed below are also
            # no-ops on an already-terminal row).
            current = await db.get_experiment(exp_id, path=self._db_path)
            if current and current["status"] == "cancelled":
                logger.info("Experiment %s cancelled", exp_id)
                return

            # Custom scripts don't write _result.json; synthesise one from the
            # exit code and the run dirs discovered under the experiment dir
            # so the existing _record_result path (and Browse/Compare
            # discovery) work.
            if kind == "script":
                _synthesise_script_result(experiment_dir, proc.returncode)

            await self._record_result(exp_id, experiment_dir, proc.returncode)
        except Exception as exc:
            logger.exception("Experiment %s: unexpected dispatcher error", exp_id)
            await db.set_failed(
                exp_id, f"Unexpected dispatcher error: {exc}", path=self._db_path
            )
        finally:
            if self._running_exp_id == exp_id:
                self._running_proc = None
                self._running_exp_id = None

    async def _record_result(
        self, exp_id: str, experiment_dir: Path, returncode: Optional[int]
    ) -> None:
        """Read ``_result.json`` and transition the experiment to its terminal
        DB status (completed / failed), logging the outcome."""
        result_path = experiment_dir / "_result.json"
        if not result_path.exists():
            logger.error(
                "Experiment %s: runner exited with code %s but no _result.json",
                exp_id,
                returncode,
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

    # External control

    async def cancel(self, exp_id: str) -> Optional[str]:
        """Cancel ``exp_id``. Returns ``'terminated'`` if running,
        ``'cancelled'`` if queued, or ``None`` if already terminal.

        The DB decides *whether* a cancel applies (F-7.2-04) — but
        ``self._running_proc`` is mutated by the separate ``_run_one``
        coroutine over the experiment's lifetime. This only reads (never writes)
        ``self._running_proc``/``self._running_exp_id`` — and double-checks
        ``self._running_exp_id == exp_id``
        before treating it as "the" process to kill, so a stale/reassigned
        handle can never cause the wrong process to be terminated.
        """
        record = await db.get_experiment(exp_id, path=self._db_path)
        if record is None:
            return None
        status = record["status"]
        if status == "running":
            if self._running_exp_id == exp_id and self._running_proc is not None:
                await _terminate_group(self._running_proc)
                await _wait_or_kill(self._running_proc)
            await db.set_cancelled(exp_id, path=self._db_path)
            return "terminated"
        elif status == "queued":
            await db.set_cancelled(exp_id, path=self._db_path)
            return "cancelled"
        return None  # already terminal
