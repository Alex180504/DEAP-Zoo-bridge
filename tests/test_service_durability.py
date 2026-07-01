"""Service robustness & durability regression tests.

Covers the following: per-experiment timeout + process-group kill, a single
filesystem error not wedging the dispatcher queue, cancel killing the whole
process tree, one JSON error envelope for unhandled exceptions, and import
restoring the archived status/timestamps/error.

These tests drive a real :class:`Dispatcher` against real subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time

import pytest

from deap_pettingzoo.service import db
from deap_pettingzoo.service.dispatcher import Dispatcher


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = db._connect(path)
    try:
        db.create_tables(conn)
    finally:
        conn.close()
    yield path
    import os
    try:
        os.unlink(path)
        os.unlink(path + "-wal")
        os.unlink(path + "-shm")
    except OSError:
        pass


def _valid_spec_dict():
    return {
        "environment": "simple_spread_v3",
        "algorithms": ["ea"],
        "budget": {"mode": "eval_episodes", "value": 16, "num_seeds": 1},
        "hyperparameters": {
            "ea": {"population_size": 4, "hidden_size": 8, "max_steps_per_episode": 10}
        },
    }


def _script_spec(script: str, label: str = "t") -> str:
    return json.dumps(
        {
            "kind": "script",
            "environment": "(custom script)",
            "label": label,
            "script": script,
        }
    )


async def _wait_until(predicate, timeout_s: float = 30.0, interval_s: float = 0.3) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval_s)
    raise AssertionError(f"condition not met within {timeout_s}s")


# Per-experiment timeout kills the process group.
class TestTimeout:
    @pytest.mark.anyio
    async def test_timeout_kills_hung_experiment_and_queue_keeps_draining(
        self, tmp_db, tmp_path, monkeypatch
    ):
        # Force a tiny ceiling so the hung script times out almost immediately
        # (script-kind specs have no budget and always get the ceiling).
        monkeypatch.setenv("DEAP_EXPERIMENT_TIMEOUT_CEILING_S", "2")

        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()
        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)

        hang_id = await db.insert_experiment(
            _script_spec("import time\ntime.sleep(120)\n", "hang"), path=tmp_db
        )
        quick_id = await db.insert_experiment(
            _script_spec("print('quick-done')\n", "quick"), path=tmp_db
        )

        await disp.start()
        try:
            deadline = time.time() + 40
            hang_status = quick_status = None
            while time.time() < deadline:
                records = {r["id"]: r for r in await db.list_experiments(path=tmp_db)}
                hang_status = records.get(hang_id, {}).get("status")
                quick_status = records.get(quick_id, {}).get("status")
                if hang_status not in ("queued", "running") and quick_status not in (
                    "queued",
                    "running",
                ):
                    break
                await asyncio.sleep(0.5)
        finally:
            await disp.shutdown()

        assert hang_status == "failed", f"expected timeout-failed, got {hang_status!r}"
        hang_record = await db.get_experiment(hang_id, path=tmp_db)
        assert "timeout" in (hang_record["error"] or "").lower()

        assert quick_status == "completed", (
            f"the queue should keep draining after a timeout, got {quick_status!r}"
        )


# A filesystem error during pre-spawn doesn't wedge the loop.
class TestFilesystemErrorResilience:
    @pytest.mark.anyio
    async def test_mkdir_failure_marks_failed_and_queue_continues(self, tmp_db, tmp_path):
        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()

        bad_id = await db.insert_experiment(
            _script_spec("print('should never run')\n", "bad"), path=tmp_db
        )
        # Pre-create a *file* at the experiment's working-dir path so
        # `experiment_dir.mkdir(parents=True, exist_ok=True)` raises
        # FileExistsError instead of creating a directory.
        (experiments_dir / bad_id).write_text("not a directory", encoding="utf-8")

        good_id = await db.insert_experiment(
            _script_spec("print('good-done')\n", "good"), path=tmp_db
        )

        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)
        await disp.start()
        try:
            deadline = time.time() + 30
            bad_record = good_record = None
            while time.time() < deadline:
                bad_record = await db.get_experiment(bad_id, path=tmp_db)
                good_record = await db.get_experiment(good_id, path=tmp_db)
                if bad_record["status"] in ("completed", "failed") and good_record[
                    "status"
                ] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.3)
        finally:
            await disp.shutdown()

        assert bad_record["status"] == "failed"
        assert good_record["status"] == "completed", (
            "a pre-spawn filesystem error on one experiment must not stop "
            "the dispatcher loop from draining the next one"
        )


# Cancel kills the whole process group, not just the parent.
class TestCancelProcessGroup:
    @pytest.mark.anyio
    async def test_cancel_kills_child_processes_too(self, tmp_db, tmp_path):
        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()
        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)
        await disp.start()

        heartbeat = tmp_path / "heartbeat.txt"
        child_code = (
            "import time\n"
            "i = 0\n"
            "while True:\n"
            f"    with open({str(heartbeat)!r}, 'w') as f:\n"
            "        f.write(str(i))\n"
            "    i += 1\n"
            "    time.sleep(0.1)\n"
        )
        parent_script = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])\n"
            "time.sleep(120)\n"
        )

        exp_id = await db.insert_experiment(
            _script_spec(parent_script, "tree"), path=tmp_db
        )

        try:
            await _wait_until(lambda: heartbeat.exists(), timeout_s=20)
            await asyncio.sleep(0.3)  # let the child write at least once more

            result = await disp.cancel(exp_id)
            assert result == "terminated"

            snapshot = heartbeat.read_text()
            await asyncio.sleep(1.5)
            assert heartbeat.read_text() == snapshot, (
                "child process kept writing heartbeats after the parent was "
                "cancelled — the whole process group was not killed"
            )
        finally:
            await disp.shutdown()


# Cancel() must verify proc identity, not just DB status.
class TestCancelDoesNotTargetWrongProcess:
    @pytest.mark.anyio
    async def test_cancel_ignores_a_stale_handle_for_a_different_experiment(
        self, tmp_db, tmp_path
    ):
        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()
        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)

        exp_a = await db.insert_experiment(
            _script_spec("print('a')\n", "a"), path=tmp_db
        )
        await db.set_running(exp_a, pid=1, path=tmp_db)  # DB says A is running.

        # Simulate the dispatcher having moved on to an unrelated experiment
        # B's live process by the time a (stale) cancel(A) call runs.
        other_proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)"
        )
        disp._running_proc = other_proc
        disp._running_exp_id = "experiment-b"

        try:
            result = await disp.cancel(exp_a)
            assert result == "terminated"
            assert other_proc.returncode is None, (
                "cancel(A) killed experiment B's process — proc identity was "
                "not checked before use"
            )
            record_a = await db.get_experiment(exp_a, path=tmp_db)
            assert record_a["status"] == "cancelled"
        finally:
            other_proc.kill()
            await other_proc.wait()


# Post-spawn errors must not wedge the queue either.
class TestPostSpawnErrorsStillTerminate:
    @pytest.mark.anyio
    async def test_unexpected_exception_after_spawn_still_marks_failed(
        self, tmp_db, tmp_path, monkeypatch
    ):
        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()
        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)

        async def _boom(self, exp_id, experiment_dir, returncode):
            raise AttributeError("simulated corrupt _result.json")

        monkeypatch.setattr(Dispatcher, "_record_result", _boom)

        exp_id = await db.insert_experiment(
            _script_spec("print('quick')\n", "quick"), path=tmp_db
        )
        pair = await db.dequeue_next(path=tmp_db)
        assert pair is not None
        await disp._run_one(*pair)

        record = await db.get_experiment(exp_id, path=tmp_db)
        assert record["status"] == "failed", (
            "an exception in post-spawn result handling left the row "
            f"stuck at {record['status']!r} instead of a terminal state"
        )
        assert "Unexpected dispatcher error" in record["error"]
        assert disp._running_proc is None
        assert disp._running_exp_id is None


# Must return single JSON error envelope, even for unhandled exceptions.
class TestErrorEnvelope:
    def test_unhandled_exception_returns_json_detail_envelope(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from deap_pettingzoo.service import db as db_mod
        from deap_pettingzoo.service.app import create_app

        db_path = str(tmp_path / "_service.db")
        experiments_dir = tmp_path / "_experiments"
        experiments_dir.mkdir()
        app = create_app(db_path=db_path, experiments_dir=experiments_dir)

        async def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(db_mod, "get_experiment", _boom)

        # raise_server_exceptions=False so the response actually reaches the client
        # instead of propagating as a Python exception (TestClient's default
        # debug behaviour).
        with TestClient(app, raise_server_exceptions=False) as tc:
            r = tc.get("/experiments/whatever")
            assert r.status_code == 500
            assert r.json() == {"detail": "Internal server error"}


# Import restores archived status / timestamps / error.
class TestImportRestoresArchivedStatus:
    @pytest.mark.anyio
    async def test_import_failed_experiment_stays_failed(self, tmp_db):
        from deap_pettingzoo.service.export_import import restore_experiment_to_db

        experiment_json = {
            "id": "orig-failed",
            "label": "broke",
            "status": "failed",
            "spec": _valid_spec_dict(),
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:05+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "error": "boom: something exploded",
        }
        new_id = await restore_experiment_to_db(
            experiment_json, run_dirs=[], db_path=tmp_db
        )
        record = await db.get_experiment(new_id, path=tmp_db)
        assert record["status"] == "failed"
        assert record["error"] == "boom: something exploded"
        assert record["origin"] == "import"
        assert record["created_at"].startswith("2026-01-01T00:00:00")
        assert record["started_at"].startswith("2026-01-01T00:00:05")
        assert record["finished_at"].startswith("2026-01-01T00:01:00")

    @pytest.mark.anyio
    async def test_import_completed_experiment_stays_completed(self, tmp_db):
        from deap_pettingzoo.service.export_import import restore_experiment_to_db

        experiment_json = {
            "status": "completed",
            "spec": _valid_spec_dict(),
            "created_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
        }
        new_id = await restore_experiment_to_db(
            experiment_json, run_dirs=["results/run1"], db_path=tmp_db
        )
        record = await db.get_experiment(new_id, path=tmp_db)
        assert record["status"] == "completed"
        assert record["run_dirs"] == ["results/run1"]

    @pytest.mark.anyio
    async def test_import_running_experiment_becomes_failed_not_completed(self, tmp_db):
        """A 'running'/'queued' export is a snapshot of a process that no
        longer exists — it must not silently re-enter the live queue or be
        presented as a (false) success."""
        from deap_pettingzoo.service.export_import import restore_experiment_to_db

        experiment_json = {
            "status": "running",
            "spec": _valid_spec_dict(),
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        new_id = await restore_experiment_to_db(
            experiment_json, run_dirs=[], db_path=tmp_db
        )
        record = await db.get_experiment(new_id, path=tmp_db)
        assert record["status"] == "failed"
        assert "no longer exists" in record["error"]
