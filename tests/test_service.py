"""Tests for the Phase 2 service package: models, db, runner, dispatcher, app.

Requires the ``fastapi``, ``pydantic``, ``httpx``, and ``uvicorn`` packages.
These are part of the ``[service]`` extra and are installed in the dev venv.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

import deap_pettingzoo.environments  # noqa: F401 — register built-in envs
from deap_pettingzoo.baselines.experiment_spec import (
    ExperimentSpec,
    build_configs,
    validate_spec,
)
from deap_pettingzoo.catalog import availability
from deap_pettingzoo.service import db
from deap_pettingzoo.service.models import (
    BudgetSpecModel,
    CoevolutionSideSpecModel,
    CoevolutionSettingsModel,
    ExperimentRecord,
    ExperimentSpecModel,
    FitnessSettingsModel,
    ParallelismSettingsModel,
    QueueState,
    RecordingSettingsModel,
    SubmitResponse,
    ValidationReportModel,
)
from deap_pettingzoo.service.runner import ProgressWriter, run_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database with tables and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = db._connect(path)
    try:
        db.create_tables(conn)
    finally:
        conn.close()
    yield path
    try:
        os.unlink(path)
        os.unlink(path + "-wal")
        os.unlink(path + "-shm")
    except OSError:
        pass


@pytest.fixture
def tmp_experiment_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _valid_spec_dict():
    return {
        "environment": "simple_spread_v3",
        "algorithms": ["ea"],
        "budget": {"mode": "eval_episodes", "value": 16, "num_seeds": 1},
        "hyperparameters": {
            "ea": {"population_size": 4, "hidden_size": 8, "max_steps_per_episode": 10}
        },
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_spec_round_trip_to_dataclass(self):
        spec = ExperimentSpecModel(**_valid_spec_dict())
        dc = spec.to_dataclass()
        assert isinstance(dc, ExperimentSpec)
        assert dc.environment == "simple_spread_v3"
        assert dc.algorithms == ["ea"]
        assert dc.budget.value == 16

    def test_label_sanitisation(self):
        spec = ExperimentSpecModel(environment="simple_v3", label="rm -rf /;><")
        assert ";" not in spec.label
        assert len(spec.label) <= 200

    def test_invalid_recording_fmt(self):
        with pytest.raises(Exception):
            RecordingSettingsModel(fmt="avi")

    def test_budget_validation(self):
        with pytest.raises(Exception):
            BudgetSpecModel(value=0)
        with pytest.raises(Exception):
            BudgetSpecModel(num_seeds=0)

    def test_validation_report_model(self):
        r = ValidationReportModel(
            ok=False,
            issues=[
                {"path": "a", "code": "x", "message": "bad", "severity": "error"}
            ],
        )
        assert not r.ok
        assert len(r.issues) == 1

    def test_submit_response(self):
        sr = SubmitResponse(id="abc123", status="queued", position=0)
        assert sr.id == "abc123"

    def test_record_from_raw_dict(self):
        """Verify _experiment_to_record maps a db row to ExperimentRecord."""
        from deap_pettingzoo.service.app import _experiment_to_record

        raw = {
            "id": "abc",
            "label": "my exp",
            "status": "queued",
            "spec": _valid_spec_dict(),
            "created_at": "2025-01-01T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "run_dirs": [],
            "error": None,
            "position": 1,
            "progress": None,
        }
        rec = _experiment_to_record(raw)
        assert rec.id == "abc"
        assert rec.status == "queued"
        assert rec.spec.algorithms == ["ea"]

    def test_coevolution_side_spec(self):
        side = CoevolutionSideSpecModel(
            name="pred", algorithm="ea", role_prefixes=["adversary"]
        )
        dc = side.to_dataclass()
        assert dc.name == "pred"
        assert dc.role_prefixes == ("adversary",)

    def test_coevolution_settings(self):
        cs = CoevolutionSettingsModel(
            enabled=True,
            mode="competitive_2pop",
            sides=[
                CoevolutionSideSpecModel(name="p", algorithm="ea", role_prefixes=["a"]),
                CoevolutionSideSpecModel(name="q", algorithm="pso", role_prefixes=["b"]),
            ],
        )
        dc = cs.to_dataclass()
        assert dc.enabled
        assert len(dc.sides) == 2


# ---------------------------------------------------------------------------
# Database (async CRUD)
# ---------------------------------------------------------------------------


class TestDatabase:
    @pytest.mark.anyio
    async def test_insert_and_get(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        assert exp_id
        record = await db.get_experiment(exp_id, path=tmp_db)
        assert record is not None
        assert record["status"] == "queued"

    @pytest.mark.anyio
    async def test_list_by_status_and_pagination(self, tmp_db):
        for i in range(3):
            await db.insert_experiment(
                json.dumps(_valid_spec_dict()), label=f"run{i}", path=tmp_db
            )
        all_ = await db.list_experiments(path=tmp_db)
        assert len(all_) == 3
        limited = await db.list_experiments(limit=2, offset=0, path=tmp_db)
        assert len(limited) == 2
        filtered = await db.list_experiments(status="queued", path=tmp_db)
        assert len(filtered) == 3

    @pytest.mark.anyio
    async def test_lifecycle_transitions(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_running(exp_id, pid=12345, path=tmp_db)
        r = await db.get_experiment(exp_id, path=tmp_db)
        assert r["status"] == "running"
        assert r["pid"] == 12345

        await db.set_completed(exp_id, ["/tmp/runs/ea_seed0"], path=tmp_db)
        r = await db.get_experiment(exp_id, path=tmp_db)
        assert r["status"] == "completed"
        assert r["run_dirs"] == ["/tmp/runs/ea_seed0"]

    @pytest.mark.anyio
    async def test_set_failed_and_cancelled(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_failed(exp_id, "something went wrong", path=tmp_db)
        r = await db.get_experiment(exp_id, path=tmp_db)
        assert r["status"] == "failed"
        assert r["error"]

        exp2 = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_cancelled(exp2, path=tmp_db)
        r2 = await db.get_experiment(exp2, path=tmp_db)
        assert r2["status"] == "cancelled"

    @pytest.mark.anyio
    async def test_delete_non_running(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        assert await db.delete_experiment(exp_id, path=tmp_db)
        assert await db.get_experiment(exp_id, path=tmp_db) is None

    @pytest.mark.anyio
    async def test_cannot_delete_running(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_running(exp_id, pid=1, path=tmp_db)
        assert not await db.delete_experiment(exp_id, path=tmp_db)

    @pytest.mark.anyio
    async def test_dequeue_next_fifo_order(self, tmp_db):
        ids = []
        for _ in range(3):
            ids.append(await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db))
        # First dequeue should be the first inserted.
        pair = await db.dequeue_next(path=tmp_db)
        assert pair is not None
        assert pair[0] == ids[0]

    @pytest.mark.anyio
    async def test_dequeue_respects_pause(self, tmp_db):
        await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_paused(True, path=tmp_db)
        assert await db.dequeue_next(path=tmp_db) is None
        await db.set_paused(False, path=tmp_db)
        assert await db.dequeue_next(path=tmp_db) is not None

    @pytest.mark.anyio
    async def test_dequeue_wont_pop_when_running(self, tmp_db):
        await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        exp2 = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        # Mark the dequeued one as running.
        pair = await db.dequeue_next(path=tmp_db)
        await db.set_running(pair[0], pid=1, path=tmp_db)
        # Now dequeue should return None (something is already running).
        assert await db.dequeue_next(path=tmp_db) is None

    @pytest.mark.anyio
    async def test_crash_recovery(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.set_running(exp_id, pid=1, path=tmp_db)
        await db.set_interrupted_to_failed(path=tmp_db)
        r = await db.get_experiment(exp_id, path=tmp_db)
        assert r["status"] == "failed"

    @pytest.mark.anyio
    async def test_queue_state(self, tmp_db):
        qs = await db.get_queue_state(path=tmp_db)
        assert not qs["paused"]
        await db.set_paused(True, path=tmp_db)
        qs2 = await db.get_queue_state(path=tmp_db)
        assert qs2["paused"]

    @pytest.mark.anyio
    async def test_update_progress(self, tmp_db):
        exp_id = await db.insert_experiment(json.dumps(_valid_spec_dict()), path=tmp_db)
        await db.update_progress(exp_id, json.dumps({"fraction": 0.5}), path=tmp_db)
        r = await db.get_experiment(exp_id, path=tmp_db)
        assert r["progress"] == {"fraction": 0.5}


# ---------------------------------------------------------------------------
# Runner (in-process + subprocess)
# ---------------------------------------------------------------------------


class TestRunner:
    def test_progress_writer_atomic(self, tmp_experiment_dir):
        pw = ProgressWriter(tmp_experiment_dir)

        class FakeEvent:
            def to_dict(self):
                return {"unit": "generation", "current": 2, "total": 10, "fraction": 0.2}

        pw(FakeEvent())
        progress_file = tmp_experiment_dir / "_progress.json"
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data["current"] == 2

    def test_runner_in_process_smoke(self, tmp_experiment_dir):
        """Run a tiny EA experiment via run_experiment (in-process path)."""
        spec = ExperimentSpec(
            environment="simple_spread_v3",
            algorithms=["ea"],
            env_config={},
            hyperparameters={
                "ea": {"population_size": 4, "hidden_size": 8, "max_steps_per_episode": 10}
            },
        )
        report = validate_spec(spec)
        assert report.ok, [i.message for i in report.errors]

        spec_file = tmp_experiment_dir / "_spec.json"
        spec_file.write_text(json.dumps(spec.to_dict()))

        run_experiment(spec_file, tmp_experiment_dir)

        result_file = tmp_experiment_dir / "_result.json"
        assert result_file.exists()
        result = json.loads(result_file.read_text())
        assert result["status"] == "completed"
        # output_dir is None → run_dirs may be empty; run_dir is None since
        # the spec didn't set an output_dir. That's normal for no-disk-output runs.

    def test_runner_validation_fails(self, tmp_experiment_dir):
        """Bad spec -> failed with validation errors."""
        spec_file = tmp_experiment_dir / "_spec.json"
        spec_file.write_text(
            json.dumps(
                {
                    "environment": "nope_v99",
                    "algorithms": ["ea"],
                    "budget": {"mode": "env_steps", "value": 10, "num_seeds": 1},
                }
            )
        )
        with pytest.raises(SystemExit) as exc_info:
            run_experiment(spec_file, tmp_experiment_dir)
        assert exc_info.value.code == 1
        result = json.loads((tmp_experiment_dir / "_result.json").read_text())
        assert result["status"] == "failed"

    def test_runner_subprocess_smoke(self, tmp_experiment_dir):
        """Run the runner as a subprocess (matches the dispatcher path)."""
        spec = ExperimentSpec(
            environment="simple_spread_v3",
            algorithms=["ea"],
            hyperparameters={
                "ea": {"population_size": 4, "hidden_size": 8, "max_steps_per_episode": 10}
            },
        )
        spec_file = tmp_experiment_dir / "_spec.json"
        spec_file.write_text(json.dumps(spec.to_dict()))

        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "deap_pettingzoo.service.runner",
                "--spec-file",
                str(spec_file),
                "--experiment-dir",
                str(tmp_experiment_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        result = json.loads((tmp_experiment_dir / "_result.json").read_text())
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI TestClient with an isolated tmp db+experiments dir."""
    from deap_pettingzoo.service.app import create_app

    db_path = str(tmp_path / "_service.db")
    experiments_dir = tmp_path / "_experiments"
    experiments_dir.mkdir()
    app = create_app(db_path=db_path, experiments_dir=experiments_dir)
    from fastapi.testclient import TestClient

    # Inject global state for the test
    import deap_pettingzoo.service.app as app_mod

    app_mod._db_path = db_path
    app_mod._experiments_dir = experiments_dir

    with TestClient(app) as tc:
        yield tc

    # Cleanup dispatcher
    import deap_pettingzoo.service.app as app_mod2

    if app_mod2._dispatcher:
        # The dispatcher was started on app startup via lifespan. In TestClient
        # the lifespan may not have run; ensure we clean up if it did.
        pass


class TestAppHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestAppCatalog:
    def test_catalog_snapshot(self, client):
        r = client.get("/catalog")
        assert r.status_code == 200
        data = r.json()
        assert "algorithms" in data
        assert "environments" in data
        assert any(a["name"] == "pso" for a in data["algorithms"])

    def test_catalog_algorithm(self, client):
        r = client.get("/catalog/algorithms/ea")
        assert r.status_code == 200
        assert r.json()["name"] == "ea"

    def test_catalog_algorithm_not_found(self, client):
        r = client.get("/catalog/algorithms/skynet")
        assert r.status_code == 404

    def test_catalog_environment(self, client):
        r = client.get("/catalog/environments/simple_spread_v3")
        assert r.status_code == 200
        assert r.json()["name"] == "simple_spread_v3"

    def test_catalog_environment_not_found(self, client):
        r = client.get("/catalog/environments/nope_v99")
        assert r.status_code == 404

    def test_catalog_compatibility(self, client):
        r = client.get("/catalog/compatibility?env=simple_spread_v3")
        assert r.status_code == 200
        data = r.json()
        assert "compatible" in data


class TestAppExperiments:
    def test_validate_ok(self, client):
        r = client.post("/experiments/validate", json=_valid_spec_dict())
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_validate_fail(self, client):
        spec = _valid_spec_dict()
        spec["environment"] = "nope_v99"
        r = client.post("/experiments/validate", json=spec)
        assert r.status_code == 200
        assert not r.json()["ok"]

    def test_submit_and_list(self, client):
        r = client.post("/experiments", json=_valid_spec_dict())
        assert r.status_code == 201
        assert r.json()["status"] == "queued"

        lst = client.get("/experiments")
        assert lst.status_code == 200
        assert len(lst.json()) == 1

    def test_get_by_id(self, client):
        sub = client.post("/experiments", json=_valid_spec_dict())
        exp_id = sub.json()["id"]
        r = client.get(f"/experiments/{exp_id}")
        assert r.status_code == 200
        assert r.json()["id"] == exp_id

    def test_get_not_found(self, client):
        r = client.get("/experiments/nonexistent")
        assert r.status_code == 404

    def test_cancel_queued(self, client):
        sub = client.post("/experiments", json=_valid_spec_dict())
        exp_id = sub.json()["id"]
        r = client.post(f"/experiments/{exp_id}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] in ("cancelled", "queued")

    def test_delete_terminal(self, client):
        sub = client.post("/experiments", json=_valid_spec_dict())
        exp_id = sub.json()["id"]
        client.post(f"/experiments/{exp_id}/cancel")
        r = client.delete(f"/experiments/{exp_id}")
        assert r.status_code == 200

    def test_422_on_invalid_spec(self, client):
        spec = _valid_spec_dict()
        spec["environment"] = ""
        r = client.post("/experiments", json=spec)
        assert r.status_code == 422


class TestAppStdout:
    def test_stdout_missing_file(self, client):
        r = client.get("/experiments/nonexistent/stdout")
        assert r.status_code == 200
        assert r.json()["eof"]


class TestAppLogs:
    def test_logs_missing_file(self, client, monkeypatch, tmp_path):
        # No service.log written yet -> empty, eof.
        monkeypatch.setenv("DEAP_LOG_DIR", str(tmp_path / "empty_logs"))
        r = client.get("/logs")
        assert r.status_code == 200
        assert r.json()["eof"]

    def test_logs_tail(self, client, monkeypatch, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "service.log").write_text(
            "line-one\nline-two\nline-three\n", encoding="utf-8"
        )
        monkeypatch.setenv("DEAP_LOG_DIR", str(log_dir))

        r = client.get("/logs", params={"lines": 2})
        assert r.status_code == 200
        body = r.json()
        assert "line-three" in body["text"]
        assert body["eof"]


class TestAppQueue:
    def test_queue_state(self, client):
        r = client.get("/queue")
        assert r.status_code == 200
        qs = r.json()
        assert "paused" in qs

    def test_pause_resume(self, client):
        r = client.post("/queue/pause")
        assert r.status_code == 200
        assert r.json()["paused"]
        r = client.get("/queue")
        assert r.json()["paused"]
        r = client.post("/queue/resume")
        assert r.status_code == 200
        assert not r.json()["paused"]


# ---------------------------------------------------------------------------
# Dispatcher smoke (without spawning real runners)
# ---------------------------------------------------------------------------


class TestDispatcherSmoke:
    @pytest.mark.anyio
    async def test_dispatcher_lifecycle(self, tmp_db):
        """Start dispatcher, submit a tiny spec, verify it completes."""
        from deap_pettingzoo.service.dispatcher import Dispatcher

        experiments_dir = Path(tempfile.mkdtemp())
        disp = Dispatcher(db_path=tmp_db, experiments_dir=experiments_dir)

        await disp.start()

        # Insert a spec and let the dispatcher run it.
        await db.insert_experiment(
            json.dumps(_valid_spec_dict()), path=tmp_db
        )
        # Wait up to 60s for the experiment to complete.
        deadline = time.time() + 60
        while time.time() < deadline:
            records = await db.list_experiments(path=tmp_db)
            if records and records[0]["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

        await disp.shutdown()

        records = await db.list_experiments(path=tmp_db)
        assert records
        final_status = records[0]["status"]
        final_error = records[0].get("error", "")
        # The runner should complete the tiny EA run successfully.
        assert final_status == "completed", (
            f"status was {final_status!r}. error: {final_error}"
        )
