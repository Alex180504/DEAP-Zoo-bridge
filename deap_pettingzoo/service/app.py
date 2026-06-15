"""FastAPI application — the REST sidecar for experiment lifecycle management.

This module exposes the full REST surface described in the UI-overhaul plan.
The Streamlit frontend connects to it via ``FASTAPI_URL`` and never imports
``torch`` / ``agilerl`` / ``jax``.

Launch::

    python -m deap_pettingzoo.service [--port 8100] [--host 127.0.0.1]

Or programmatically::

    from deap_pettingzoo.service.app import create_app
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8100)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from deap_pettingzoo.service import db
from deap_pettingzoo.service.dispatcher import Dispatcher
from deap_pettingzoo.service.models import (
    ExperimentRecord,
    ExperimentSpecModel,
    HealthResponse,
    QueueState,
    StdoutChunk,
    SubmitResponse,
    ValidationReportModel,
)

try:
    from fastapi import FastAPI, HTTPException, Query, status
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    raise ImportError(
        "The service package requires fastapi. "
        "Install with: pip install deap-pettingzoo[service]"
    )


# ---------------------------------------------------------------------------
# Globals (set by lifespan / create_app)
# ---------------------------------------------------------------------------

_dispatcher: Optional[Dispatcher] = None
_db_path: Optional[str] = None
_experiments_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _dispatcher, _db_path

    # Resolve the DB path early — create_app() may have been called without
    # arguments (uvicorn + factory=True), leaving _db_path as None.
    _db_path = _db_path or db._DEFAULT_DB_PATH

    # Ensure tables exist before the dispatcher queries them.
    conn = db._connect(_db_path)
    try:
        db.create_tables(conn)
    finally:
        conn.close()

    _dispatcher = Dispatcher(db_path=_db_path, experiments_dir=_experiments_dir)
    await _dispatcher.start()
    yield
    await _dispatcher.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    db_path: Optional[str] = None,
    experiments_dir: Optional[Path] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: SQLite database path; defaults to
            ``<DEAP_RESULTS_ROOT>/_service.db``.
        experiments_dir: Directory for experiment scratch space; defaults to
            ``<DEAP_RESULTS_ROOT>/_experiments/``.
    """
    global _db_path, _experiments_dir
    _db_path = db_path
    _experiments_dir = experiments_dir

    app = FastAPI(
        title="DEAP-bridge Experiment Service",
        version="0.1.0",
        lifespan=_lifespan,
    )
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _validate_spec(spec: ExperimentSpecModel) -> ValidationReportModel:
    """Run catalog-level validation on the spec (defense in depth)."""
    dc = spec.to_dataclass()
    from deap_pettingzoo.baselines.experiment_spec import validate_spec
    report = validate_spec(dc)
    return ValidationReportModel(
        ok=report.ok,
        issues=[
            {
                "path": i.path,
                "code": i.code,
                "message": i.message,
                "severity": i.severity,
            }
            for i in report.issues
        ],
    )


def _experiment_to_record(
    raw: Dict[str, Any],
    queued_count_ahead: int = 0,
) -> ExperimentRecord:
    """Map a db row dict to an ExperimentRecord response model."""
    spec = raw.get("spec")
    if isinstance(spec, dict):
        spec_model = ExperimentSpecModel(**spec)
    else:
        spec_model = ExperimentSpecModel(environment="unknown")

    return ExperimentRecord(
        id=raw["id"],
        label=raw.get("label", ""),
        status=raw["status"],
        spec=spec_model,
        created_at=raw.get("created_at", ""),
        started_at=raw.get("started_at"),
        finished_at=raw.get("finished_at"),
        run_dirs=raw.get("run_dirs", []),
        error=raw.get("error"),
        position=raw.get("position", 0),
        progress=raw.get("progress"),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:

    @app.get("/health", response_model=HealthResponse)
    async def health():
        from deap_pettingzoo.catalog.schema import CATALOG_SCHEMA_VERSION
        return HealthResponse(catalog_version=CATALOG_SCHEMA_VERSION)

    # -- catalog -----------------------------------------------------------

    @app.get("/catalog")
    async def get_catalog():
        from deap_pettingzoo import catalog
        return catalog.to_json()

    @app.get("/catalog/algorithms")
    async def get_catalog_algorithms():
        from deap_pettingzoo import catalog
        return [
            a.to_dict(available=catalog.is_algorithm_available(a.name))
            for a in catalog.list_algorithms()
        ]

    @app.get("/catalog/algorithms/{name}")
    async def get_catalog_algorithm(name: str):
        from deap_pettingzoo import catalog
        try:
            spec = catalog.get_algorithm(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown algorithm {name!r}")
        return spec.to_dict(available=catalog.is_algorithm_available(spec.name))

    @app.get("/catalog/environments")
    async def get_catalog_environments():
        from deap_pettingzoo import catalog
        return [
            e.to_dict(available=catalog.is_environment_available(e.name))
            for e in catalog.list_environments()
        ]

    @app.get("/catalog/environments/{name}")
    async def get_catalog_environment(name: str):
        from deap_pettingzoo import catalog
        try:
            spec = catalog.get_environment(name)
        except ValueError:
            raise HTTPException(
                status_code=404, detail=f"Unknown environment {name!r}"
            )
        return spec.to_dict(available=catalog.is_environment_available(spec.name))

    @app.get("/catalog/coevolution")
    async def get_catalog_coevolution():
        from deap_pettingzoo import catalog
        return catalog.get_coevolution_options()

    @app.get("/catalog/compatibility")
    async def get_catalog_compatibility(env: str, algorithms: Optional[str] = None):
        from deap_pettingzoo import catalog

        algo_list = (
            [a.strip() for a in algorithms.split(",") if a.strip()]
            if algorithms
            else None
        )
        return catalog.compatibility_matrix(env, algo_list)

    # -- experiments -------------------------------------------------------

    @app.post("/experiments", status_code=201, response_model=SubmitResponse)
    async def submit_experiment(spec: ExperimentSpecModel):
        report = _validate_spec(spec)
        if not report.ok:
            return JSONResponse(
                status_code=422,
                content=report.model_dump(),
            )

        exp_id = await db.insert_experiment(
            spec_json=json.dumps(spec.model_dump()),
            label=spec.label,
            path=_db_path,
        )
        # Compute queue position.
        ahead = await db.count_queued_before(exp_id, path=_db_path)
        return SubmitResponse(id=exp_id, status="queued", position=max(0, ahead))

    @app.post("/experiments/validate", response_model=ValidationReportModel)
    async def validate_experiment(spec: ExperimentSpecModel):
        return _validate_spec(spec)

    @app.get("/experiments", response_model=List[ExperimentRecord])
    async def list_experiments(
        status_filter: Optional[str] = Query(default=None, alias="status"),
        limit: int = 50,
        offset: int = 0,
    ):
        raw_list = await db.list_experiments(
            status=status_filter, limit=limit, offset=offset, path=_db_path
        )
        return [_experiment_to_record(r) for r in raw_list]

    @app.get("/experiments/{exp_id}", response_model=ExperimentRecord)
    async def get_experiment(exp_id: str):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Unknown experiment {exp_id!r}")
        ahead = await db.count_queued_before(exp_id, path=_db_path)
        raw["position"] = max(0, ahead)
        return _experiment_to_record(raw)

    @app.post("/experiments/{exp_id}/cancel")
    async def cancel_experiment(exp_id: str):
        if _dispatcher is None:
            raise HTTPException(status_code=503, detail="Dispatcher not running")
        result = await _dispatcher.cancel(exp_id)
        if result is None:
            # Already terminal — fetch it and return the record.
            raw = await db.get_experiment(exp_id, path=_db_path)
            if raw is None:
                raise HTTPException(
                    status_code=404, detail=f"Unknown experiment {exp_id!r}"
                )
            return _experiment_to_record(raw)
        raw = await db.get_experiment(exp_id, path=_db_path)
        return _experiment_to_record(raw or {"id": exp_id, "status": "cancelled"})

    @app.delete("/experiments/{exp_id}")
    async def delete_experiment(exp_id: str):
        ok = await db.delete_experiment(exp_id, path=_db_path)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {exp_id!r} not found or is running",
            )
        return {"ok": True}

    @app.get("/experiments/{exp_id}/progress", response_model=Dict[str, Any])
    async def get_experiment_progress(exp_id: str):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Unknown experiment {exp_id!r}")
        prog = raw.get("progress")
        if prog is None:
            # Fall back to the _progress.json file.
            exp_dir = (_experiments_dir or Path("results/_experiments")) / exp_id
            progress_file = exp_dir / "_progress.json"
            if progress_file.exists():
                try:
                    prog = json.loads(progress_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
        return prog or {}

    @app.get("/experiments/{exp_id}/stdout", response_model=StdoutChunk)
    async def get_experiment_stdout(
        exp_id: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        lines: Optional[int] = None,
    ):
        exp_dir = (_experiments_dir or Path("results/_experiments")) / exp_id
        log_path = exp_dir / "_stdout.log"
        if not log_path.exists():
            return StdoutChunk(text="", next_offset=0, eof=True)

        text = ""
        next_offset = offset or 0
        eof = False

        try:
            if lines is not None:
                # Return the last N lines.
                raw = log_path.read_text(encoding="utf-8", errors="replace")
                all_lines = raw.splitlines()
                wanted = all_lines[-lines:]
                text = "\n".join(wanted)
                next_offset = len(raw)
                eof = True
            else:
                with open(log_path, "rb") as fh:
                    start = offset if offset is not None else 0
                    fh.seek(0, 2)  # end
                    file_size = fh.tell()
                    if start >= file_size:
                        text = ""
                        next_offset = file_size
                        eof = True
                    else:
                        fh.seek(start)
                        chunk_size = min(limit or 65536, file_size - start)
                        text = fh.read(chunk_size).decode("utf-8", errors="replace")
                        next_offset = start + chunk_size
                        eof = next_offset >= file_size
        except OSError:
            text = ""
            next_offset = 0
            eof = True

        return StdoutChunk(text=text, next_offset=next_offset, eof=eof)

    # -- queue control -----------------------------------------------------

    @app.get("/queue", response_model=QueueState)
    async def get_queue():
        return QueueState(**(await db.get_queue_state(path=_db_path)))

    @app.post("/queue/pause")
    async def pause_queue():
        await db.set_paused(True, path=_db_path)
        return {"paused": True}

    @app.post("/queue/resume")
    async def resume_queue():
        await db.set_paused(False, path=_db_path)
        return {"paused": False}
