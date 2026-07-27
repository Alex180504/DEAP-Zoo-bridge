"""FastAPI application — the REST sidecar for experiment lifecycle management.

This module exposes the full REST API surface for inter-service communication.
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
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from deap_pettingzoo.service import db

logger = logging.getLogger(__name__)
from deap_pettingzoo.service.dispatcher import Dispatcher
from deap_pettingzoo.service.models import (
    DeletedResponse,
    ExperimentRecord,
    ExperimentSpecModel,
    HealthResponse,
    ImportResponse,
    InstallRequest,
    InstallStartResponse,
    InstallStatusResponse,
    PausedResponse,
    QueueState,
    RunsDeleteResponse,
    ScriptSubmitModel,
    SpecResponse,
    StdoutChunk,
    SubmitResponse,
    ValidationReportModel,
)

try:
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # pragma: no cover
    raise ImportError(
        "The service package requires fastapi. "
        "Install with: pip install deap-pettingzoo[service]"
    )

# OpenAPI metadata (surfaces in /docs, /redoc, and /openapi.json)

_API_DESCRIPTION = (
    "REST API for experiment lifecycle management. The Streamlit frontend "
    "consumes this API; third-party clients can use it directly.\n\n"
    "**Base URL** defaults to `http://127.0.0.1:8100` (override with `--host` "
    "/ `--port` or the `FASTAPI_URL` environment variable).\n\n"
    "**Content type** is `application/json` for every request and response "
    "body except the ZIP download / upload endpoints.\n\n"
    '**Error model** — every error response is `{"detail": "..."}`; '
    "request-validation failures (422) return a list of per-field errors "
    "under `detail`."
)

_OPENAPI_TAGS = [
    {"name": "Health", "description": "Liveness probe and service metadata."},
    {
        "name": "Catalog",
        "description": "Read-only algorithm / environment / coevolution "
        "catalog and compatibility checks. No auth.",
    },
    {
        "name": "Experiments",
        "description": "Submit, validate, list, inspect, cancel, and delete "
        "experiments.",
    },
    {
        "name": "Progress & Logs",
        "description": "Live progress snapshots and stdout / service-log " "tailing.",
    },
    {
        "name": "Export / Import",
        "description": "Download experiments or run directories as ZIP "
        "archives, and re-import them.",
    },
    {
        "name": "Runs",
        "description": "Operations on run directories under the results root.",
    },
    {
        "name": "Queue",
        "description": "Inspect and control the experiment dispatch queue.",
    },
    {
        "name": "Advanced",
        "description": "Dependency install and custom-script submission. "
        "Arbitrary code / command execution by design (trusted-input model), "
        "gated behind `DEAP_ENABLE_ADVANCED` — every route returns 403 when "
        "disabled.",
    },
]

# Reusable error-response docs.
_NOT_FOUND = {"description": "Resource not found."}

# Globals (set by lifespan / create_app)

_dispatcher: Optional[Dispatcher] = None
_db_path: Optional[str] = None
_experiments_dir: Optional[Path] = None


# Lifespan


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

    logger.info(
        "Service starting (db=%s, experiments_dir=%s)", _db_path, _experiments_dir
    )
    _dispatcher = Dispatcher(db_path=_db_path, experiments_dir=_experiments_dir)
    await _dispatcher.start()
    yield
    await _dispatcher.shutdown()
    logger.info("Service stopped")


# App factory


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
        description=_API_DESCRIPTION,
        openapi_tags=_OPENAPI_TAGS,
        lifespan=_lifespan,
    )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        """Map every otherwise-unhandled (500) error to the one documented
        ``{"detail": "..."}`` JSON envelope.

        FastAPI's default handlers cover ``HTTPException`` /
        ``RequestValidationError`` before this ever runs, so this only sees
        genuinely-unhandled errors. Without it, Starlette's
        ``ServerErrorMiddleware`` returns a plain-text body instead.
        """
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    _register_routes(app)
    return app


# Routes


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
        origin=raw.get("origin", "ui"),
    )


# Log / stdout tailing helper (shared by GET /experiments/{id}/stdout and /logs)


from deap_pettingzoo.service.logtail import read_tail as _read_tail  # noqa: E402

# Health


def _register_routes(app: FastAPI) -> None:

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Liveness check",
        description=(
            "Service liveness probe — the Docker `HEALTHCHECK` uses this. "
            "`advanced_enabled` reflects `DEAP_ENABLE_ADVANCED` and tells the "
            "UI whether to show the Setup -> Advanced panel."
        ),
    )
    async def health():
        from deap_pettingzoo.catalog.schema import CATALOG_SCHEMA_VERSION
        from deap_pettingzoo.service.advanced import advanced_enabled

        return HealthResponse(
            catalog_version=CATALOG_SCHEMA_VERSION,
            advanced_enabled=advanced_enabled(),
        )

    # Catalog

    @app.get(
        "/catalog",
        tags=["Catalog"],
        summary="Full catalog snapshot",
        description=(
            "Complete catalog — algorithms, environments, coevolution options, "
            "budget modes, and optional-dependency availability. The frontend "
            "fetches this once on first render."
        ),
    )
    async def get_catalog():
        from deap_pettingzoo import catalog

        return catalog.to_json()

    @app.get(
        "/catalog/algorithms",
        tags=["Catalog"],
        summary="List algorithms",
        description="Every registered algorithm with its availability flag.",
    )
    async def get_catalog_algorithms():
        from deap_pettingzoo import catalog

        return [
            a.to_dict(available=catalog.is_algorithm_available(a.name))
            for a in catalog.list_algorithms()
        ]

    @app.get(
        "/catalog/algorithms/{name}",
        tags=["Catalog"],
        summary="Get one algorithm",
        description="A single algorithm with its full hyperparameter schema.",
        responses={404: _NOT_FOUND},
    )
    async def get_catalog_algorithm(name: str):
        from deap_pettingzoo import catalog

        try:
            spec = catalog.get_algorithm(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown algorithm {name!r}")
        return spec.to_dict(available=catalog.is_algorithm_available(spec.name))

    @app.get(
        "/catalog/environments",
        tags=["Catalog"],
        summary="List environments",
        description="Every registered environment with its availability flag.",
    )
    async def get_catalog_environments():
        from deap_pettingzoo import catalog

        return [
            e.to_dict(available=catalog.is_environment_available(e.name))
            for e in catalog.list_environments()
        ]

    @app.get(
        "/catalog/environments/{name}",
        tags=["Catalog"],
        summary="Get one environment",
        description="A single environment with its config params and team " "metadata.",
        responses={404: _NOT_FOUND},
    )
    async def get_catalog_environment(name: str):
        from deap_pettingzoo import catalog

        try:
            spec = catalog.get_environment(name)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Unknown environment {name!r}")
        return spec.to_dict(available=catalog.is_environment_available(spec.name))

    @app.get(
        "/catalog/coevolution",
        tags=["Catalog"],
        summary="Coevolution options",
        description="Modes, schedules, samplers, aggregation methods, and "
        "per-side optimizer eligibility.",
    )
    async def get_catalog_coevolution():
        from deap_pettingzoo import catalog

        return catalog.get_coevolution_options()

    @app.get(
        "/catalog/compatibility",
        tags=["Catalog"],
        summary="Algorithm/environment compatibility",
        description="Which algorithms are compatible with the given "
        "environment, plus any warnings.",
    )
    async def get_catalog_compatibility(
        env: str = Query(description="Environment name."),
        algorithms: Optional[str] = Query(
            default=None,
            description="Comma-separated algorithm names. Omit to check all "
            "registered algorithms.",
        ),
    ):
        from deap_pettingzoo import catalog

        algo_list = (
            [a.strip() for a in algorithms.split(",") if a.strip()]
            if algorithms
            else None
        )
        return catalog.compatibility_matrix(env, algo_list)

    # Experiments

    @app.post(
        "/experiments",
        status_code=201,
        response_model=SubmitResponse,
        tags=["Experiments"],
        summary="Submit an experiment",
        description="Validate the spec server-side (defense in depth) and, if "
        "it passes, enqueue it. Returns the new id and queue position.",
        responses={
            422: {
                "description": "Validation failed — field-level "
                "errors are returned in the body."
            }
        },
    )
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

    @app.post(
        "/experiments/validate",
        response_model=ValidationReportModel,
        tags=["Experiments"],
        summary="Dry-run validation",
        description="Validate a spec without enqueuing it. The Setup page "
        "calls this behind the 'Validate' button.",
    )
    async def validate_experiment(spec: ExperimentSpecModel):
        return _validate_spec(spec)

    @app.get(
        "/experiments",
        response_model=List[ExperimentRecord],
        tags=["Experiments"],
        summary="List experiments",
        description="Experiments ordered by submission time (newest first), "
        "with optional status filtering and pagination.",
    )
    async def list_experiments(
        status_filter: Optional[str] = Query(
            default=None,
            alias="status",
            description="Filter by status: queued, running, completed, failed, "
            "cancelled. Omit for all.",
        ),
        limit: int = Query(default=50, description="Maximum records to return."),
        offset: int = Query(default=0, description="Pagination offset."),
    ):
        raw_list = await db.list_experiments(
            status=status_filter, limit=limit, offset=offset, path=_db_path
        )
        return [_experiment_to_record(r) for r in raw_list]

    @app.get(
        "/experiments/{exp_id}",
        response_model=ExperimentRecord,
        tags=["Experiments"],
        summary="Get one experiment",
        description="A single experiment by id, including its live queue " "position.",
        responses={404: _NOT_FOUND},
    )
    async def get_experiment(exp_id: str):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )
        ahead = await db.count_queued_before(exp_id, path=_db_path)
        raw["position"] = max(0, ahead)
        return _experiment_to_record(raw)

    _TERMINAL_STATUSES = ("completed", "failed", "cancelled")

    @app.post(
        "/experiments/{exp_id}/cancel",
        response_model=ExperimentRecord,
        tags=["Experiments"],
        summary="Cancel an experiment",
        description="Queued experiments are marked cancelled immediately; a "
        "running experiment has its runner subprocess terminated.",
        responses={
            404: _NOT_FOUND,
            409: {"description": "Experiment is already in a terminal state."},
            503: {"description": "Dispatcher is not running."},
        },
    )
    async def cancel_experiment(exp_id: str):
        if _dispatcher is None:
            raise HTTPException(status_code=503, detail="Dispatcher not running")
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )
        if raw["status"] in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Experiment {exp_id!r} is already {raw['status']} and cannot be cancelled",
            )
        await _dispatcher.cancel(exp_id)
        raw = await db.get_experiment(exp_id, path=_db_path)
        return _experiment_to_record(raw or {"id": exp_id, "status": "cancelled"})

    @app.delete(
        "/experiments/{exp_id}",
        response_model=DeletedResponse,
        tags=["Experiments"],
        summary="Delete an experiment",
        description="Remove an experiment record. Only terminal or queued "
        "experiments can be deleted — running experiments must be cancelled "
        "first. Also removes the experiment scratch directory.",
        responses={
            404: _NOT_FOUND,
            409: {"description": "Experiment is running — cancel it first."},
        },
    )
    async def delete_experiment(exp_id: str):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )
        if raw["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Experiment {exp_id!r} is running — cancel it first",
            )
        ok = await db.delete_experiment(exp_id, path=_db_path)
        if not ok:
            # Status flipped to 'running' between our check and the delete
            # (race) — surface the same 409 rather than a misleading 404.
            raise HTTPException(
                status_code=409,
                detail=f"Experiment {exp_id!r} is running — cancel it first",
            )
        # Remove the experiment scratch directory (stdout log, spec, result).
        import shutil as _shutil

        exp_dir = (_experiments_dir or Path("results/_experiments")) / exp_id
        if exp_dir.is_dir():
            _shutil.rmtree(exp_dir, ignore_errors=True)
        return {"deleted": True}

    @app.get(
        "/experiments/{exp_id}/progress",
        response_model=Dict[str, Any],
        tags=["Progress & Logs"],
        summary="Live progress snapshot",
        description="The current sub-run's progress. The Queue page polls this "
        "roughly every 2s. Returns an empty object when nothing is running.",
        responses={404: _NOT_FOUND},
    )
    async def get_experiment_progress(exp_id: str):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )
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

    @app.get(
        "/experiments/{exp_id}/stdout",
        response_model=StdoutChunk,
        tags=["Progress & Logs"],
        summary="Tail experiment stdout",
        description="Tail the runner subprocess's stdout log. Pass `lines` for "
        "the last N lines, or `offset`/`limit` to stream forward. A missing "
        "log returns an empty chunk (not 404).",
    )
    async def get_experiment_stdout(
        exp_id: str,
        offset: Optional[int] = Query(
            default=None, description="Byte offset to start reading from."
        ),
        limit: Optional[int] = Query(
            default=None, description="Maximum number of bytes to return."
        ),
        lines: Optional[int] = Query(
            default=None,
            description="If set, return the last N lines instead (overrides "
            "offset/limit).",
        ),
    ):
        exp_dir = (_experiments_dir or Path("results/_experiments")) / exp_id
        text, next_offset, eof = _read_tail(
            exp_dir / "_stdout.log", offset, limit, lines
        )
        return StdoutChunk(text=text, next_offset=next_offset, eof=eof)

    @app.get(
        "/logs",
        response_model=StdoutChunk,
        tags=["Progress & Logs"],
        summary="Tail the service log",
        description="Tail the backend service diagnostic log "
        "(`<DEAP_LOG_DIR>/service.log`) — queue lifecycle, warnings, and "
        "exceptions from the app process. Same tailing contract as "
        "`/experiments/{id}/stdout`; a missing log returns an empty chunk.",
    )
    async def get_service_logs(
        offset: Optional[int] = Query(
            default=None, description="Byte offset to start reading from."
        ),
        limit: Optional[int] = Query(
            default=None, description="Maximum number of bytes to return."
        ),
        lines: Optional[int] = Query(
            default=None,
            description="If set, return the last N lines instead (overrides "
            "offset/limit).",
        ),
    ):
        """Tail the backend service log (``<DEAP_LOG_DIR>/service.log``).

        Mirrors ``GET /experiments/{id}/stdout``: pass ``lines`` for the last N
        lines, or ``offset``/``limit`` to stream forward.
        """
        from deap_pettingzoo.service.logging_config import default_log_dir

        log_path = Path(default_log_dir()) / "service.log"
        text, next_offset, eof = _read_tail(log_path, offset, limit, lines)
        return StdoutChunk(text=text, next_offset=next_offset, eof=eof)

    # -- export / import --------------------------------------------------

    @app.get(
        "/experiments/{exp_id}/export",
        tags=["Export / Import"],
        summary="Export an experiment (ZIP)",
        description="Stream a ZIP bundle of the experiment: spec, manifests, "
        "metrics, and optionally frames, TensorBoard logs, and trajectories.",
        responses={
            200: {
                "content": {"application/zip": {}},
                "description": "ZIP archive (attachment).",
            },
            404: _NOT_FOUND,
        },
    )
    async def export_experiment(
        exp_id: str,
        include_frames: bool = Query(
            default=False, description="Include recorded video frames."
        ),
        include_tb: bool = Query(
            default=False, description="Include TensorBoard event files."
        ),
        include_trajectories: bool = Query(
            default=False, description="Include trajectory / replay data."
        ),
    ):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )

        run_dirs_raw = raw.get("run_dirs", [])
        run_dirs = [Path(d) for d in run_dirs_raw if d]
        experiment_dir = (_experiments_dir or Path("results/_experiments")) / exp_id

        from deap_pettingzoo.service.export_import import (
            create_experiment_export_zip_generator,
            export_filename,
        )

        filename = export_filename(exp_id, raw.get("label", ""))
        generator = create_experiment_export_zip_generator(
            exp_id=exp_id,
            db_record=raw,
            experiment_dir=experiment_dir,
            run_dirs=run_dirs,
            include_frames=include_frames,
            include_tb=include_tb,
            include_trajectories=include_trajectories,
        )
        return StreamingResponse(
            generator,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get(
        "/runs/export",
        tags=["Export / Import"],
        summary="Export run directories (ZIP)",
        description="Stream a ZIP bundle of one or more run directories. Paths "
        "that resolve outside the results root are rejected with 400.",
        responses={
            200: {
                "content": {"application/zip": {}},
                "description": "ZIP archive (attachment).",
            },
            400: {
                "description": "No run_dirs given, or a path is outside the "
                "results root."
            },
        },
    )
    async def export_runs(
        run_dirs: str = Query(
            default="", description="Comma-separated run directory paths."
        ),
        include_frames: bool = Query(
            default=False, description="Include recorded video frames."
        ),
        include_tb: bool = Query(
            default=False, description="Include TensorBoard event files."
        ),
        include_trajectories: bool = Query(
            default=False, description="Include trajectory / replay data."
        ),
    ):
        paths = [Path(p.strip()) for p in run_dirs.split(",") if p.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="No run_dirs provided.")

        # Reject any path that resolves outside the results root.
        results_root_resolved = Path(
            os.environ.get("DEAP_RESULTS_ROOT", "results")
        ).resolve()
        for p in paths:
            try:
                p.resolve().relative_to(results_root_resolved)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Path '{p}' is outside the results root and cannot be exported.",
                )

        import io
        import zipfile

        from deap_pettingzoo.service.export_import import (
            _iter_run_files,
            build_export_manifest,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = build_export_manifest(
                exp_id="",
                label="",
                run_dirs=[p.name for p in paths if p.is_dir()],
                include_frames=include_frames,
                include_tb=include_tb,
                include_trajectories=include_trajectories,
            )
            zf.writestr("deap_export_manifest.json", json.dumps(manifest, indent=2))
            for run_dir in paths:
                if not run_dir.is_dir():
                    continue
                rel_base = run_dir.name
                for arcname, filepath in _iter_run_files(
                    run_dir, include_frames, include_tb, include_trajectories
                ):
                    zf.write(filepath, f"{rel_base}/{arcname}")
        buf.seek(0)
        zip_bytes = buf.getvalue()
        return StreamingResponse(
            iter([zip_bytes]),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="deap_runs_export.zip"'
            },
        )

    @app.post(
        "/experiments/import",
        response_model=ImportResponse,
        tags=["Export / Import"],
        summary="Import experiments (ZIP upload)",
        description="Upload a previously exported ZIP, validate it, and restore "
        "its experiment records and run directories.",
        responses={422: {"description": "The uploaded archive failed validation."}},
    )
    async def import_experiments(
        file: UploadFile = File(
            ..., description="A ZIP produced by an export endpoint."
        ),
        overwrite_existing: bool = Query(
            default=False,
            description="Overwrite run directories that already exist.",
        ),
    ):
        import tempfile

        from starlette.concurrency import run_in_threadpool

        from deap_pettingzoo.service.export_import import (
            extract_import_zip,
            restore_experiment_to_db,
            validate_import_zip,
        )

        # Stream upload to a temp file. Opening and writing happen on a worker
        # thread: at 8 MiB a chunk, writing inline would block the event loop
        # for the duration of every write and stall every other request.
        tmp = await run_in_threadpool(
            tempfile.NamedTemporaryFile, suffix=".zip", delete=False
        )
        try:
            while chunk := await file.read(8 * 1024 * 1024):  # 8 MiB chunks
                await run_in_threadpool(tmp.write, chunk)
        finally:
            await run_in_threadpool(tmp.close)
        tmp_path = Path(tmp.name)

        try:
            results_root = Path(os.environ.get("DEAP_RESULTS_ROOT", "results"))

            # Validate.
            validation = validate_import_zip(tmp_path, results_root)
            if not validation.ok:
                return JSONResponse(
                    status_code=422,
                    content={
                        "ok": False,
                        "issues": [
                            {"path": e.path, "code": e.code, "message": e.message}
                            for e in validation.errors
                        ]
                        + [
                            {
                                "path": w.path,
                                "code": w.code,
                                "message": w.message,
                                "severity": "warning",
                            }
                            for w in validation.warnings
                        ],
                    },
                )

            # Read experiment.json from the ZIP in memory; do not rely on
            # it being written to disk (extract_import_zip skips metadata files).
            import zipfile as _zipfile

            exp_json: dict = {}
            try:
                with _zipfile.ZipFile(tmp_path, "r") as _zf:
                    exp_json = json.loads(_zf.read("experiment.json"))
            except (KeyError, json.JSONDecodeError, OSError):
                pass

            # Extract run directories only (metadata files are skipped).
            extracted = extract_import_zip(tmp_path, results_root, overwrite_existing)

            # Restore to DB.
            imported: list = []
            for exp_info in extracted:
                new_id = await restore_experiment_to_db(
                    experiment_json=exp_json,
                    run_dirs=exp_info["run_dirs"],
                    db_path=_db_path or db._DEFAULT_DB_PATH,
                )
                imported.append(
                    {
                        "original_exp_id": exp_info["experiment_id"],
                        "new_exp_id": new_id,
                        "label": exp_info["label"],
                        "run_count": len(exp_info["run_dirs"]),
                        "warnings": exp_info.get("warnings", []),
                    }
                )

            return {"ok": True, "imported": imported}

        finally:
            # Clean up temp file.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @app.get(
        "/experiments/{exp_id}/spec",
        response_model=SpecResponse,
        tags=["Experiments"],
        summary="Get the raw experiment spec",
        description="The raw stored spec JSON, used by the UI to pre-populate "
        "the Setup form for a re-run.",
        responses={404: _NOT_FOUND},
    )
    async def get_experiment_spec(exp_id: str):
        """Return the raw ExperimentSpec JSON for re-run pre-population."""
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown experiment {exp_id!r}"
            )
        return {"spec": raw.get("spec", {})}

    # Run management

    def _results_root() -> Path:
        return Path(os.environ.get("DEAP_RESULTS_ROOT", "results")).resolve()

    def _check_under_results_root(p: Path) -> None:
        """Raise HTTPException 400 if *p* resolves outside the results root."""
        try:
            p.resolve().relative_to(_results_root())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Path '{p}' is outside the results root and cannot be accessed.",
            )

    @app.post(
        "/runs/delete",
        response_model=RunsDeleteResponse,
        tags=["Runs"],
        summary="Delete run directories",
        description="Delete one or more run directories from disk. Each path "
        "must resolve under the results root and contain a manifest.json; "
        "per-path failures are reported without failing the whole request.",
        responses={400: {"description": "No run_dirs provided."}},
    )
    async def delete_runs(
        run_dirs: str = Query(
            default="", description="Comma-separated run directory paths."
        ),
    ):
        """Delete one or more run directories from disk.

        Only accessible from the UI container on the shared Docker network;
        the service container has write access to the results volume.
        """
        import shutil

        paths = [Path(p.strip()) for p in run_dirs.split(",") if p.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="No run_dirs provided.")
        deleted: list = []
        errors: list = []
        for p in paths:
            try:
                _check_under_results_root(p)
                if not p.is_dir():
                    errors.append(
                        {"path": str(p), "error": "Not found or not a directory"}
                    )
                    continue
                if not (p / "manifest.json").is_file():
                    errors.append(
                        {
                            "path": str(p),
                            "error": "Not a run directory (no manifest.json)",
                        }
                    )
                    continue
                shutil.rmtree(p)
                deleted.append(str(p))
            except HTTPException:
                errors.append(
                    {"path": str(p), "error": "Path is outside the results root"}
                )
            except OSError as exc:
                errors.append({"path": str(p), "error": str(exc)})
        return {"ok": len(errors) == 0, "deleted": deleted, "errors": errors}

    # Queue control

    @app.get(
        "/queue",
        response_model=QueueState,
        tags=["Queue"],
        summary="Queue state",
        description="Which experiment is running, what is waiting, and whether "
        "the dispatcher is paused.",
    )
    async def get_queue():
        return QueueState(**(await db.get_queue_state(path=_db_path)))

    @app.post(
        "/queue/pause",
        response_model=PausedResponse,
        tags=["Queue"],
        summary="Pause the dispatcher",
        description="Pause dequeuing. The currently-running experiment "
        "continues; no new experiments start until resumed.",
    )
    async def pause_queue():
        await db.set_paused(True, path=_db_path)
        return {"paused": True}

    @app.post(
        "/queue/resume",
        response_model=PausedResponse,
        tags=["Queue"],
        summary="Resume the dispatcher",
        description="Resume dequeuing of queued experiments.",
    )
    async def resume_queue():
        await db.set_paused(False, path=_db_path)
        return {"paused": False}

    # Advanced: dependency install + custom-script submission
    #
    # Arbitrary code/command execution by design — gated behind
    # DEAP_ENABLE_ADVANCED (default on).

    def _require_advanced() -> None:
        from deap_pettingzoo.service.advanced import advanced_enabled

        if not advanced_enabled():
            raise HTTPException(
                status_code=403,
                detail="Advanced features are disabled (DEAP_ENABLE_ADVANCED).",
            )

    @app.post(
        "/env/install",
        response_model=InstallStartResponse,
        tags=["Advanced"],
        summary="Start a dependency install",
        description="Start a dependency-install job in the backend environment. "
        "Only an allowlist is accepted: pip install, pip uninstall, git clone. "
        "Commands are spawned with an explicit argument list (never a shell).",
        responses={
            400: {"description": "Disallowed operation or forbidden argument."},
            403: {"description": "Advanced features are disabled."},
        },
    )
    async def env_install(req: InstallRequest):
        _require_advanced()
        from deap_pettingzoo.service import install_jobs

        try:
            job_id = await install_jobs.start_install(req.manager, req.action, req.args)
        except install_jobs.InstallValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return InstallStartResponse(job_id=job_id, status="running")

    @app.get(
        "/env/install/{job_id}",
        response_model=InstallStatusResponse,
        tags=["Advanced"],
        summary="Poll an install job",
        description="Poll an install job's status and tail its output (same "
        "tailing contract as the stdout endpoint).",
        responses={
            403: {"description": "Advanced features are disabled."},
            404: {"description": "Unknown job id."},
        },
    )
    async def env_install_status(
        job_id: str,
        offset: Optional[int] = Query(
            default=None, description="Byte offset to start reading from."
        ),
        limit: Optional[int] = Query(
            default=None, description="Maximum number of bytes to return."
        ),
        lines: Optional[int] = Query(
            default=None,
            description="If set, return the last N lines instead (overrides "
            "offset/limit).",
        ),
    ):
        _require_advanced()
        from deap_pettingzoo.service import install_jobs

        status_dict = install_jobs.get_status(job_id, offset, limit, lines)
        if status_dict is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}")
        return InstallStatusResponse(**status_dict)

    @app.post(
        "/experiments/script/validate",
        response_model=ValidationReportModel,
        tags=["Advanced"],
        summary="Validate a custom script",
        description="Static validation of a custom experiment script — confirms "
        "it parses and looks runnable. It cannot prove the script trains "
        "correctly.",
        responses={403: {"description": "Advanced features are disabled."}},
    )
    async def validate_script(req: ScriptSubmitModel):
        _require_advanced()
        from deap_pettingzoo.service.script_validate import validate_script_source

        report = validate_script_source(req.script)
        return ValidationReportModel(ok=report["ok"], issues=report["issues"])

    @app.post(
        "/experiments/script",
        status_code=201,
        response_model=SubmitResponse,
        tags=["Advanced"],
        summary="Submit a custom script",
        description="Submit a hand-written experiment script as a queued "
        "experiment. The dispatcher runs it as `python user_script.py` in the "
        "experiment working directory.",
        responses={
            422: {"description": "Script failed static validation."},
            403: {"description": "Advanced features are disabled."},
        },
    )
    async def submit_script(req: ScriptSubmitModel):
        _require_advanced()
        from deap_pettingzoo.service.script_validate import validate_script_source

        report = validate_script_source(req.script)
        if not report["ok"]:
            return JSONResponse(status_code=422, content=report)

        # Store the script as a discriminated "script"-kind spec. The dispatcher
        # branches on the "kind" field and runs the source as a subprocess.
        # NOTE: serialise the dict directly (not via ExperimentSpecModel) so
        # the extra `script`/`kind`/`entry` keys are preserved in spec_json.
        spec_json = json.dumps(
            {
                "kind": "script",
                "environment": "(custom script)",
                "label": req.label,
                "script": req.script,
                "entry": req.entry,
            }
        )
        exp_id = await db.insert_experiment(
            spec_json=spec_json, label=req.label, path=_db_path
        )
        ahead = await db.count_queued_before(exp_id, path=_db_path)
        return SubmitResponse(id=exp_id, status="queued", position=max(0, ahead))
