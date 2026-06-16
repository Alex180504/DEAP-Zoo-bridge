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
    from fastapi import FastAPI, HTTPException, Query, UploadFile, File, status
    from fastapi.responses import JSONResponse, StreamingResponse
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
        origin=raw.get("origin", "ui"),
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

    # -- export / import --------------------------------------------------

    @app.get("/experiments/{exp_id}/export")
    async def export_experiment(
        exp_id: str,
        include_frames: bool = Query(default=False),
        include_tb: bool = Query(default=False),
        include_trajectories: bool = Query(default=False),
    ):
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Unknown experiment {exp_id!r}")

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

    @app.get("/runs/export")
    async def export_runs(
        run_dirs: str = Query(default="", description="Comma-separated run directory paths"),
        include_frames: bool = Query(default=False),
        include_tb: bool = Query(default=False),
        include_trajectories: bool = Query(default=False),
    ):
        paths = [Path(p.strip()) for p in run_dirs.split(",") if p.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="No run_dirs provided.")

        from deap_pettingzoo.service.export_import import (
            build_export_manifest,
            _iter_run_files,
        )
        import io
        import zipfile

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
                "Content-Disposition": "attachment; filename=\"deap_runs_export.zip\""
            },
        )

    @app.post("/experiments/import")
    async def import_experiments(
        file: UploadFile = File(...),
        overwrite_existing: bool = Query(default=False),
    ):
        import tempfile
        from deap_pettingzoo.service.export_import import (
            validate_import_zip,
            extract_import_zip,
            restore_experiment_to_db,
        )

        # Stream upload to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            while chunk := await file.read(8 * 1024 * 1024):  # 8 MiB chunks
                tmp.write(chunk)
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
                        ] + [
                            {"path": w.path, "code": w.code, "message": w.message, "severity": "warning"}
                            for w in validation.warnings
                        ],
                    },
                )

            # Extract.
            extracted = extract_import_zip(tmp_path, results_root, overwrite_existing)

            # Restore to DB.
            imported: list = []
            for exp_info in extracted:
                # Read experiment.json from the extracted location.
                exp_json_path = results_root / "experiment.json"
                exp_json: dict = {}
                if exp_json_path.exists():
                    try:
                        exp_json = json.loads(exp_json_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        pass

                new_id = await restore_experiment_to_db(
                    experiment_json=exp_json,
                    run_dirs=exp_info["run_dirs"],
                    db_path=_db_path or db._DEFAULT_DB_PATH,
                )
                imported.append({
                    "original_exp_id": exp_info["experiment_id"],
                    "new_exp_id": new_id,
                    "label": exp_info["label"],
                    "run_count": len(exp_info["run_dirs"]),
                    "warnings": exp_info.get("warnings", []),
                })

            return {"ok": True, "imported": imported}

        finally:
            # Clean up temp file.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @app.get("/experiments/{exp_id}/spec")
    async def get_experiment_spec(exp_id: str):
        """Return the raw ExperimentSpec JSON for re-run pre-population."""
        raw = await db.get_experiment(exp_id, path=_db_path)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Unknown experiment {exp_id!r}")
        return {"spec": raw.get("spec", {})}

    # -- run management ----------------------------------------------------

    @app.post("/runs/delete")
    async def delete_runs(
        run_dirs: str = Query(default="", description="Comma-separated run directory paths"),
    ):
        """Delete one or more run directories from disk.

        Only accessible from the UI container on the shared Docker network;
        the service container has write access to the results volume.
        """
        paths = [Path(p.strip()) for p in run_dirs.split(",") if p.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="No run_dirs provided.")
        deleted: list = []
        errors: list = []
        for p in paths:
            try:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p)
                    deleted.append(str(p))
                else:
                    errors.append({"path": str(p), "error": "Not found or not a directory"})
            except OSError as exc:
                errors.append({"path": str(p), "error": str(exc)})
        return {"ok": len(errors) == 0, "deleted": deleted, "errors": errors}

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
