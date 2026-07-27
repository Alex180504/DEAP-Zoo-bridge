"""Experiment export (ZIP) and import utilities.

All exports are streaming (generator-based) so the FastAPI
``StreamingResponse`` can serve multi-GB archives without loading them
into memory.  The UI uses ``st.link_button`` pointing at the streaming
endpoints — the browser handles the download natively.

Import validates the archive structure, checks for path-traversal attacks
and available disk space, extracts the runs under ``DEAP_RESULTS_ROOT``,
and optionally restores the experiment record into the SQLite database.

Export ZIP structure (experiment level)::

    deap_export_{exp_id}_{label}_{date}.zip
    ├── deap_export_manifest.json    # schema version, timestamp, contents
    ├── experiment.json              # DB record (id, label, spec, timestamps)
    ├── _spec.json                   # experiment spec (from experiment_dir)
    ├── _result.json                 # runner result (run_dirs, final_metrics)
    ├── EA/
    │   └── seed_0/
    │       └── <run_id>/
    │           ├── manifest.json
    │           ├── metrics.csv
    │           ├── summary.json
    │           └── frames/  (optional)
    └── MADDPG/
        └── seed_0/
            └── <run_id>/
                └── ...

Export ZIP structure (individual runs)::

    deap_runs_export_{date}.zip
    ├── deap_export_manifest.json
    └── <run_dir_rel>/
        ├── manifest.json
        ├── metrics.csv
        └── ...

"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Constants

EXPORT_MANIFEST_SCHEMA_VERSION = 1
"""Bump when the export manifest format changes incompatibly."""

DEAP_VERSION = "0.2.0"  # Keep in sync with pyproject.toml

# Files / directories that are *always* included in a run export.
_ALWAYS_INCLUDE = {"manifest.json", "metrics.csv", "summary.json", "episodes.csv"}

# Optional artifact directories / patterns keyed by the user-facing flag.
_OPTIONAL_ARTIFACTS: Dict[str, List[str]] = {
    "frames": ["frames", "frames/*.png", "frames/*.jpg"],
    "tensorboard": ["tb", "tb/*"],
    "trajectories": ["trajectories.parquet", "*.parquet"],
}

# Hard upper bound on zip extraction to prevent zip-bomb attacks.
_MAX_EXTRACT_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB sanity cap


# Result types


@dataclass
class ImportValidationError:
    path: str  # e.g. "deap_export_manifest.json" or ""
    code: str  # machine-readable, e.g. "missing_manifest"
    message: str


@dataclass
class ImportValidationResult:
    ok: bool
    errors: List[ImportValidationError] = field(default_factory=list)
    warnings: List[ImportValidationError] = field(default_factory=list)
    estimated_bytes: int = 0
    experiment_count: int = 0
    run_count: int = 0


@dataclass
class ImportItem:
    original_exp_id: str
    new_exp_id: str
    label: str
    run_count: int


# Helpers


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_filename(label: str) -> str:
    """Sanitise a label for use in a zip filename."""
    return "".join(ch if (ch.isalnum() or ch in "._- ") else "_" for ch in label)[:80]


# Export manifest


def build_export_manifest(
    exp_id: str,
    label: str,
    run_dirs: List[str],
    include_frames: bool,
    include_tb: bool,
    include_trajectories: bool,
    spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``deap_export_manifest.json`` payload."""
    return {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "deap_version": DEAP_VERSION,
        "exported_at": _now_iso(),
        "experiment_id": exp_id,
        "label": label,
        "run_dirs": [str(d) for d in run_dirs],
        "artifact_flags": {
            "frames": include_frames,
            "tensorboard": include_tb,
            "trajectories": include_trajectories,
        },
        "spec": spec,
    }


# Run-level export


def _iter_run_files(
    run_dir: Path,
    include_frames: bool,
    include_tb: bool,
    include_trajectories: bool,
) -> Generator[Tuple[str, Path], None, None]:
    """Yield ``(arcname, file_path)`` pairs for every file to include from *run_dir*."""
    if not run_dir.is_dir():
        return

    for entry in sorted(run_dir.rglob("*")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(run_dir)
        name = str(rel).replace("\\", "/")

        # Always include the core metadata files.
        if name in _ALWAYS_INCLUDE:
            yield (name, entry)
            continue

        # Optional artifacts.
        if include_frames and (name.startswith("frames/") or name == "frames"):
            yield (name, entry)
            continue
        if include_tb and (name.startswith("tb/") or name == "tb"):
            yield (name, entry)
            continue
        if include_trajectories and (
            name.endswith(".parquet") or name.startswith("trajectories")
        ):
            yield (name, entry)
            continue


def create_run_export_zip(
    run_dir: Path,
    include_frames: bool = False,
    include_tb: bool = False,
    include_trajectories: bool = False,
) -> bytes:
    """Create an in-memory ZIP of a single run directory.

    Returns the raw ZIP bytes (suitable for small/medium runs; large
    experiment-level exports should use the streaming generator instead).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write the export manifest.
        manifest = build_export_manifest(
            exp_id="",
            label=run_dir.name,
            run_dirs=[str(run_dir)],
            include_frames=include_frames,
            include_tb=include_tb,
            include_trajectories=include_trajectories,
        )
        zf.writestr("deap_export_manifest.json", json.dumps(manifest, indent=2))

        # Walk and add files.
        run_rel_base = run_dir.name
        for arcname, filepath in _iter_run_files(
            run_dir, include_frames, include_tb, include_trajectories
        ):
            zf.write(filepath, f"{run_rel_base}/{arcname}")

    return buf.getvalue()


def archive_run_dir(run_dir: Path, *, remove_source: bool = False) -> Path:
    """Archive one completed run directory as an import-ready ZIP on disk.

    Produces ``<run_dir>.zip`` next to the directory, in the same layout the
    UI import endpoint expects (``deap_export_manifest.json`` at the root
    plus the run's files under ``<run_dir.name>/``), so the archive can be
    uploaded straight into Browse -> "Import experiments". Unlike the UI's
    :func:`create_run_export_zip` (which streams a *selection* of artifacts
    to a download), this includes **every** file in the run directory —
    videos, checkpoints, parquet — because the archive is the transport for
    the whole run (HPC nodes, backups).

    Args:
        run_dir: A run directory containing ``manifest.json``.
        remove_source: When ``True``, delete the uncompressed directory
            after the ZIP has been written successfully. A partially
            written archive is removed and the source left intact on error.

    Returns:
        Path of the written ZIP.

    Raises:
        ValueError: If ``run_dir`` is not a run directory (no
            ``manifest.json``) — refuses to archive (and delete) arbitrary
            directories.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(
            f"{run_dir} does not contain manifest.json — not a run directory."
        )
    zip_path = run_dir.with_name(run_dir.name + ".zip")
    manifest = build_export_manifest(
        exp_id="",
        label=run_dir.name,
        run_dirs=[run_dir.name],
        include_frames=True,
        include_tb=True,
        include_trajectories=True,
    )
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("deap_export_manifest.json", json.dumps(manifest, indent=2))
            for entry in sorted(run_dir.rglob("*")):
                if entry.is_file():
                    rel = str(entry.relative_to(run_dir)).replace("\\", "/")
                    zf.write(entry, f"{run_dir.name}/{rel}")
    except BaseException:
        zip_path.unlink(missing_ok=True)
        raise
    if remove_source:
        shutil.rmtree(run_dir)
    return zip_path


def archive_runs_under(root: Path, *, remove_source: bool = False) -> List[Path]:
    """Archive every run directory found (recursively) under *root*.

    A run directory is any directory containing ``manifest.json``. Each one
    becomes a sibling ``<run_dir>.zip`` via :func:`archive_run_dir`.

    Returns the list of written ZIP paths (empty when no runs were found).
    """
    root = Path(root)
    return [
        archive_run_dir(p.parent, remove_source=remove_source)
        for p in sorted(root.rglob("manifest.json"))
    ]


# Experiment-level export (streaming)


def _iter_experiment_zip(
    exp_id: str,
    db_record: Dict[str, Any],
    experiment_dir: Path,
    run_dirs: List[Path],
    include_frames: bool,
    include_tb: bool,
    include_trajectories: bool,
) -> Generator[bytes, None, None]:
    """Generator that yields ZIP chunks for a full experiment archive.

    Builds the ZIP in a temp file (so ``zipfile`` can seek as needed),
    then streams it out in 1 MiB chunks.  The caller wraps this in a
    ``StreamingResponse``.
    """
    import tempfile

    # Build the zip in a temp file — zipfile needs a seekable backing store.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Export manifest — use paths relative to experiment_dir
            #    so they match the ZIP layout.
            label = db_record.get("label", "")
            run_dir_rel = []
            for rd in run_dirs:
                try:
                    run_dir_rel.append(
                        str(rd.relative_to(experiment_dir)).replace("\\", "/")
                    )
                except ValueError:
                    run_dir_rel.append(rd.name)
            manifest = build_export_manifest(
                exp_id=exp_id,
                label=label,
                run_dirs=run_dir_rel,
                include_frames=include_frames,
                include_tb=include_tb,
                include_trajectories=include_trajectories,
                spec=db_record.get("spec"),
            )
            zf.writestr("deap_export_manifest.json", json.dumps(manifest, indent=2))

            # 2. Experiment DB record.
            exp_record = {
                "id": exp_id,
                "label": label,
                "status": db_record.get("status"),
                "spec": db_record.get("spec"),
                "created_at": db_record.get("created_at"),
                "started_at": db_record.get("started_at"),
                "finished_at": db_record.get("finished_at"),
                "error": db_record.get("error"),
                "origin": db_record.get("origin", "ui"),
            }
            zf.writestr("experiment.json", json.dumps(exp_record, indent=2))

            # 3. Service-level artifacts from the experiment directory.
            for svc_file in ["_spec.json", "_result.json"]:
                svc_path = experiment_dir / svc_file
                if svc_path.is_file():
                    zf.write(svc_path, svc_file)

            # 4. Each run directory (preserving relative structure).
            for run_dir in run_dirs:
                try:
                    rel_base = str(run_dir.relative_to(experiment_dir)).replace(
                        "\\", "/"
                    )
                except ValueError:
                    rel_base = run_dir.name
                for arcname, filepath in _iter_run_files(
                    run_dir, include_frames, include_tb, include_trajectories
                ):
                    zf.write(filepath, f"{rel_base}/{arcname}")

        # Stream the temp file in chunks.
        chunk_size = 1024 * 1024  # 1 MiB
        with open(tmp_path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def create_experiment_export_zip_generator(
    exp_id: str,
    db_record: Dict[str, Any],
    experiment_dir: Path,
    run_dirs: List[Path],
    include_frames: bool = False,
    include_tb: bool = False,
    include_trajectories: bool = False,
) -> Generator[bytes, None, None]:
    """Streaming ZIP generator for a full experiment export.

    Use with FastAPI ``StreamingResponse``::

        return StreamingResponse(
            create_experiment_export_zip_generator(...),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=..."},
        )
    """
    return _iter_experiment_zip(
        exp_id=exp_id,
        db_record=db_record,
        experiment_dir=experiment_dir,
        run_dirs=run_dirs,
        include_frames=include_frames,
        include_tb=include_tb,
        include_trajectories=include_trajectories,
    )


def export_filename(exp_id: str, label: str) -> str:
    """Return a sanitised download filename for an experiment export."""
    safe_label = _safe_filename(label) if label else exp_id
    return f"deap_export_{exp_id}_{safe_label}.zip"


# Import validation


def validate_import_zip(
    zip_path: Path,
    results_root: Path,
) -> ImportValidationResult:
    """Validate a DEAP export ZIP before extraction.

    Checks performed:
    - Is it a valid ZIP?
    - Does it contain ``deap_export_manifest.json`` at the root?
    - Is the manifest schema version supported?
    - Are referenced run dirs present and do they contain ``manifest.json``?
    - No path-traversal attacks (``..`` or absolute paths).
    - Estimated extraction size fits within available disk space.
    """
    errors: List[ImportValidationError] = []
    warnings: List[ImportValidationError] = []

    # 1. Is it a valid ZIP?
    if not zipfile.is_zipfile(zip_path):
        errors.append(
            ImportValidationError(
                path="",
                code="not_a_zip",
                message="The uploaded file is not a valid ZIP archive.",
            )
        )
        return ImportValidationResult(ok=False, errors=errors, warnings=warnings)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())

            # 2. Must have deap_export_manifest.json at root.
            if "deap_export_manifest.json" not in names:
                errors.append(
                    ImportValidationError(
                        path="deap_export_manifest.json",
                        code="missing_manifest",
                        message="ZIP does not contain deap_export_manifest.json at the root. "
                        "This does not appear to be a DEAP-bridge export.",
                    )
                )
                return ImportValidationResult(
                    ok=False, errors=errors, warnings=warnings
                )

            # 3. Parse and validate manifest.
            try:
                manifest_raw = json.loads(zf.read("deap_export_manifest.json"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(
                    ImportValidationError(
                        path="deap_export_manifest.json",
                        code="manifest_parse_error",
                        message=f"Cannot parse deap_export_manifest.json: {exc}",
                    )
                )
                return ImportValidationResult(
                    ok=False, errors=errors, warnings=warnings
                )

            schema_version = manifest_raw.get("schema_version", 0)
            if schema_version > EXPORT_MANIFEST_SCHEMA_VERSION:
                errors.append(
                    ImportValidationError(
                        path="deap_export_manifest.json",
                        code="schema_version_mismatch",
                        message=f"Export schema version {schema_version} is newer than "
                        f"this version of DEAP-bridge (supports ≤ {EXPORT_MANIFEST_SCHEMA_VERSION}).",
                    )
                )
                return ImportValidationResult(
                    ok=False, errors=errors, warnings=warnings
                )

            exp_id = manifest_raw.get("experiment_id", "unknown")
            run_dir_names = manifest_raw.get("run_dirs", [])

            # 4. Check every referenced run dir has a manifest.json in the zip.
            run_count = 0
            for rd in run_dir_names:
                rd_clean = str(rd).replace("\\", "/").rstrip("/")
                manifest_path = f"{rd_clean}/manifest.json"
                if manifest_path in names:
                    run_count += 1
                else:
                    warnings.append(
                        ImportValidationError(
                            path=rd_clean,
                            code="run_missing_manifest",
                            message=f"Run directory '{rd_clean}' in manifest does not contain "
                            f"manifest.json in the ZIP; skipping.",
                        )
                    )

            if run_count == 0 and len(run_dir_names) > 0:
                errors.append(
                    ImportValidationError(
                        path="",
                        code="no_valid_runs",
                        message="None of the run directories in the export contain a manifest.json.",
                    )
                )

            # 5. Path-traversal check.  Normalise to forward slashes first so
            #    that Windows format "..\\" entries are also caught.
            for name in names:
                norm = name.replace("\\", "/")
                if norm.startswith("/") or ".." in norm.split("/"):
                    errors.append(
                        ImportValidationError(
                            path=name,
                            code="path_traversal",
                            message=f"Path '{name}' appears to be absolute or contains '..' — rejected.",
                        )
                    )
                    break

            # 6. Estimate extraction size and check disk space.
            total_size = sum(
                info.file_size for info in zf.infolist() if not info.is_dir()
            )
            if total_size > _MAX_EXTRACT_BYTES:
                errors.append(
                    ImportValidationError(
                        path="",
                        code="too_large",
                        message=f"Estimated extraction size ({total_size / 1e9:.1f} GB) "
                        f"exceeds the safety limit of {_MAX_EXTRACT_BYTES / 1e9:.0f} GB.",
                    )
                )

            usage = shutil.disk_usage(results_root)
            available = usage.free
            if total_size > available * 0.9:  # leave 10% headroom
                errors.append(
                    ImportValidationError(
                        path="",
                        code="disk_full",
                        message=f"Not enough disk space. Need ~{total_size / 1e9:.1f} GB, "
                        f"have {available / 1e9:.1f} GB available on {results_root}.",
                    )
                )

            ok = len(errors) == 0
            return ImportValidationResult(
                ok=ok,
                errors=errors,
                warnings=warnings,
                estimated_bytes=total_size,
                experiment_count=1,
                run_count=run_count,
            )

    except (zipfile.BadZipFile, OSError) as exc:
        errors.append(
            ImportValidationError(
                path="",
                code="zip_read_error",
                message=f"Cannot read ZIP file: {exc}",
            )
        )
        return ImportValidationResult(ok=False, errors=errors, warnings=warnings)


# Import extraction


def extract_import_zip(
    zip_path: Path,
    results_root: Path,
    overwrite_existing: bool = False,
) -> List[Dict[str, Any]]:
    """Extract a validated DEAP export ZIP under *results_root*.

    Returns a list of dicts describing what was extracted::

        [{"experiment_id": ..., "label": ..., "run_dirs": [...], "warnings": [...]}]

    Handles duplicate run directories by either overwriting or skipping
    (with a warning).
    """
    # Service-metadata files live at the ZIP root and must not be extracted
    # into the results tree — they are read from the ZIP in memory by the
    # caller (app.py) and passed directly to restore_experiment_to_db.
    _SERVICE_META_FILES = frozenset(
        {
            "deap_export_manifest.json",
            "experiment.json",
            "_spec.json",
            "_result.json",
        }
    )

    extracted: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Read the manifest for metadata.
        manifest_raw = json.loads(zf.read("deap_export_manifest.json"))
        exp_id = manifest_raw.get("experiment_id", "unknown")
        label = manifest_raw.get("label", "")

        run_dirs_extracted: List[str] = []
        warnings: List[str] = []

        for info in zf.infolist():
            if info.is_dir():
                continue

            # Skip service-metadata files — these are consumed in memory, not
            # written to the results tree.
            if info.filename in _SERVICE_META_FILES:
                continue

            dest = results_root / info.filename

            # Sanity: ensure dest is still under results_root (double-check).
            try:
                dest.resolve().relative_to(results_root.resolve())
            except ValueError:
                warnings.append(
                    f"Skipping '{info.filename}' — resolves outside results root."
                )
                continue

            # Handle duplicates.
            if dest.exists() and not overwrite_existing:
                # Check if this is a run directory (contains manifest.json).
                if dest.name == "manifest.json":
                    run_dir = dest.parent
                    existing_manifest = run_dir / "manifest.json"
                    if existing_manifest.exists():
                        warnings.append(
                            f"Run directory '{run_dir.relative_to(results_root)}' already exists — skipped."
                        )
                continue

            # Extract.
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src:
                with open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            # Track run dirs (directories containing manifest.json).
            if dest.name == "manifest.json":
                run_dirs_extracted.append(str(dest.parent))

        extracted.append(
            {
                "experiment_id": exp_id,
                "label": label,
                "run_dirs": run_dirs_extracted,
                "warnings": warnings,
            }
        )

    return extracted


# DB restore


def _parse_iso_to_epoch(value: Optional[str]) -> Optional[float]:
    """Reverse :func:`deap_pettingzoo.service.db._iso` — parse an archived
    ISO-8601 timestamp back to a Unix epoch float. Returns ``None`` for
    missing/unparseable values (e.g. a 'queued' run's never-set ``started_at``)."""
    if not value:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


async def restore_experiment_to_db(
    experiment_json: Dict[str, Any],
    run_dirs: List[str],
    db_path: str,
) -> str:
    """Insert a restored experiment record into the SQLite database.

    Generates a **new** experiment ID to avoid collisions and sets
    ``origin='import'``. Restores the *archived* status (completed / failed /
    cancelled) and timestamps/error status as an imported ``failed`` run must still read as failed.
    A 'queued' or 'running' export is a snapshot of a process that no longer
    exists, so it is restored as ``failed`` with an explanatory message.

    Args:
        experiment_json: The ``experiment.json`` payload from the ZIP.
        run_dirs: The list of run directories as they now exist on disk.
        db_path: Path to the SQLite database.

    Returns:
        The new experiment ID.
    """
    from deap_pettingzoo.service import db as db_module

    spec = experiment_json.get("spec", {})
    if not spec:
        spec = {"environment": "unknown", "label": experiment_json.get("label", "")}
    spec_json = json.dumps(spec)

    new_id = await db_module.insert_experiment(
        spec_json=spec_json,
        label=experiment_json.get("label", ""),
        path=db_path,
        origin="import",
    )

    status = experiment_json.get("status") or "completed"
    error = experiment_json.get("error")
    if status == "completed":
        await db_module.set_completed(new_id, run_dirs, path=db_path)
    elif status == "cancelled":
        await db_module.set_cancelled(new_id, path=db_path)
    elif status == "failed":
        await db_module.set_failed(
            new_id, error or "Unknown error", run_dirs, path=db_path
        )
    else:
        # 'queued' / 'running' (or any unrecognised value) describe a
        # process that no longer exists — surface as failed rather than
        # re-entering the live queue or silently presenting as completed.
        await db_module.set_failed(
            new_id,
            error
            or f"Imported from a {status!r} state; the original process no longer exists",
            run_dirs,
            path=db_path,
        )

    await db_module.restore_timestamps(
        new_id,
        created_at=_parse_iso_to_epoch(experiment_json.get("created_at")),
        started_at=_parse_iso_to_epoch(experiment_json.get("started_at")),
        finished_at=_parse_iso_to_epoch(experiment_json.get("finished_at")),
        path=db_path,
    )

    return new_id
