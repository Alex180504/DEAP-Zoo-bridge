"""Filesystem management operations for run directories.

All mutations are performed atomically where possible (write-to-temp then
rename) and include safety guards to prevent accidental modification of
non-run directories.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# Only alphanumerics, hyphens, underscores, and dots are permitted in folder
# names to prevent path traversal and filesystem issues.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _require_manifest(run_dir: Path) -> Path:
    """Return the manifest path, raising descriptive errors if invalid."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"Directory {run_dir} does not contain manifest.json "
            "and is not a recognised run directory."
        )
    return manifest


def rename_run(run_dir: Path, display_name: str) -> None:
    """Set a human-readable ``display_name`` label in ``manifest.json``.

    The write is atomic: the new content is written to a ``.tmp`` file
    alongside the manifest, then ``os.replace``-d into place so a crash
    mid-write never leaves a corrupt manifest.

    Args:
        run_dir: Path to the run directory (must contain ``manifest.json``).
        display_name: New label.  May be empty string to clear the label.
    """
    import os

    manifest_path = _require_manifest(run_dir)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if display_name:
        data["display_name"] = display_name
    elif "display_name" in data:
        del data["display_name"]

    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def delete_run(run_dir: Path) -> None:
    """Permanently remove a run directory and all its contents.

    Args:
        run_dir: Path to the run directory.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
        ValueError: If ``run_dir`` does not contain ``manifest.json``
            (safety guard against accidental deletion of non-run directories).
    """
    _require_manifest(run_dir)
    shutil.rmtree(run_dir)


def create_folder(parent: Path, name: str) -> Path:
    """Create a subdirectory under ``parent`` with the given ``name``.

    Args:
        parent: Existing directory to create the folder inside.
        name: Folder name. Must match ``[A-Za-z0-9._-]+`` to prevent
            path traversal.

    Returns:
        Path to the newly created (or already-existing) directory.

    Raises:
        ValueError: If ``name`` contains unsafe characters.
    """
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Folder name {name!r} contains invalid characters. "
            "Use only letters, digits, hyphens, underscores, and dots."
        )
    target = parent / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def move_run(run_dir: Path, target_dir: Path) -> None:
    """Move a run directory into ``target_dir``.

    After this call the run lives at ``target_dir / run_dir.name``.

    Args:
        run_dir: Source run directory (must contain ``manifest.json``).
        target_dir: Destination parent directory (created if missing).

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
        ValueError: If ``run_dir`` does not contain ``manifest.json``.
        FileExistsError: If a directory with the same name already exists
            inside ``target_dir``.
    """
    _require_manifest(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / run_dir.name
    if dest.exists():
        raise FileExistsError(
            f"A directory named {run_dir.name!r} already exists in {target_dir}."
        )
    shutil.move(str(run_dir), str(dest))
