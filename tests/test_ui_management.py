"""Tests for ``deap_pettingzoo.ui.services.management``.

All tests are pure filesystem operations using ``tmp_path``; no
``streamlit`` import is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deap_pettingzoo.ui.services.management import (
    create_folder,
    delete_run,
    move_run,
    rename_run,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_run_dir(parent: Path, name: str = "my_run") -> Path:
    """Create a minimal run directory with a manifest.json."""
    run_dir = parent / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": name,
        "algorithm": "ea",
        "env_name": "test_env",
        "seed": 0,
        "schema_version": 1,
        "started_at": "2024-01-01T00:00:00",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# rename_run
# ---------------------------------------------------------------------------


def test_rename_run_writes_display_name(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    rename_run(run_dir, "My Experiment")
    data = json.loads((run_dir / "manifest.json").read_text())
    assert data["display_name"] == "My Experiment"


def test_rename_run_clears_display_name_when_empty(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    rename_run(run_dir, "Initial name")
    rename_run(run_dir, "")
    data = json.loads((run_dir / "manifest.json").read_text())
    assert "display_name" not in data


def test_rename_run_is_atomic(tmp_path: Path):
    """No .json.tmp file should remain after a successful rename."""
    run_dir = _make_run_dir(tmp_path)
    rename_run(run_dir, "Atomic write test")
    assert not (run_dir / "manifest.json.tmp").exists()


def test_rename_run_raises_if_no_manifest(tmp_path: Path):
    run_dir = tmp_path / "not_a_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="manifest.json"):
        rename_run(run_dir, "Should fail")


def test_rename_run_raises_if_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        rename_run(tmp_path / "does_not_exist", "Should fail")


# ---------------------------------------------------------------------------
# delete_run
# ---------------------------------------------------------------------------


def test_delete_run_removes_directory(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    assert run_dir.exists()
    delete_run(run_dir)
    assert not run_dir.exists()


def test_delete_run_raises_if_no_manifest(tmp_path: Path):
    run_dir = tmp_path / "no_manifest"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="manifest.json"):
        delete_run(run_dir)


def test_delete_run_raises_if_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        delete_run(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# create_folder
# ---------------------------------------------------------------------------


def test_create_folder_creates_directory(tmp_path: Path):
    result = create_folder(tmp_path, "experiment-1")
    assert result.is_dir()
    assert result == tmp_path / "experiment-1"


def test_create_folder_idempotent(tmp_path: Path):
    create_folder(tmp_path, "experiment-1")
    # Second call must not raise.
    result = create_folder(tmp_path, "experiment-1")
    assert result.is_dir()


def test_create_folder_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid characters"):
        create_folder(tmp_path, "../escape")


def test_create_folder_rejects_slash_in_name(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid characters"):
        create_folder(tmp_path, "foo/bar")


def test_create_folder_rejects_empty_name(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid characters"):
        create_folder(tmp_path, "")


# ---------------------------------------------------------------------------
# move_run
# ---------------------------------------------------------------------------


def test_move_run_moves_directory(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path, "run_to_move")
    target_parent = tmp_path / "experiment-A"
    target_parent.mkdir()

    move_run(run_dir, target_parent)

    assert not run_dir.exists()
    assert (target_parent / "run_to_move").is_dir()
    assert (target_parent / "run_to_move" / "manifest.json").is_file()


def test_move_run_creates_target_dir_if_missing(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path, "run_x")
    target_parent = tmp_path / "new_experiment"  # does not exist yet

    move_run(run_dir, target_parent)

    assert (target_parent / "run_x").is_dir()


def test_move_run_raises_if_no_manifest(tmp_path: Path):
    run_dir = tmp_path / "not_a_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="manifest.json"):
        move_run(run_dir, tmp_path / "target")


def test_move_run_raises_if_dest_exists(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path, "dup_run")
    target_parent = tmp_path / "dest"
    # Pre-create a collision.
    (target_parent / "dup_run").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        move_run(run_dir, target_parent)
