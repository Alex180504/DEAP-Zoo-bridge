"""Tests for the export/import utilities (Phase 2 of export/import feature).

Covers: export manifest, run ZIP creation, experiment ZIP streaming,
import validation (valid, missing manifest, path traversal, corrupted,
schema mismatch), import extraction, DB restore, and spec conversion.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from deap_pettingzoo.service.export_import import (
    EXPORT_MANIFEST_SCHEMA_VERSION,
    ImportValidationError,
    ImportValidationResult,
    build_export_manifest,
    create_run_export_zip,
    export_filename,
    extract_import_zip,
    validate_import_zip,
)
from deap_pettingzoo.ui.services.spec_utils import (
    extract_loaded_spec,
    spec_to_form_defaults,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_results_root():
    """Temporary directory to act as DEAP_RESULTS_ROOT."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_run_dir(tmp_results_root):
    """Create a minimal run directory with manifest.json and metrics.csv."""
    run_dir = tmp_results_root / "ea_simple_spread_s0_20250101T000000_a1b2c3"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "ea_simple_spread_s0_20250101T000000_a1b2c3",
        "algorithm": "ea",
        "env_name": "simple_spread_v3",
        "seed": 0,
        "started_at": "2025-01-01T00:00:00Z",
        "schema_version": 1,
        "env_config": {},
        "hyperparams": {"population_size": 10},
        "budget_mode": "eval_episodes",
        "budget_value": 100,
        "notes": "",
        "environment": {"python": "3.11"},
        "artifacts": {},
        "final_summary": {"best_fitness": 95.5},
        "wall_time_s": 12.3,
        "finished_at": "2025-01-01T00:00:12Z",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.csv").write_text("iteration,fitness\n0,10\n1,20\n")
    (run_dir / "summary.json").write_text('{"best_fitness": 95.5}')
    return run_dir


@pytest.fixture
def sample_run_dir_with_frames(sample_run_dir):
    """Run dir with optional artifacts."""
    frames_dir = sample_run_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    tb_dir = sample_run_dir / "tb"
    tb_dir.mkdir()
    (tb_dir / "events.out.tfevents").write_text("tensorboard_data")
    (sample_run_dir / "trajectories.parquet").write_bytes(b"PAR1" + b"\x00" * 20)
    return sample_run_dir


@pytest.fixture
def sample_db_record():
    return {
        "id": "abcd1234efgh5678",
        "label": "test experiment",
        "status": "completed",
        "spec": {
            "environment": "simple_spread_v3",
            "algorithms": ["ea"],
            "budget": {"mode": "eval_episodes", "value": 100, "num_seeds": 1},
        },
        "created_at": "2025-01-01T00:00:00Z",
        "started_at": "2025-01-01T00:00:01Z",
        "finished_at": "2025-01-01T00:00:13Z",
        "run_dirs": [],
        "origin": "ui",
    }


# ---------------------------------------------------------------------------
# Export manifest tests
# ---------------------------------------------------------------------------


class TestExportManifest:
    def test_structure(self):
        m = build_export_manifest(
            exp_id="test123",
            label="my experiment",
            run_dirs=["/path/to/run1", "/path/to/run2"],
            include_frames=True,
            include_tb=False,
            include_trajectories=True,
        )
        assert m["schema_version"] == EXPORT_MANIFEST_SCHEMA_VERSION
        assert m["experiment_id"] == "test123"
        assert m["label"] == "my experiment"
        assert len(m["run_dirs"]) == 2
        assert m["artifact_flags"]["frames"] is True
        assert m["artifact_flags"]["tensorboard"] is False
        assert m["artifact_flags"]["trajectories"] is True
        assert "exported_at" in m
        assert "deap_version" in m

    def test_optional_spec(self):
        spec = {"environment": "simple_tag_v3"}
        m = build_export_manifest(
            exp_id="x", label="", run_dirs=[], include_frames=False,
            include_tb=False, include_trajectories=False, spec=spec,
        )
        assert m["spec"] == spec

    def test_no_spec(self):
        m = build_export_manifest(
            exp_id="x", label="", run_dirs=[], include_frames=False,
            include_tb=False, include_trajectories=False,
        )
        assert m["spec"] is None


# ---------------------------------------------------------------------------
# Run ZIP creation tests
# ---------------------------------------------------------------------------


class TestRunExportZip:
    def test_always_includes_core_files(self, sample_run_dir):
        data = create_run_export_zip(sample_run_dir)
        assert isinstance(data, bytes)
        assert len(data) > 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "deap_export_manifest.json" in names
            # Core files should be present under the run dir name.
            run_name = sample_run_dir.name
            assert f"{run_name}/manifest.json" in names
            assert f"{run_name}/metrics.csv" in names
            assert f"{run_name}/summary.json" in names

    def test_excludes_frames_by_default(self, sample_run_dir_with_frames):
        data = create_run_export_zip(sample_run_dir_with_frames)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            run_name = sample_run_dir_with_frames.name
            assert f"{run_name}/frames/frame_0000.png" not in names
            assert f"{run_name}/tb/events.out.tfevents" not in names
            assert f"{run_name}/trajectories.parquet" not in names

    def test_includes_frames_when_flagged(self, sample_run_dir_with_frames):
        data = create_run_export_zip(
            sample_run_dir_with_frames, include_frames=True,
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            run_name = sample_run_dir_with_frames.name
            assert f"{run_name}/frames/frame_0000.png" in names

    def test_includes_tb_when_flagged(self, sample_run_dir_with_frames):
        data = create_run_export_zip(
            sample_run_dir_with_frames, include_tb=True,
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            run_name = sample_run_dir_with_frames.name
            assert f"{run_name}/tb/events.out.tfevents" in names

    def test_includes_trajectories_when_flagged(self, sample_run_dir_with_frames):
        data = create_run_export_zip(
            sample_run_dir_with_frames, include_trajectories=True,
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            run_name = sample_run_dir_with_frames.name
            assert f"{run_name}/trajectories.parquet" in names

    def test_empty_dir_does_not_crash(self, tmp_results_root):
        empty_dir = tmp_results_root / "empty_run"
        empty_dir.mkdir()
        data = create_run_export_zip(empty_dir)
        # Should produce a valid zip with just the manifest.
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "deap_export_manifest.json" in zf.namelist()


# ---------------------------------------------------------------------------
# Export filename tests
# ---------------------------------------------------------------------------


class TestExportFilename:
    def test_with_label(self):
        name = export_filename("abc123", "my cool experiment")
        assert name.startswith("deap_export_abc123")
        assert name.endswith(".zip")
        assert "my cool experiment" in name

    def test_without_label(self):
        name = export_filename("abc123", "")
        assert "abc123" in name
        assert name.endswith(".zip")

    def test_sanitises_special_chars(self):
        name = export_filename("abc123", "exp / with : weird * chars")
        assert "/" not in name
        assert ":" not in name
        assert "*" not in name


# ---------------------------------------------------------------------------
# Import validation tests
# ---------------------------------------------------------------------------


def _make_valid_export_zip(dest: Path, run_dirs=None) -> Path:
    """Create a minimal valid DEAP export ZIP at *dest*."""
    if run_dirs is None:
        run_dirs = ["ea_simple_spread_s0_20250101T000000_a1b2c3"]
    zip_path = dest / "test_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        manifest = {
            "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            "deap_version": "0.2.0",
            "exported_at": "2025-01-01T00:00:00Z",
            "experiment_id": "test123",
            "label": "test",
            "run_dirs": run_dirs,
            "artifact_flags": {"frames": False, "tensorboard": False, "trajectories": False},
        }
        zf.writestr("deap_export_manifest.json", json.dumps(manifest))
        zf.writestr("experiment.json", json.dumps({"id": "test123", "label": "test"}))
        for rd in run_dirs:
            run_manifest = {"run_id": rd, "algorithm": "ea", "env_name": "simple_spread_v3"}
            zf.writestr(f"{rd}/manifest.json", json.dumps(run_manifest))
    return zip_path


class TestValidateImportZip:
    def test_valid_zip_passes(self, tmp_results_root):
        zip_path = _make_valid_export_zip(tmp_results_root)
        result = validate_import_zip(zip_path, tmp_results_root)
        assert result.ok
        assert len(result.errors) == 0
        assert result.run_count == 1

    def test_not_a_zip(self, tmp_results_root):
        fake = tmp_results_root / "not_a_zip.zip"
        fake.write_text("this is not a zip file")
        result = validate_import_zip(fake, tmp_results_root)
        assert not result.ok
        assert any(e.code == "not_a_zip" for e in result.errors)

    def test_missing_manifest(self, tmp_results_root):
        zip_path = tmp_results_root / "no_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("some_file.txt", "hello")
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "missing_manifest" for e in result.errors)

    def test_schema_version_too_high(self, tmp_results_root):
        zip_path = tmp_results_root / "future_export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {"schema_version": 999, "run_dirs": []}
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "schema_version_mismatch" for e in result.errors)

    def test_path_traversal_rejected(self, tmp_results_root):
        zip_path = tmp_results_root / "traversal.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "run_dirs": ["../etc/passwd"],
            }
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))
            zf.writestr("../etc/passwd/manifest.json", "{}")
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "path_traversal" for e in result.errors)

    def test_absolute_path_rejected(self, tmp_results_root):
        zip_path = tmp_results_root / "absolute.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "run_dirs": ["/etc/passwd"],
            }
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))
            zf.writestr("/etc/passwd/manifest.json", "{}")
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "path_traversal" for e in result.errors)

    def test_run_missing_manifest_warns(self, tmp_results_root):
        zip_path = tmp_results_root / "missing_run_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "run_dirs": ["some_run"],
            }
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))
            zf.writestr("experiment.json", "{}")
            # No some_run/manifest.json written.
        result = validate_import_zip(zip_path, tmp_results_root)
        # No valid runs → should error.
        assert not result.ok
        assert any(e.code == "no_valid_runs" for e in result.errors)
        assert any(w.code == "run_missing_manifest" for w in result.warnings)

    def test_manifest_parse_error(self, tmp_results_root):
        zip_path = tmp_results_root / "bad_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("deap_export_manifest.json", "not valid json {{{")
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "manifest_parse_error" for e in result.errors)

    def test_too_large_zip_bomb(self, tmp_results_root, monkeypatch):
        """Check that estimated extraction size beyond the safety limit is caught."""
        zip_path = tmp_results_root / "big.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "run_dirs": [],
            }
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))

        # Monkey-patch _MAX_EXTRACT_BYTES to a tiny value so the check fires.
        monkeypatch.setattr(
            "deap_pettingzoo.service.export_import._MAX_EXTRACT_BYTES", 1
        )
        # Even a tiny zip will exceed a 1-byte limit.
        result = validate_import_zip(zip_path, tmp_results_root)
        assert not result.ok
        assert any(e.code == "too_large" for e in result.errors)


# ---------------------------------------------------------------------------
# Import extraction tests
# ---------------------------------------------------------------------------


class TestExtractImportZip:
    def test_extracts_correctly(self, tmp_results_root):
        zip_path = _make_valid_export_zip(tmp_results_root)
        extracted = extract_import_zip(zip_path, tmp_results_root)
        assert len(extracted) == 1
        assert extracted[0]["experiment_id"] == "test123"
        assert len(extracted[0]["run_dirs"]) == 1
        # Verify manifest.json was extracted.
        run_dir = Path(extracted[0]["run_dirs"][0])
        assert (run_dir / "manifest.json").exists()

    def test_skip_existing_duplicate(self, tmp_results_root):
        zip_path = _make_valid_export_zip(tmp_results_root)
        # First extraction.
        extract_import_zip(zip_path, tmp_results_root)
        # Second extraction with overwrite=False.
        extracted2 = extract_import_zip(zip_path, tmp_results_root, overwrite_existing=False)
        # Should warn about duplicate.
        assert len(extracted2[0].get("warnings", [])) > 0

    def test_overwrite_existing(self, tmp_results_root):
        zip_path = _make_valid_export_zip(tmp_results_root)
        extract_import_zip(zip_path, tmp_results_root)
        extracted2 = extract_import_zip(zip_path, tmp_results_root, overwrite_existing=True)
        # Should succeed without warnings.
        assert len(extracted2[0].get("warnings", [])) == 0
        assert len(extracted2[0]["run_dirs"]) == 1

    def test_path_traversal_skipped_during_extraction(self, tmp_results_root):
        zip_path = tmp_results_root / "traversal_extract.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            manifest = {
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
                "run_dirs": ["safe_run"],
            }
            zf.writestr("deap_export_manifest.json", json.dumps(manifest))
            zf.writestr("safe_run/manifest.json", '{"run_id":"safe"}')
            # Add a traversal path.
            zf.writestr("../escape/manifest.json", '{"run_id":"escape"}')
        extracted = extract_import_zip(zip_path, tmp_results_root)
        # Only the safe run should be extracted.
        assert len(extracted[0].get("warnings", [])) >= 1


# ---------------------------------------------------------------------------
# Spec-to-form conversion tests
# ---------------------------------------------------------------------------


class TestSpecToFormDefaults:
    def test_basic_conversion(self):
        spec = {
            "environment": "simple_spread_v3",
            "algorithms": ["ea", "cma-es"],
            "env_config": {"max_cycles": 25},
            "hyperparameters": {
                "ea": {"population_size": 50},
                "cma-es": {"sigma": 0.5},
            },
            "budget": {"mode": "eval_episodes", "value": 200, "num_seeds": 3},
            "fitness": {"aggregation_method": "mean"},
            "recording": {"enabled": True, "fps": 30, "fmt": "mp4"},
            "parallelism": {"num_workers": 4, "parallel_backend": "process"},
            "coevolution": {"enabled": False},
            "raw_config": {},
            "label": "my test",
        }
        form = spec_to_form_defaults(spec)
        assert form["env"] == "simple_spread_v3"
        assert form["algos"] == ["ea", "cma-es"]
        assert form["env_values"] == {"max_cycles": 25}
        assert form["algo_values"]["ea"]["population_size"] == 50
        assert form["algo_values"]["cma-es"]["sigma"] == 0.5
        assert form["budget"]["mode"] == "eval_episodes"
        assert form["budget"]["value"] == 200
        assert form["budget"]["num_seeds"] == 3
        assert form["fitness"]["aggregation_method"] == "mean"
        assert form["recording"]["enabled"] is True
        assert form["recording"]["fps"] == 30
        assert form["parallelism"]["num_workers"] == 4
        assert form["coevolution"]["enabled"] is False

    def test_empty_spec_defaults(self):
        form = spec_to_form_defaults({})
        assert form["env"] == ""
        assert form["algos"] == []
        assert form["budget"]["mode"] == "env_steps"
        assert form["budget"]["value"] == 50000
        assert form["raw_text"] == ""

    def test_raw_config_to_text(self):
        spec = {"raw_config": {"wrapper": "some.module.Wrapper", "param": 42}}
        form = spec_to_form_defaults(spec)
        assert "some.module.Wrapper" in form["raw_text"]
        assert "42" in form["raw_text"]

    def test_returns_serializable(self):
        spec = {
            "environment": "simple_spread_v3",
            "algorithms": ["ea"],
            "budget": {"mode": "env_steps", "value": 1000},
        }
        form = spec_to_form_defaults(spec)
        # Must be JSON-serializable.
        json.dumps(form)


class TestExtractLoadedSpec:
    def test_extracts_valid_spec(self):
        spec = {"environment": "simple_spread_v3", "algorithms": ["ea"]}
        setup_config = json.dumps({
            "env": "simple_spread_v3",
            "algos": ["ea"],
            "_loaded_spec": spec,
        })
        result = extract_loaded_spec(setup_config)
        assert result == spec

    def test_returns_none_for_missing_spec(self):
        setup_config = json.dumps({"env": "simple_spread_v3", "algos": ["ea"]})
        result = extract_loaded_spec(setup_config)
        assert result is None

    def test_returns_none_for_invalid_json(self):
        assert extract_loaded_spec("not json") is None

    def test_returns_none_for_none_input(self):
        assert extract_loaded_spec(None) is None

    def test_rejects_spec_without_environment(self):
        setup_config = json.dumps({"_loaded_spec": {"algorithms": ["ea"]}})
        result = extract_loaded_spec(setup_config)
        assert result is None


# ---------------------------------------------------------------------------
# Streaming export generator test
# ---------------------------------------------------------------------------


class TestExperimentExportStreaming:
    def test_generator_produces_valid_zip(self, tmp_results_root, sample_db_record):
        from deap_pettingzoo.service.export_import import (
            create_experiment_export_zip_generator,
        )
        # Create experiment dir structure.
        exp_dir = tmp_results_root / "_experiments" / "test123"
        exp_dir.mkdir(parents=True)
        (exp_dir / "_spec.json").write_text('{"environment":"simple_spread_v3"}')
        (exp_dir / "_result.json").write_text('{"status":"completed","run_dirs":[]}')

        # Create a sub-run.
        run_dir = exp_dir / "ea" / "seed_0" / "ea_simple_s0_20250101_a1b2c3"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": "ea_simple_s0_20250101_a1b2c3",
            "algorithm": "ea",
            "env_name": "simple_spread_v3",
        }))
        (run_dir / "metrics.csv").write_text("iteration,fitness\n0,10\n")

        gen = create_experiment_export_zip_generator(
            exp_id="test123",
            db_record=sample_db_record,
            experiment_dir=exp_dir,
            run_dirs=[run_dir],
            include_frames=False,
            include_tb=False,
            include_trajectories=False,
        )
        # Collect all chunks into a temp file (streaming writes to BytesIO
        # piecewise — the caller (StreamingResponse) streams them to the client).
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            for chunk in gen:
                tmp.write(chunk)
            tmp_path = Path(tmp.name)

        try:
            # Verify it's a valid zip.
            assert zipfile.is_zipfile(tmp_path)
            with zipfile.ZipFile(tmp_path) as zf:
                names = zf.namelist()
                assert "deap_export_manifest.json" in names
                assert "experiment.json" in names
                assert "_spec.json" in names
                assert "_result.json" in names
                # Run dir should be preserved relative to experiment_dir.
                rel = str(run_dir.relative_to(exp_dir)).replace("\\", "/")
                assert f"{rel}/manifest.json" in names
                assert f"{rel}/metrics.csv" in names
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
