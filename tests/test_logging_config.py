"""Tests for the centralised diagnostic logging config
(:mod:`deap_pettingzoo.service.logging_config`)."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pytest

from deap_pettingzoo.service import logging_config as lc


@pytest.fixture(autouse=True)
def _restore_logging():
    """Snapshot/restore root logging + relevant env vars so configuring the
    global logging state in one test does not leak into the others."""
    import os

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_env = {
        k: os.environ.get(k)
        for k in ("DEAP_LOG_LEVEL", "DEAP_LOG_FORMAT", "DEAP_LOG_DIR", "DEAP_RESULTS_ROOT")
    }
    for k in saved_env:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for h in root.handlers[:]:
            if h not in saved_handlers:
                h.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        logging.captureWarnings(False)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Level / format resolution
# ---------------------------------------------------------------------------


class TestResolveLevel:
    def test_default_is_warning(self):
        assert lc.resolve_level() == logging.WARNING

    def test_verbose_count(self):
        assert lc.resolve_level(verbose=1) == logging.INFO
        assert lc.resolve_level(verbose=2) == logging.DEBUG

    def test_explicit_overrides_verbose_and_env(self, monkeypatch):
        monkeypatch.setenv("DEAP_LOG_LEVEL", "debug")
        assert lc.resolve_level(verbose=1, explicit="error") == logging.ERROR

    def test_env_used_when_no_args(self, monkeypatch):
        monkeypatch.setenv("DEAP_LOG_LEVEL", "info")
        assert lc.resolve_level() == logging.INFO

    def test_unknown_falls_back_to_default(self):
        assert lc.resolve_level(explicit="bogus") == logging.WARNING


class TestResolveFormat:
    def test_default_plain(self):
        assert lc.resolve_format() == "plain"

    def test_json(self):
        assert lc.resolve_format("json") == "json"

    def test_env(self, monkeypatch):
        monkeypatch.setenv("DEAP_LOG_FORMAT", "json")
        assert lc.resolve_format() == "json"


class TestDefaultLogDir:
    def test_explicit_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEAP_LOG_DIR", str(tmp_path / "logs"))
        assert lc.default_log_dir() == str(tmp_path / "logs")

    def test_under_results_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEAP_RESULTS_ROOT", str(tmp_path))
        assert lc.default_log_dir() == str(tmp_path / "_logs")


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_creates_rotating_file_and_dir(self, tmp_path):
        log_file = tmp_path / "nested" / "service.log"
        lc.configure_logging(log_level="info", log_file=str(log_file))

        root = logging.getLogger()
        assert root.level == logging.INFO
        # A RotatingFileHandler should be attached and its directory created.
        assert log_file.parent.is_dir()
        classes = {type(h).__name__ for h in root.handlers}
        assert "RotatingFileHandler" in classes
        assert "StreamHandler" in classes

        logging.getLogger("deap_pettingzoo.test").info("hello-info")
        for h in root.handlers:
            h.flush()
        assert "hello-info" in log_file.read_text(encoding="utf-8")

    def test_default_level_suppresses_info(self, tmp_path):
        log_file = tmp_path / "service.log"
        lc.configure_logging(log_file=str(log_file))  # default WARNING

        log = logging.getLogger("deap_pettingzoo.test")
        log.info("should-not-appear")
        log.warning("should-appear")
        for h in logging.getLogger().handlers:
            h.flush()
        text = log_file.read_text(encoding="utf-8")
        assert "should-not-appear" not in text
        assert "should-appear" in text

    def test_json_format_is_parseable(self, tmp_path):
        log_file = tmp_path / "service.log"
        lc.configure_logging(log_level="info", log_format="json", log_file=str(log_file))

        logging.getLogger("deap_pettingzoo.test").info("json-line")
        for h in logging.getLogger().handlers:
            h.flush()
        lines = [
            ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        record = json.loads(lines[-1])
        assert record["level"] == "INFO"
        assert record["msg"] == "json-line"
        assert record["logger"] == "deap_pettingzoo.test"

    def test_captures_warnings(self, tmp_path):
        log_file = tmp_path / "service.log"
        lc.configure_logging(log_level="warning", log_file=str(log_file))

        warnings.warn("captured-warning", RuntimeWarning)
        for h in logging.getLogger().handlers:
            h.flush()
        assert "captured-warning" in log_file.read_text(encoding="utf-8")

    def test_idempotent_no_handler_accumulation(self, tmp_path):
        log_file = tmp_path / "service.log"
        lc.configure_logging(log_level="info", log_file=str(log_file))
        first = len(logging.getLogger().handlers)
        lc.configure_logging(log_level="info", log_file=str(log_file))
        assert len(logging.getLogger().handlers) == first
