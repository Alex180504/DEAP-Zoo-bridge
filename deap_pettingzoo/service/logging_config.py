"""Centralised diagnostic logging for the DEAP-bridge backend service.

This is *application / diagnostic* logging — exceptions, warnings, experiment
lifecycle, and which components are active.  It is deliberately **separate**
from :mod:`deap_pettingzoo.utils.logging`, which is the *experiment-results*
framework (CSV / Parquet / manifest sinks for metrics and checkpoints).

Two backend processes configure logging from here:

* the FastAPI app + dispatcher process (:func:`deap_pettingzoo.service.run`),
  which logs to **stdout** (captured by ``docker logs``) *and* a rotating
  ``<DEAP_LOG_DIR>/service.log`` file on the results volume, and
* the per-experiment **runner** subprocess (:func:`runner.main`), which logs to
  stdout only — the dispatcher already redirects that to
  ``<experiment-dir>/_stdout.log`` so it is retrievable per experiment.

Behaviour is driven by environment variables (so it works unchanged inside a
container) and by the ``--verbose`` CLI switch:

==================  =========================================================
``DEAP_LOG_LEVEL``  ``debug`` / ``info`` / ``warning`` (default) / ``error``
``DEAP_LOG_FORMAT`` ``plain`` (default) or ``json``
``DEAP_LOG_DIR``    directory for ``service.log`` (default
                    ``<DEAP_RESULTS_ROOT>/_logs``)
==================  =========================================================

Level semantics (the verbose switch):

* default ``WARNING`` — warnings (incl. captured ``warnings.warn``) and
  exceptions only.
* ``--verbose`` / ``-v`` → ``INFO`` — also normal operation: experiment
  lifecycle and which components are active.
* ``-vv`` / ``--log-level debug`` → ``DEBUG`` — also per-request / resolved
  hyperparameters / config-merge detail.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "JsonFormatter",
    "build_dict_config",
    "configure_logging",
    "default_log_dir",
    "resolve_format",
    "resolve_level",
]

DEFAULT_LEVEL = logging.WARNING

_LEVEL_NAMES = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


def resolve_level(
    verbose: Optional[int] = None, explicit: Optional[str] = None
) -> int:
    """Resolve the effective root log level.

    Precedence: ``explicit`` (e.g. ``--log-level``) > ``verbose`` count
    (``-v`` → INFO, ``-vv`` → DEBUG) > ``DEAP_LOG_LEVEL`` env > default WARNING.
    Unknown names fall back to the default.
    """
    if explicit:
        return _LEVEL_NAMES.get(explicit.strip().lower(), DEFAULT_LEVEL)
    if verbose:
        return logging.DEBUG if verbose >= 2 else logging.INFO
    env = os.environ.get("DEAP_LOG_LEVEL")
    if env:
        return _LEVEL_NAMES.get(env.strip().lower(), DEFAULT_LEVEL)
    return DEFAULT_LEVEL


def resolve_format(fmt: Optional[str] = None) -> str:
    """Return ``"json"`` or ``"plain"`` from arg or ``DEAP_LOG_FORMAT`` env."""
    value = (fmt or os.environ.get("DEAP_LOG_FORMAT") or "plain").strip().lower()
    return "json" if value == "json" else "plain"


def default_log_dir() -> str:
    """Directory for ``service.log`` — ``DEAP_LOG_DIR`` or
    ``<DEAP_RESULTS_ROOT>/_logs`` (which sits on the results volume)."""
    explicit = os.environ.get("DEAP_LOG_DIR")
    if explicit:
        return explicit
    results_root = os.environ.get("DEAP_RESULTS_ROOT", str(Path("results")))
    return str(Path(results_root) / "_logs")


def build_dict_config(
    level: int, fmt: str, log_file: Optional[str]
) -> Dict[str, Any]:
    """Build a :func:`logging.config.dictConfig` dict.

    ``disable_existing_loggers`` is ``False`` so module loggers created at
    import time keep working.  The ``uvicorn*`` loggers are given no handlers of
    their own and ``propagate=True`` so their records flow through our root
    handlers (sharing the format + level); pass ``log_config=None`` to
    ``uvicorn.run`` so uvicorn does not re-install its default config on top.
    """
    if fmt == "json":
        formatter: Dict[str, Any] = {
            "()": f"{__name__}.JsonFormatter",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    else:
        formatter = {
            "format": "%(asctime)s %(levelname)-7s %(name)s  %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }

    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
            "level": level,
        }
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": int(
                os.environ.get("DEAP_LOG_FILE_MAX_BYTES", 10 * 1024 * 1024)
            ),
            "backupCount": int(os.environ.get("DEAP_LOG_FILE_BACKUPS", 3)),
            "encoding": "utf-8",
            "formatter": "default",
            "level": level,
        }

    handler_names = list(handlers)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "handlers": handlers,
        "root": {"level": level, "handlers": handler_names},
        "loggers": {
            # Our own package — inherit the root handlers.
            "deap_pettingzoo": {"level": level, "handlers": [], "propagate": True},
            # uvicorn: route through our handlers, do not double-emit.
            "uvicorn": {"level": level, "handlers": [], "propagate": True},
            "uvicorn.error": {"level": level, "handlers": [], "propagate": True},
            "uvicorn.access": {"level": level, "handlers": [], "propagate": True},
            # warnings.warn(...) records (see logging.captureWarnings).
            "py.warnings": {
                "level": logging.WARNING,
                "handlers": [],
                "propagate": True,
            },
        },
    }


def configure_logging(
    *,
    verbose: Optional[int] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Configure root logging for a backend process and return the dict config.

    Idempotent — :func:`logging.config.dictConfig` rebuilds the root handlers
    each call rather than accumulating them.  Also enables
    :func:`logging.captureWarnings` so every ``warnings.warn(...)`` in the stack
    is routed to the ``py.warnings`` logger at ``WARNING``.

    The runner subprocess passes ``log_file=None`` (console only — its stdout is
    captured to ``_stdout.log``); the app process passes the path to
    ``service.log``.
    """
    level = resolve_level(verbose, log_level)
    fmt = resolve_format(log_format)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    config = build_dict_config(level, fmt, log_file)
    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    return config
