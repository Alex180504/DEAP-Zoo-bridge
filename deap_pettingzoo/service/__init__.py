"""DEAP-bridge experiment service — FastAPI sidecar for the UI overhaul.

Serves the catalog as JSON, manages experiment submission / queue lifecycle,
and provides live progress + stdout streaming.

Quick-start::

    python -m deap_pettingzoo.service [--port 8100] [--host 127.0.0.1]
"""

from __future__ import annotations

from typing import Optional


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 8100,
    results_root: Optional[str] = None,
    verbose: int = 0,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """Launch the FastAPI service programmatically."""
    import logging
    import os
    from pathlib import Path

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the service. "
            "Install with: pip install deap-pettingzoo[service]"
        ) from None

    from deap_pettingzoo.service.logging_config import (
        configure_logging,
        default_log_dir,
        resolve_format,
        resolve_level,
    )

    if results_root:
        os.environ["DEAP_RESULTS_ROOT"] = results_root

    # Resolve once and export to the environment so the runner subprocess
    # (spawned with the inherited env) logs at the same level / format.
    level = resolve_level(verbose, log_level)
    fmt = resolve_format(log_format)
    os.environ["DEAP_LOG_LEVEL"] = logging.getLevelName(level).lower()
    os.environ["DEAP_LOG_FORMAT"] = fmt

    log_file = str(Path(default_log_dir()) / "service.log")
    configure_logging(log_level=os.environ["DEAP_LOG_LEVEL"], log_format=fmt,
                      log_file=log_file)

    logger = logging.getLogger("deap_pettingzoo.service")
    logger.info(
        "Starting DEAP-bridge service on %s:%s (level=%s, format=%s, log_file=%s)",
        host, port, os.environ["DEAP_LOG_LEVEL"], fmt, log_file,
    )

    # log_config=None keeps our configuration intact (uvicorn would otherwise
    # re-apply its default dictConfig and clobber our handlers / levels).
    uvicorn.run(
        "deap_pettingzoo.service.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_config=None,
    )


__all__ = ["run"]
