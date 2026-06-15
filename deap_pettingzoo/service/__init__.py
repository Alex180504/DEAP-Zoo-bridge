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
) -> None:
    """Launch the FastAPI service programmatically."""
    import os
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the service. "
            "Install with: pip install deap-pettingzoo[service]"
        ) from None

    if results_root:
        os.environ["DEAP_RESULTS_ROOT"] = results_root

    uvicorn.run(
        "deap_pettingzoo.service.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="info",
    )


__all__ = ["run"]
