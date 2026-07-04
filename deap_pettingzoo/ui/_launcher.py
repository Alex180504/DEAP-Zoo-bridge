"""Subprocess launcher for the Streamlit UI.

Kept in a private module so ``deap_pettingzoo.ui.__init__`` stays free
of the heavy import side effects of pulling in ``streamlit`` itself, so that
the import only executes when :func:`run` is actually called.

Supports two modes:

1. **Frontend-only** (default) — ``streamlit run`` against the bundled
   :file:`app.py`. The Setup/Queue pages will connect to ``FASTAPI_URL``
   (default ``http://127.0.0.1:8100``). Start the service separately with
   ``python -m deap_pettingzoo.service``.

2. **Integrated** (``--with-service``) — spawns the FastAPI service as a
   background process, sets ``FASTAPI_URL`` accordingly, and launches
   Streamlit. The service is automatically terminated when Streamlit exits.
   ``--service-port PORT`` overrides the default (8100).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_UI_DIR = Path(__file__).resolve().parent
_APP_FILE = _UI_DIR / "app.py"
_DEFAULT_SERVICE_PORT = 8100


class StreamlitNotInstalled(RuntimeError):
    """Raised when ``streamlit`` is not importable.

    The UI extra is opt-in (``pip install deap-pettingzoo[ui]``) — this
    error guides the user to install it without crashing the rest of the
    package.
    """


def _ensure_streamlit_installed() -> None:
    try:
        import streamlit  # noqa: F401  (just probing import)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise StreamlitNotInstalled(
            "The UI requires the optional 'ui' extra. Install with:\n"
            "    pip install deap-pettingzoo[ui]"
        ) from exc


def run(
    results_root: Optional[str] = None,
    *,
    extra_args: Optional[Sequence[str]] = None,
    with_service: bool = False,
    service_port: int = _DEFAULT_SERVICE_PORT,
) -> int:
    """Launch ``streamlit run`` against the bundled :file:`app.py`.

    Parameters
    ----------
    results_root:
        Initial value for ``DEAP_RESULTS_ROOT`` — the env var consumed
        by :mod:`deap_pettingzoo.ui.services.config`. When ``None``,
        the existing environment is left untouched.
    extra_args:
        Additional CLI arguments forwarded to the ``streamlit run``
        invocation (after the app path), e.g. ``["--server.port=8502"]``.
    with_service:
        When ``True`` and ``FASTAPI_URL`` is not already set in the
        environment, the FastAPI experiment service is spawned as a
        background subprocess. It is automatically terminated when
        Streamlit exits.
    service_port:
        Port for the auto-started service (ignored unless
        ``with_service=True`` and ``FASTAPI_URL`` is unset).

    Returns
    -------
    int
        The exit code of the ``streamlit`` subprocess.
    """
    _ensure_streamlit_installed()
    env = os.environ.copy()
    if results_root is not None:
        env["DEAP_RESULTS_ROOT"] = str(results_root)

    service_proc = None
    if with_service and not env.get("FASTAPI_URL"):
        print(
            f"[deap-bridge] starting experiment service on port {service_port} ...",
            flush=True,
        )
        try:
            service_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "deap_pettingzoo.service",
                    "--port",
                    str(service_port),
                    "--host",
                    "127.0.0.1",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(
                f"[deap-bridge] WARNING: could not start service: {exc}",
                flush=True,
            )
            service_proc = None
        else:
            env["FASTAPI_URL"] = f"http://127.0.0.1:{service_port}"
            # Give the service a moment to bind.
            time.sleep(1.5)

    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP_FILE)]
    if extra_args:
        cmd.extend(extra_args)

    exit_code = subprocess.call(cmd, env=env)

    if service_proc is not None:
        print("[deap-bridge] stopping experiment service ...", flush=True)
        service_proc.terminate()
        try:
            service_proc.wait(timeout=5)
        except subprocess.TimeoutError:
            service_proc.kill()
        print("[deap-bridge] service stopped.", flush=True)

    return exit_code


def main() -> int:
    """Entry point for ``python -m deap_pettingzoo.ui``.

    Forwards ``sys.argv[1:]`` to ``streamlit run`` so users can do, e.g.::

        python -m deap_pettingzoo.ui --server.port=8502 --with-service

    Recognised flags (consumed by the launcher, not forwarded to Streamlit):

    * ``--with-service`` — auto-start the FastAPI service in the background.
    * ``--service-port PORT`` — override the default service port (8100).
    """
    args = list(sys.argv[1:]) if len(sys.argv) > 1 else []
    with_service = False
    service_port = _DEFAULT_SERVICE_PORT

    # Consume launcher-specific flags.
    extra: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--with-service":
            with_service = True
            i += 1
        elif args[i] == "--service-port" and i + 1 < len(args):
            try:
                service_port = int(args[i + 1])
            except ValueError:
                print(
                    f"[deap-bridge] ignoring non-integer --service-port {args[i+1]!r}",
                    flush=True,
                )
            i += 2
        else:
            extra.append(args[i])
            i += 1

    return run(
        extra_args=extra or None,
        with_service=with_service,
        service_port=service_port,
    )
