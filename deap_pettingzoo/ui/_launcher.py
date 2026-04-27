"""Subprocess launcher for the Streamlit UI.

Kept in a private module so ``deap_pettingzoo.ui.__init__`` stays free
of the heavy import side effects of pulling in ``streamlit`` itself —
the import only executes when :func:`run` is actually called.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


_UI_DIR = Path(__file__).resolve().parent
_APP_FILE = _UI_DIR / "app.py"


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

    Returns
    -------
    int
        The exit code of the ``streamlit`` subprocess.
    """
    _ensure_streamlit_installed()
    env = os.environ.copy()
    if results_root is not None:
        env["DEAP_RESULTS_ROOT"] = str(results_root)
    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP_FILE)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, env=env)


def main() -> int:
    """Entry point for ``python -m deap_pettingzoo.ui``.

    Forwards ``sys.argv[1:]`` to ``streamlit run`` so users can do, e.g.::

        python -m deap_pettingzoo.ui --server.port=8502
    """
    return run(extra_args=sys.argv[1:] or None)
