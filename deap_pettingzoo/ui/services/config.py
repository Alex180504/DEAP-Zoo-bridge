"""UI configuration — env var and default directory handling."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Environment variable that overrides the default results root.
DEFAULT_ROOT_ENV_VAR = "DEAP_RESULTS_ROOT"

#: Directory checked when the env var is unset and no CLI override
#: was given.
DEFAULT_RESULTS_ROOT = Path("results")


@dataclass(frozen=True)
class UISettings:
    """Resolved UI settings.

    Attributes
    ----------
    results_root:
        Directory the UI scans for run subdirectories. Defaults to
        ``$DEAP_RESULTS_ROOT`` if set, else ``./results``.
    page_title:
        Browser tab title used by :file:`app.py`.
    """

    results_root: Path
    page_title: str = "DEAP-bridge results"

    @property
    def results_root_exists(self) -> bool:
        return self.results_root.is_dir()


def load_settings(*, results_root: Optional[str] = None) -> UISettings:
    """Resolve UI settings from CLI override -> env var -> default.

    Parameters
    ----------
    results_root:
        Optional explicit override; takes precedence over the env var.
    """
    if results_root:
        root = Path(results_root)
    else:
        env_value = os.environ.get(DEFAULT_ROOT_ENV_VAR)
        root = Path(env_value) if env_value else DEFAULT_RESULTS_ROOT
    return UISettings(results_root=root.expanduser().resolve())


def resolve_output_dir(raw: str | Path, results_root: Path) -> Path:
    """Resolve a report output directory against ``results_root``.

    Relative paths are anchored to the results root rather than the
    process working directory, which is unreliable when the UI is
    launched from a protected or read-only location (e.g. the Docker UI
    image, where ``/app`` is root-owned). Absolute paths and ``~`` are
    honoured as-is.
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = results_root / path
    return path.resolve()


def pick_writable_output_dir(requested: Path) -> tuple[Path, bool]:
    """Return a directory that can actually be written to.

    Returns ``(directory, fell_back)``. When ``requested`` cannot be
    created or probed for writes — e.g. the Docker UI mounts the results
    root read-only, so ``/app/results/reports`` raises ``EROFS`` — fall
    back to a fresh temporary directory so the inline report preview still
    works.
    """
    try:
        requested.mkdir(parents=True, exist_ok=True)
        probe = requested / ".deap-write-probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
    except OSError:
        return Path(tempfile.mkdtemp(prefix="deap-report-")), True
    return requested, False
