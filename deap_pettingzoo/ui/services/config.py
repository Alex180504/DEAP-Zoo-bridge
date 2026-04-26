"""UI configuration — env var and default directory handling."""

from __future__ import annotations

import os
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
    """Resolve UI settings from CLI override → env var → default.

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
