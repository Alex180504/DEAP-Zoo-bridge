"""UI services — pure-Python helpers shared across pages.

Each module is import-safe without ``streamlit`` so it can be
unit-tested directly (cache decorators are no-ops outside Streamlit's
runtime).
"""

from __future__ import annotations

from deap_pettingzoo.ui.services.config import (
    DEFAULT_ROOT_ENV_VAR,
    UISettings,
    load_settings,
    pick_writable_output_dir,
    resolve_output_dir,
)
from deap_pettingzoo.ui.services.data import (
    discover_runs,
    load_experiment_cached,
)
from deap_pettingzoo.ui.services.management import (
    create_folder,
    delete_run,
    move_run,
    rename_run,
)
from deap_pettingzoo.ui.services.state import UIState

__all__ = [
    "DEFAULT_ROOT_ENV_VAR",
    "UISettings",
    "UIState",
    "create_folder",
    "delete_run",
    "discover_runs",
    "load_experiment_cached",
    "load_settings",
    "move_run",
    "pick_writable_output_dir",
    "rename_run",
    "resolve_output_dir",
]
