"""Reusable UI components for the experiment-setup and queue pages.

Modules:
* :mod:`.param_widgets` — map a :class:`~deap_pettingzoo.catalog.schema.ParamSpec`
  JSON dict to a Streamlit widget.
* :mod:`.forms` — tabbed dynamic-form scaffolding that produces
  ``(partial_dict, touched_keys)``.
* :mod:`.validation_view` — render a :class:`ValidationReport` grouped by tab.
"""

from __future__ import annotations

from deap_pettingzoo.ui.components.param_widgets import (
    WidgetContext,
    coerce_param,
    coerce_touched,
    render_param,
)
from deap_pettingzoo.ui.components.forms import (
    build_env_form,
    build_algo_form,
)
from deap_pettingzoo.ui.components.validation_view import (
    render_validation_report,
)

__all__ = [
    "WidgetContext",
    "build_algo_form",
    "build_env_form",
    "coerce_param",
    "coerce_touched",
    "render_param",
    "render_validation_report",
]
