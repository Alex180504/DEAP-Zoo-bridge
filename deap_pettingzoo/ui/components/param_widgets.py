"""Map a :class:`~deap_pettingzoo.catalog.schema.ParamSpec` JSON dict to a
Streamlit widget, with pure Python coercion helpers for unit testing.

The ``render_param`` function is the **only** function that imports
``streamlit``. ``coerce_param`` and ``coerce_touched`` are streamlit-free
and can be tested without a Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Streamlit-free Python coercion helpers (unit-testable)

# Parameter type names expected in the JSON from the catalog
_PARAM_TYPE_COERCERS: Dict[str, Callable[[Any], Any]] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


def coerce_param(param_spec: Dict[str, Any], value: Any) -> Any:
    """Convert a user-supplied value to the param's declared Python type.

    Handles the round-trip edge cases a form produces: ``st.number_input``
    returns ``int``/``float`` natively, ``st.selectbox`` returns the choice
    value directly, ``st.checkbox`` returns ``bool``. The conversion is
    idempotent — calling it twice with an already-correct value is a no-op.

    Args:
        param_spec: A :class:`ParamSpec` dict from the catalog JSON
            (containing at least ``"type"`` and ``"nullable"``).
        value: The raw widget output.

    Returns:
        The coerced value, or ``None`` if ``nullable`` is ``True`` and
        ``value`` is ``None`` / empty string.

    Raises:
        ValueError: If the value cannot be coerced.
    """
    if value is None or value == "":
        if param_spec.get("nullable", False):
            return None
    ptype = param_spec.get("type", "str")
    if ptype == "enum":
        choices = param_spec.get("choices", [])
        if value in choices:
            return value
        # Try string match
        str_val = str(value)
        for c in choices:
            if str(c) == str_val:
                return c
        if choices and value is not None:
            raise ValueError(
                f"Expected one of {choices!r}, got {value!r}"
            )
        return value
    if ptype == "list":
        if isinstance(value, list):
            return value
        if isinstance(value, (str, bytes)):
            raise ValueError(f"Expected a list, got a string")
        try:
            return list(value)
        except TypeError:
            raise ValueError(f"Expected a list, got {value!r}")
    if ptype in _PARAM_TYPE_COERCERS:
        if ptype == "int" and isinstance(value, bool):
            raise ValueError(f"Expected an integer, got bool")
        return _PARAM_TYPE_COERCERS[ptype](value)
    return value


def _is_default(param_spec: Dict[str, Any], value: Any) -> bool:
    """Return ``True`` when ``value`` equals the spec's default."""
    default = param_spec.get("default")
    nullable = param_spec.get("nullable", False)
    if default is None and nullable and value is None:
        return True
    if default is None and not nullable:
        return value is None
    try:
        return coerce_param(param_spec, value) == coerce_param(param_spec, default)
    except (ValueError, TypeError):
        return False


def coerce_touched(
    param_specs: List[Dict[str, Any]],
    values: Dict[str, Any],
) -> Tuple[Dict[str, Any], Set[str]]:
    """Given a list of param specs and a dict of raw form values, return a
    tuple ``(coerced_dict, touched_keys)``.

    * ``coerced_dict`` contains only non-default, not-None, well-typed
      entries.  This is exactly the payload the Setup page sends to the
      backend - no default values are transmitted (the catalog fills them).
    * ``touched_keys`` is the set of param names the user explicitly changed
      away from the default.  Used for the 3-layer merge: explicit form
      beats raw config.
    """
    coerced: Dict[str, Any] = {}
    touched: Set[str] = set()
    for spec in param_specs:
        name = spec["name"]
        if name not in values:
            continue
        raw = values[name]
        try:
            cv = coerce_param(spec, raw)
        except (ValueError, TypeError):
            cv = raw
        if not _is_default(spec, cv):
            touched.add(name)
            coerced[name] = cv
    return coerced, touched


# WidgetContext


@dataclass
class WidgetContext:
    """Mutable bag of form-local state shared across widget renders.

    Attributes:
        disabled_reason: When non-empty, every widget is disabled and this
            string is shown as a tooltip / caption.
        touched: Accumulated set of explicitly-changed param names.
        values: Accumulated coerced values (propagated per-tab).
        prefix: Namespace prefix included in every widget key so that two
            contexts rendered on the same page (e.g. EA and CMA-ES forms side
            by side) never produce duplicate Streamlit element IDs even when
            they share parameter names like ``population_size``.
    """

    disabled_reason: str = ""
    touched: Set[str] = field(default_factory=set)
    values: Dict[str, Any] = field(default_factory=dict)
    prefix: str = ""
    _counter: int = 0

    def _key(self, name: str) -> str:
        self._counter += 1
        ns = f"{self.prefix}_" if self.prefix else ""
        return f"_w_{ns}{self._counter}_{name}"


# render_param (Streamlit-dependent)


def render_param(
    param_spec: Dict[str, Any],
    ctx: WidgetContext,
    *,
    current_value: Any = None,
) -> Any:
    """Render a single parameter as a Streamlit widget.

    This is the **only** function in this module that imports ``streamlit``.

    Args:
        param_spec: :class:`ParamSpec` dict from the catalog JSON.
        ctx: Shared :class:`WidgetContext` carrying ``disabled_reason``,
            ``touched``, and ``values``.
        current_value: Previously-set value (for restoring from stored config
            or the raw config overlay).

    Returns:
        The widget's current value (already coerced).
    """
    import streamlit as st  # noqa: WPS433 / F811  (lazy, UI context)

    name = param_spec["name"]
    label = param_spec.get("label", name.replace("_", " ").title())
    help_text = param_spec.get("help", "")
    ptype = param_spec.get("type", "str")
    disabled = bool(ctx.disabled_reason)
    key = ctx._key(name)

    # Resolve the initial value: stored > current_value > default
    stored = ctx.values.get(name)
    if stored is not None:
        current_value = stored
    if current_value is None and not param_spec.get("nullable"):
        current_value = param_spec.get("default")
    # Coerce for display (handle None gracefully)
    try:
        display_value = coerce_param(param_spec, current_value)
    except (ValueError, TypeError):
        display_value = current_value

    if ptype == "int":
        minimum = param_spec.get("minimum")
        maximum = param_spec.get("maximum")
        step = param_spec.get("step", 1)
        v = st.number_input(
            label,
            value=int(display_value) if display_value is not None else int(minimum or 0),
            min_value=minimum,
            max_value=maximum,
            step=step,
            help=help_text,
            disabled=disabled,
            key=key,
        )
    elif ptype == "float":
        minimum = param_spec.get("minimum")
        maximum = param_spec.get("maximum")
        step = param_spec.get("step")
        v = st.number_input(
            label,
            value=float(display_value) if display_value is not None else 0.0,
            min_value=minimum,
            max_value=maximum,
            step=step if step is not None else 0.1,
            help=help_text,
            disabled=disabled,
            key=key,
        )
    elif ptype == "bool":
        v = st.checkbox(
            label,
            value=bool(display_value) if display_value is not None else False,
            help=help_text,
            disabled=disabled,
            key=key,
        )
    elif ptype == "enum":
        choices = param_spec.get("choices", [])
        choice_labels = [str(c) for c in choices]
        default_str = str(display_value) if display_value is not None else None
        default_idx = 0
        if default_str is not None:
            try:
                default_idx = choice_labels.index(default_str)
            except ValueError:
                default_idx = 0
        v = st.selectbox(
            label,
            options=choices,
            index=default_idx,
            format_func=str,
            help=help_text,
            disabled=disabled,
            key=key,
        )
    elif ptype == "str":
        v = st.text_input(
            label,
            value=str(display_value) if display_value is not None else "",
            help=help_text,
            disabled=disabled,
            key=key,
        )
    elif ptype == "list":
        v = st.multiselect(
            label,
            options=[],
            default=list(display_value) if isinstance(display_value, list) else [],
            help=help_text,
            disabled=disabled,
            key=key,
        )
    else:
        v = display_value

    # Coerce and store
    try:
        cv = coerce_param(param_spec, v)
    except (ValueError, TypeError):
        cv = v

    ctx.values[name] = cv
    return cv
