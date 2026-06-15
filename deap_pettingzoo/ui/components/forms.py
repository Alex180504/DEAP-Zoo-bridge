"""Tabbed dynamic-form builders.

Each builder consumes the catalog JSON (as fetched from the FastAPI
``GET /catalog/*`` endpoints) and renders a tab's worth of widgets,
returning a ``(values_dict, touched_keys)`` tuple.

The builders are **streamlit-dependent** (they call ``render_param``
which calls ``st.*`` widgets), but the return-value shape is pure data
so the Setup page's orchestration logic can be tested without widgets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from deap_pettingzoo.ui.components.param_widgets import (
    WidgetContext,
    coerce_touched,
    render_param,
)


# ---------------------------------------------------------------------------
# Environment tab
# ---------------------------------------------------------------------------


def build_env_form(
    env_spec: Dict[str, Any],
    ctx: WidgetContext,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Render environment configuration widgets.

    Args:
        env_spec: Environment catalog entry (the dict from
            ``GET /catalog/environments/{name}``).
        ctx: Shared :class:`WidgetContext`.

    Returns:
        ``(env_config_values, touched_keys)`` — only non-default values.
    """
    import streamlit as st  # noqa: WPS433

    st.markdown(f"**Environment:** `{env_spec['name']}`")
    notes = env_spec.get("notes")
    if notes:
        st.caption(notes)
    teams = env_spec.get("teams", [])
    if teams:
        team_labels = ", ".join(
            f"{t['name']} ({', '.join(t.get('agent_prefixes', []))})"
            for t in teams
        )
        st.caption(f"Teams: {team_labels}")

    params = env_spec.get("env_config", [])
    if not params:
        st.info("This environment has no configurable parameters.")
        return {}, set()

    # Group params.
    _render_param_group(params, ctx, group_label=None)
    return coerce_touched(params, ctx.values)


# ---------------------------------------------------------------------------
# Algorithm tab(s)
# ---------------------------------------------------------------------------


def build_algo_form(
    algo_spec: Dict[str, Any],
    ctx: WidgetContext,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Render the hyperparameter form for one algorithm.

    Only non-advanced parameters are shown upfront; advanced ones are hidden
    inside a ``st.expander("Advanced")``.  The return values follow the same
    ``(dict, set)`` contract as :func:`build_env_form`.
    """
    import streamlit as st  # noqa: WPS433

    if not algo_spec.get("available", True):
        st.warning(
            f"Algorithm **{algo_spec['label']}** is not installed. "
            f"Required extras: {', '.join(algo_spec.get('required_extras', []) or ['unknown'])}."
        )
        return {}, set()

    params = algo_spec.get("hyperparameters", [])
    if not params:
        return {}, set()

    normal = [p for p in params if not p.get("advanced")]
    advanced = [p for p in params if p.get("advanced")]

    _render_param_group(normal, ctx, group_label=None)
    if advanced:
        with st.expander("Advanced", expanded=False):
            _render_param_group(advanced, ctx, group_label=None)

    return coerce_touched(params, ctx.values)


# ---------------------------------------------------------------------------
# Fitness tab
# ---------------------------------------------------------------------------


def build_fitness_form(
    available_aggregations: List[str],
    multi_team: bool,
    ctx: WidgetContext,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Render fitness/aggregation options."""
    import streamlit as st  # noqa: WPS433

    method = st.selectbox(
        "Aggregation method",
        options=[a for a in available_aggregations if a != "team" or multi_team],
        index=0,
        help="How to collapse per-agent rewards into a single fitness value.",
        disabled=bool(ctx.disabled_reason),
    )
    # Team mapping is not surfaced in the simple form — it's raw-config territory
    # (auto-inferred from the env's declared teams).
    ctx.values["aggregation_method"] = method
    return {"aggregation_method": method}, {"aggregation_method"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_param_group(
    params: List[Dict[str, Any]],
    ctx: WidgetContext,
    *,
    group_label: Optional[str] = None,
) -> None:
    """Render a flat list of params, optionally under a group label."""
    if not params:
        return
    if group_label:
        import streamlit as st

        st.subheader(group_label)
    for ps in params:
        render_param(ps, ctx)
