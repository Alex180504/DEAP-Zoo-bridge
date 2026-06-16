"""Setup — configure and submit an experiment.

This page fetches the catalog JSON from the backend (service must be
running — a yellow banner is shown if unreachable), builds a multi-tab
dynamic form, validates the experiment spec, and submits it to the queue.

Tabs: Environment / Algorithm(s) / Budget / Fitness / Recording /
Coevolution / Parallelism / Raw config.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.api_client import (
    ApiClient,
    ApiUnavailable,
    ApiValidationError,
)
from deap_pettingzoo.ui.components import (
    WidgetContext,
    build_algo_form,
    build_env_form,
    build_fitness_form,
    coerce_touched,
    render_validation_report,
)
from deap_pettingzoo.ui.components.param_widgets import render_param
from deap_pettingzoo.ui.components.forms import _render_param_group
from deap_pettingzoo.ui.services.spec_utils import extract_loaded_spec, spec_to_form_defaults

st.set_page_config(page_title="Experiment setup", layout="wide")
st.title("Experiment setup")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

client = ApiClient()
try:
    catalog = client.get_catalog()
except (ApiUnavailable, Exception):
    st.warning(
        "Cannot reach the experiment service backend. "
        "Start it with `python -m deap_pettingzoo.service` "
        "or set `FASTAPI_URL` to point at a running instance."
    )
    st.stop()

env_list = {e["name"]: e for e in catalog.get("environments", [])}
algo_list = {a["name"]: a for a in catalog.get("algorithms", [])}
all_algo_names = list(algo_list.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Budget-mode param specs (hand-authored — not from the catalog, since
#: budget modes are a top-level concept).
_BUDGET_PARAMS: List[Dict[str, Any]] = [
    {
        "name": "mode",
        "type": "enum",
        "choices": ["env_steps", "eval_episodes"],
        "default": "env_steps",
        "label": "Budget mode",
        "help": "env_steps = total environment steps across all agents; eval_episodes = number of evaluation episodes.",
    },
    {
        "name": "value",
        "type": "int",
        "default": 50000,
        "minimum": 1,
        "step": 1000,
        "label": "Budget value",
        "help": "How much budget to allocate.",
    },
    {
        "name": "num_seeds",
        "type": "int",
        "default": 1,
        "minimum": 1,
        "step": 1,
        "label": "Number of seeds",
        "help": "Independent training runs with different random seeds.",
    },
]

#: Recording-mode param specs.
_RECORDING_PARAMS: List[Dict[str, Any]] = [
    {
        "name": "enabled",
        "type": "bool",
        "default": False,
        "label": "Record champion rollouts",
        "help": "Save MP4/GIF videos of the best policy during training.",
    },
    {
        "name": "max_recordings",
        "type": "int",
        "default": None,
        "nullable": True,
        "minimum": 1,
        "step": 1,
        "group": "Advanced",
        "advanced": True,
        "label": "Max recordings",
        "help": "Cap on total recordings (first→gen 1, last→final, rest spaced evenly). None = cadence-based.",
    },
    {
        "name": "fps",
        "type": "int",
        "default": 20,
        "minimum": 1,
        "step": 5,
        "group": "Advanced",
        "advanced": True,
        "label": "FPS",
    },
    {
        "name": "fmt",
        "type": "enum",
        "choices": ["mp4", "gif"],
        "default": "mp4",
        "group": "Advanced",
        "advanced": True,
        "label": "Format",
    },
    {
        "name": "max_dim",
        "type": "int",
        "default": None,
        "nullable": True,
        "minimum": 1,
        "step": 100,
        "group": "Advanced",
        "advanced": True,
        "label": "Max dimension",
        "help": "Downscale longer side of rendered frames to this many pixels.",
    },
]

#: Parallelism param specs.
_PARALLELISM_PARAMS: List[Dict[str, Any]] = [
    {
        "name": "num_workers",
        "type": "int",
        "default": 1,
        "minimum": 0,
        "step": 1,
        "label": "Parallel workers",
        "help": "Number of parallel evaluation workers (1 = serial).",
    },
    {
        "name": "parallel_backend",
        "type": "enum",
        "choices": ["process", "scoop", "dask"],
        "default": "process",
        "advanced": True,
        "label": "Backend",
        "help": "Parallel executor backend.",
    },
]


def _build_submit_spec(
    env_name: str,
    selected_algos: List[str],
    env_values: Dict[str, Any],
    algo_values: Dict[str, Dict[str, Any]],
    budget_values: Dict[str, Any],
    fitness_values: Dict[str, Any],
    recording_values: Dict[str, Any],
    parallelism_values: Dict[str, Any],
    coevolution_values: Dict[str, Any],
    raw_config_text: str,
) -> Dict[str, Any]:
    """Assemble the full experiment-spec dict for submission."""
    spec: Dict[str, Any] = {
        "environment": env_name,
        "algorithms": selected_algos,
        "env_config": env_values,
        "hyperparameters": algo_values,
        "budget": budget_values,
        "fitness": fitness_values,
        "recording": recording_values,
        "parallelism": parallelism_values,
        "coevolution": coevolution_values,
        "label": env_name,
    }
    if raw_config_text.strip():
        try:
            import yaml
            raw = yaml.safe_load(raw_config_text)
            if isinstance(raw, dict):
                spec["raw_config"] = raw
        except Exception:
            pass
    return spec


# ---------------------------------------------------------------------------
# Restore from stored state (survives hot-reload / tab switch)
# ---------------------------------------------------------------------------

_stored = state.setup_config
_prev_env = ""
_prev_algos: List[str] = []
_prev_env_vals: Dict[str, Any] = {}
_prev_algo_vals: Dict[str, Dict[str, Any]] = {}
_prev_budget: Dict[str, Any] = {}
_prev_fitness: Dict[str, Any] = {}
_prev_recording: Dict[str, Any] = {}
_prev_parallelism: Dict[str, Any] = {}
_prev_coev: Dict[str, Any] = {}
_prev_raw_text = ""
if _stored:
    try:
        saved = json.loads(_stored)
        _prev_env = saved.get("env", "")
        _prev_algos = saved.get("algos", [])
        _prev_env_vals = saved.get("env_values", {})
        _prev_algo_vals = saved.get("algo_values", {})
        _prev_budget = saved.get("budget", {})
        _prev_fitness = saved.get("fitness", {})
        _prev_recording = saved.get("recording", {})
        _prev_parallelism = saved.get("parallelism", {})
        _prev_coev = saved.get("coevolution", {})
        _prev_raw_text = saved.get("raw_text", "")

        # Detect loaded spec from re-run.
        _loaded_spec = extract_loaded_spec(_stored)
        if _loaded_spec:
            loaded_label = _loaded_spec.get("label", "")
    except (json.JSONDecodeError, KeyError):
        _loaded_spec = None
else:
    _loaded_spec = None

# --- Banner for loaded spec ------------------------------------------------
if _loaded_spec:
    loaded_label = _loaded_spec.get("label", "")
    cols = st.columns([8, 1])
    cols[0].info(
        f"📋 **Loaded spec from experiment** ***'{loaded_label}'***. "
        "Review and modify before submitting."
    )
    if cols[1].button("✕ Clear", key="clear_loaded_spec"):
        state.setup_config = ""
        state.persist()
        st.rerun()


# ===========================================================================
# Tab 1 — Environment
# ===========================================================================

tab_env, tab_algos, tab_budget, tab_fitness, tab_rec, tab_coev, tab_par, tab_raw = (
    st.tabs(
        [
            "1. Environment",
            "2. Algorithm(s)",
            "3. Budget",
            "4. Fitness",
            "5. Recording",
            "6. Coevolution",
            "7. Parallelism",
            "8. Raw config",
        ]
    )
)

with tab_env:
    env_names = sorted(env_list.keys())
    env_sel = st.selectbox(
        "Environment",
        options=env_names,
        index=env_names.index(_prev_env) if _prev_env in env_names else 0,
        help="Select a registered PettingZoo environment.",
    )
    env_spec = env_list.get(env_sel, {})

    # Environment-specific config
    env_pctx = WidgetContext(values=dict(_prev_env_vals))
    env_params = env_spec.get("env_config", [])
    if env_params:
        _render_param_group(env_params, env_pctx, group_label="Parameters")
    env_values, _ = coerce_touched(env_params, env_pctx.values)

    # Compatibility-driven algorithm suggestions.
    if env_sel:
        try:
            compat = client.get_compatibility(env_sel)
            compatible_algos = compat.get("compatible", [])
        except Exception:
            compatible_algos = all_algo_names
    else:
        compatible_algos = all_algo_names

# ===========================================================================
# Tab 2 — Algorithm(s)
# ===========================================================================

with tab_algos:
    st.markdown(f"**Environment:** `{env_sel}` — compatible algorithms highlighted below.")
    # Show only installed algos, with compatible ones preselected.
    available_algos = [
        (name, spec) for name, spec in algo_list.items()
        if spec.get("available", False) and name != "coevolution"
    ]
    default_algos = _prev_algos or [n for n, _ in available_algos if n in compatible_algos][:1]

    selected_algos = st.multiselect(
        "Algorithm(s)",
        options=[n for n, _ in available_algos],
        default=default_algos,
        help="Select one or more algorithms to train.",
    )

    algo_forms: Dict[str, Dict[str, Any]] = {}
    touched_algos: Dict[str, Set[str]] = {}

    for algo_name in selected_algos:
        a_spec = algo_list.get(algo_name)
        if a_spec is None:
            continue
        st.subheader(a_spec.get("label", algo_name))
        if algo_name not in compatible_algos:
            st.warning(
                f"**{algo_name}** may be incompatible with **{env_sel}**. "
                "The experiment may fail at runtime.",
            )
        algo_ctx = WidgetContext(
            values=dict(_prev_algo_vals.get(algo_name, {}))
        )
        vals, touched = build_algo_form(a_spec, algo_ctx)
        algo_forms[algo_name] = vals
        touched_algos[algo_name] = touched

    # Also show uninstalled algorithms (greyed out).
    unavailable = [
        (name, spec) for name, spec in algo_list.items()
        if not spec.get("available", False)
    ]
    if unavailable:
        with st.expander("Unavailable algorithms (not installed)", expanded=False):
            for name, spec in unavailable:
                extras = ", ".join(spec.get("required_extras", []) or ["unknown"])
                st.caption(f"**{spec.get('label', name)}** — install `{extras}`")

# ===========================================================================
# Tab 3 — Budget
# ===========================================================================

with tab_budget:
    budget_ctx = WidgetContext(values=dict(_prev_budget))
    _render_param_group(_BUDGET_PARAMS, budget_ctx)
    budget_values, _ = coerce_touched(_BUDGET_PARAMS, budget_ctx.values)

# ===========================================================================
# Tab 4 — Fitness
# ===========================================================================

with tab_fitness:
    multi_team = len(env_spec.get("teams", [])) >= 2
    # All selected algorithms share an aggregation interface.
    agg_options = {"sum", "mean", "min", "max"}
    if multi_team:
        agg_options.add("team")
    fitness_ctx = WidgetContext(values=dict(_prev_fitness))
    build_fitness_form(sorted(agg_options), multi_team, fitness_ctx)
    fitness_values, _ = coerce_touched([], fitness_ctx.values)  # aggregation_method already in ctx
    if "aggregation_method" in fitness_ctx.values:
        fitness_values["aggregation_method"] = fitness_ctx.values["aggregation_method"]

# ===========================================================================
# Tab 5 — Recording
# ===========================================================================

with tab_rec:
    rec_ctx = WidgetContext(values=dict(_prev_recording))
    _render_param_group(_RECORDING_PARAMS, rec_ctx)
    recording_values, _ = coerce_touched(_RECORDING_PARAMS, rec_ctx.values)

# ===========================================================================
# Tab 6 — Coevolution
# ===========================================================================

with tab_coev:
    st.checkbox(
        "Enable co-evolution",
        value=_prev_coev.get("enabled", False),
        key="_coev_enabled",
        help="Pair populations in a competitive / self-play setup.",
        disabled=not multi_team,
    )
    if not multi_team:
        st.caption("Co-evolution requires a multi-team environment (≥2 teams).")
    coevolution_values: Dict[str, Any] = {"enabled": st.session_state.get("_coev_enabled", False)}
    if coevolution_values["enabled"]:
        coev_opts = catalog.get("coevolution", {})
        modes = [m["name"] for m in coev_opts.get("modes", [])]
        if modes:
            coevolution_values["mode"] = st.selectbox(
                "Mode", options=modes, index=0, key="_coev_mode"
            )
        schedules = coev_opts.get("schedules", [])
        if schedules:
            coevolution_values["schedule"] = st.selectbox(
                "Schedule", options=schedules, index=0, key="_coev_sched"
            )
        coevolution_values["k_opponents"] = st.number_input(
            "Opponents per evaluation", value=3, min_value=1, key="_coev_k"
        )
        coevolution_values["aggregation"] = st.selectbox(
            "Aggregation", options=list(coev_opts.get("aggregations", ["mean"])), key="_coev_agg"
        )

        # Build simple side spec from the env's declared teams.
        teams = env_spec.get("teams", [])
        sides = []
        for i, team in enumerate(teams):
            team_name = team.get("name", f"team_{i}")
            st.markdown(f"**Side `{team_name}`** — agents: `{', '.join(team.get('agent_prefixes', []))}`")
            side_algo = st.selectbox(
                f"Algorithm for {team_name}",
                options=all_algo_names,
                index=0,
                key=f"_side_algo_{i}",
            )
            sides.append({
                "name": team_name,
                "algorithm": side_algo,
                "role_prefixes": team.get("agent_prefixes", []),
                "hyperparameters": {},
            })
        coevolution_values["sides"] = sides

# ===========================================================================
# Tab 7 — Parallelism
# ===========================================================================

with tab_par:
    par_ctx = WidgetContext(values=dict(_prev_parallelism))
    _render_param_group(_PARALLELISM_PARAMS, par_ctx)
    parallelism_values, _ = coerce_touched(_PARALLELISM_PARAMS, par_ctx.values)

# ===========================================================================
# Tab 8 — Raw config
# ===========================================================================

with tab_raw:
    raw_text = st.text_area(
        "Raw YAML / JSON config overlay",
        value=_prev_raw_text,
        height=200,
        help=(
            "Free-form config that is deep-merged on top of the form "
            "defaults. Explicit form values still take precedence."
        ),
    )
    if raw_text.strip():
        try:
            import yaml
            parsed = yaml.safe_load(raw_text)
            if isinstance(parsed, dict):
                st.caption(f"Parsed {len(parsed)} top-level keys.")
            else:
                st.warning("Raw config must be a YAML/JSON object (dict).")
        except Exception as exc:
            st.error(f"Parse error: {exc}")

# ---------------------------------------------------------------------------
# Validate + Submit
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("Validate", use_container_width=True):
        spec = _build_submit_spec(
            env_sel, selected_algos, env_values, algo_forms,
            budget_values, fitness_values, recording_values,
            parallelism_values, coevolution_values, raw_text,
        )
        with st.spinner("Validating..."):
            try:
                report = client.validate(spec)
                render_validation_report(report)
            except ApiValidationError as exc:
                render_validation_report(exc.report)
            except ApiUnavailable:
                st.error("Backend unreachable.")

with col2:
    if st.button("Submit to queue", use_container_width=True, type="primary"):
        spec = _build_submit_spec(
            env_sel, selected_algos, env_values, algo_forms,
            budget_values, fitness_values, recording_values,
            parallelism_values, coevolution_values, raw_text,
        )
        with st.spinner("Validating and submitting..."):
            try:
                result = client.submit(spec)
                st.success(
                    f"Experiment **{result['id']}** submitted! "
                    f"Position in queue: {result['position']}. "
                    f"Go to the **Queue** page to watch progress."
                )
            except ApiValidationError as exc:
                render_validation_report(exc.report)
                st.error("Fix the issues above before submitting.")
            except ApiUnavailable:
                st.error("Backend unreachable.")

# ---------------------------------------------------------------------------
# Persist form state for hot-reload / tab-switch survival
# ---------------------------------------------------------------------------

_persist = {
    "env": env_sel,
    "algos": selected_algos,
    "env_values": env_values,
    "algo_values": algo_forms,
    "budget": budget_values,
    "fitness": fitness_values,
    "recording": recording_values,
    "parallelism": parallelism_values,
    "coevolution": coevolution_values,
    "raw_text": raw_text,
}
state.setup_config = json.dumps(_persist, default=str)
state.persist()
