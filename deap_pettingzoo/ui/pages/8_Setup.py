"""Setup: configure and submit an experiment.

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

from deap_pettingzoo.ui.components import (
    WidgetContext,
    build_algo_form,
    build_env_form,
    build_fitness_form,
    coerce_touched,
    render_validation_report,
)
from deap_pettingzoo.ui.components.forms import _render_param_group
from deap_pettingzoo.ui.components.param_widgets import render_param
from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.api_client import (
    ApiClient,
    ApiClientError,
    ApiUnavailable,
    ApiValidationError,
    get_client,
)
from deap_pettingzoo.ui.services.spec_utils import (
    extract_loaded_spec,
    spec_to_form_defaults,
)

st.set_page_config(page_title="Experiment setup", layout="wide")
st.title("Experiment setup")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

# API client

client = get_client()
try:
    catalog = client.get_catalog()
except (ApiUnavailable, Exception):
    st.warning(
        "Cannot reach the experiment service backend. "
        "Set `FASTAPI_URL` to point at a running backend service or container."
    )
    st.stop()

env_list = {e["name"]: e for e in catalog.get("environments", [])}
algo_list = {a["name"]: a for a in catalog.get("algorithms", [])}
all_algo_names = list(algo_list.keys())

# Select whether the backend exposes the advanced (install + custom-script) surface.
try:
    advanced_enabled = bool(client.get_health().get("advanced_enabled", False))
except Exception:
    advanced_enabled = False


# Helpers

#: Budget-mode param specs (hand-authored — not from the catalog, since budget modes are a top-level concept).
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
    # NOTE: ``mode`` + its companion field (``max_recordings`` / ``every_n_steps``)
    # are rendered with custom widgets in the Recording tab so the companion
    # field can be shown conditionally on the chosen mode.
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
    # NOTE: ``max_dim`` is rendered with custom widgets in the Recording tab
    # (a "full resolution" toggle + a 512-default cap) because the generic
    # nullable-int widget cannot emit ``None`` and would force a tiny cap.
]

#: Default downscale cap (longest frame side, px) for UI recordings.
_DEFAULT_MAX_DIM = 512

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
        "choices": ["process", "dask"],
        "default": "process",
        "advanced": True,
        "label": "Backend",
        "help": "Parallel executor backend.",
    },
    {
        "name": "scheduler_address",
        "type": "str",
        "default": None,
        "nullable": True,
        "advanced": True,
        "label": "Dask scheduler",
        "help": (
            "Remote Dask scheduler for the 'dask' backend: a "
            "'tcp://host:8786' URL or a scheduler-file path. Leave empty "
            "to auto-build a local cluster."
        ),
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
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Assemble the full experiment-spec dict for submission.

    Returns ``(spec, raw_error)`` where ``raw_error`` is non-None when the
    raw-config overlay failed to parse.  Callers must show the error and
    should not submit if it is set.
    """
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
    raw_error: Optional[str] = None
    if raw_config_text.strip():
        try:
            import yaml

            raw = yaml.safe_load(raw_config_text)
            if isinstance(raw, dict):
                spec["raw_config"] = raw
            else:
                raw_error = "Raw config must be a YAML/JSON object (dict), not a scalar or list."
        except Exception as exc:
            raw_error = f"Raw config parse error: {exc}"
    return spec, raw_error


# Restore from stored state (survives hot-reload / tab switch)

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

# Banner for loaded spec
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
# Advanced section (dependency install + custom experiment script)
# ===========================================================================

_GITHUB_HINT = (
    "Looks like a bare GitHub URL. pip can't install that directly — use the "
    "`git+https://…` form, "
    "(or a `.../archive/refs/heads/main.zip` link)."
)


def _looks_like_bare_github(args: List[str]) -> bool:
    """True if an arg is a plain github.com URL pip can't install as-is."""
    archive_suffixes = (".zip", ".tar.gz", ".tgz", ".whl")
    for a in args:
        low = a.lower()
        if (
            "github.com/" in low
            and not low.startswith("git+")
            and not low.endswith(archive_suffixes)
        ):
            return True
    return False


@st.fragment(run_every=2)
def _install_output_fragment(client: ApiClient, job_id: str) -> None:
    """Poll an install job and render its (scrollable, bounded) log."""
    try:
        info = client.get_install(job_id, lines=400)
    except Exception as exc:  # noqa: BLE001 — best-effort poll
        st.caption(f"Unable to fetch install output: {exc}")
        return
    status = info.get("status", "running")
    badge = {
        "running": "🔄 Running…",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
    }.get(status, status)
    st.markdown(f"**`{info.get('command', '')}`** — {badge}")
    # Fixed-height scrollable box so a long pip build log can't blow up the page.
    with st.container(height=320):
        st.code(info.get("text") or "(no output yet)", language="text")


def _render_install_panel(client: ApiClient) -> None:
    st.subheader("Install dependencies")
    st.caption(
        "Install a custom environment package into the backend. Allowed: "
        "`pip install`, `pip uninstall`, `git clone`. For a GitHub repo use "
        "`pip install` with a `git+https://…` URL. **Note:** in a containerised "
        "deployment these installs are lost when the container is rebuilt — add "
        "the package to `Dockerfile.server` for a permanent environment."
    )
    op = st.selectbox(
        "Operation",
        options=["pip install", "pip uninstall", "git clone"],
        index=0,
    )
    spec_text = st.text_input(
        "Package / spec / URL",
        placeholder="marl-aquarium   ·   git+https://github.com/Alex180504/marl-aquarium.git",
    )
    if st.button("Run", type="primary"):
        manager, action = op.split(" ", 1)
        args = spec_text.split()
        if not args:
            st.error("Enter a package spec or URL.")
        elif manager == "pip" and _looks_like_bare_github(args):
            st.warning(_GITHUB_HINT)
        else:
            try:
                res = client.start_install(manager, action, args)
                st.session_state["_install_job_id"] = res.get("job_id")
            except ApiUnavailable:
                st.error("Backend unreachable.")
            except ApiClientError as exc:
                st.error(str(exc))

    job_id = st.session_state.get("_install_job_id")
    if job_id:
        _install_output_fragment(client, job_id)


def _render_script_editor(client: ApiClient, state: UIState) -> None:
    st.subheader("Custom experiment script")
    st.caption(
        "Paste a full Python experiment script. It is queued and run as `python user_script.py`; its relative "
        "`results/…` outputs are linked to the experiment and appear under "
        "Browse. Validation is **static only** — it checks syntax/structure, not "
        "that the script trains correctly."
    )
    label = st.text_input("Label (optional)", value="")

    # Optional rich editor (syntax-highlighted) if streamlit-code-editor is installed.
    use_rich = st.checkbox(
        "Rich code editor",
        value=False,
        help="Syntax-highlighted editor (needs streamlit-code-editor).",
    )
    src = state.advanced_script
    if use_rich:
        try:
            from code_editor import code_editor  # type: ignore

            _ed = code_editor(
                state.advanced_script,
                lang="python",
                height=[18, 30],
                key="_adv_code_editor",
                # Emit text on blur + debounced typing so a button click (which blurs the editor) captures the latest content.
                response_mode=["debounce", "blur"],
            )
            returned = (_ed or {}).get("text", "")
            # Fall back to the persisted text when the editor hasn't emitted yet.
            src = returned if returned.strip() else state.advanced_script
            st.caption("Tip: click outside the editor before Validate/Submit.")
        except Exception as exc:  # noqa: BLE001 — fall back to a plain area
            st.caption(
                f"Error: Rich editor unavailable ({exc}); using plain text area."
            )
            src = st.text_area(
                "Python script",
                value=state.advanced_script,
                height=400,
                label_visibility="collapsed",
            )
    else:
        src = st.text_area(
            "Python script",
            value=state.advanced_script,
            height=400,
            label_visibility="collapsed",
        )
    state.advanced_script = src

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Validate script", use_container_width=True, key="_adv_validate"):
            try:
                render_validation_report(client.validate_script(src))
            except ApiValidationError as exc:
                render_validation_report(exc.report)
            except ApiUnavailable:
                st.error("Backend unreachable.")
    with c2:
        if st.button(
            "Submit a script to queue",
            use_container_width=True,
            type="primary",
            key="_adv_submit",
        ):
            try:
                result = client.submit_script(src, label=label)
                st.success(
                    f"Script experiment **{result['id']}** submitted! "
                    f"Position in queue: {result['position']}. "
                    f"Go to the **Queue** page to watch progress."
                )
            except ApiValidationError as exc:
                render_validation_report(exc.report)
                st.error("Fix the issues above before submitting.")
            except ApiUnavailable:
                st.error("Backend unreachable.")


def _render_advanced_section(client: ApiClient, state: UIState) -> None:
    """Render the install panel / script editor (one at a time)."""
    st.caption(
        "⚠️ These tools run arbitrary commands / code in the backend "
        "environment (trusted-input model — see docs/known_issues.md)."
    )
    view = st.radio(
        "Tool",
        options=["install", "script"],
        format_func=lambda v: (
            "📦 Install dependencies" if v == "install" else "🐍 Script editor"
        ),
        index=0 if state.advanced_view != "script" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    state.advanced_view = view
    if view == "install":
        _render_install_panel(client)
    else:
        _render_script_editor(client, state)


# Tab 1: Environment

# Top-level mode: the spec form (8 tabs + Validate/Submit) OR the Advanced tools.
# Rendering Advanced as a mutually-exclusive mode (not a 9th tab) keeps the
# page-level Validate/Submit buttons from duplicating alongside the script editor's own buttons
if advanced_enabled:
    page_mode = st.radio(
        "Page mode",
        options=["Configure experiment", "Advanced"],
        horizontal=True,
        key="_setup_page_mode",
        label_visibility="collapsed",
    )
else:
    page_mode = "Configure experiment"

if page_mode == "Advanced":
    _render_advanced_section(client, state)
    state.persist()
    st.stop()

(
    tab_env,
    tab_algos,
    tab_budget,
    tab_fitness,
    tab_rec,
    tab_coev,
    tab_par,
    tab_raw,
) = st.tabs(
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

# Tab 2: Algorithm(s)

with tab_algos:
    st.markdown(
        f"**Environment:** `{env_sel}` — compatible algorithms highlighted below."
    )
    # Show only installed algos, with compatible ones preselected.
    available_algos = [
        (name, spec)
        for name, spec in algo_list.items()
        if spec.get("available", False) and name != "coevolution"
    ]
    default_algos = (
        _prev_algos or [n for n, _ in available_algos if n in compatible_algos][:1]
    )

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
            values=dict(_prev_algo_vals.get(algo_name, {})),
            prefix=algo_name,
        )
        vals, touched = build_algo_form(a_spec, algo_ctx)
        algo_forms[algo_name] = vals
        touched_algos[algo_name] = touched

    # Also show uninstalled algorithms (greyed out).
    unavailable = [
        (name, spec)
        for name, spec in algo_list.items()
        if not spec.get("available", False)
    ]
    if unavailable:
        with st.expander("Unavailable algorithms (not installed)", expanded=False):
            for name, spec in unavailable:
                extras = ", ".join(spec.get("required_extras", []) or ["unknown"])
                st.caption(f"**{spec.get('label', name)}** — install `{extras}`")

# Tab 3: Budget

with tab_budget:
    budget_ctx = WidgetContext(values=dict(_prev_budget))
    _render_param_group(_BUDGET_PARAMS, budget_ctx)
    budget_values, _ = coerce_touched(_BUDGET_PARAMS, budget_ctx.values)

    # budget_on_consumed — only meaningful under env_steps mode.
    _boc_mode = budget_values.get("mode", "env_steps")
    _boc_default = bool(_prev_budget.get("budget_on_consumed", False))
    if _boc_mode == "env_steps":
        _budget_on_consumed = st.checkbox(
            "Budget on consumed steps",
            value=_boc_default,
            key="_budget_on_consumed",
            help=(
                "Run until the **real** consumed env-step count reaches the "
                "budget value instead of stopping after a precomputed number of "
                "generations. Supported by EA (strategy='simple'), CMA-ES, PSO, "
                "and co-evolution (shared pool across all sides). Rejected with a "
                "clear error for EA mu±lambda, NEAT, and QDax."
            ),
        )
        if _budget_on_consumed and selected_algos:
            _evo_selected = [
                a
                for a in selected_algos
                if algo_list.get(a, {}).get("family") == "evolutionary"
            ]
            if not _evo_selected:
                st.warning(
                    "`Budget on consumed steps` has no effect on the selected "
                    "algorithm(s) — it only applies to EA (strategy='simple'), "
                    "CMA-ES, and PSO."
                )
            elif any(a in ("ea", "genetic", "ga") for a in selected_algos):
                st.info(
                    "For EA, `Budget on consumed steps` requires "
                    "strategy='simple'. The mu±lambda strategies run a "
                    "fixed generation count and will be rejected at runtime."
                )
        budget_values["budget_on_consumed"] = _budget_on_consumed
    else:
        budget_values["budget_on_consumed"] = False

# Tab 4: Fitness

with tab_fitness:
    multi_team = len(env_spec.get("teams", [])) >= 2
    # All selected algorithms share an aggregation interface.
    agg_options = {"sum", "mean", "min", "max"}
    if multi_team:
        agg_options.add("team")
    fitness_ctx = WidgetContext(values=dict(_prev_fitness))
    build_fitness_form(sorted(agg_options), multi_team, fitness_ctx)
    fitness_values, _ = coerce_touched(
        [], fitness_ctx.values
    )  # aggregation_method already in ctx
    if "aggregation_method" in fitness_ctx.values:
        fitness_values["aggregation_method"] = fitness_ctx.values["aggregation_method"]

# Tab 5: Recording

with tab_rec:
    rec_ctx = WidgetContext(values=dict(_prev_recording))
    _render_param_group(_RECORDING_PARAMS, rec_ctx)
    recording_values, _ = coerce_touched(_RECORDING_PARAMS, rec_ctx.values)

    # Cadence mode (algorithm-agnostic)
    # One concept across EA generations and RL iterations: choose how often to
    # capture the champion rollout. The companion field (count / env-step interval) is shown conditionally.
    _MODE_LABELS = {
        "count": "N evenly spread over the run",
        "every": "Every generation / iteration",
        "interval": "Every N env-steps",
        "final": "Only the final rollout",
    }
    _mode_keys = list(_MODE_LABELS)
    _prev_mode = str(_prev_recording.get("mode", "count"))
    _mode_idx = _mode_keys.index(_prev_mode) if _prev_mode in _mode_keys else 0
    rec_mode = st.selectbox(
        "Recording mode",
        options=_mode_keys,
        index=_mode_idx,
        format_func=lambda k: _MODE_LABELS[k],
        help="How often to record a champion rollout. Same meaning for every algorithm.",
    )
    recording_values["mode"] = rec_mode
    if rec_mode == "count":
        _prev_n = _prev_recording.get("max_recordings") or 10
        recording_values["max_recordings"] = int(
            st.number_input(
                "Max recordings",
                value=int(_prev_n),
                min_value=1,
                step=1,
                help="First → start of run, last → end, the rest spaced evenly.",
            )
        )
    elif rec_mode == "interval":
        _prev_step = _prev_recording.get("every_n_steps") or 10000
        recording_values["every_n_steps"] = int(
            st.number_input(
                "Every N env-steps",
                value=int(_prev_step),
                min_value=1,
                step=1000,
                help="Record one rollout each time this many env steps elapse.",
            )
        )
    elif rec_mode == "every":
        st.warning(
            "Recording every generation/iteration can produce a very large "
            "number of videos (disk + time) on long runs. Prefer 'N evenly "
            "spread' unless you need every step."
        )

    # Resolution control (custom: the generic int widget can't emit None)
    # ``max_dim`` downscales the longer side of each rendered frame. Default to
    # 512 px and offer a full-resolution toggle (max_dim=None) for pixel-exact
    # replays, gated behind a memory/performance warning.
    _prev_max_dim = _prev_recording.get("max_dim", _DEFAULT_MAX_DIM)
    _prev_full_res = "max_dim" in _prev_recording and _prev_max_dim is None
    full_res = st.checkbox(
        "Record without compression (full resolution)",
        value=bool(_prev_full_res),
        help=(
            "Skip frame downscaling and record at the environment's native "
            "render resolution."
        ),
    )
    if full_res:
        st.warning(
            "Full-resolution recording buffers every frame uncompressed in "
            "RAM before encoding. In large / high-resolution environments "
            "this can consume a lot of memory "
            "and slow training noticeably. Prefer a max-dimension cap unless "
            "you need pixel-exact replays."
        )
        recording_values["max_dim"] = None
    else:
        _md_default = (
            int(_prev_max_dim)
            if isinstance(_prev_max_dim, int) and _prev_max_dim > 0
            else _DEFAULT_MAX_DIM
        )
        max_dim = st.number_input(
            "Max dimension (px)",
            value=_md_default,
            min_value=64,
            step=64,
            help="Downscale the longer side of rendered frames to this many pixels.",
        )
        recording_values["max_dim"] = int(max_dim)

# Tab 6: Coevolution

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
    coevolution_values: Dict[str, Any] = {
        "enabled": st.session_state.get("_coev_enabled", False)
    }
    if coevolution_values["enabled"]:
        st.info(
            "When co-evolution is enabled the **Algorithm(s)** tab selection is not used — "
            "configure each side's algorithm below instead."
        )
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
            "Aggregation",
            options=list(coev_opts.get("aggregations", ["mean"])),
            key="_coev_agg",
        )

        # Only offer algorithms that are installed and are not the "coevolution" pseudo-algorithm entry.
        coev_algo_options = [
            n
            for n, s in algo_list.items()
            if s.get("available", False) and n != "coevolution"
        ]

        # Build simple side spec from the env's declared teams.
        teams = env_spec.get("teams", [])
        sides = []
        for i, team in enumerate(teams):
            team_name = team.get("name", f"team_{i}")
            st.markdown(
                f"**Side `{team_name}`** — agents: `{', '.join(team.get('agent_prefixes', []))}`"
            )
            side_algo = st.selectbox(
                f"Algorithm for {team_name}",
                options=coev_algo_options,
                index=0,
                key=f"_side_algo_{i}",
            )
            sides.append(
                {
                    "name": team_name,
                    "algorithm": side_algo,
                    "role_prefixes": team.get("agent_prefixes", []),
                    "hyperparameters": {},
                }
            )
        coevolution_values["sides"] = sides

# Tab 7: Parallelism

with tab_par:
    par_ctx = WidgetContext(values=dict(_prev_parallelism))
    _render_param_group(_PARALLELISM_PARAMS, par_ctx)
    parallelism_values, _ = coerce_touched(_PARALLELISM_PARAMS, par_ctx.values)

# Tab 8: Raw config

with tab_raw:
    raw_text = st.text_area(
        "Raw YAML / JSON config overlay",
        value=_prev_raw_text,
        height=200,
        help=(
            "Free-form config that is merged on top of the form "
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

# Validate + Submit

col1, col2 = st.columns(2)

with col1:
    if st.button("Validate", use_container_width=True):
        spec, raw_error = _build_submit_spec(
            env_sel,
            selected_algos,
            env_values,
            algo_forms,
            budget_values,
            fitness_values,
            recording_values,
            parallelism_values,
            coevolution_values,
            raw_text,
        )
        if raw_error:
            st.error(f"Raw config tab: {raw_error}")
        else:
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
        spec, raw_error = _build_submit_spec(
            env_sel,
            selected_algos,
            env_values,
            algo_forms,
            budget_values,
            fitness_values,
            recording_values,
            parallelism_values,
            coevolution_values,
            raw_text,
        )
        if raw_error:
            st.error(f"Raw config tab: {raw_error}")
        else:
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
                    st.error("Error: Backend connection failed.")

# Persist form state for hot-reload / tab-switch survival

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
