"""Browse — filter discovered runs and select which to compare/replay.

Selection writes through to :class:`UIState.selected_run_ids`, which the
Compare / Diff / Replay / Report pages all read from.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from deap_pettingzoo.ui.services import (
    UIState,
    create_folder,
    discover_runs,
    load_settings,
    move_run,
    rename_run,
)
from deap_pettingzoo.ui.services.api_client import ApiValidationError, get_client
from deap_pettingzoo.ui.services.spec_utils import spec_to_form_defaults

st.set_page_config(page_title="Browse runs", layout="wide")
st.title("Browse runs")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)
state.results_root = settings.results_root

if not settings.results_root_exists:
    st.warning(
        f"Results root **{settings.results_root}** does not exist. "
        "Set it on the Home page first."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _cached_discover(root: str, mtime_bucket: float):
    """Re-runs only when root or any manifest.json mtime changes."""
    return discover_runs(Path(root))


def _root_mtime_bucket(root: Path) -> float:
    try:
        return max(
            (m.stat().st_mtime for m in root.rglob("manifest.json")),
            default=0.0,
        )
    except OSError:
        return 0.0


root = settings.results_root
runs = _cached_discover(str(root), _root_mtime_bucket(root))


def _run_import(file_bytes: bytes, file_id: str) -> bool:
    """Upload the ZIP to the backend; show inline status. Returns True on success."""
    with st.status("Importing…", expanded=True) as _status:
        st.write("Uploading archive — this may take a moment for large files…")
        try:
            result = get_client().import_experiments(file_bytes)
        except ApiValidationError as exc:
            _status.update(label="Validation failed", state="error")
            for issue in exc.report.get("issues", []):
                sev = issue.get("severity", "error")
                msg = f"**{issue.get('path', '')}**: {issue.get('message', '')}"
                (st.warning if sev == "warning" else st.error)(msg)
            return False
        except Exception as exc:
            _status.update(label="Import failed", state="error")
            st.error(f"Import failed: {exc}")
            return False
        if result.get("ok"):
            n = len(result.get("imported", []))
            _status.update(label=f"Imported {n} experiment(s)", state="complete")
            st.session_state["_last_import_id"] = file_id
            return True
        _status.update(label="Import completed with errors", state="error")
        for err in result.get("errors", []):
            st.error(f"{err.get('path', '')}: {err.get('message', '')}")
        for warn in result.get("warnings", []):
            st.warning(f"{warn.get('path', '')}: {warn.get('message', '')}")
        return False


def _import_widget(uploader_key: str, btn_key: str) -> None:
    """File-uploader + Import button; clears the discover cache and reruns on success."""
    uploaded = st.file_uploader(
        "Upload a DEAP export (.zip)",
        type=["zip"],
        key=uploader_key,
        label_visibility="collapsed",
    )
    if uploaded is None:
        return
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_last_import_id") == file_id:
        st.info("This file was already imported. Upload a different archive to import again.")
        return
    if st.button("📥 Import", key=btn_key):
        if _run_import(uploaded.getvalue(), file_id):
            # Drop filter widget state so the next render re-initialises
            # with default=<all>, exposing newly-imported algorithms/envs.
            for _k in ("browse_filter_algos", "browse_filter_envs", "browse_date_range"):
                st.session_state.pop(_k, None)
            _cached_discover.clear()
            st.rerun()


if not runs:
    st.info(f"No runs found under **{root}**. Import a previously exported experiment below.")
    st.markdown("### 📦 Import experiments")
    _import_widget("browse_import_empty_uploader", "browse_import_empty_btn")
    st.stop()

# Sidebar filters & management 
algos = sorted({r.algorithm for r in runs})
envs = sorted({r.env_name for r in runs})

# Collect dates that have at least one run for min/max defaults
run_dates = [r.started_at.date() for r in runs if r.started_at is not None]

with st.sidebar:
    st.header("Filters")
    sel_algos = st.multiselect("Algorithms", algos, default=algos, key="browse_filter_algos")
    sel_envs = st.multiselect("Environments", envs, default=envs, key="browse_filter_envs")

    # Date range — only shown when there is timestamp data
    date_range: list = []
    if run_dates:
        min_date = min(run_dates)
        max_date = max(run_dates)
        date_range = list(
            st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="browse_date_range",
            )
        )

    st.divider()

    # Group by experiment
    group_by_exp = st.checkbox("Group by experiment", key="browse_group_by_exp")

    st.divider()
    st.subheader("Sort")
    sort_col = st.selectbox(
        "Sort by",
        ["started_at", "final_fitness", "wall_time_s", "algorithm", "env_name", "seed"],
        index=0,
        key="browse_sort_col",
    )
    sort_asc = st.radio("Order", ["↓ Newest / Highest", "↑ Oldest / Lowest"], key="browse_sort_order") == "↑ Oldest / Lowest"

    st.divider()
    with st.expander("📁 Directory management"):
        st.markdown("**Create folder**")
        new_folder_name = st.text_input("Folder name", key="browse_new_folder_name", placeholder="e.g. experiment-A")
        if st.button("Create", key="browse_create_folder"):
            if new_folder_name:
                try:
                    new_path = create_folder(root, new_folder_name)
                    st.success(f"Created: {new_path.relative_to(root)}")
                except ValueError as exc:
                    st.error(str(exc))
            else:
                st.warning("Enter a folder name first.")

        st.markdown("**Move selected runs to folder**")
        # List first-level subdirs that are not run dirs (no manifest.json)
        try:
            subdirs = sorted(
                p.name for p in root.iterdir()
                if p.is_dir() and not (p / "manifest.json").is_file()
            )
        except OSError:
            subdirs = []
        if subdirs:
            move_target_name = st.selectbox("Target folder", subdirs, key="browse_move_target")
            if st.button("Move selected here", key="browse_move_btn"):
                target_dir = root / move_target_name
                moved, errors = 0, []
                for rid in state.selected_run_ids:
                    matching = [r for r in runs if r.run_id == rid]
                    if not matching:
                        continue
                    try:
                        move_run(matching[0].run_dir, target_dir)
                        moved += 1
                    except (FileNotFoundError, ValueError, FileExistsError) as exc:
                        errors.append(str(exc))
                if moved:
                    st.success(f"Moved {moved} run(s) to '{move_target_name}'.")
                    state.update_selected([])
                    state.persist()
                    _cached_discover.clear()
                    st.rerun()
                for err in errors:
                    st.error(err)
        else:
            st.caption("No subfolders found. Create one above first.")


# Apply filters & sort
filtered = [
    r for r in runs
    if r.algorithm in sel_algos
    and r.env_name in sel_envs
]

if len(date_range) == 2:
    lo, hi = date_range[0], date_range[1]
    filtered = [
        r for r in filtered
        if r.started_at is None or (lo <= r.started_at.date() <= hi)
    ]

# Sort
_SORT_NONE_LAST = {
    "started_at": lambda r: (r.started_at is None, r.started_at),
    "final_fitness": lambda r: (
        r.final_summary.get("final_fitness") is None
        and r.final_summary.get("best_fitness") is None
        and r.final_summary.get("mean_episode_reward") is None,
        r.final_summary.get("final_fitness")
        or r.final_summary.get("best_fitness")
        or r.final_summary.get("mean_episode_reward"),
    ),
    "wall_time_s": lambda r: (r.wall_time_s is None, r.wall_time_s),
    "algorithm": lambda r: (False, r.algorithm),
    "env_name": lambda r: (False, r.env_name),
    "seed": lambda r: (r.seed is None, r.seed),
}
filtered.sort(key=_SORT_NONE_LAST.get(sort_col, _SORT_NONE_LAST["started_at"]), reverse=not sort_asc)

st.caption(f"{len(filtered)} of {len(runs)} runs match.")

# Table
rows = [
    {
        "name": r.display_name or r.run_id,
        "run_id": r.run_id,
        "algorithm": r.algorithm,
        "env_name": r.env_name,
        "seed": r.seed,
        "started_at": r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else None,
        "wall_time_s": r.wall_time_s,
        "final_fitness": r.final_summary.get("final_fitness")
        or r.final_summary.get("best_fitness")
        or r.final_summary.get("mean_episode_reward"),
        "run_dir": str(r.run_dir),
        "origin": f"UI ({r.experiment_id})" if r.experiment_id else "",
        "exp_id": r.experiment_id or "",
    }
    for r in filtered
]

# Pre-select rows whose run_id is already in state.selected_run_ids
preselected_indices = [
    i for i, row in enumerate(rows) if row["run_id"] in state.selected_run_ids
]
_table_key = "browse_table"

# Store row data so the on_select callback can resolve indices → run IDs
st.session_state["_browse_rows"] = rows


def _on_table_select():
    """Persist table selection to UI state immediately (before rerun).

    Reads the widget state directly from session state (which Streamlit
    updates before the callback fires) and writes the resolved run IDs
    into ``_deap_ui_state_v1``.  This guarantees that Compare / Inspect /
    Replay / Report see the selection even if the user navigates away
    before the rerun completes.
    """
    table_state = st.session_state.get(_table_key, {})
    sel_rows = list(table_state.get("selection", {}).get("rows", []))
    rows_data = st.session_state.get("_browse_rows", [])
    selected_ids = [
        rows_data[i]["run_id"] for i in sel_rows if 0 <= i < len(rows_data)
    ]
    # Update session state directly
    st.session_state.setdefault("_deap_ui_state_v1", {})[
        "selected_run_ids"
    ] = selected_ids


# Initialise widget state for pre-selection on first render.
# Streamlit uses a {selection: {rows: [...], columns: [...]}} dict.
if _table_key not in st.session_state:
    st.session_state[_table_key] = {
        "selection": {"rows": preselected_indices, "columns": []}
    }

event = st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True,
    on_select=_on_table_select,
    selection_mode="multi-row",
    key=_table_key,
)

# Read selection directly from widget state (NOT from event.selection, which can be stale during the on_select-triggered rerun).
_table_state = st.session_state.get(_table_key, {})
selected_indices = list(
    _table_state.get("selection", {}).get("rows", [])
)
selected_run_ids = [rows[i]["run_id"] for i in selected_indices if 0 <= i < len(rows)]
selected_run_dirs = {
    rows[i]["run_id"]: Path(rows[i]["run_dir"])
    for i in selected_indices
    if 0 <= i < len(rows)
}

# Re-sync from session state (callback may have updated it before this rerun)
# and use the widget-state read as the primary source.
# If the widget stateis empty but the callback already persisted a non-empty selection, keep
# the callback's value to guard against stale widget-state reads.
stored = st.session_state.get("_deap_ui_state_v1", {})
callback_ids = list(stored.get("selected_run_ids") or [])
if selected_run_ids:
    state.update_selected(selected_run_ids)
elif callback_ids:
    # Widget state read was empty; trust the callback's persisted value.
    selected_run_ids = callback_ids
    state.selected_run_ids = callback_ids
else:
    state.update_selected([])
state.persist()

# Delete confirmation dialog
if st.session_state.get("_browse_confirm_delete"):
    n = len(state.selected_run_ids)
    st.warning(
        f"**Permanently delete {n} run(s)?** This will remove all files on disk and cannot be undone."
    )
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("✅ Confirm delete", key="_confirm_del_yes"):
        errors = []
        run_dirs_to_delete = []
        for rid in state.selected_run_ids:
            run_dir = selected_run_dirs.get(rid)
            if run_dir is None:
                matching = [r for r in runs if r.run_id == rid]
                run_dir = matching[0].run_dir if matching else None
            if run_dir is None:
                continue
            run_dirs_to_delete.append(str(run_dir))

        if run_dirs_to_delete:
            try:
                result = get_client().delete_runs(run_dirs_to_delete)
                for err in result.get("errors", []):
                    errors.append(f"{err.get('path', '?')}: {err.get('error', 'unknown')}")
            except Exception as exc:
                errors.append(str(exc))
        st.session_state["_browse_confirm_delete"] = False
        state.update_selected([])
        state.persist()
        # Clear the dataframe widget's row-index selection from session state.
        # Without this, stale indices point to different runs after the table
        # reshuffles (deleted rows removed), producing phantom selections.
        st.session_state.pop("browse_table", None)
        _cached_discover.clear()
        if errors:
            for err in errors:
                st.error(err)
        else:
            st.rerun()
    if c2.button("✗ Cancel", key="_confirm_del_no"):
        st.session_state["_browse_confirm_delete"] = False
        st.rerun()

# Manage selected runs
if selected_run_ids:
    with st.expander(f"⚙️ Manage {len(selected_run_ids)} selected run(s)", expanded=False):
        st.markdown("**Rename** — set a human-readable display name.")
        col_name, col_btn = st.columns([3, 1])
        new_name = col_name.text_input(
            "New display name",
            key="browse_rename_input",
            placeholder="Leave blank to clear",
            label_visibility="collapsed",
        )
        if col_btn.button("Rename", key="browse_rename_btn"):
            renamed, errors = 0, []
            for rid in selected_run_ids:
                run_dir = selected_run_dirs.get(rid)
                if run_dir is None:
                    matching = [r for r in runs if r.run_id == rid]
                    run_dir = matching[0].run_dir if matching else None
                if run_dir is None:
                    continue
                try:
                    rename_run(run_dir, new_name)
                    renamed += 1
                except (FileNotFoundError, ValueError) as exc:
                    errors.append(str(exc))
            if renamed:
                st.success(f"Renamed {renamed} run(s).")
                _cached_discover.clear()
                st.rerun()
            for err in errors:
                st.error(err)

        st.divider()
        st.markdown("**Delete** — permanently remove the run directory from disk.")
        if st.button(f"🗑 Delete {len(selected_run_ids)} selected run(s)", key="browse_delete_btn", type="secondary"):
            st.session_state["_browse_confirm_delete"] = True
            st.rerun()

        # --- Re-run (UI-origin experiments only) --------------------------
        st.divider()
        st.markdown("**Re-run** — load an experiment's spec into the Setup page.")
        exp_ids = sorted({
            r.experiment_id for r in runs
            if r.run_id in selected_run_ids and r.experiment_id
        })
        if exp_ids:
            selected_exp = st.selectbox(
                "Experiment", exp_ids,
                key="browse_rerun_exp",
                format_func=lambda eid: f"Experiment {eid}",
            )
            if st.button("🔄 Load spec into Setup", key="browse_rerun_btn"):
                try:
                    client = get_client()
                    result = client.get_experiment_spec(selected_exp)
                    spec = result.get("spec", {})
                    if spec and spec.get("environment"):
                        import json
                        form_defaults = spec_to_form_defaults(spec)
                        form_defaults["_loaded_spec"] = spec
                        state.setup_config = json.dumps(form_defaults)
                        state.persist()
                        st.success(
                            f"Spec loaded from experiment {selected_exp}. "
                            "Go to the Setup page to review and submit."
                        )
                        st.page_link("pages/8_Setup.py", label="Go to Setup page →")
                    else:
                        st.warning("No valid spec found for this experiment.")
                except Exception as exc:
                    st.error(f"Failed to load spec: {exc}")
        else:
            st.caption("Select runs from a UI-origin experiment to enable re-run.")

st.divider()

# --- Export (after table — sees fresh selection immediately) ------------
with st.expander("📦 Export selected runs", expanded=bool(state.selected_run_ids)):
    exp_frames = st.checkbox("Include video recordings", key="browse_exp_frames_main")
    exp_tb = st.checkbox("Include TensorBoard logs", key="browse_exp_tb_main")
    exp_traj = st.checkbox("Include trajectory data", key="browse_exp_traj_main")
    if selected_run_ids:
        run_dirs_list = [
            str(r.run_dir) for r in runs if r.run_id in selected_run_ids
        ]
        if run_dirs_list:
            client = get_client()
            url = client.build_run_export_url(
                run_dirs_list,
                include_frames=exp_frames,
                include_tb=exp_tb,
                include_trajectories=exp_traj,
            )
            st.link_button("⬇ Download selected runs", url)
    else:
        st.caption("Select runs in the table above first.")

# --- Import (main content, below export) --------------------------------
with st.expander("📦 Import experiments"):
    st.markdown("**Import an experiment**")
    _import_widget("browse_import_main_uploader", "browse_import_main_btn")

st.divider()
col_a, col_b = st.columns(2)
col_a.metric("Selected", len(state.selected_run_ids))
if state.selected_run_ids:
    col_b.markdown(
        "Now visit **Compare**, **Diff**, **Replay**, or **Report** in the sidebar."
    )
