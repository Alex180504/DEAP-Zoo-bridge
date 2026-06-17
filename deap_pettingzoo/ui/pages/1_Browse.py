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
    delete_run,
    discover_runs,
    load_settings,
    move_run,
    rename_run,
)

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


@st.cache_resource(show_spinner=False)
def _cached_discover(root: str, mtime_bucket: float):
    """Re-runs only when ``root`` or any ``manifest.json`` mtime changes."""
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
if not runs:
    st.info(f"No runs found under **{root}**.")
    st.stop()

# Sidebar filters & management 
algos = sorted({r.algorithm for r in runs})
envs = sorted({r.env_name for r in runs})

# Collect dates that have at least one run for min/max defaults
run_dates = [r.started_at.date() for r in runs if r.started_at is not None]

with st.sidebar:
    st.header("Filters")
    sel_algos = st.multiselect("Algorithms", algos, default=algos)
    sel_envs = st.multiselect("Environments", envs, default=envs)

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
        # List first-level subdirs that are NOT run dirs (no manifest.json)
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
                for rid in list(state.selected_run_ids):
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
                    st.cache_resource.clear()
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
    }
    for r in filtered
]

# Pre-select rows whose run_id is already in state.selected_run_ids.
preselected_indices = [
    i for i, row in enumerate(rows) if row["run_id"] in state.selected_run_ids
]
_table_key = "browse_table"
if _table_key not in st.session_state:
    st.session_state[_table_key] = {
        "selection": {"rows": preselected_indices, "columns": []}
    }

event = st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    key=_table_key,
)

selected_indices = list(event.selection.rows) if event and event.selection else []
selected_run_ids = [rows[i]["run_id"] for i in selected_indices if 0 <= i < len(rows)]
selected_run_dirs = {
    rows[i]["run_id"]: Path(rows[i]["run_dir"])
    for i in selected_indices
    if 0 <= i < len(rows)
}
state.update_selected(selected_run_ids)
state.persist()

# Delete confirmation dialog
if st.session_state.get("_browse_confirm_delete"):
    n = len(state.selected_run_ids)
    st.warning(
        f"**Permanently delete {n} run(s)?** This removes all files on disk and cannot be undone."
    )
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("✅ Confirm delete", key="_confirm_del_yes"):
        errors = []
        for rid in list(state.selected_run_ids):
            run_dir = selected_run_dirs.get(rid)
            if run_dir is None:
                # Locate from full run list
                matching = [r for r in runs if r.run_id == rid]
                run_dir = matching[0].run_dir if matching else None
            if run_dir is None:
                continue
            try:
                delete_run(run_dir)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
        st.session_state["_browse_confirm_delete"] = False
        state.update_selected([])
        state.persist()
        st.cache_resource.clear()
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
        st.markdown("**Rename** — sets a human-readable display name (stored in manifest).")
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
                st.cache_resource.clear()
                st.rerun()
            for err in errors:
                st.error(err)

        st.divider()
        st.markdown("**Delete** — permanently removes the run directory from disk.")
        if st.button(f"🗑 Delete {len(selected_run_ids)} selected run(s)", key="browse_delete_btn", type="secondary"):
            st.session_state["_browse_confirm_delete"] = True
            st.rerun()

st.divider()
col_a, col_b = st.columns(2)
col_a.metric("Selected", len(state.selected_run_ids))
if state.selected_run_ids:
    col_b.markdown(
        "Now visit **Compare**, **Diff**, **Replay**, or **Report** in the sidebar."
    )
