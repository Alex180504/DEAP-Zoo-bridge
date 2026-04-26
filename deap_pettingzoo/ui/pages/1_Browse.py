"""Browse — filter discovered runs and select which to compare/replay.

Selection writes through to :class:`UIState.selected_run_ids`, which the
Compare / Diff / Replay / Live / Report pages all read from.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from deap_pettingzoo.ui.services import (
    UIState,
    discover_runs,
    load_settings,
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

# ----- filters -------------------------------------------------------------
algos = sorted({r.algorithm for r in runs})
envs = sorted({r.env_name for r in runs})

with st.sidebar:
    st.header("Filters")
    sel_algos = st.multiselect("Algorithms", algos, default=algos)
    sel_envs = st.multiselect("Environments", envs, default=envs)

filtered = [r for r in runs if r.algorithm in sel_algos and r.env_name in sel_envs]
st.caption(f"{len(filtered)} of {len(runs)} runs match.")

# ----- table ---------------------------------------------------------------
# We use ``st.dataframe`` with native multi-row selection rather than
# ``st.data_editor`` checkboxes. The data_editor approach is prone to a
# Streamlit quirk where the cached "edited_rows" diff is re-applied on top of
# new input, so a second checkbox click can be silently dropped or misattributed
# when the underlying ``rows`` list changes between reruns. Native row
# selection is purpose-built for this and stores indices, not diffs.
rows = [
    {
        "run_id": r.run_id,
        "algorithm": r.algorithm,
        "env_name": r.env_name,
        "seed": r.seed,
        "wall_time_s": r.wall_time_s,
        "final_fitness": r.final_summary.get("final_fitness")
        or r.final_summary.get("best_fitness")
        or r.final_summary.get("mean_episode_reward"),
        "run_dir": str(r.run_dir),
    }
    for r in filtered
]

# Pre-select the rows whose run_id is already in state.selected_run_ids so the
# selection persists across page navigation.
preselected_indices = [
    i for i, row in enumerate(rows) if row["run_id"] in state.selected_run_ids
]
# Streamlit reads the initial selection from session_state under
# "<widget_key>.selection". Set it once if the widget hasn't been rendered yet
# this session, otherwise leave the user's most recent selection untouched.
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
selected = [rows[i]["run_id"] for i in selected_indices if 0 <= i < len(rows)]
state.update_selected(selected)
state.persist()

st.divider()
col_a, col_b = st.columns(2)
col_a.metric("Selected", len(state.selected_run_ids))
if state.selected_run_ids:
    col_b.markdown(
        "Now visit **Compare**, **Diff**, **Replay**, **Live Rollout**, "
        "or **Report** in the sidebar."
    )
