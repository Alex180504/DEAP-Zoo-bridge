"""Streamlit Home / landing page for deap-bridge results.

Run with::

    python -m deap_pettingzoo.ui            # convenience launcher
    streamlit run deap_pettingzoo/ui/app.py # equivalent

Uses :func:`streamlit.navigation` to organise pages into three groups:

* **Experiment** — Setup (configure + submit), Queue (monitor runs).
* **Visualize** — Browse, Compare, Replay, Report (read-only artefacts).
* **Inspect** — Diff + Sweep (side-by-side and hyperparam effect analysis).

This module imports ``streamlit`` at top level on purpose: it is the
file that Streamlit *executes*, not one we import from elsewhere. The
backend, analysis library, and tests never import this file.
"""

from __future__ import annotations

import streamlit as st

from deap_pettingzoo.ui.services import (
    DEFAULT_ROOT_ENV_VAR,
    UIState,
    discover_runs,
    load_settings,
)

st.set_page_config(
    page_title="DEAP-bridge results",
    page_icon="🧬",
    layout="wide",
)

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)
state.results_root = settings.results_root

# ---------------------------------------------------------------------------
# Page definitions (st.navigation groups)
# ---------------------------------------------------------------------------

_PAGES_DIR = "deap_pettingzoo/ui/pages"

pages = {
    "Experiment": [
        st.Page(f"{_PAGES_DIR}/8_Setup.py", title="Setup", icon="⚙️"),
        st.Page(f"{_PAGES_DIR}/9_Queue.py", title="Queue", icon="📋"),
    ],
    "Visualize": [
        st.Page(f"{_PAGES_DIR}/1_Browse.py", title="Browse", icon="🔍"),
        st.Page(f"{_PAGES_DIR}/2_Compare.py", title="Compare", icon="📊"),
        st.Page(f"{_PAGES_DIR}/6_Replay.py", title="Replay", icon="🎬"),
        st.Page(f"{_PAGES_DIR}/7_Report.py", title="Report", icon="📄"),
    ],
    "Inspect": [
        st.Page(f"{_PAGES_DIR}/3_Inspect.py", title="Inspect", icon="🔬"),
    ],
}

pg = st.navigation(pages, position="sidebar")

# ---------------------------------------------------------------------------
# Sidebar — results-root picker (below navigation)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.header("Results root")
    new_root = st.text_input(
        "Directory",
        value=str(settings.results_root),
        help=f"Override via env var ${DEFAULT_ROOT_ENV_VAR} or by typing here.",
    )
    if new_root and new_root != str(settings.results_root):
        from pathlib import Path as _Path
        state.results_root = _Path(new_root).expanduser().resolve()
        settings = load_settings(results_root=new_root)

# ---------------------------------------------------------------------------
# Home / landing body (shown when no page is selected, i.e. base URL)
# ---------------------------------------------------------------------------

st.title("DEAP-bridge")
st.caption(
    "Browse, compare, and replay experiment runs. "
    "Configure and submit new experiments via the **Setup** page, "
    "or monitor the queue on **Queue**."
)

if not settings.results_root_exists:
    st.warning(
        f"Results root **{settings.results_root}** does not exist yet. "
        "Train a run with the experiment-logging stack, or update the "
        "directory in the sidebar."
    )
else:
    runs = discover_runs(settings.results_root)
    if not runs:
        st.info(
            f"No `manifest.json` files found under **{settings.results_root}** "
            "(checked the directory and one level deep)."
        )
    else:
        st.subheader(f"{len(runs)} run(s) discovered")
        algorithms = sorted({r.algorithm for r in runs})
        envs = sorted({r.env_name for r in runs})
        col1, col2, col3 = st.columns(3)
        col1.metric("Algorithms", len(algorithms))
        col2.metric("Environments", len(envs))
        col3.metric("Total wall time (s)",
                    f"{sum((r.wall_time_s or 0.0) for r in runs):.1f}")
        st.markdown(
            "Use the **Browse** page in the sidebar to filter and select runs."
        )

# ---------------------------------------------------------------------------
# Run the selected page (if any — home body stays visible when none selected)
# ---------------------------------------------------------------------------

if pg is not None:
    pg.run()

state.persist()
