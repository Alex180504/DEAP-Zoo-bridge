"""Streamlit Home / landing page for deap-bridge results.

Run with::

    python -m deap_pettingzoo.ui            # convenience launcher
    streamlit run deap_pettingzoo/ui/app.py # equivalent

Subsequent page files (``pages/1_Browse.py``, ``pages/2_Compare.py`` …)
are auto-discovered by Streamlit's multipage app convention.

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

st.title("DEAP-bridge results")
st.caption(
    "Browse, compare, and replay experiment runs produced by "
    "`deap_pettingzoo.utils.logging`. The backend never depends on this UI."
)

with st.sidebar:
    st.header("Results root")
    new_root = st.text_input(
        "Directory",
        value=str(settings.results_root),
        help=(
            f"Override via env var ${DEFAULT_ROOT_ENV_VAR} or by typing here."
        ),
    )
    if new_root and new_root != str(settings.results_root):
        from pathlib import Path as _Path

        state.results_root = _Path(new_root).expanduser().resolve()
        settings = load_settings(results_root=new_root)

    st.divider()
    st.markdown(
        "Pages\n\n"
        "- **Browse** – pick runs to compare\n"
        "- **Compare** – aggregate curves + pairwise tests\n"
        "- **Diff** – side-by-side manifest / hyperparam diff\n"
        "- **Hyperparams** – effect-size sweeps\n"
        "- **Replay** – play recorded frames\n"
        "- **Live Rollout** – run a saved policy in a fresh env\n"
        "- **Report** – render the HTML/PDF report"
    )

# ----- Body ----------------------------------------------------------------
if not settings.results_root_exists:
    st.warning(
        f"Results root **{settings.results_root}** does not exist yet. "
        "Train a run with the experiment-logging stack, or update the "
        "directory in the sidebar."
    )
    st.stop()

runs = discover_runs(settings.results_root)
if not runs:
    st.info(
        f"No `manifest.json` files found under **{settings.results_root}** "
        "(checked the directory and one level deep)."
    )
    st.stop()

st.subheader(f"{len(runs)} run(s) discovered")
algorithms = sorted({r.algorithm for r in runs})
envs = sorted({r.env_name for r in runs})
col1, col2, col3 = st.columns(3)
col1.metric("Algorithms", len(algorithms))
col2.metric("Environments", len(envs))
col3.metric("Total wall time (s)",
            f"{sum((r.wall_time_s or 0.0) for r in runs):.1f}")

st.markdown("Use the **Browse** page (left sidebar) to filter and select runs.")
state.persist()
