"""Replay — play recorded video frames for a selected run."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs

st.set_page_config(page_title="Replay", layout="wide")
st.title("Replay frames")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if not state.selected_run_ids:
    st.info("Pick a run on the **Browse** page first.")
    st.stop()

all_runs = {r.run_id: r for r in discover_runs(settings.results_root)}
options = [rid for rid in state.selected_run_ids if rid in all_runs]
if not options:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

picked = st.selectbox("Run", options)
run_dir = all_runs[picked].run_dir
frames_dir = run_dir / "frames"
if not frames_dir.is_dir():
    st.info(
        f"No `frames/` directory under {run_dir}. Enable frame recording "
        "in your trainer config to populate it."
    )
    st.stop()

video_files = sorted(p for p in frames_dir.iterdir() if p.suffix in {".mp4", ".gif"})
if not video_files:
    st.info(f"`{frames_dir}` is empty.")
    st.stop()

choice = st.selectbox("Episode", video_files, format_func=lambda p: p.name)
if choice.suffix == ".mp4":
    st.video(str(choice))
else:
    st.image(str(choice))
