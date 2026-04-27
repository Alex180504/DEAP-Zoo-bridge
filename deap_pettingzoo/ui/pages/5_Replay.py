"""Replay — play recorded video frames for a selected run."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs

# Matches FrameRecorder filenames `{prefix}episode_NNNN.{ext}`. The
# prefix is captured non-greedily and may be empty (e.g. legacy
# `episode_0001.mp4`). Trainers currently emit `train_`; live-rollout
# capture uses `live_`.
_EPISODE_RE = re.compile(r"^(?P<prefix>\w*?)episode_\d+\.(?:mp4|gif)$")


def _prefix_of(path: Path) -> str:
    """Return the FrameRecorder prefix encoded in *path*'s filename."""
    match = _EPISODE_RE.match(path.name)
    return match.group("prefix") if match else ""

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

video_files = sorted(
    (p for p in frames_dir.iterdir() if p.suffix in {".mp4", ".gif"}),
    key=lambda p: (_prefix_of(p), p.name),
)
if not video_files:
    st.info(f"`{frames_dir}` is empty.")
    st.stop()

# Group artefacts by their FrameRecorder prefix so training captures
# (`train_`) and future live-rollout captures (`live_`) cluster in the
# UI without tripping over each other.
prefixes = sorted({_prefix_of(p) for p in video_files})
if len(prefixes) > 1:
    selected_prefix = st.radio(
        "Source",
        options=prefixes,
        format_func=lambda pfx: pfx.rstrip("_") if pfx else "(unprefixed)",
        horizontal=True,
    )
    video_files = [p for p in video_files if _prefix_of(p) == selected_prefix]

choice = st.selectbox("Episode", video_files, format_func=lambda p: p.name)
if choice.suffix == ".mp4":
    st.video(str(choice))
else:
    st.image(str(choice))
