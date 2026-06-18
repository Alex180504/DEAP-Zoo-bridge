"""Replay — play recorded video frames for a selected run.

Recordings written by :class:`FrameRecorder` since the iteration-metadata
feature carry a ``frames/recordings.json`` sidecar mapping each artefact to
the training iteration/generation (and env-step count) it captured. When
present, the episode selector is annotated with that iteration and a headline
fitness value, and a detail panel below the player shows the full metrics row
plus a trend chart with the recording's iteration marked. Recordings made
before the feature (no sidecar entry) play back with their bare filename and
no iteration association.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs

# Matches FrameRecorder filenames `{prefix}episode_NNNN.{ext}`. The
# prefix is captured non-greedily and may be empty (e.g. legacy
# `episode_0001.mp4`). Trainers currently emit `train_`; live-rollout
# capture uses `live_`.
_EPISODE_RE = re.compile(r"^(?P<prefix>.*?)episode_\d+\.(?:mp4|gif)$")

# Columns tried (in order) for the one-line headline value shown next to a
# recording in the selector.
_HEADLINE_COLS = ("best_fitness", "max", "mean", "qd_score")


def _prefix_of(path: Path) -> str:
    """Return the FrameRecorder prefix encoded in *path*'s filename."""
    match = _EPISODE_RE.match(path.name)
    return match.group("prefix") if match else ""


@st.cache_data(show_spinner=False)
def _load_recordings_meta(frames_dir_str: str, mtime: float) -> Dict[str, dict]:
    """Load ``frames/recordings.json`` into a ``{filename: entry}`` map.

    ``mtime`` is part of the cache key so edits invalidate the cache.
    Returns an empty map when the sidecar is absent or unreadable.
    """
    path = Path(frames_dir_str) / "recordings.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, dict] = {}
    for entry in data if isinstance(data, list) else []:
        name = entry.get("file") if isinstance(entry, dict) else None
        if name:
            out[str(name)] = entry
    return out


@st.cache_data(show_spinner=False)
def _load_metrics(run_dir_str: str, mtime: float):
    """Load the run's per-iteration metrics as a DataFrame (or ``None``).

    Goes through :func:`analysis.loaders.load_run` so RL runs get their
    synthesized metrics frame just like the rest of the UI.
    """
    from deap_pettingzoo.analysis.loaders import load_run

    try:
        rec = load_run(Path(run_dir_str))
    except (FileNotFoundError, ValueError):
        return None
    return rec.metrics


def _metrics_rows(metrics, entry: Optional[dict]):
    """Return the metrics row(s) tied to *entry*, or ``None``.

    Matches by ``iteration`` first (exact — EA/CMA/PSO/NEAT/coevolution),
    falling back to the nearest ``env_steps`` row (RL/QDax, whose
    ``iteration`` axis is the env-step count). Coevolution frames keep a
    ``side`` column, so multiple rows may be returned for one iteration.
    """
    if metrics is None or entry is None or len(metrics) == 0:
        return None
    it = entry.get("iteration")
    if it is not None and "iteration" in metrics.columns:
        rows = metrics[metrics["iteration"] == it]
        if not rows.empty:
            return rows
    es = entry.get("env_steps")
    if es is not None and "env_steps" in metrics.columns:
        idx = (metrics["env_steps"] - es).abs().idxmin()
        return metrics.loc[[idx]]
    return None


def _headline(rows) -> Optional[str]:
    """Pick a short ``col value`` headline from the matched row(s)."""
    if rows is None or rows.empty:
        return None
    for col in _HEADLINE_COLS:
        if col in rows.columns:
            try:
                return f"{col} {float(rows[col].iloc[0]):.3g}"
            except (TypeError, ValueError):
                continue
    return None


def _label_for(path: Path, entry: Optional[dict], rows) -> str:
    """Selector label: filename plus iteration/headline when known."""
    if entry is None:
        return path.name
    it = entry.get("iteration")
    bits = []
    if it is not None:
        bits.append(f"iter {it}")
    head = _headline(rows)
    if head:
        bits.append(head)
    return f"{path.name} — {' · '.join(bits)}" if bits else path.name


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

# Sidecar iteration metadata + per-iteration metrics (both best-effort).
sidecar_path = frames_dir / "recordings.json"
sidecar_mtime = sidecar_path.stat().st_mtime if sidecar_path.is_file() else 0.0
recordings_meta = _load_recordings_meta(str(frames_dir), sidecar_mtime)
manifest_path = run_dir / "manifest.json"
manifest_mtime = manifest_path.stat().st_mtime if manifest_path.is_file() else 0.0
metrics = _load_metrics(str(run_dir), manifest_mtime)

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

# Precompute labels so the selectbox shows iteration + headline fitness.
labels = {
    p: _label_for(p, recordings_meta.get(p.name), _metrics_rows(metrics, recordings_meta.get(p.name)))
    for p in video_files
}

choice = st.selectbox("Episode", video_files, format_func=lambda p: labels[p])

# Replay window is capped to a default width so it doesn't fill the wide
# layout; the slider lets the user resize it (down to a thumbnail or up to
# near-full-width).
video_width = st.slider(
    "Replay size (px)",
    min_value=320,
    max_value=1280,
    value=720,
    step=20,
    help="Cap and resize the replay window width.",
)
# Mid-gray border so the window edges are visible against MPE's white
# background in both light and dark themes. st.video takes no style arg,
# so style the video element via CSS. A bare `video` selector is used
# (rather than a version-specific data-testid) since this page renders a
# single replay video.
_REPLAY_BORDER = "2px solid #888"
st.markdown(
    f"<style>video{{border:{_REPLAY_BORDER};border-radius:4px;"
    f"box-sizing:border-box;}}</style>",
    unsafe_allow_html=True,
)
if choice.suffix == ".mp4":
    try:
        st.video(str(choice), width=video_width)
    except TypeError:
        # Older Streamlit (<1.56) has no `width` arg on st.video.
        st.video(str(choice))
else:
    # st.image() strips animation data when decoding through Pillow.
    # Embed the GIF as a base64 data-URI so the browser animates it.
    b64 = base64.b64encode(choice.read_bytes()).decode()
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:{video_width}px;max-width:100%;height:auto;'
        f'border:{_REPLAY_BORDER};border-radius:4px;box-sizing:border-box">',
        unsafe_allow_html=True,
    )

# ---- Iteration metrics detail panel -----------------------------------
st.divider()
st.subheader("Iteration metrics")

entry = recordings_meta.get(choice.name)
rows = _metrics_rows(metrics, entry)
if entry is None:
    st.caption("No iteration metadata recorded for this artefact.")
else:
    meta_bits = []
    if entry.get("iteration") is not None:
        meta_bits.append(f"**iteration** {entry['iteration']}")
    if entry.get("env_steps") is not None:
        meta_bits.append(f"**env steps** {entry['env_steps']:,}")
    if entry.get("wall_time") is not None:
        meta_bits.append(f"**wall time** {entry['wall_time']:.1f}s")
    if meta_bits:
        st.markdown(" · ".join(meta_bits))

    if rows is None or rows.empty:
        st.caption("No matching metrics row for this iteration.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Trend chart: metric curve over training with this iteration marked.
        try:
            import plotly.graph_objects as go

            x_col = "iteration" if "iteration" in metrics.columns else "env_steps"
            y_col = next(
                (c for c in ("mean", "max", "best_fitness", "qd_score")
                 if c in metrics.columns),
                None,
            )
            if x_col in metrics.columns and y_col is not None:
                fig = go.Figure()
                if "side" in metrics.columns:
                    for side, grp in metrics.groupby("side"):
                        fig.add_scatter(
                            x=grp[x_col], y=grp[y_col], mode="lines", name=str(side)
                        )
                else:
                    fig.add_scatter(
                        x=metrics[x_col], y=metrics[y_col], mode="lines", name=y_col
                    )
                marker_x = rows[x_col].iloc[0]
                fig.add_vline(
                    x=marker_x, line_dash="dash", line_color="crimson",
                    annotation_text="this recording",
                )
                fig.update_layout(
                    xaxis_title=x_col, yaxis_title=y_col, height=360,
                    margin={"l": 40, "r": 20, "t": 20, "b": 40},
                )
                st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install `plotly` for the trend chart.")
        except Exception as exc:  # noqa: BLE001 - chart is best-effort
            st.caption(f"Trend chart unavailable: {exc}")
