"""Queue — monitor the experiment queue and running experiments.

Polls the backend (FastAPI service) every ~2 seconds to show:
- A queue-status bar (paused state, running experiment id).
- All experiments with status badges, algorithm/env, submit time, and
  cancel buttons.
- For the currently-running experiment: a progress bar (both overall
  run k/N and per-run generation/env-step progress), live metrics,
  and a scrollable stdout log.
- Queue-level pause / resume buttons.

Completed experiments are immediately discoverable by the existing
Browse / Compare / Replay pages — no migration needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.api_client import ApiClient, ApiUnavailable
from deap_pettingzoo.ui.services.queue_helpers import (
    algo_label as _algo_label,
    env_label as _env_label,
    format_time as _format_time,
    progress_description,
    short_id as _short_id,
    status_badge as _status_badge,
)

st.set_page_config(page_title="Experiment queue", layout="wide")
st.title("Experiment queue")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

client = ApiClient()

# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------

POLL_INTERVAL = 2  # seconds for the auto-refresh fragment


# ---------------------------------------------------------------------------
# Auto-refreshing fragment — the body of the page re-runs every 2 s.
# ---------------------------------------------------------------------------

@st.fragment(run_every=POLL_INTERVAL)
def _queue_body() -> None:
    # -------------------------------------------------------------------
    # Fetch data
    # -------------------------------------------------------------------

    experiments: List[Dict[str, Any]] = []
    queue_state: Dict[str, Any] = {
        "paused": False, "running_id": None, "queued_ids": [], "worker_limit": 1,
    }

    try:
        experiments = client.list_experiments(limit=200)
        queue_state = client.get_queue()
    except ApiUnavailable:
        st.warning(
            "Cannot reach the experiment service backend. "
            "Start it with `python -m deap_pettingzoo.service` "
            "or set `FASTAPI_URL` to point at a running instance."
        )
        st.stop()
    except Exception:
        pass  # use the defaults above

    # -------------------------------------------------------------------
    # Queue control bar
    # -------------------------------------------------------------------

    col_pause, col_resume, col_spacer = st.columns([1, 1, 6])

    paused: bool = queue_state.get("paused", False)
    with col_pause:
        if st.button("⏸ Pause queue", disabled=paused, use_container_width=True):
            try:
                client.pause_queue()
            except Exception:
                st.toast("Failed to pause queue.")
            st.rerun()

    with col_resume:
        if st.button("▶ Resume queue", disabled=not paused, use_container_width=True):
            try:
                client.resume_queue()
            except Exception:
                st.toast("Failed to resume queue.")
            st.rerun()

    running_id: Optional[str] = queue_state.get("running_id")
    queued_ids: List[str] = queue_state.get("queued_ids", [])

    if paused:
        st.info("Queue is **paused** — no new experiments will start.")
    if running_id:
        st.caption(f"Running: `{_short_id(running_id)}` | Queued: {len(queued_ids)}")
    elif queued_ids:
        st.caption(f"Queued: {len(queued_ids)} experiments waiting.")

    # -------------------------------------------------------------------
    # Active experiment (progress + stdout)
    # -------------------------------------------------------------------

    running_exp: Optional[Dict[str, Any]] = None
    for exp in experiments:
        if exp.get("status") == "running":
            running_exp = exp
            break

    if running_exp:
        exp_id = running_exp["id"]
        spec = running_exp.get("spec") or {}
        st.subheader(
            f"Active: {_algo_label(spec)} on {_env_label(spec)} "
            f"(`{_short_id(exp_id)}`)",
            divider="blue",
        )

        # Fetch live progress + stdout.
        try:
            progress_data = client.get_progress(exp_id)
        except Exception:
            progress_data = running_exp.get("progress") or {}

        col_prog, col_metrics = st.columns([2, 1])

        with col_prog:
            if progress_data:
                unit = progress_data.get("unit", "generation")
                current = int(progress_data.get("current", 0))
                total = int(progress_data.get("total", 1))
                run_idx = int(progress_data.get("run_index", 0)) + 1
                total_runs = int(progress_data.get("total_runs", 1))
                algo = progress_data.get("algorithm", "")
                seed = progress_data.get("seed", "")

                # Compute fraction from current/total (runner doesn't send it).
                fraction = (current / total) if total > 0 else 0.0

                sub_label = (
                    f"Run {run_idx}/{total_runs} — {algo} s{seed}  "
                    f"({unit} {current}/{total})"
                )
                st.progress(fraction, text=sub_label)
            else:
                st.info("Waiting for first progress update…")

        with col_metrics:
            if progress_data:
                bf = progress_data.get("best_fitness")
                mf = progress_data.get("mean_fitness")
                wt = progress_data.get("wall_time_s", 0)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Best fitness", f"{bf:.3f}" if bf is not None else "—")
                with col_b:
                    st.metric("Mean fitness", f"{mf:.3f}" if mf is not None else "—")
                st.caption(f"Elapsed: {wt:.0f}s")

        # Stdout log
        with st.expander("Live output (_stdout.log)", expanded=False):
            try:
                chunk = client.get_stdout(exp_id, lines=50)
                if chunk.get("text"):
                    st.code(chunk["text"], language="text")
                else:
                    st.caption("No output yet…")
            except Exception:
                st.caption("Unable to fetch stdout.")

        # Cancel button
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🛑 Cancel experiment", type="secondary", key=f"cancel_{exp_id}"):
                try:
                    client.cancel_experiment(exp_id)
                    st.toast("Cancellation requested.")
                except Exception:
                    st.toast("Failed to cancel experiment.")
                st.rerun()

        st.divider()

    # -------------------------------------------------------------------
    # All experiments list
    # -------------------------------------------------------------------

    st.subheader("All experiments", divider="gray")

    if not experiments:
        st.info(
            "No experiments submitted yet. "
            "Go to the **Setup** page to configure and submit one."
        )
    else:
        # Build a compact table.
        rows: List[Dict[str, Any]] = []
        for exp in experiments:
            spec = exp.get("spec") or {}
            rows.append(
                {
                    "ID": _short_id(exp["id"]),
                    "Status": _status_badge(exp.get("status", "?")),
                    "Algorithms": _algo_label(spec),
                    "Environment": _env_label(spec),
                    "Created": _format_time(exp.get("created_at")),
                    "position": exp.get("position", -1),
                    "_id": exp["id"],
                    "_status": exp.get("status", ""),
                }
            )

        # Separate active+queued from terminal.
        active_rows = [r for r in rows if r["_status"] in ("queued", "running")]
        terminal_rows = [r for r in rows if r["_status"] not in ("queued", "running")]

        def _render_table(title: str, table_rows: List[Dict[str, Any]]) -> None:
            if not table_rows:
                return
            st.markdown(f"**{title}** ({len(table_rows)})")
            for row in table_rows:
                c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 2, 3, 3, 2])
                with c1:
                    st.caption(row["ID"])
                with c2:
                    st.caption(row["Status"])
                with c3:
                    st.caption(row["Algorithms"])
                with c4:
                    st.caption(row["Environment"])
                with c5:
                    st.caption(row["Created"])
                with c6:
                    cancelable = row["_status"] in ("queued", "running")
                    if cancelable:
                        if st.button("Cancel", key=f"cancel_list_{row['_id']}"):
                            try:
                                client.cancel_experiment(row["_id"])
                                st.toast(f"Cancelled {row['_id'][:8]}…")
                            except Exception:
                                st.toast("Cancel failed.")
                            st.rerun()
                    elif row["_status"] == "completed":
                        st.caption("✅ Done")
                    else:
                        st.caption(row["Status"])

        _render_table("Active / queued", active_rows)
        if terminal_rows:
            with st.expander("Completed / failed / cancelled", expanded=False):
                _render_table("Past", terminal_rows)

        st.caption(
            "Completed experiments appear in the **Browse** page automatically — "
            "their run directories are under your results root."
        )


# ---------------------------------------------------------------------------
# Run the auto-refreshing body + persist state
# ---------------------------------------------------------------------------

_queue_body()
state.persist()
