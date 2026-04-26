"""Diff — show manifest / hyperparam differences between two selected runs.

Highlights only fields that *differ*, with a side-by-side table.
"""

from __future__ import annotations

import json

import streamlit as st

from deap_pettingzoo.analysis.loaders import load_run
from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs

st.set_page_config(page_title="Diff runs", layout="wide")
st.title("Diff runs")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if len(state.selected_run_ids) < 2:
    st.info("Select at least two runs on the **Browse** page first.")
    st.stop()

all_runs = {r.run_id: r for r in discover_runs(settings.results_root)}
options = [rid for rid in state.selected_run_ids if rid in all_runs]
if len(options) < 2:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

col_l, col_r = st.columns(2)
with col_l:
    left_id = st.selectbox("Left run", options, index=0, key="diff_left")
with col_r:
    right_id = st.selectbox(
        "Right run",
        options,
        index=1 if len(options) > 1 else 0,
        key="diff_right",
    )

if left_id == right_id:
    st.warning("Pick two distinct runs.")
    st.stop()

left = load_run(all_runs[left_id].run_dir)
right = load_run(all_runs[right_id].run_dir)


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        else:
            out[key] = v
    return out


def _diff(a: dict, b: dict) -> list[dict]:
    fa = _flatten(a)
    fb = _flatten(b)
    keys = sorted(set(fa) | set(fb))
    rows = []
    for k in keys:
        va = fa.get(k, "—")
        vb = fb.get(k, "—")
        if va == vb:
            continue
        rows.append({"key": k, "left": _to_str(va), "right": _to_str(vb)})
    return rows


def _to_str(v) -> str:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, sort_keys=True)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


tab_h, tab_m, tab_s = st.tabs(["Hyperparams", "Manifest", "Summary"])

with tab_h:
    rows = _diff(
        left.manifest.get("hyperparams") or {},
        right.manifest.get("hyperparams") or {},
    )
    if not rows:
        st.success("Hyperparameters are identical.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_m:
    # Skip noisy fields that always differ (run_id, timestamps, paths).
    skip = {"run_id", "wall_time_s", "schema_version", "artifacts"}
    a = {k: v for k, v in left.manifest.items() if k not in skip}
    b = {k: v for k, v in right.manifest.items() if k not in skip}
    rows = _diff(a, b)
    if not rows:
        st.success("Manifests are identical (excluding run_id/timestamps).")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_s:
    rows = _diff(left.summary, right.summary)
    if not rows:
        st.success("Final summaries are identical.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)
