"""Hyperparams — group selected runs by hyperparam value, show effect size."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from deap_pettingzoo.analysis import iqm, mean_with_ci
from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs, load_experiment_cached

st.set_page_config(page_title="Hyperparams", layout="wide")
st.title("Hyperparam sweep")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if not state.selected_run_ids:
    st.info("Pick runs on the **Browse** page first.")
    st.stop()


all_runs = {r.run_id: r for r in discover_runs(settings.results_root)}
selected = [all_runs[r] for r in state.selected_run_ids if r in all_runs]
if not selected:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

experiment = load_experiment_cached([s.run_dir for s in selected])

# Collect hyperparam keys across all runs, with their distinct value counts.
key_values: dict[str, set[str]] = {}
for run in experiment:
    for k, v in (run.manifest.get("hyperparams") or {}).items():
        key_values.setdefault(str(k), set()).add(str(v))

if not key_values:
    st.info("None of the selected runs recorded hyperparams.")
    st.stop()

# Only keys that actually vary are useful sweep targets.
varying_keys = sorted(k for k, vs in key_values.items() if len(vs) > 1)
if not varying_keys:
    st.warning(
        "None of the recorded hyperparams vary across the selected runs — "
        "every run uses the same value for each key, so a sweep would just "
        "produce a single bar. Select runs from a sweep to use this page."
    )
    with st.expander("Recorded hyperparams (constant across all selected runs)"):
        st.json({k: next(iter(vs)) for k, vs in sorted(key_values.items())})
    st.stop()

# Determine y-metric candidates: prefer summary keys present in any run.
_metric_candidates: list[str] = []
_seen_metrics: set[str] = set()
for run in experiment:
    for k, v in (run.summary or {}).items():
        if k in _seen_metrics:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            _metric_candidates.append(k)
            _seen_metrics.add(k)
_preferred_metric = next(
    (m for m in ("final_fitness", "best_fitness", "mean_episode_reward",
                 "mean", "max") if m in _seen_metrics),
    _metric_candidates[0] if _metric_candidates else "final_fitness",
)

with st.sidebar:
    sweep_key = st.selectbox("Sweep on", varying_keys)
    if _metric_candidates:
        metric = st.selectbox(
            "Metric", _metric_candidates,
            index=_metric_candidates.index(_preferred_metric),
        )
    else:
        metric = st.text_input("Metric (final-summary key)",
                                value="final_fitness")

# Group runs by the chosen key's value.
groups: Dict[str, List[float]] = {}
for run in experiment:
    hp = run.manifest.get("hyperparams") or {}
    if sweep_key not in hp:
        continue
    bucket = str(hp[sweep_key])
    val = run.summary.get(metric)
    if val is None:
        val = run.final_fitness
    if val is None:
        continue
    try:
        groups.setdefault(bucket, []).append(float(val))
    except (TypeError, ValueError):
        continue

if not groups:
    st.warning(
        f"No runs had both `{sweep_key}` set and a numeric `{metric}` value."
    )
    st.stop()

rows = []
for bucket, vals in sorted(groups.items()):
    mean, lo, hi = mean_with_ci(
        vals,
        n_bootstrap=int(state.n_resamples),
        ci=float(state.ci_level),
    )
    rows.append(
        {
            sweep_key: bucket,
            "n_runs": len(vals),
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "iqm": iqm(vals),
            "min": min(vals),
            "max": max(vals),
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

try:
    import plotly.graph_objects as go

    fig = go.Figure()
    # Box (or single-point marker) plus per-run jittered dots so single-value
    # buckets are still visible rather than rendering as flat zero-width bars.
    for r in rows:
        bucket = str(r[sweep_key])
        vals = groups[bucket]
        fig.add_trace(
            go.Box(
                y=vals,
                name=bucket,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(size=8, opacity=0.85),
                line=dict(width=1.2),
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title=sweep_key,
        yaxis_title=metric,
        height=420,
        boxmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    pass
