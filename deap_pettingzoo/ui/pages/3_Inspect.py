"""Inspect: diff runs and sweep hyperparameters.

Merges the former *Diff* and *Hyperparams* pages into one page with two tabs:

* **Diff** — side-by-side manifest / hyperparameter / summary comparison of
  two selected runs. Only differing fields are shown.
* **Sweep** — group selected runs by a varying hyperparameter value and
  show effect-size box plots with bootstrap CIs.

Both tabs operate on the same selected-run pool from the **Browse** page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

from deap_pettingzoo.analysis import iqm, mean_with_ci
from deap_pettingzoo.analysis.loaders import load_run
from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs, load_experiment_cached

st.set_page_config(page_title="Inspect runs", layout="wide")
st.title("Inspect runs")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if not state.selected_run_ids:
    st.info("Select runs on the **Browse** page first.")
    st.stop()

all_runs = {r.run_id: r for r in discover_runs(settings.results_root)}
selected_ids = [rid for rid in state.selected_run_ids if rid in all_runs]
if not selected_ids:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

selected_runs = [all_runs[rid] for rid in selected_ids]
experiment = load_experiment_cached([s.run_dir for s in selected_runs])

# Helpers

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
        va = fa.get(k, "\u2014")
        vb = fb.get(k, "\u2014")
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

# Tabs

tab_diff, tab_sweep = st.tabs(["Diff", "Sweep"])

# Diff tab

with tab_diff:
    if len(selected_ids) < 2:
        st.info("Select at least two runs on the **Browse** page to diff them.")
    else:
        options = selected_ids
        col_l, col_r = st.columns(2)
        with col_l:
            left_id = st.selectbox("Left run", options, index=0, key="diff_left")
        with col_r:
            right_id = st.selectbox(
                "Right run", options,
                index=min(1, len(options) - 1),
                key="diff_right",
            )

        if left_id == right_id:
            st.warning("Pick two distinct runs to compare.")
        else:
            left = load_run(all_runs[left_id].run_dir)
            right = load_run(all_runs[right_id].run_dir)

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
                skip = {"run_id", "wall_time_s", "schema_version", "artifacts"}
                a = {k: v for k, v in left.manifest.items() if k not in skip}
                b = {k: v for k, v in right.manifest.items() if k not in skip}
                rows = _diff(a, b)
                if not rows:
                    st.success(
                        "Manifests are identical (excluding run_id/timestamps)."
                    )
                else:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

            with tab_s:
                rows = _diff(left.summary, right.summary)
                if not rows:
                    st.success("Final summaries are identical.")
                else:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

# Hyperparameter sweep tab

with tab_sweep:
    # Collect hyperparam keys across all runs, with their distinct value counts.
    key_values: dict[str, set[str]] = {}
    for run in experiment:
        for k, v in (run.manifest.get("hyperparams") or {}).items():
            key_values.setdefault(str(k), set()).add(str(v))

    if not key_values:
        st.info("None of the selected runs recorded hyperparams.")
    else:
        varying_keys = sorted(k for k, vs in key_values.items() if len(vs) > 1)
        if not varying_keys:
            st.warning(
                "None of the recorded hyperparams vary across the selected "
                "runs — every run uses the same value for each key, so a "
                "sweep would just produce a single bar. Select runs from a "
                "sweep to use this tab."
            )
            with st.expander(
                "Recorded hyperparams (constant across all selected runs)"
            ):
                st.json(
                    {k: next(iter(vs)) for k, vs in sorted(key_values.items())}
                )
        else:
            # Determine y-metric candidates.
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
                (
                    m
                    for m in (
                        "final_fitness",
                        "best_fitness",
                        "mean_episode_reward",
                        "mean",
                        "max",
                    )
                    if m in _seen_metrics
                ),
                _metric_candidates[0] if _metric_candidates else "final_fitness",
            )

            with st.sidebar:
                st.subheader("Sweep controls")
                sweep_key = st.selectbox("Sweep on", varying_keys)
                if _metric_candidates:
                    metric = st.selectbox(
                        "Metric",
                        _metric_candidates,
                        index=_metric_candidates.index(_preferred_metric)
                        if _preferred_metric in _metric_candidates
                        else 0,
                    )
                else:
                    metric = st.text_input(
                        "Metric (final-summary key)", value="final_fitness"
                    )

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
                    f"No runs had both `{sweep_key}` set and a numeric "
                    f"`{metric}` value."
                )
            else:
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

state.persist()
