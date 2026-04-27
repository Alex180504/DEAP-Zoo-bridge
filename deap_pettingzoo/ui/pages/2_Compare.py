"""Compare — aggregate curves and pairwise tests across selected runs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st
from deap_pettingzoo.analysis import (
    learning_curve_with_ci,
    pairwise_table,
)
from deap_pettingzoo.ui.services import (
    UIState,
    load_experiment_cached,
    load_settings,
)

st.set_page_config(page_title="Compare runs", layout="wide")
st.title("Compare runs")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if not state.selected_run_ids:
    st.info("Pick runs on the **Browse** page first.")
    st.stop()


@st.cache_resource(show_spinner="Loading runs…")
def _cached_experiment(run_dir_strs: tuple[str, ...]):
    return load_experiment_cached([Path(p) for p in run_dir_strs])


# Resolve selected run_ids → run_dirs by re-discovering (cheap, cached above).
from deap_pettingzoo.ui.services.data import discover_runs

all_runs = discover_runs(settings.results_root)
selected_run_dirs = [
    r.run_dir for r in all_runs if r.run_id in state.selected_run_ids
]
if not selected_run_dirs:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

experiment = _cached_experiment(tuple(str(p) for p in selected_run_dirs))

# ----- controls ------------------------------------------------------------
# Discover columns actually present across the selected runs' metrics.csv so
# the user can't pick a non-existent y/x column (which would surface as
# "No runs contained both '<x>' and '<y>'").
_y_candidates: list[str] = []
_x_candidates: list[str] = []
_seen_y: set[str] = set()
_seen_x: set[str] = set()
for _r in experiment:
    if _r.metrics is None:
        continue
    for _c in _r.metrics.columns:
        # numeric-only for y; everything for x (env_steps/iteration/wall_time
        # are all numeric in practice anyway).
        if _c in _seen_y:
            continue
        try:
            if _r.metrics[_c].dtype.kind in "fiu":
                _y_candidates.append(_c)
                _seen_y.add(_c)
        except Exception:
            pass
        if _c not in _seen_x:
            _x_candidates.append(_c)
            _seen_x.add(_c)

# Sensible defaults: prefer best_fitness/best/mean/max for y; env_steps for x.
_preferred_y = ["best_fitness", "best", "mean", "max", "reward"]
_default_y = next(
    (c for c in _preferred_y if c in _seen_y),
    _y_candidates[0] if _y_candidates else "mean",
)
_preferred_x = ["env_steps", "iteration", "wall_time"]
_x_options = [c for c in _preferred_x if c in _seen_x] or _x_candidates or [
    "env_steps"
]

with st.sidebar:
    st.header("Aggregation")
    if _y_candidates:
        _y_options = _y_candidates
        _y_index = (
            _y_options.index(state.metric)
            if state.metric in _y_options
            else _y_options.index(_default_y)
        )
        state.metric = st.selectbox(
            "Y column (in metrics.csv)", _y_options, index=_y_index
        )
    else:
        state.metric = st.text_input(
            "Y column (in metrics.csv)", value=state.metric or _default_y
        )
        st.caption("No metrics.csv columns detected in the selected runs.")
    x_col = st.selectbox("X column", _x_options)
    state.statistic = st.selectbox(
        "Statistic", ["mean", "iqm"], index=["mean", "iqm"].index(state.statistic)
        if state.statistic in {"mean", "iqm"}
        else 0,
    )
    state.ci_level = st.slider("CI", 0.5, 0.99, float(state.ci_level), 0.01)
    state.n_resamples = st.number_input(
        "Bootstrap resamples", 200, 20000, int(state.n_resamples), step=200
    )
state.persist()

# ----- learning curves -----------------------------------------------------
_envs_in_selection = list(experiment.by_env().keys())
_is_multitask = len(_envs_in_selection) >= 2
if _is_multitask:
    tab_curves, tab_pairwise, tab_multitask, tab_summary = st.tabs(
        ["Curves", "Pairwise", "Multi-task", "Summary"]
    )
else:
    tab_curves, tab_pairwise, tab_summary = st.tabs(
        ["Curves", "Pairwise", "Summary"]
    )
    tab_multitask = None

with tab_curves:
    st.subheader(f"{state.statistic.upper()} of `{state.metric}` vs `{x_col}`")
    try:
        for algo, runs in experiment.by_algorithm().items():
            if not runs:
                continue
            df = learning_curve_with_ci(
                runs,
                x_col=x_col,
                y_col=state.metric,
                statistic=state.statistic,
                n_bootstrap=int(state.n_resamples),
                ci=float(state.ci_level),
            )
            if df is None or df.empty:
                st.caption(f"{algo}: no usable data for `{state.metric}`.")
                continue
            try:
                import plotly.graph_objects as go

                y_col = "iqm" if state.statistic == "iqm" else "mean"
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col], y=df[y_col], name=f"{algo} ({y_col})",
                        mode="lines",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=list(df[x_col]) + list(df[x_col][::-1]),
                        y=list(df["hi"]) + list(df["lo"][::-1]),
                        fill="toself", showlegend=False, opacity=0.2,
                        name=f"{algo} CI",
                        line=dict(width=0),
                    )
                )
                fig.update_layout(
                    xaxis_title=x_col, yaxis_title=state.metric,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(df, use_container_width=True)
    except Exception as exc:  # surface library errors but don't crash the page
        st.error(f"Curve aggregation failed: {exc}")

with tab_pairwise:
    st.subheader("Pairwise tests on final-fitness")
    try:
        if len(experiment.by_algorithm()) < 2:
            st.info("Need at least two algorithms to run pairwise tests.")
        else:
            tbl = pairwise_table(
                experiment,
                metric="final_fitness",
                test=state.pairwise_test,
            )
            st.dataframe(tbl, use_container_width=True)
    except Exception as exc:
        st.error(f"Pairwise table failed: {exc}")

if tab_multitask is not None:
    with tab_multitask:
        st.subheader("Cross-task aggregation (rliable)")
        st.caption(
            "Selected runs span "
            f"{len(_envs_in_selection)} environments: "
            + ", ".join(f"`{e}`" for e in _envs_in_selection)
        )
        algos_in_selection = sorted(experiment.by_algorithm().keys())
        norm_mode = st.radio(
            "Normalisation",
            ["auto min-max", "reference algorithm", "off"],
            horizontal=True,
        )
        ref_algo = None
        if norm_mode == "reference algorithm":
            ref_algo = st.selectbox("Reference algorithm", algos_in_selection)

        try:
            from deap_pettingzoo.analysis import (  # noqa: WPS433
                NormalizationSpec,
                aggregate_metrics_forest,
                collect_score_matrices,
                normalize_score_matrices,
            )
            from deap_pettingzoo.analysis.rliable_bridge import (  # noqa: WPS433
                performance_profile,
                probability_of_improvement,
            )
            from deap_pettingzoo.analysis import plots as _plots  # noqa: WPS433

            matrices, env_names = collect_score_matrices(
                experiment, metric="final_fitness"
            )
            if not matrices or len(env_names) < 2:
                st.info("No multi-task scores available across the selection.")
            else:
                if norm_mode == "auto min-max":
                    spec = NormalizationSpec.from_minmax(matrices, env_names)
                    normed = normalize_score_matrices(matrices, env_names, spec)
                elif norm_mode == "reference algorithm" and ref_algo:
                    spec = NormalizationSpec.from_reference_algorithm(
                        matrices, env_names, ref_algo
                    )
                    normed = normalize_score_matrices(matrices, env_names, spec)
                else:
                    normed = matrices

                st.markdown("**Aggregate metrics (95% bootstrap CI)**")
                forest = aggregate_metrics_forest(normed, reps=2000)
                st.dataframe(forest, use_container_width=True)
                fig_forest = _plots.plot_aggregate_metrics_forest(forest)
                st.pyplot(fig_forest)

                st.markdown("**Performance profile (normalised)**")
                taus, point, intervals = performance_profile(
                    score_matrices=normed, reps=1000, n_tau=40
                )
                fig_pp = _plots.plot_normalized_performance_profile(
                    taus, point, intervals
                )
                st.pyplot(fig_pp)

                st.markdown("**Per-env score distributions (raw)**")
                fig_env = _plots.plot_per_env_eval(matrices, env_names)
                st.pyplot(fig_env)

                st.markdown("**Probability of improvement (normalised)**")
                poi = probability_of_improvement(
                    score_matrices=normed, reps=1000
                )
                st.dataframe(poi, use_container_width=True)
        except ImportError as exc:
            st.error(
                "Multi-task analysis requires the `[analysis]` extra "
                f"({exc})."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Multi-task analysis failed: {exc}")

with tab_summary:
    try:
        st.dataframe(experiment.summary_table(), use_container_width=True)
    except Exception as exc:
        st.error(f"Summary table failed: {exc}")
