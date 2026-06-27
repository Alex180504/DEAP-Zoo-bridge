"""Compare page: aggregate curves and pairwise tests across the selected runs."""

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


def _pairwise_display_table(tbl):
    """Render pairwise comparisons as x/y rows under one algorithm column."""
    if getattr(tbl, "empty", True):
        return tbl
    required = {"algo_x", "algo_y", "n_x", "n_y"}
    if not required.issubset(set(tbl.columns)):
        return tbl

    import pandas as pd

    rows = []
    for pair in tbl.to_dict("records"):
        shared = {
            key: value
            for key, value in pair.items()
            if key not in {"algo_x", "algo_y", "n_x", "n_y"}
        }
        rows.append(
            {
                "algo": pair["algo_x"],
                "n": pair["n_x"],
                **shared,
            }
        )
        rows.append(
            {
                "algo": pair["algo_y"],
                "n": pair["n_y"],
                **shared,
            }
        )
    return pd.DataFrame(rows)


_COMPARE_GROUP_OPTIONS = {
    "Aggregate by algorithm": "algorithm",
    "Aggregate by parent folder": "folder",
    "Aggregate by display name": "display_name",
    "Individual runs": "run",
}


def _display_name(run) -> str | None:
    label = str(run.manifest.get("display_name") or "").strip()
    return label or None


def _run_label(run) -> str:
    label = _display_name(run) or run.run_id
    if run.seed is not None and f"seed={run.seed}" not in label:
        return f"{label} (seed={run.seed})"
    return label


def _parent_group_label(run, root: Path) -> str:
    parent = Path(run.run_dir).resolve().parent
    try:
        label = parent.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        label = parent.name
    if not label or label == ".":
        label = parent.name or run.algorithm
    return label


def _compare_group_label(run, group_mode: str, root: Path) -> str:
    if group_mode == "run":
        return _run_label(run)
    if group_mode == "display_name":
        return _display_name(run) or _parent_group_label(run, root)
    if group_mode == "folder":
        return _parent_group_label(run, root)
    return run.algorithm


def _default_compare_group_mode(experiment) -> str:
    if len(experiment.by_algorithm()) > 1:
        return "algorithm"
    display_labels = {
        _display_name(run) for run in experiment if _display_name(run) is not None
    }
    if len(display_labels) > 1:
        return "display_name"
    folder_labels = {_parent_group_label(run, experiment.root) for run in experiment}
    if len(folder_labels) > 1:
        return "folder"
    return "algorithm"


def _build_compare_experiment(experiment, group_mode: str):
    if group_mode == "algorithm":
        return experiment

    from dataclasses import replace as _dc_replace
    from deap_pettingzoo.analysis.loaders import Experiment as _Exp

    proxy_runs = []
    for run in experiment:
        label = _compare_group_label(run, group_mode, experiment.root)
        proxy_runs.append(
            _dc_replace(run, manifest={**run.manifest, "algorithm": label})
        )
    return _Exp(root=experiment.root, runs=proxy_runs)


def _grouped_summary_table(experiment, group_mode: str):
    raw = experiment.summary_table()
    if getattr(raw, "empty", True):
        return raw

    raw = raw.copy()
    raw.insert(
        0,
        "group",
        [
            _compare_group_label(run, group_mode, experiment.root)
            for run in experiment
        ],
    )
    if group_mode == "run":
        return raw

    import pandas as pd

    rows = []
    for group_label in raw["group"].drop_duplicates().tolist():
        subset = raw[raw["group"] == group_label]
        row = {
            "group": group_label,
            "n_runs": int(len(subset)),
            "algorithm": ", ".join(sorted(set(subset["algorithm"].astype(str)))),
            "env_name": ", ".join(sorted(set(subset["env_name"].astype(str)))),
        }
        for column in subset.columns:
            if column in {"group", "run_id", "algorithm", "env_name"}:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if not values.empty:
                row[column] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


# Resolve selected run_ids -> run_dirs by re-discovering (cheap, cached above).
from deap_pettingzoo.ui.services.data import discover_runs

all_runs = discover_runs(settings.results_root)
selected_run_dirs = [
    r.run_dir for r in all_runs if r.run_id in state.selected_run_ids
]
if not selected_run_dirs:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

experiment = _cached_experiment(tuple(str(p) for p in selected_run_dirs))

# Controls

# Discover columns actually present across the selected run's metrics.csv so that
# the user can't pick a non-existent y/x column (which would surface as "No runs contained both '<x>' and '<y>'").
_y_candidates: list[str] = []
_x_candidates: list[str] = []
_seen_y: set[str] = set()
_seen_x: set[str] = set()
for _r in experiment:
    if _r.metrics is None:
        continue
    for _c in _r.metrics.columns:
        # Numeric-only for y; everything for x (env_steps/iteration/wall_time are all numeric in practice anyway).
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

# Defaults: prefer best_fitness/best/mean/max for y; env_steps for x.
_preferred_y = ["best_fitness", "best", "mean", "max", "reward"]
_default_y = next(
    (c for c in _preferred_y if c in _seen_y),
    _y_candidates[0] if _y_candidates else "mean",
)

# Prefer 'iteration' over 'env_steps' for coevolution runs where env_steps
# is degenerate (always 0 because coevolution logging doesn't track it).
_env_steps_degenerate = (
    "env_steps" in _seen_x
    and all(
        r.metrics is not None
        and "env_steps" in r.metrics.columns
        and (r.metrics["env_steps"] == 0).all()
        for r in experiment
        if r.metrics is not None and "env_steps" in r.metrics.columns
    )
    and any(
        r.metrics is not None and "env_steps" in r.metrics.columns
        for r in experiment
    )
)
_preferred_x = (
    ["iteration", "env_steps", "wall_time"]
    if _env_steps_degenerate
    else ["env_steps", "iteration", "wall_time"]
)
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
    _default_group_label = next(
        label
        for label, value in _COMPARE_GROUP_OPTIONS.items()
        if value == _default_compare_group_mode(experiment)
    )
    compare_group_label = st.selectbox(
        "Compare selected runs as",
        list(_COMPARE_GROUP_OPTIONS.keys()),
        index=list(_COMPARE_GROUP_OPTIONS.keys()).index(_default_group_label),
        key="compare_group_mode_label",
    )
    st.caption(
        "Use Browse > Rename to give multiple seed runs the same display name "
        "and compare them as one experiment."
    )
state.persist()

compare_group_mode = _COMPARE_GROUP_OPTIONS[compare_group_label]
compare_experiment = _build_compare_experiment(experiment, compare_group_mode)
compare_groups = compare_experiment.by_algorithm()
compare_group_desc = {
    "algorithm": "algorithm",
    "folder": "parent folder",
    "display_name": "display name",
    "run": "run",
}[compare_group_mode]

# Learning curves
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
    st.caption(f"Grouped by {compare_group_desc}.")
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        any_data = False
        if compare_group_mode == "run":
            for run in experiment:
                if run.metrics is None:
                    continue
                cols = set(run.metrics.columns)
                if x_col not in cols or state.metric not in cols:
                    continue
                df_r = run.metrics[[x_col, state.metric]].dropna().sort_values(x_col)
                if df_r.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df_r[x_col],
                        y=df_r[state.metric],
                        name=_compare_group_label(run, "run", experiment.root),
                        mode="lines",
                    )
                )
                any_data = True
        else:
            y_col_stat = "iqm" if state.statistic == "iqm" else "mean"
            for group_label, runs in compare_groups.items():
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
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col_stat],
                        name=f"{group_label} ({y_col_stat})",
                        mode="lines",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=list(df[x_col]) + list(df[x_col][::-1]),
                        y=list(df["hi"]) + list(df["lo"][::-1]),
                        fill="toself",
                        showlegend=False,
                        opacity=0.2,
                        name=f"{group_label} CI",
                        line={"width": 0},
                    )
                )
                any_data = True
        if any_data:
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=state.metric,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"No usable data for `{state.metric}`.")
    except ImportError:
        st.info("Install `plotly` for interactive charts.")
    except Exception as exc:  # surface library errors but don't crash the page
        st.error(f"Curve aggregation failed: {exc}")

with tab_pairwise:
    st.subheader("Pairwise tests on final-fitness")
    try:
        if len(compare_groups) < 2:
            st.info(
                "Select runs spanning at least two comparison groups, or "
                "switch the grouping mode to `Individual runs`."
            )
        else:
            tbl = pairwise_table(
                compare_experiment,
                metric="final_fitness",
                test=state.pairwise_test,
            )
            st.dataframe(
                _pairwise_display_table(tbl),
                use_container_width=True,
            )
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
        algos_in_selection = sorted(compare_groups.keys())
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
                compare_experiment, metric="final_fitness"
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
        st.dataframe(
            _grouped_summary_table(experiment, compare_group_mode),
            use_container_width=True,
        )
        if compare_group_mode != "run":
            with st.expander("Raw runs", expanded=False):
                st.dataframe(experiment.summary_table(), use_container_width=True)
    except Exception as exc:
        st.error(f"Summary table failed: {exc}")
