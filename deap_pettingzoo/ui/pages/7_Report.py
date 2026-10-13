"""Report: render the analysis package's HTML report inline."""

from __future__ import annotations

import streamlit as st

from deap_pettingzoo.ui.services import (
    UIState,
    load_settings,
    pick_writable_output_dir,
    resolve_output_dir,
)
from deap_pettingzoo.ui.services.data import discover_runs, load_experiment_cached

try:
    from deap_pettingzoo.analysis.report import build_report
except ImportError as exc:
    st.error(
        "The report builder needs the `analysis` extra. Install with "
        "`pip install deap-pettingzoo[analysis]`. "
        f"(Underlying error: {exc})"
    )
    st.stop()

st.set_page_config(page_title="Report", layout="wide")
st.title("Render report")

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

# Per-generation metrics download (CSV) experiment.runs are coevolution-expanded (one RunRecord per side),
# so that each download is a single, flat per-iteration metrics table.
st.subheader("Per-generation metrics (CSV)")
runs_with_metrics = [
    r for r in experiment.runs if r.metrics is not None and len(r.metrics) > 0
]
if not runs_with_metrics:
    st.caption("No per-iteration metrics found for the selected runs.")
else:
    import pandas as pd

    combined_parts = []
    cols = st.columns(min(3, len(runs_with_metrics)))
    for i, rec in enumerate(runs_with_metrics):
        safe_id = rec.run_id.replace("/", "_")
        cols[i % len(cols)].download_button(
            f"⬇ {rec.run_id}",
            data=rec.metrics.to_csv(index=False),
            file_name=f"{safe_id}_metrics.csv",
            mime="text/csv",
            key=f"dl_{safe_id}",
            use_container_width=True,
        )
        part = rec.metrics.copy()
        part.insert(0, "algorithm", rec.algorithm)
        part.insert(0, "run_id", rec.run_id)
        combined_parts.append(part)

    if len(combined_parts) > 1:
        combined = pd.concat(combined_parts, ignore_index=True)
        st.download_button(
            "⬇ Combined (all selected runs)",
            data=combined.to_csv(index=False),
            file_name="combined_metrics.csv",
            mime="text/csv",
            key="dl_combined",
        )

st.divider()

out_dir_raw = st.text_input(
    "Output directory",
    value=str(settings.results_root / "reports"),
    help=(
        "Report and figures are written here. Relative paths resolve "
        "under the results root."
    ),
)
fmt = st.selectbox("Format", ["html", "md"], index=0)
metric = st.text_input("Metric", value="final_fitness")

if st.button("Render"):
    out_dir = resolve_output_dir(out_dir_raw, settings.results_root)
    out_dir, fell_back = pick_writable_output_dir(out_dir)
    try:
        out_path = build_report(experiment, out_dir, fmt=fmt, metric=metric)
    except Exception as exc:  # noqa: BLE001 - surface any rendering error
        st.error(f"Report rendering failed: {exc}")
        st.stop()
    if fell_back:
        st.info(
            f"The requested output directory is read-only, so the report was "
            f"written to a temporary directory: **{out_dir}**"
        )
    st.success(f"Report written to **{out_path}**")
    if fmt == "html":
        try:
            html_text = out_path.read_text(encoding="utf-8")
        except OSError:
            st.caption("Open the file directly to inspect the report.")
        else:
            # Inline relative image references as base64 data URIs so the iframe (which has no filesystem access) can render the figures.
            import base64
            import mimetypes
            import re

            def _inline(match: "re.Match[str]") -> str:
                src = match.group(1)
                if src.startswith(("data:", "http://", "https://", "file://")):
                    return match.group(0)
                fig_path = (out_path.parent / src).resolve()
                if not fig_path.is_file():
                    return match.group(0)
                mime, _ = mimetypes.guess_type(fig_path.name)
                mime = mime or "image/png"
                b64 = base64.b64encode(fig_path.read_bytes()).decode("ascii")
                return f'src="data:{mime};base64,{b64}"'

            html_text = re.sub(r'src="([^"]+)"', _inline, html_text)
            st.components.v1.html(html_text, height=800, scrolling=True)
    else:
        try:
            st.markdown(out_path.read_text(encoding="utf-8"))
        except OSError:
            st.caption("Open the file directly to inspect the report.")
