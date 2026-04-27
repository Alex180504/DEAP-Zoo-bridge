"""Report — render the analysis package's HTML report inline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs, load_experiment_cached

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

try:
    from deap_pettingzoo.analysis.report import build_report
except ImportError as exc:
    st.error(
        "The report builder needs the `analysis` extra. Install with "
        "`pip install deap-pettingzoo[analysis]`. "
        f"(Underlying error: {exc})"
    )
    st.stop()

out_dir = Path(st.text_input("Output directory", value=".report-out"))
fmt = st.selectbox("Format", ["html", "md"], index=0)
metric = st.text_input("Metric", value="final_fitness")

if st.button("Render"):
    try:
        out_path = build_report(
            experiment, out_dir, fmt=fmt, metric=metric
        )
    except Exception as exc:  # noqa: BLE001 - surface any rendering error
        st.error(f"Report rendering failed: {exc}")
        st.stop()
    st.success(f"Report written to **{out_path}**")
    if fmt == "html":
        try:
            html_text = out_path.read_text(encoding="utf-8")
        except OSError:
            st.caption("Open the file directly to inspect the report.")
        else:
            # Inline relative image references as base64 data URIs so the
            # iframe (which has no filesystem access) can render the figures.
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
