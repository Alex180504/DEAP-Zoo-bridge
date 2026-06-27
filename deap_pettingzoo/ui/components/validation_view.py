"""Render a :class:`ValidationReport` as inline error/warning blocks,
grouped by the top-level config tab the issue's path belongs to.

Streamlit-dependent (calls ``st.error``/``st.warning``/``st.success``).
The grouping logic is streamlit-free and unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

# Known path prefixes -> tab label mapping
_TAB_MAP = {
    "environment": "Environment",
    "algorithms": "Algorithm(s)",
    "hyperparameters": "Hyperparameters",
    "budget": "Budget",
    "fitness": "Fitness",
    "recording": "Recording",
    "coevolution": "Coevolution",
    "parallelism": "Parallelism",
    "raw_config": "Raw config",
}


def _tab_for_path(path: str) -> str:
    """Map an issue path to the tab it belongs to."""
    for prefix, label in _TAB_MAP.items():
        if path.startswith(prefix):
            return label
    return "General"


def group_issues_by_tab(
    issues: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group a flat list of issue dicts by tab label."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for issue in issues:
        tab = _tab_for_path(issue.get("path", ""))
        groups.setdefault(tab, []).append(issue)
    return groups


def render_validation_report(report: Dict[str, Any]) -> None:
    """Display a validation report (the dict from ``GET /experiments/validate``
    or the error body of a 422 response) as colour-coded blocks.

    ``report["ok"]`` returns ``True`` -> green success banner.
    Otherwise, red errors per tab and orange warnings."""
    import streamlit as st  # noqa: WPS433

    issues = report.get("issues", [])
    if report.get("ok"):
        st.success("Experiment spec is valid.")
        return

    errors = [i for i in issues if i.get("severity", "error") == "error"]
    warnings = [i for i in issues if i.get("severity", "error") == "warning"]

    if errors:
        grouped = group_issues_by_tab(errors)
        for tab, items in grouped.items():
            with st.container(border=True):
                st.error(f"**{tab}**  ({len(items)} issue(s))")
                for item in items:
                    st.caption(
                        f"`{item.get('path', '?')}` — {item.get('message', '')}"
                    )
    if warnings:
        with st.container(border=True):
            st.warning(f"{len(warnings)} warning(s)")
            for item in warnings:
                st.caption(
                    f"`{item.get('path', '?')}` — {item.get('message', '')}"
                )
