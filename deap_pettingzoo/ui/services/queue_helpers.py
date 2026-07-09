"""Streamlit-free helpers for the Queue page.

These are separated from ``ui/pages/9_Queue.py`` so they can be
unit-tested without triggering the module-level backend-connection code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Status -> emoji mapping for the list badges.
_STATUS_BADGE: Dict[str, str] = {
    "queued": "\u23f3",
    "running": "\U0001f504",
    "completed": "\u2705",
    "failed": "\u274c",
    "cancelled": "\U0001f6ab",
}


def status_badge(status: str) -> str:
    """Return an emoji + capitalised status label."""
    emoji = _STATUS_BADGE.get(status, "\u2753")
    return f"{emoji} {status.capitalize()}"


def format_time(iso: Optional[str]) -> str:
    """Return a compact display time from an ISO-8601 string."""
    if not iso:
        return "\u2014"
    try:
        return iso.replace("T", " ")[:19]
    except Exception:
        return iso


def short_id(exp_id: str) -> str:
    """Truncate a long experiment id for display."""
    return exp_id[:8] + "\u2026" if len(exp_id) > 10 else exp_id


def algo_label(spec: Optional[Dict[str, Any]]) -> str:
    """Extract algorithm names from a spec dict."""
    if not spec:
        return "\u2014"
    algos = spec.get("algorithms") or []
    if not algos:
        return spec.get("environment", "\u2014")
    return ", ".join(str(a) for a in algos[:3])


def env_label(spec: Optional[Dict[str, Any]]) -> str:
    """Extract the environment name from a spec dict."""
    if not spec:
        return "\u2014"
    return spec.get("environment", "\u2014")


def progress_description(progress: Optional[Dict[str, Any]]) -> str:
    """Build a one-line progress label from a progress snapshot dict."""
    if not progress:
        return ""
    unit = progress.get("unit", "generation")
    current = int(progress.get("current", 0))
    total = int(progress.get("total", 0))
    run_idx = int(progress.get("run_index", 0)) + 1
    total_runs = int(progress.get("total_runs", 1))
    algo = progress.get("algorithm", "")
    seed = progress.get("seed", "")
    return (
        f"Run {run_idx}/{total_runs} \u2014 {algo} s{seed}  "
        f"({unit} {current}/{total})"
    )
