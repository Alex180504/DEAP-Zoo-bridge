"""Typed accessors over Streamlit's session state.

Pages share state (selected runs, metric, CI level, …) through this
class instead of touching ``st.session_state`` directly, so the
schema stays discoverable and we get IDE autocomplete + mypy coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, Optional


@dataclass
class UIState:
    """Snapshot view of the UI's session state.

    Pages instantiate this via :meth:`from_streamlit` at the top of
    their render function, mutate fields freely, then call
    :meth:`persist` before returning to push changes back into
    ``st.session_state``.
    """

    results_root: Optional[Path] = None
    selected_run_ids: List[str] = field(default_factory=list)
    metric: str = "best_fitness"
    statistic: str = "iqm"
    ci_level: float = 0.95
    n_resamples: int = 2000
    pairwise_test: str = "mann_whitney"
    #: The last experiment config edited on the Setup page (serialised to JSON
    #: before persisting so it survives hot-reloads).
    setup_config: Optional[str] = None
    #: Which experiment detail is expanded on the Queue page.
    queue_expanded: str = ""

    _backing: Optional[MutableMapping[str, Any]] = None

    _KEY = "_deap_ui_state_v1"

    @classmethod
    def from_mapping(cls, state: MutableMapping[str, Any]) -> "UIState":
        """Rebuild a :class:`UIState` from a ``st.session_state``-like
        mapping. Missing keys fall back to defaults."""
        stored = state.get(cls._KEY) or {}
        if not isinstance(stored, dict):
            stored = {}
        instance = cls(
            results_root=Path(stored["results_root"])
            if stored.get("results_root")
            else None,
            selected_run_ids=list(stored.get("selected_run_ids") or []),
            metric=str(stored.get("metric", "best_fitness")),
            statistic=str(stored.get("statistic", "iqm")),
            ci_level=float(stored.get("ci_level", 0.95)),
            n_resamples=int(stored.get("n_resamples", 2000)),
            pairwise_test=str(stored.get("pairwise_test", "mann_whitney")),
            setup_config=stored.get("setup_config"),
            queue_expanded=str(stored.get("queue_expanded", "")),
        )
        instance._backing = state
        return instance

    @classmethod
    def from_streamlit(cls) -> "UIState":  # pragma: no cover - needs streamlit runtime
        """Convenience: pull state from the live ``st.session_state``."""
        import streamlit as st

        return cls.from_mapping(st.session_state)

    def persist(self) -> None:
        """Write the current values back to the bound mapping (if any)."""
        if self._backing is None:
            return
        self._backing[self._KEY] = {
            "results_root": str(self.results_root) if self.results_root else None,
            "selected_run_ids": list(self.selected_run_ids),
            "metric": self.metric,
            "statistic": self.statistic,
            "ci_level": float(self.ci_level),
            "n_resamples": int(self.n_resamples),
            "pairwise_test": self.pairwise_test,
            "setup_config": self.setup_config,
            "queue_expanded": self.queue_expanded,
        }

    def update_selected(self, run_ids: Iterable[str]) -> None:
        """Replace the selected-run list (de-duped, order preserved)."""
        seen: set[str] = set()
        unique: List[str] = []
        for rid in run_ids:
            if rid not in seen:
                seen.add(rid)
                unique.append(rid)
        self.selected_run_ids = unique
