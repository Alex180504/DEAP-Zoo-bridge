"""Run discovery + cached experiment loading for the UI.

This module is the single place the UI talks to
:mod:`deap_pettingzoo.analysis.loaders`. Everything else (page modules,
services) goes through :func:`discover_runs` /
:func:`load_experiment_cached` so caching policy and error handling
stay consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from deap_pettingzoo.analysis.loaders import Experiment, RunRecord


@dataclass(frozen=True)
class RunSummary:
    """Lightweight per-run row used by the Browse page table.

    Heavier fields (metrics, episode rows) are loaded lazily on demand.
    """

    run_id: str
    run_dir: Path
    algorithm: str
    env_name: str
    seed: Optional[int]
    wall_time_s: Optional[float]
    final_summary: dict


def discover_runs(root: Path, max_depth: int = 6) -> List[RunSummary]:
    """Recursively scan ``root`` (up to ``max_depth`` levels) for run dirs.

    A "run dir" is any directory containing :file:`manifest.json`.
    Nested experiment layouts like
    ``results/<experiment>/<algo>/seed_<n>/<run>/manifest.json`` are
    supported. Once a directory is identified as a run dir its
    subdirectories are not descended into further.

    Returns rows sorted newest-first by manifest mtime.
    """
    from deap_pettingzoo.analysis.loaders import load_run

    if not root.is_dir():
        return []

    candidates: List[Path] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if (entry / "manifest.json").is_file():
                candidates.append(entry)
                continue
            _walk(entry, depth + 1)

    _walk(root, 0)

    out: List[RunSummary] = []
    for run_dir in candidates:
        try:
            rec = load_run(run_dir)
        except (FileNotFoundError, ValueError):
            continue
        out.append(
            RunSummary(
                run_id=rec.run_id,
                run_dir=Path(rec.run_dir),
                algorithm=rec.algorithm,
                env_name=rec.env_name,
                seed=rec.seed,
                wall_time_s=rec.wall_time_s,
                final_summary=dict(rec.summary or {}),
            )
        )

    def _mtime(s: RunSummary) -> float:
        try:
            return (s.run_dir / "manifest.json").stat().st_mtime
        except OSError:
            return 0.0

    out.sort(key=_mtime, reverse=True)
    return out


def load_experiment_cached(run_dirs: List[Path]) -> "Experiment":
    """Hydrate ``run_dirs`` into a single :class:`Experiment`.

    Caching is delegated to Streamlit when this function is called from
    a page module (via ``st.cache_resource``); when imported from a
    plain Python context (tests, scripts) it just calls the loader
    directly. Keeping the cache decorator inside the page module rather
    than here means this module is import-safe without ``streamlit``.
    """
    from deap_pettingzoo.analysis.loaders import Experiment, load_run

    runs: List["RunRecord"] = []
    resolved_parents: List[Path] = []
    for run_dir in run_dirs:
        try:
            runs.append(load_run(run_dir))
        except (FileNotFoundError, ValueError):
            continue
        resolved_parents.append(Path(run_dir).resolve().parent)

    if resolved_parents:
        common_root: Path = Path(os.path.commonpath([str(p) for p in resolved_parents]))
    else:
        common_root = Path.cwd()
    return Experiment(root=common_root, runs=runs)
