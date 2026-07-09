"""Formatting helpers for QDax progress reporting.

This module intentionally avoids importing JAX/QDax so both the standalone
QDax trainers and the generic co-evolution trainer can reuse the same
formatting without pulling heavy optional dependencies at module import time.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def summarise_fitness_batch(
    fitnesses: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(batch_best, batch_mean)`` over finite fitness values."""
    if fitnesses is None:
        return None, None
    arr = np.asarray(fitnesses, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(finite.max()), float(finite.mean())


def summarise_qdax_archive(
    fitnesses: Any,
) -> Tuple[float, float, float, int, int]:
    """Return ``(best, qd_score, coverage, filled, total)`` for an archive."""
    arr = np.asarray(fitnesses, dtype=np.float64).reshape(-1)
    total = int(arr.size)
    if total == 0:
        return float("nan"), 0.0, 0.0, 0, 0

    mask = np.isfinite(arr)
    filled = int(mask.sum())
    if filled == 0:
        return float("nan"), 0.0, 0.0, 0, total

    finite = arr[mask]
    return (
        float(finite.max()),
        float(finite.sum()),
        100.0 * filled / total,
        filled,
        total,
    )


def format_qdax_progress_fields(
    *,
    archive_best: float,
    qd_score: float,
    coverage: float,
    filled_niches: Optional[int] = None,
    total_niches: Optional[int] = None,
    batch_best: Optional[float] = None,
    batch_mean: Optional[float] = None,
) -> str:
    """Format shared QDax progress fields for console/log output."""
    parts = []
    if batch_best is not None:
        parts.append(f"batch_best={batch_best:.3f}")
    if batch_mean is not None:
        parts.append(f"batch_mean={batch_mean:.3f}")
    parts.append(f"archive_best={archive_best:.3f}")
    parts.append(f"qd={qd_score:.3f}")
    if filled_niches is not None and total_niches is not None and total_niches > 0:
        parts.append(f"coverage={filled_niches}/{total_niches} ({coverage:.3f}%)")
    else:
        parts.append(f"coverage={coverage:.3f}%")
    return "  ".join(parts)
