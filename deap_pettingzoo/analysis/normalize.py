"""Per-task score normalisation for multi-task aggregation.

Cross-environment aggregation is only meaningful on a comparable
scale. The standard recipe is

    norm = (score - low) / (high - low)

per task, where ``(low, high)`` come from one of:

- explicit user-provided dict (e.g. published random / expert scores),
- per-task ``(min, max)`` across all observed runs (``from_minmax``),
- a reference algorithm's per-task IQM/mean (``from_reference_algorithm``).

NaNs are preserved through normalisation so that downstream
``rliable`` calls keep their stratified-bootstrap behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

__all__ = [
    "NormalizationSpec",
    "normalize_score_matrices",
]


@dataclass(frozen=True)
class NormalizationSpec:
    """Per-environment ``(low, high)`` reference scores.

    Attributes
    ----------
    low, high:
        Mappings keyed by environment name. Missing keys mean that
        environment is left **un-normalised** (passes through verbatim,
        with a warning logged at apply time).
    """

    low: Mapping[str, float]
    high: Mapping[str, float]

    # Factories

    @classmethod
    def from_dict(
        cls,
        low: Mapping[str, float],
        high: Mapping[str, float],
    ) -> "NormalizationSpec":
        return cls(low=dict(low), high=dict(high))

    @classmethod
    def from_minmax(
        cls,
        score_matrices: Mapping[str, np.ndarray],
        env_names: Sequence[str],
    ) -> "NormalizationSpec":
        """Per-task min / max across **all** algorithms and seeds.

        Environments where every observation is NaN, or where min == max,
        are omitted from the spec (downstream apply will skip them).
        """
        if not score_matrices:
            return cls(low={}, high={})

        stacked = np.concatenate(list(score_matrices.values()), axis=0)
        if stacked.shape[1] != len(env_names):
            raise ValueError(
                f"score matrices have {stacked.shape[1]} columns but "
                f"{len(env_names)} env_names were given"
            )

        low: Dict[str, float] = {}
        high: Dict[str, float] = {}
        for j, env in enumerate(env_names):
            col = stacked[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            lo = float(col.min())
            hi = float(col.max())
            if hi == lo:
                continue
            low[env] = lo
            high[env] = hi
        return cls(low=low, high=high)

    @classmethod
    def from_reference_algorithm(
        cls,
        score_matrices: Mapping[str, np.ndarray],
        env_names: Sequence[str],
        algorithm: str,
        *,
        baseline_low: float = 0.0,
    ) -> "NormalizationSpec":
        """Use ``algorithm``'s per-task mean as ``high``; ``baseline_low`` as low.

        Useful when there is no canonical "random" baseline but a known
        reference run (e.g. published EA seeds == 1.0).
        """
        if algorithm not in score_matrices:
            raise KeyError(f"reference algorithm {algorithm!r} not in score_matrices")
        ref = score_matrices[algorithm]
        if ref.shape[1] != len(env_names):
            raise ValueError(
                f"reference matrix has {ref.shape[1]} columns but "
                f"{len(env_names)} env_names were given"
            )
        low: Dict[str, float] = {}
        high: Dict[str, float] = {}
        for j, env in enumerate(env_names):
            col = ref[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            hi = float(col.mean())
            if hi == baseline_low:
                continue
            low[env] = float(baseline_low)
            high[env] = hi
        return cls(low=low, high=high)


def normalize_score_matrices(
    score_matrices: Mapping[str, np.ndarray],
    env_names: Sequence[str],
    spec: NormalizationSpec,
) -> Dict[str, np.ndarray]:
    """Apply ``(score - low) / (high - low)`` per task.

    Environments missing from ``spec`` (or where ``high == low``) are
    passed through unchanged. NaNs are preserved.
    """
    out: Dict[str, np.ndarray] = {}
    for algo, mat in score_matrices.items():
        if mat.shape[1] != len(env_names):
            raise ValueError(
                f"matrix for {algo!r} has {mat.shape[1]} columns but "
                f"{len(env_names)} env_names were given"
            )
        normed = mat.astype(float, copy=True)
        for j, env in enumerate(env_names):
            if env not in spec.low or env not in spec.high:
                continue
            lo = spec.low[env]
            hi = spec.high[env]
            denom = hi - lo
            if denom == 0:
                continue
            normed[:, j] = (normed[:, j] - lo) / denom
        out[algo] = normed
    return out
