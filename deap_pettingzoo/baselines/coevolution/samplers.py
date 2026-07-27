"""Opponent-sampling strategies for :mod:`coevolution.pools`.

A *sampler* is a pure callable

    sampler(size, fitnesses, rng) -> int

returning an index into ``range(size)``. ``fitnesses`` is an optional
1-D ``np.ndarray`` of per-item scalar fitnesses (same ordering as the
underlying pool items) or ``None`` when the sampler does not require
them (e.g. :func:`uniform_sampler`). ``rng`` is a
:class:`numpy.random.Generator`; samplers must consume it so runs are
reproducible under a fixed seed.

Samplers are intentionally state-free and source-agnostic: they do not
know whether the items came from a DEAP population, a Hall-of-Fame,
a QDax repertoire etc. The pool calling them is responsible for tracking.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

Sampler = Callable[[int, Optional[np.ndarray], np.random.Generator], int]


def uniform_sampler(
    size: int,
    fitnesses: Optional[np.ndarray],  # noqa: ARG001 — interface parity
    rng: np.random.Generator,
) -> int:
    """Sample an index uniformly from ``range(size)``.

    Raises:
        ValueError: If ``size <= 0``.
    """
    if size <= 0:
        raise ValueError("uniform_sampler: pool is empty (size=0).")
    return int(rng.integers(0, size))


def latest_sampler(
    size: int,
    fitnesses: Optional[np.ndarray],  # noqa: ARG001
    rng: np.random.Generator,  # noqa: ARG001
) -> int:
    """Always return ``size - 1`` (the most recently added item).

    Useful for pools backed by :class:`RingHallOfFame` when the caller
    wants to *always* face the previous generation's champion.
    """
    if size <= 0:
        raise ValueError("latest_sampler: pool is empty (size=0).")
    return size - 1


def fitness_proportional_sampler(
    size: int,
    fitnesses: Optional[np.ndarray],
    rng: np.random.Generator,
) -> int:
    """Sample an index proportional to non-negative fitness.

    Negative fitnesses are shifted by ``-min(fitnesses) + epsilon`` so
    the distribution is valid; if all fitnesses are equal, falls back
    to uniform sampling.

    Raises:
        ValueError: If ``fitnesses`` is ``None`` or has a length
            different from ``size``.
    """
    if size <= 0:
        raise ValueError("fitness_proportional_sampler: pool is empty (size=0).")
    if fitnesses is None:
        raise ValueError(
            "fitness_proportional_sampler requires non-None fitnesses; "
            "the pool did not supply per-item scores."
        )
    arr = np.asarray(fitnesses, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(
            f"fitnesses shape {arr.shape} does not match pool size ({size},)."
        )
    shifted = arr - arr.min() + 1e-9
    total = shifted.sum()
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.integers(0, size))
    probs = shifted / total
    return int(rng.choice(size, p=probs))


def tournament_sampler(k: int = 3) -> Sampler:
    """Build a *k-tournament* sampler.

    The returned sampler draws ``k`` random indices uniformly and returns
    the one with the highest fitness. Ties are broken by the first
    occurrence, matching :func:`deap.tools.selTournament` semantics.

    Args:
        k: Tournament size (>= 1).

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"tournament_sampler: k must be >= 1, got {k}.")

    def _sampler(
        size: int,
        fitnesses: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> int:
        if size <= 0:
            raise ValueError("tournament_sampler: pool is empty (size=0).")
        if fitnesses is None:
            raise ValueError(
                "tournament_sampler requires non-None fitnesses; the pool "
                "did not supply per-item scores."
            )
        arr = np.asarray(fitnesses, dtype=np.float64)
        if arr.shape != (size,):
            raise ValueError(
                f"fitnesses shape {arr.shape} does not match pool size ({size},)."
            )
        effective_k = min(k, size)
        candidates = rng.integers(0, size, size=effective_k)
        scores = arr[candidates]
        winner_offset = int(np.argmax(scores))
        return int(candidates[winner_offset])

    return _sampler


__all__ = [
    "Sampler",
    "uniform_sampler",
    "latest_sampler",
    "fitness_proportional_sampler",
    "tournament_sampler",
]
