"""Hall-of-Fame archives for co-evolution.

A *Hall-of-Fame* stores past champions so future generations can be
evaluated against a diverse set of historical opponents (a standard
countermeasure against "opponent forgetting" in competitive
co-evolution — see Rosin & Belew, 1997; Ficici & Pollack, 2003).

Archives store opaque **snapshots** — any Python object — and expose
them via ``items`` / ``fitnesses`` for consumption by an
:class:`~deap_pettingzoo.baselines.coevolution.pools.ArchivePool`. The
pool is responsible for converting snapshots back into
:class:`PolicyProtocol` instances via an injected policy factory.

This keeps the archive itself source-agnostic: snapshots can be cloned
genomes, frozen RL policies, or (in the future) QDax repertoire cells.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

import numpy as np


class HallOfFame(ABC):
    """Abstract Hall-of-Fame archive.

    Concrete implementations must provide :meth:`add`; read-access via
    :attr:`items`, :attr:`fitnesses`, and ``len(...)`` is available by
    default.
    """

    def __init__(self) -> None:
        self._items: List[Any] = []
        self._fitnesses: List[float] = []

    @abstractmethod
    def add(self, snapshot: Any, fitness: float) -> None:
        """Insert ``snapshot`` with its ``fitness`` score.

        Concrete subclasses choose whether the new entry displaces an
        existing one.
        """

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def items(self) -> List[Any]:
        """Ordered list of stored snapshots (shallow copy)."""
        return list(self._items)

    @property
    def fitnesses(self) -> np.ndarray:
        """Per-snapshot fitness scores as a ``float64`` array."""
        return np.asarray(self._fitnesses, dtype=np.float64)

    def __len__(self) -> int:  # noqa: D401 — container protocol
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        """Whether the archive currently has zero entries."""
        return not self._items

    def extend(self, snapshots: Iterable[Any], fitnesses: Iterable[float]) -> None:
        """Bulk-insert multiple ``(snapshot, fitness)`` pairs via :meth:`add`."""
        for snap, fit in zip(snapshots, fitnesses):
            self.add(snap, fit)

    def clear(self) -> None:
        """Remove every entry (useful between independent runs)."""
        self._items.clear()
        self._fitnesses.clear()


class RingHallOfFame(HallOfFame):
    """FIFO ring buffer of the last ``capacity`` snapshots.

    On overflow the oldest entry is evicted. Canonical choice when the
    research interest is "how well does my population play against its
    recent past" rather than "against the strongest-ever opponent".
    """

    def __init__(self, capacity: int) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError(f"RingHallOfFame: capacity must be >= 1, got {capacity}.")
        self.capacity = int(capacity)

    def add(self, snapshot: Any, fitness: float) -> None:
        self._items.append(snapshot)
        self._fitnesses.append(float(fitness))
        while len(self._items) > self.capacity:
            self._items.pop(0)
            self._fitnesses.pop(0)


class ElitistHallOfFame(HallOfFame):
    """Top-``capacity`` fitness-elitist archive.

    Stores the best-ever snapshots (by scalar fitness). New entries
    displace the current minimum only if their fitness is strictly
    greater. Ties leave the archive unchanged.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError(
                f"ElitistHallOfFame: capacity must be >= 1, got {capacity}."
            )
        self.capacity = int(capacity)

    def add(self, snapshot: Any, fitness: float) -> None:
        fit = float(fitness)
        if len(self._items) < self.capacity:
            self._items.append(snapshot)
            self._fitnesses.append(fit)
            return
        worst_idx = int(np.argmin(self._fitnesses))
        if fit > self._fitnesses[worst_idx]:
            self._items[worst_idx] = snapshot
            self._fitnesses[worst_idx] = fit


__all__ = ["HallOfFame", "RingHallOfFame", "ElitistHallOfFame"]
