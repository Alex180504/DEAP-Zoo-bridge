"""Opponent pools — source-agnostic sampling of opponent policies.

The central contract is :class:`OpponentPool` — a runtime-checkable
protocol with a single required method::

    sample_opponent(rng: np.random.Generator) -> PolicyProtocol

Any object that can produce a policy-like thing on demand can satisfy
this protocol: DEAP populations, Hall-of-Fame archives, static
pre-trained policies (scripted or frozen RL), or — in the future — a
QDax :class:`MapElitesRepertoire` wrapped by a thin adapter.

This is the *only* integration point co-evolution code needs to depend
on for opponent sourcing; the trainer, the evaluator, and downstream
samplers remain unaware of where opponents ultimately come from.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol

from .archive import HallOfFame
from .samplers import Sampler, uniform_sampler


#: A *policy factory* maps a pool's stored item (opaque type) to a
#: :class:`PolicyProtocol` instance. Pools hold items (e.g. cloned
#: genomes, snapshots) without knowing how to turn them into executable
#: policies; the factory bridges that gap.
PolicyFactory = Callable[[Any], PolicyProtocol]


@runtime_checkable
class OpponentPool(Protocol):
    """Source-agnostic opponent-sampling interface.

    Implementations hold an opaque *population-like* object and produce
    :class:`PolicyProtocol` instances on demand. The protocol is
    deliberately narrow so QDax repertoires, RL snapshot buffers, and
    scripted policy sets can all plug in without touching trainer code.
    """

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        """Return a fresh opponent policy sampled from this pool."""

    def sample_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[PolicyProtocol]:
        """Sample ``k`` opponent policies (may include duplicates)."""

    def update(self, population_like: Any) -> None:  # noqa: D401 — protocol method
        """Replace or extend the backing population-like object."""

    def __len__(self) -> int:  # noqa: D401 — container protocol
        ...


# ---------------------------------------------------------------------------
# Helpers shared by concrete pools
# ---------------------------------------------------------------------------


def _default_sample_opponents(
    pool: "_BasePool", rng: np.random.Generator, k: int
) -> List[PolicyProtocol]:
    """Default ``k``-sample loop reused by every concrete pool."""
    if k < 1:
        raise ValueError(f"sample_opponents: k must be >= 1, got {k}.")
    return [pool.sample_opponent(rng) for _ in range(k)]


class _BasePool:
    """Thin shared base providing :meth:`sample_opponents` and introspection.

    Not part of the public API — concrete pools subclass this for
    convenience but client code should type-hint against
    :class:`OpponentPool`.
    """

    def sample_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[PolicyProtocol]:
        return _default_sample_opponents(self, rng, k)

    def update(self, population_like: Any) -> None:  # pragma: no cover — override
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover — override
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


# ---------------------------------------------------------------------------
# Concrete pools
# ---------------------------------------------------------------------------


class PopulationPool(_BasePool):
    """Opponent pool backed by a DEAP-style population.

    The ``population_like`` argument is expected to be an iterable of
    individuals exposing ``.fitness.values`` (DEAP convention). The
    *policy factory* is called on each sampled individual to produce a
    fresh :class:`PolicyProtocol` instance — typically a closure that
    wraps the individual's ``.genome`` in a
    :class:`FeedforwardNeuralPolicy`.

    Fitnesses default to ``individual.fitness.values[0]`` and are
    forwarded to the sampler. Individuals without a valid fitness are
    treated as ``-inf`` so they never "win" a tournament but remain
    sampleable uniformly.
    """

    def __init__(
        self,
        population_like: Sequence[Any],
        policy_factory: PolicyFactory,
        sampler: Sampler = uniform_sampler,
    ) -> None:
        self._policy_factory = policy_factory
        self._sampler = sampler
        self._population: List[Any] = []
        self.update(population_like)

    def update(self, population_like: Any) -> None:
        self._population = list(population_like) if population_like is not None else []

    def _fitnesses(self) -> Optional[np.ndarray]:
        if not self._population:
            return None
        vals = np.empty(len(self._population), dtype=np.float64)
        for i, ind in enumerate(self._population):
            try:
                vals[i] = float(ind.fitness.values[0])
            except (AttributeError, IndexError, TypeError):
                vals[i] = -np.inf
        return vals

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        if not self._population:
            raise ValueError(
                "PopulationPool.sample_opponent: pool is empty. Ensure the "
                "backing population has been populated before sampling."
            )
        idx = self._sampler(len(self._population), self._fitnesses(), rng)
        return self._policy_factory(self._population[idx])

    def __len__(self) -> int:
        return len(self._population)


class ArchivePool(_BasePool):
    """Opponent pool backed by a :class:`HallOfFame`.

    The archive stores opaque snapshots (e.g. cloned genomes); the
    ``policy_factory`` converts one into a :class:`PolicyProtocol` at
    sample time.
    """

    def __init__(
        self,
        archive: HallOfFame,
        policy_factory: PolicyFactory,
        sampler: Sampler = uniform_sampler,
    ) -> None:
        self._archive = archive
        self._policy_factory = policy_factory
        self._sampler = sampler

    @property
    def archive(self) -> HallOfFame:
        """The wrapped Hall-of-Fame (exposed for trainer snapshotting)."""
        return self._archive

    def update(self, population_like: Any) -> None:
        """No-op.

        Archive state is mutated through :meth:`HallOfFame.add` by the
        trainer's snapshot loop, not through the pool. Implemented to
        satisfy the :class:`OpponentPool` protocol.
        """

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        size = len(self._archive)
        if size == 0:
            raise ValueError(
                "ArchivePool.sample_opponent: archive is empty. The "
                "trainer must snapshot at least one champion before "
                "this pool can be sampled."
            )
        idx = self._sampler(size, self._archive.fitnesses, rng)
        return self._policy_factory(self._archive.items[idx])

    def __len__(self) -> int:
        return len(self._archive)


class FrozenPolicyPool(_BasePool):
    """Static list of pre-built :class:`PolicyProtocol` instances.

    Use cases:

    * Scripted opponents (e.g. APF evaders).
    * Pre-trained RL policies frozen for EA-vs-RL co-evolution
      (e.g. a MADDPG :class:`AgileRLPolicyWrapper` instance).
    * Replay of hand-picked historical champions from a previous run.

    Fitnesses may optionally be supplied alongside the policies so
    fitness-aware samplers (e.g. :func:`tournament_sampler`) can be
    used. When omitted, samplers that require scores will raise.
    """

    def __init__(
        self,
        policies: Sequence[PolicyProtocol],
        fitnesses: Optional[Sequence[float]] = None,
        sampler: Sampler = uniform_sampler,
    ) -> None:
        self._policies: List[PolicyProtocol] = list(policies)
        if fitnesses is not None:
            fit_list = list(fitnesses)
            if len(fit_list) != len(self._policies):
                raise ValueError(
                    "FrozenPolicyPool: fitnesses length "
                    f"{len(fit_list)} does not match policies length "
                    f"{len(self._policies)}."
                )
            self._fitnesses: Optional[np.ndarray] = np.asarray(
                fit_list, dtype=np.float64
            )
        else:
            self._fitnesses = None
        self._sampler = sampler

    def update(self, population_like: Any) -> None:
        """Replace the frozen policy list (population_like: sequence)."""
        self._policies = list(population_like) if population_like is not None else []
        # Size-change invalidates any stale fitness cache.
        self._fitnesses = None

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        if not self._policies:
            raise ValueError("FrozenPolicyPool.sample_opponent: pool is empty.")
        idx = self._sampler(len(self._policies), self._fitnesses, rng)
        return self._policies[idx]

    def __len__(self) -> int:
        return len(self._policies)


class CompositePool(_BasePool):
    """Weighted mixture of sub-pools.

    On each :meth:`sample_opponent` call, one sub-pool is selected
    proportional to ``weights``; sampling is then delegated to that
    sub-pool. Empty sub-pools are skipped (their effective weight is
    re-distributed over the remaining pools).

    Canonical use: 50/50 blend of the current-generation
    :class:`PopulationPool` and a :class:`ArchivePool` wrapping a
    Hall-of-Fame — the PSRO-ish "current + historical" opponent set.

    >>> pool = CompositePool([pop_pool, hof_pool], weights=[0.5, 0.5])
    """

    def __init__(
        self,
        pools: Sequence[OpponentPool],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not pools:
            raise ValueError("CompositePool: 'pools' must be non-empty.")
        self._pools: Tuple[OpponentPool, ...] = tuple(pools)
        if weights is None:
            weights = [1.0] * len(self._pools)
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (len(self._pools),):
            raise ValueError(
                f"CompositePool: weights shape {weights_arr.shape} does "
                f"not match pools length ({len(self._pools)},)."
            )
        if np.any(weights_arr < 0):
            raise ValueError("CompositePool: weights must be non-negative.")
        if weights_arr.sum() <= 0:
            raise ValueError("CompositePool: weights must sum to > 0.")
        self._weights = weights_arr

    def update(self, population_like: Any) -> None:
        """No-op at composite level — update underlying pools directly."""

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        # Mask out empty pools and renormalise.
        sizes = np.array([len(p) for p in self._pools], dtype=np.int64)
        mask = sizes > 0
        if not mask.any():
            raise ValueError("CompositePool.sample_opponent: all sub-pools are empty.")
        effective = self._weights * mask
        total = effective.sum()
        if total <= 0:
            # All non-empty pools happen to have zero weight — fall back
            # to uniform over non-empty pools for robustness.
            effective = mask.astype(np.float64)
            total = effective.sum()
        probs = effective / total
        choice = int(rng.choice(len(self._pools), p=probs))
        return self._pools[choice].sample_opponent(rng)

    def __len__(self) -> int:
        return sum(len(p) for p in self._pools)

    @property
    def sub_pools(self) -> Tuple[OpponentPool, ...]:
        """The underlying sub-pools (exposed for introspection / testing)."""
        return self._pools


__all__ = [
    "PolicyFactory",
    "OpponentPool",
    "PopulationPool",
    "ArchivePool",
    "FrozenPolicyPool",
    "CompositePool",
]
