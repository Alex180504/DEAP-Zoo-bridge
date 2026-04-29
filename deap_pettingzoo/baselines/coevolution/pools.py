"""Opponent pools — source-agnostic sampling of opponent policies.

The central contract is :class:`OpponentPool` — a runtime-checkable
protocol with a single required method::

    sample_opponent(rng: np.random.Generator) -> PolicyProtocol

Any object that can produce a policy-like object on demand can satisfy
this protocol: DEAP populations, Hall-of-Fame archives, static
pre-trained policies (scripted or frozen RL), or — in the future — a
QDax :class:`MapElitesRepertoire` wrapped by a thin adapter.

This is the *only* integration point co-evolution code needs to depend
on for opponent sourcing; the trainer, the evaluator, and downstream
samplers remain unaware of where opponents ultimately come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
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


#: A *policy factory* maps a pool's stored item (opaque type) to a class:`PolicyProtocol` instance. 
#: Pools hold items (e.g. cloned genomes, snapshots) without knowing how to turn them into executable policies; the factory bridges that gap.
PolicyFactory = Callable[[Any], PolicyProtocol]


# Picklable opponent snapshots

@dataclass
class OpponentSnapshot:
    """Picklable per-opponent payload shipped to worker processes.

    A snapshot decouples *sampling* (driver-side, mutates pool RNG /
    archive cursors) from *materialisation* (worker-side, builds a
    live :class:`PolicyProtocol`). The driver calls
    :meth:`OpponentPool.snapshot_opponents` to obtain a
    ``List[OpponentSnapshot]``, then ships them across the process
    boundary as part of an episode payload. Workers call
    :func:`rebuild_opponent_policy` to materialise each one.

    Attributes:
        kind: Dispatch tag interpreted by
            :func:`rebuild_opponent_policy`. Built-in kinds:

            * ``"feedforward"`` — payload ``{"genome": FixedTopologyGenome,
              "masked": bool}``. Worker builds
              :class:`FeedforwardNeuralPolicy`.
            * ``"frozen"`` — payload ``{"policy": PolicyProtocol}``. The
              policy is shipped by value; it must be picklable. For
              non-picklable policies (e.g. torch models), use
              :func:`register_frozen_policy` and snapshot kind
              ``"frozen_registered"``.
            * ``"frozen_registered"`` — payload ``{"name": str}``.
              Worker resolves via the global frozen-policy registry.
            * ``"qdax_flax"`` — payload ``{"flat_params": np.ndarray,
              "layer_sizes": tuple, "activation": str}``. Worker
              rebuilds :class:`FlaxPolicyAdapter` (requires the QDax
              extra installed in the worker).
        payload: kind-specific dict (see above). Must be picklable
            with the standard library pickler — no closures, no JAX
            traced values, no torch tensors.
    """

    kind: str
    payload: Dict[str, Any]


# Module-global registry for non-picklable frozen policies (e.g. torch/AgileRL wrappers).
_FROZEN_POLICY_REGISTRY: Dict[str, Callable[[], PolicyProtocol]] = {}


def register_frozen_policy(name: str, factory: Callable[[], PolicyProtocol]) -> None:
    """Register a zero-arg factory so workers can rebuild a frozen policy.

    Use this when a frozen policy cannot survive a pickle round-trip
    (e.g. a torch / AgileRL wrapper). The factory must:

    * live at module import time (not inside a function) so it executes
      on every worker process when that module is imported, and
    * be deterministic / cheap (called once per worker on first lookup,
      cached thereafter).

    Args:
        name: Stable identifier; the snapshot's
            ``payload["name"]`` must match.
        factory: Zero-arg callable returning a fresh
            :class:`PolicyProtocol`.

    Raises:
        ValueError: if ``name`` is already registered with a different
            factory (re-registering the same factory is idempotent).
    """
    existing = _FROZEN_POLICY_REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(
            f"register_frozen_policy: {name!r} already registered with a "
            "different factory. Use a unique name or call "
            "unregister_frozen_policy first."
        )
    _FROZEN_POLICY_REGISTRY[name] = factory


def unregister_frozen_policy(name: str) -> None:
    """Remove ``name`` from the registry (no-op if not present)."""
    _FROZEN_POLICY_REGISTRY.pop(name, None)


def _resolve_frozen_policy(name: str) -> PolicyProtocol:
    factory = _FROZEN_POLICY_REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"OpponentSnapshot(kind='frozen_registered', name={name!r}): "
            f"not found in registry. Known: {sorted(_FROZEN_POLICY_REGISTRY)}. "
            "Call register_frozen_policy(name, factory) at module import "
            "time so workers see it on spawn."
        )
    return factory()


# Worker-side cache for QDax module objects keyed on (layer_sizes, activation).
# Rebuilding the Flax module is cheap but not free;
# cache avoids redoing it for every opponent snapshot in every episode.
_QDAX_MODULE_CACHE: Dict[Tuple[Tuple[int, ...], str], Any] = {}


def _resolve_qdax_module(layer_sizes: Tuple[int, ...], activation: str) -> Any:
    key = (tuple(int(s) for s in layer_sizes), activation)
    cached = _QDAX_MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    from deap_pettingzoo.contrib.qdax.policy import make_flax_mlp

    module = make_flax_mlp(key[0], activation=key[1])
    _QDAX_MODULE_CACHE[key] = module
    return module


def rebuild_opponent_policy(
    snapshot: OpponentSnapshot, *, action_space: Any
) -> PolicyProtocol:
    """Materialise a live policy from a picklable :class:`OpponentSnapshot`.

    Dispatches on ``snapshot.kind``. Called by worker processes inside
    :func:`_episode_rollout`; safe to call on the driver too (the
    serial fallback path uses it as well).

    Args:
        snapshot: A snapshot returned by an
            :class:`OpponentPool.snapshot_opponents` implementation.
        action_space: Gym/Gymnasium action space of the agent the
            policy will drive. Resolved on the worker from a freshly
            built :class:`PettingZooWrapper` so it is independent of
            any driver-side env handle.

    Raises:
        ValueError: on unknown ``snapshot.kind``.
    """
    kind = snapshot.kind
    payload = snapshot.payload

    if kind == "feedforward":
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        genome = payload["genome"]
        masked = bool(payload.get("masked", False))
        return FeedforwardNeuralPolicy(
            genome, action_space=action_space, masked=masked
        )

    if kind == "frozen":
        return payload["policy"]

    if kind == "frozen_registered":
        return _resolve_frozen_policy(payload["name"])

    if kind == "qdax_flax":
        from deap_pettingzoo.contrib.qdax.policy import (
            FlaxPolicyAdapter,
            flat_to_pytree,
        )

        layer_sizes = tuple(int(s) for s in payload["layer_sizes"])
        module = _resolve_qdax_module(layer_sizes, payload.get("activation", "tanh"))
        params = flat_to_pytree(np.asarray(payload["flat_params"]), layer_sizes)
        return FlaxPolicyAdapter(module, params, action_space=action_space)

    raise ValueError(
        f"rebuild_opponent_policy: unknown snapshot kind {kind!r}. "
        "Built-in kinds: 'feedforward', 'frozen', 'frozen_registered', "
        "'qdax_flax'."
    )


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

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        """Sample ``k`` picklable opponent payloads for parallel evaluation.

        Sampling happens on the **driver** so any
        pool RNG / archive cursors mutate exactly once per evaluation,
        regardless of how many worker processes consume the result.
        Workers rebuild policies via :func:`rebuild_opponent_policy`.
        """

    def update(self, population_like: Any) -> None:  # noqa: D401 — protocol method
        """Replace or extend the backing population-like object."""

    def __len__(self) -> int:  # noqa: D401 — container protocol
        ...


# Helpers shared by concrete pools


def _default_sample_opponents(
    pool: "_BasePool", rng: np.random.Generator, k: int
) -> List[PolicyProtocol]:
    """Default ``k``-sample loop reused by every concrete pool."""
    if k < 1:
        raise ValueError(f"sample_opponents: k must be >= 1, got {k}.")
    return [pool.sample_opponent(rng) for _ in range(k)]


def _snapshot_for_item(
    item: Any, kind: str, metadata: Dict[str, Any]
) -> OpponentSnapshot:
    """Build an :class:`OpponentSnapshot` from a Population/Archive item.

    ``item`` is whatever the pool stores — typically a DEAP individual
    (with ``.genome``) or a bare :class:`FixedTopologyGenome`. We
    extract a clean snapshot payload here so all kind-specific logic
    lives in one place rather than spread across pool classes.
    """
    if kind == "feedforward":
        # Accept either a DEAP individual (with .genome) or a bare genome.
        # Clone so that subsequent population mutation cannot affect the shipped snapshot.
        genome = item.genome if hasattr(item, "genome") else item
        cloned = genome.clone() if hasattr(genome, "clone") else genome
        payload: Dict[str, Any] = {"genome": cloned}
        if "masked" in metadata:
            payload["masked"] = bool(metadata["masked"])
        return OpponentSnapshot(kind="feedforward", payload=payload)
    if kind in {"frozen", "frozen_registered", "qdax_flax"}:
        raise ValueError(
            f"_snapshot_for_item: kind={kind!r} is not produced by "
            "Population/Archive pools. Use a pool subclass that owns "
            "this kind directly."
        )
    raise ValueError(
        f"_snapshot_for_item: unknown snapshot kind {kind!r}. "
        "Built-in kinds: 'feedforward'. For custom kinds, override "
        "snapshot_opponents on your pool class."
    )


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

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        """Default: live-sample, then attempt to ship policies by pickle.

        Concrete pools should override with a payload that minimises
        what crosses the process boundary (e.g. a flat genome array
        instead of a live :class:`PolicyProtocol`). The default exists
        so user-supplied :class:`OpponentPool` implementations remain
        compatible — a pickling failure becomes a clear runtime error
        attributable to the pool, not to the executor plumbing.
        """
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        return [
            OpponentSnapshot(
                kind="frozen",
                payload={"policy": self.sample_opponent(rng)},
            )  # type: ignore[attr-defined]
            for _ in range(k)
        ]

    def update(self, population_like: Any) -> None:  # pragma: no cover — override
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover — override
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


# Concrete pools


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
        *,
        snapshot_kind: str = "feedforward",
        snapshot_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._policy_factory = policy_factory
        self._sampler = sampler
        self._population: List[Any] = []
        # Defaults set to prams used by EA/CMA; override only if you wire a non-MLP factory.
        self._snapshot_kind = snapshot_kind
        self._snapshot_metadata: Dict[str, Any] = dict(snapshot_metadata or {})
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

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        if not self._population:
            raise ValueError(
                "PopulationPool.snapshot_opponents: pool is empty."
            )
        fitnesses = self._fitnesses()
        snaps: List[OpponentSnapshot] = []
        for _ in range(k):
            idx = self._sampler(len(self._population), fitnesses, rng)
            item = self._population[idx]
            snaps.append(_snapshot_for_item(item, self._snapshot_kind, self._snapshot_metadata))
        return snaps

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
        *,
        snapshot_kind: str = "feedforward",
        snapshot_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._archive = archive
        self._policy_factory = policy_factory
        self._sampler = sampler
        self._snapshot_kind = snapshot_kind
        self._snapshot_metadata: Dict[str, Any] = dict(snapshot_metadata or {})

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

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        size = len(self._archive)
        if size == 0:
            raise ValueError(
                "ArchivePool.snapshot_opponents: archive is empty."
            )
        snaps: List[OpponentSnapshot] = []
        for _ in range(k):
            idx = self._sampler(size, self._archive.fitnesses, rng)
            item = self._archive.items[idx]
            snaps.append(_snapshot_for_item(item, self._snapshot_kind, self._snapshot_metadata))
        return snaps

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
        *,
        registry_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._policies: List[PolicyProtocol] = list(policies)
        if registry_names is not None:
            reg_list = list(registry_names)
            if len(reg_list) != len(self._policies):
                raise ValueError(
                    "FrozenPolicyPool: registry_names length "
                    f"{len(reg_list)} does not match policies length "
                    f"{len(self._policies)}."
                )
            self._registry_names: Optional[List[str]] = reg_list
        else:
            self._registry_names = None
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
        # The user is responsible for re-supplying registry_names via a fresh constructor if they were using the registry path.
        self._registry_names = None

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        if not self._policies:
            raise ValueError("FrozenPolicyPool.sample_opponent: pool is empty.")
        idx = self._sampler(len(self._policies), self._fitnesses, rng)
        return self._policies[idx]

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        if not self._policies:
            raise ValueError("FrozenPolicyPool.snapshot_opponents: pool is empty.")
        snaps: List[OpponentSnapshot] = []
        for _ in range(k):
            idx = self._sampler(len(self._policies), self._fitnesses, rng)
            if self._registry_names is not None:
                snaps.append(
                    OpponentSnapshot(
                        kind="frozen_registered",
                        payload={"name": self._registry_names[idx]},
                    )
                )
            else:
                # Try to ship the policy object directly. 
                # Fails fast with a clear PicklingError if the user supplied a non-picklable wrapper.
                # In this case, switch to the registry path (see ``registry_names=`` ctor kwarg).
                snaps.append(
                    OpponentSnapshot(
                        kind="frozen", payload={"policy": self._policies[idx]}
                    )
                )
        return snaps

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
        choice = self._select_subpool(rng)
        return self._pools[choice].sample_opponent(rng)

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        snaps: List[OpponentSnapshot] = []
        for _ in range(k):
            choice = self._select_subpool(rng)
            snaps.extend(self._pools[choice].snapshot_opponents(rng, 1))
        return snaps

    def _select_subpool(self, rng: np.random.Generator) -> int:
        # Mask out empty pools and renormalise.
        sizes = np.array([len(p) for p in self._pools], dtype=np.int64)
        mask = sizes > 0
        if not mask.any():
            raise ValueError("CompositePool: all sub-pools are empty.")
        effective = self._weights * mask
        total = effective.sum()
        if total <= 0:
            # All non-empty pools happen to have zero weight — fall back to uniform over non-empty pools for robustness.
            effective = mask.astype(np.float64)
            total = effective.sum()
        probs = effective / total
        return int(rng.choice(len(self._pools), p=probs))

    def __len__(self) -> int:
        return sum(len(p) for p in self._pools)

    @property
    def sub_pools(self) -> Tuple[OpponentPool, ...]:
        """The underlying sub-pools (exposed for introspection / testing)."""
        return self._pools


# RL pool (degenerate size-1; wraps a live AgileRLPolicyWrapper)

class RLPolicyPool(_BasePool):
    """Single-element opponent pool wrapping a live RL policy snapshot.

    Used by :class:`~deap_pettingzoo.baselines.coevolution.rl_side.RLSideState`
    to expose its current AgileRL agent to *other* sides during
    bidirectional EA-RL co-evolution. Conceptually a degenerate pool
    of size 1: every :meth:`sample_opponent` call returns the same
    cached policy object (the RL side updates it once per generation).

    The pool stays serial-only in the current implementation —
    :meth:`snapshot_opponents` raises because ``AgileRLPolicyWrapper``
    holds live torch state that does not survive ``spawn``-pickle.
    Co-evolution runs that include an RL side fall back to serial
    evaluation on the EA side; see :class:`CoEvolutionConfig` docstring
    for the runtime warning emitted in that case.
    """

    def __init__(self, policy: PolicyProtocol) -> None:
        if policy is None:
            raise ValueError("RLPolicyPool: policy must not be None.")
        self._policy = policy

    def update(self, population_like: Any) -> None:
        """Swap in a fresh policy (typically called once per generation).

        Args:
            population_like: A :class:`PolicyProtocol` instance. Other
                pool implementations accept iterables; this one is
                always size-1 so we accept the policy directly.
        """
        if population_like is None:
            raise ValueError("RLPolicyPool.update: policy must not be None.")
        self._policy = population_like

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        # rng intentionally unused — degenerate pool of size 1.
        return self._policy

    def sample_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[PolicyProtocol]:
        if k < 1:
            raise ValueError(f"sample_opponents: k must be >= 1, got {k}.")
        return [self._policy] * k

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        raise NotImplementedError(
            "RLPolicyPool.snapshot_opponents: parallel co-evolution with an "
            "RL side is not yet supported because AgileRLPolicyWrapper "
            "holds live torch state that does not pickle across spawn. "
            "Set CoEvolutionConfig.num_workers=1 (or rely on the EA-side "
            "fallback warning) when an RL side is in play."
        )

    def snapshot_sync(self) -> "RLPolicyPool":
        """Return a frozen copy of this pool for the synchronous schedule.

        The synchronous co-evolution schedule snapshots every side's
        opponent pool at the start of a generation so that all sides
        evaluate against the same frozen view of their opponents. EA
        pools satisfy this naturally: their underlying population list
        is replaced (not mutated) at the end of ``step_generation``,
        so the snapshot reference still points at the previous
        population.

        RL pools are different — both the pool object **and** the
        underlying torch agent's weights are mutated in place by the
        RL side's ``learn()`` step. Returning ``self`` would leak the
        post-learn policy back into the same generation's evaluations
        that should have used the pre-learn policy. We therefore
        return a fresh ``RLPolicyPool`` wrapping a deep copy of the
        wrapper (which transitively deep-copies the underlying torch
        modules).

        This method runs once per generation whenever the synchronous
        schedule snapshots pools — regardless of which side is the next
        learner. The RL side's own ``step_generation`` reads its live
        wrapper directly via the driver context and does not go
        through this snapshot path.
        """
        import copy
        import warnings

        try:
            frozen = copy.deepcopy(self._policy)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"RLPolicyPool.snapshot_sync: deep-copy of RL policy failed "
                f"({exc!r}); falling back to the live policy reference. "
                "The synchronous-schedule isolation contract (opponents see "
                "pre-learn weights) is best-effort for this snapshot.",
                RuntimeWarning,
                stacklevel=2,
            )
            frozen = self._policy
        return RLPolicyPool(frozen)

    def __len__(self) -> int:
        return 1

    @property
    def policy(self) -> PolicyProtocol:
        """Currently-cached policy (exposed for introspection / testing)."""
        return self._policy


__all__ = [
    "PolicyFactory",
    "OpponentPool",
    "OpponentSnapshot",
    "PopulationPool",
    "ArchivePool",
    "FrozenPolicyPool",
    "CompositePool",
    "RLPolicyPool",
    "rebuild_opponent_policy",
    "register_frozen_policy",
    "unregister_frozen_policy",
]
