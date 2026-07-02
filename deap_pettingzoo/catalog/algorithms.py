"""Algorithm catalog — the single source of truth for trainable algorithms.

This module assembles an :class:`~deap_pettingzoo.catalog.schema.AlgorithmSpec`
for every standalone algorithm the UI can configure. Each spec combines two
sources (the hybrid strategy):

* **Hyperparameters** come from the trainer's ``describe_hyperparameters()``
  declarative overlay, with defaults refreshed against the live constructor /
  ``DEFAULT_PARAMS`` via :mod:`deap_pettingzoo.catalog.introspection`.
* **Cross-cutting metadata** (capabilities, co-evolution mapping, optional-extra
  gating, supported budget modes / aggregations, aliases) is declared here in
  :data:`_ALGORITHM_DEFINITIONS`.

Keeping the cross-cutting facts here — rather than scattered across the trainer
registry, the compatibility checker and the co-evolution validator — is exactly
the "deeper refactor unifying the three registries into one catalog" the plan
calls for. :mod:`deap_pettingzoo.baselines.comparison` and
:mod:`deap_pettingzoo.baselines.compatibility` are refactored to source their
algorithm enumeration from here.

Importing this module is cheap and dependency-light: it never imports the
trainer modules at top level (only lazily, when a spec is actually built) and
never imports ``torch`` / ``agilerl`` / ``jax``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from deap_pettingzoo.catalog.availability import all_modules_available
from deap_pettingzoo.catalog.introspection import collect_defaults, refresh_defaults
from deap_pettingzoo.catalog.schema import AlgorithmSpec, ParamSpec

# Aggregation sets reused below.
_POP_AGGREGATIONS: Tuple[str, ...] = ("sum", "mean", "min", "max", "team")
_MARL_AGGREGATIONS: Tuple[str, ...] = ()  # MARL uses env reward directly.
_BUDGET_MODES: Tuple[str, ...] = ("env_steps", "eval_episodes")
# Population trainers (EA / CMA-ES / PSO) additionally accept a native
# generation budget; the step-based AgileRL trainers do not.
_POP_BUDGET_MODES: Tuple[str, ...] = ("env_steps", "eval_episodes", "generations")


@dataclass(frozen=True)
class _AlgorithmDefinition:
    """Static, dependency-free metadata for one algorithm.

    Holds everything needed to build an :class:`AlgorithmSpec` *except* the
    hyperparameter overlay, which is pulled from the trainer class lazily via
    :func:`_load_hyperparameters`.

    Attributes:
        name: Canonical algorithm identifier used as the key in
            :data:`_REGISTRY` and everywhere in the UI / experiment specs
            (e.g. ``"ea"``, ``"cma-es"``, ``"maddpg"``).
        label: Human-readable display name (e.g. ``"Evolutionary Algorithm"``,
            ``"CMA-ES"``).
        family: Coarse grouping — ``"evolutionary"`` or ``"marl"``.
        trainer_module: Dotted module path where the trainer class lives
            (e.g. ``"deap_pettingzoo.baselines.ea_trainer"``). Imported lazily
            by :func:`_load_hyperparameters` and surfaced by
            :func:`iter_trainer_bindings`.
        trainer_class: Class name (or nested ``__qualname__``) within
            ``trainer_module`` (e.g. ``"EATrainer"``, ``"GroupedIPPOTrainer"``).
            Resolved via :func:`_getattr_qualname`, which walks dot-separated
            parts to support nested class definitions.
        capabilities: The trainer's ``get_capabilities()`` dict mapping —
            ``discrete_action``, ``continuous_action``,
            ``multidiscrete_action``, ``needs_global_state``, ``on_policy``,
            ``supports_turn_based``.
        aliases: Alternative names that resolve to this algorithm (e.g.
            ``"cma"`` → ``"cma-es"``, ``"genetic"`` → ``"ea"``). Populated
            into :data:`_ALIAS_TO_CANONICAL`.
        coevolution_optimizer: The ``Side.optimizer`` token for co-evolution,
            looked up in :data:`~deap_pettingzoo.baselines.coevolution.sides._SIDE_STATE_FACTORIES`.
            ``None`` if the algorithm cannot participate in co-evolution.
        required_modules: Importable module names whose presence gates
            availability (e.g. ``("agilerl", "torch")``, ``("deap",)``).
            Consumed by :func:`is_algorithm_available` via
            :func:`~deap_pettingzoo.catalog.availability.all_modules_available`.
        required_extras: pip extras that provide :attr:`required_modules`
            (e.g. ``("rl-baselines",)``). Surfaced as install hints in the UI.
        supported_budget_modes: Budget modes the trainer accepts. Default
            ``("env_steps", "eval_episodes")``; population-based trainers
            (EA, CMA-ES, PSO) also include ``"generations"``.
        supported_aggregations: Fitness aggregation methods the trainer can
            apply. Default ``("sum", "mean", "min", "max", "team")`` for
            population trainers; MARL trainers use ``()`` (environment reward
            directly).
        description: One-line summary for tooltips and UI display.
        side_state: Optional dotted ``"module:QualName"`` binding to a
            :class:`~deap_pettingzoo.baselines.coevolution.sides.SideState`
            subclass. When set, :func:`register_algorithm` forwards it to
            :func:`~deap_pettingzoo.baselines.coevolution.sides.register_side_state`
            under :attr:`coevolution_optimizer`, so one registration feeds both
            the standalone-algorithm catalog and the co-evolution side factory.
            ``None`` for built-ins (their optimizer tokens are pre-seeded in
            the side factory) and for algorithms that cannot co-evolve.
    """

    name: str
    label: str
    family: str
    trainer_module: str
    trainer_class: str
    capabilities: Mapping[str, bool]
    aliases: Tuple[str, ...] = ()
    coevolution_optimizer: Optional[str] = None
    required_modules: Tuple[str, ...] = ()
    required_extras: Tuple[str, ...] = ()
    supported_budget_modes: Tuple[str, ...] = _BUDGET_MODES
    supported_aggregations: Tuple[str, ...] = _POP_AGGREGATIONS
    description: str = ""
    side_state: Optional[str] = None


def _caps(
    discrete: bool,
    continuous: bool,
    needs_global_state: bool,
    on_policy: bool,
    supports_turn_based: bool,
    *,
    multidiscrete: bool = False,
) -> Dict[str, bool]:
    """Build a capability dict in the trainer ``get_capabilities()`` shape."""
    return {
        "discrete_action": discrete,
        "continuous_action": continuous,
        "multidiscrete_action": multidiscrete,
        "needs_global_state": needs_global_state,
        "on_policy": on_policy,
        "supports_turn_based": supports_turn_based,
    }


# The authoritative algorithm table.

_BUILTIN_DEFINITIONS: Tuple[_AlgorithmDefinition, ...] = (
    _AlgorithmDefinition(
        name="ea",
        label="Evolutionary Algorithm",
        family="evolutionary",
        trainer_module="deap_pettingzoo.baselines.ea_trainer",
        trainer_class="EATrainer",
        capabilities=_caps(True, True, False, False, True, multidiscrete=True),
        aliases=("genetic", "ga"),
        coevolution_optimizer="ea",
        required_modules=("deap",),
        supported_budget_modes=_POP_BUDGET_MODES,
        description="DEAP genetic algorithm over flat MLP weight vectors.",
    ),
    _AlgorithmDefinition(
        name="cma-es",
        label="CMA-ES",
        family="evolutionary",
        trainer_module="deap_pettingzoo.baselines.cma_trainer",
        trainer_class="CMAESTrainer",
        capabilities=_caps(True, True, False, False, True, multidiscrete=True),
        aliases=("cma", "cmaes"),
        coevolution_optimizer="cma",
        required_modules=("deap",),
        supported_budget_modes=_POP_BUDGET_MODES,
        description="Covariance Matrix Adaptation Evolution Strategy.",
    ),
    _AlgorithmDefinition(
        name="pso",
        label="Particle Swarm Optimisation",
        family="evolutionary",
        trainer_module="deap_pettingzoo.baselines.pso_trainer",
        trainer_class="PSOTrainer",
        capabilities=_caps(True, True, False, False, True, multidiscrete=True),
        coevolution_optimizer="pso",
        required_modules=("deap",),
        supported_budget_modes=_POP_BUDGET_MODES,
        description="Particle swarm over flat MLP weight vectors.",
    ),
    _AlgorithmDefinition(
        name="maddpg",
        label="MADDPG",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_trainer",
        trainer_class="MADDPGTrainer",
        capabilities=_caps(True, True, False, False, False),
        coevolution_optimizer="rl",
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Multi-Agent DDPG (off-policy, AgileRL).",
    ),
    _AlgorithmDefinition(
        name="matd3",
        label="MATD3",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_trainer",
        trainer_class="MATD3Trainer",
        capabilities=_caps(True, True, False, False, False),
        coevolution_optimizer="rl",
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Multi-Agent Twin-Delayed DDPG (off-policy, AgileRL).",
    ),
    _AlgorithmDefinition(
        name="ippo",
        label="IPPO",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_trainer",
        trainer_class="IPPOTrainer",
        capabilities=_caps(True, True, False, True, True, multidiscrete=True),
        coevolution_optimizer="rl",
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Independent PPO (on-policy, AgileRL).",
    ),
    _AlgorithmDefinition(
        name="ippo-grouped",
        label="IPPO (grouped)",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_trainer",
        trainer_class="GroupedIPPOTrainer",
        capabilities=_caps(True, True, False, True, True, multidiscrete=True),
        coevolution_optimizer="rl",
        aliases=("grouped-ippo",),
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Prefix-grouped IPPO (on-policy, AgileRL).",
    ),
    # AgileRL evolutionary-HPO variants. Capabilities mirror
    # ``EvolveHPOTrainer.get_capabilities()`` (notably supports_turn_based=False
    # for every variant). ``coevolution_optimizer='rl_hpo'`` routes them to
    # ``RLEvolveHPOSideState`` when used as a co-evolution side.
    _AlgorithmDefinition(
        name="maddpg-hpo",
        label="MADDPG (HPO)",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_hpo_trainer",
        trainer_class="MADDPGHPOTrainer",
        capabilities=_caps(True, True, False, False, False),
        coevolution_optimizer="rl_hpo",
        aliases=("maddpg_hpo",),
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="MADDPG with evolutionary HPO (architecture + HP mutations).",
    ),
    _AlgorithmDefinition(
        name="matd3-hpo",
        label="MATD3 (HPO)",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_hpo_trainer",
        trainer_class="MATD3HPOTrainer",
        capabilities=_caps(True, True, False, False, False),
        coevolution_optimizer="rl_hpo",
        aliases=("matd3_hpo",),
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="MATD3 with evolutionary HPO (architecture + HP mutations).",
    ),
    _AlgorithmDefinition(
        name="ippo-hpo",
        label="IPPO (HPO)",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_hpo_trainer",
        trainer_class="IPPOHPOTrainer",
        capabilities=_caps(True, True, False, True, False, multidiscrete=True),
        coevolution_optimizer="rl_hpo",
        aliases=("ippo_hpo",),
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Independent PPO with evolutionary HPO (on-policy, AgileRL).",
    ),
    _AlgorithmDefinition(
        name="ippo-grouped-hpo",
        label="IPPO grouped (HPO)",
        family="marl",
        trainer_module="deap_pettingzoo.baselines.agilerl_hpo_trainer",
        trainer_class="GroupedIPPOHPOTrainer",
        capabilities=_caps(True, True, False, True, False, multidiscrete=True),
        coevolution_optimizer="rl_hpo",
        aliases=("grouped_ippo_hpo", "ippo-grouped_hpo"),
        required_modules=("agilerl", "torch"),
        required_extras=("rl-baselines",),
        supported_aggregations=_MARL_AGGREGATIONS,
        description="Prefix-grouped IPPO with evolutionary HPO (on-policy, AgileRL).",
    ),
)


# Open registry state
#
# ``_REGISTRY`` is the authoritative, *mutable* algorithm table (an
# insertion-ordered dict, seeded from ``_BUILTIN_DEFINITIONS`` and extended by
# ``register_algorithm`` / entry-point plugins). ``_BY_NAME`` and
# ``_ALIAS_TO_CANONICAL`` are derived caches rebuilt on every mutation.

_REGISTRY: Dict[str, _AlgorithmDefinition] = {}
_BY_NAME: Dict[str, _AlgorithmDefinition] = {}
_ALIAS_TO_CANONICAL: Dict[str, str] = {}

#: Entry-point groups a third-party package declares to auto-register an
#: algorithm (``deap_pettingzoo.algorithms``) or a co-evolution-only side
#: (``deap_pettingzoo.coevolution_sides``). Each entry point names a zero-arg
#: ``register()`` callable that calls :func:`register_algorithm` /
#: :func:`...coevolution.sides.register_side_state`.
_ALGORITHM_ENTRY_POINT_GROUP = "deap_pettingzoo.algorithms"
_SIDE_ENTRY_POINT_GROUP = "deap_pettingzoo.coevolution_sides"

#: Load-once guard so entry-point plugins are discovered exactly once.
_plugins_discovered = False


def _rebuild_lookups() -> None:
    """Rebuild the canonical-name / alias caches from :data:`_REGISTRY`."""
    _BY_NAME.clear()
    _ALIAS_TO_CANONICAL.clear()
    for defn in _REGISTRY.values():
        _BY_NAME[defn.name] = defn
        _ALIAS_TO_CANONICAL[defn.name] = defn.name
        for alias in defn.aliases:
            _ALIAS_TO_CANONICAL[alias] = defn.name


def canonical_name(name: str) -> Optional[str]:
    """Resolve an algorithm name or alias to its canonical form.

    ``"cma"`` → ``"cma-es"``; ``"ea"`` → ``"ea"``; unknown → ``None``.
    """
    return _ALIAS_TO_CANONICAL.get(name)


def list_algorithm_names() -> List[str]:
    """Return the canonical names of every catalog algorithm (declared order)."""
    return [d.name for d in _REGISTRY.values()]


def iter_trainer_bindings() -> List[Tuple[str, str, str]]:
    """Return ``(name, module_path, class_name)`` for every algorithm.

    Consumed by :func:`deap_pettingzoo.baselines.comparison._register_builtins`
    (and its self-healing :func:`..._sync_from_catalog`) so the trainer
    registry and the catalog never drift apart. Does **not** import the
    trainer modules.
    """
    return [(d.name, d.trainer_module, d.trainer_class) for d in _REGISTRY.values()]


def _getattr_qualname(obj: object, dotted_qualname: str) -> object:
    """Resolve a possibly-nested ``"Outer.Inner"`` attribute path off ``obj``.

    A plain ``getattr`` breaks on a nested ``__qualname__`` (``"Outer.Inner"``);
    walking the dot-separated parts resolves both top-level and nested classes,
    matching :func:`...coevolution.sides._import_dotted`.
    """
    for part in dotted_qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _load_hyperparameters(defn: _AlgorithmDefinition) -> Tuple[ParamSpec, ...]:
    """Import the trainer lazily and build its refreshed hyperparameter tuple.

    Returns an empty tuple (with a warning) if the trainer module cannot be
    imported or does not expose ``describe_hyperparameters`` — the algorithm is
    still listed (so the UI can show it as unavailable) but carries no form.
    """
    try:
        module = import_module(defn.trainer_module)
        trainer_cls = _getattr_qualname(module, defn.trainer_class)
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        warnings.warn(
            f"Catalog could not import trainer for {defn.name!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()

    describe = getattr(trainer_cls, "describe_hyperparameters", None)
    if describe is None:  # pragma: no cover - all trainers define it
        return ()

    specs = tuple(describe())
    discovered = collect_defaults(
        getattr(trainer_cls, "__init__", None),
        getattr(trainer_cls, "DEFAULT_PARAMS", None),
    )
    return tuple(refresh_defaults(specs, discovered))


@lru_cache(maxsize=None)
def get_algorithm(name: str) -> AlgorithmSpec:
    """Return the assembled :class:`AlgorithmSpec` for ``name`` (alias-aware).

    Raises:
        KeyError: If ``name`` is not a known algorithm or alias.
    """
    canonical = canonical_name(name)
    if canonical is None:
        raise KeyError(
            f"Unknown algorithm {name!r}. Known: {sorted(_ALIAS_TO_CANONICAL)}"
        )
    defn = _BY_NAME[canonical]
    return AlgorithmSpec(
        name=defn.name,
        label=defn.label,
        family=defn.family,
        aliases=defn.aliases,
        hyperparameters=_load_hyperparameters(defn),
        capabilities=dict(defn.capabilities),
        coevolution_optimizer=defn.coevolution_optimizer,
        required_modules=defn.required_modules,
        required_extras=defn.required_extras,
        supported_budget_modes=defn.supported_budget_modes,
        supported_aggregations=defn.supported_aggregations,
        description=defn.description,
    )


def is_algorithm_available(name: str) -> bool:
    """Return ``True`` if every optional module the algorithm needs is present."""
    canonical = canonical_name(name)
    if canonical is None:
        return False
    return all_modules_available(_BY_NAME[canonical].required_modules)


def list_algorithms(*, only_available: bool = False) -> List[AlgorithmSpec]:
    """Return every algorithm spec, optionally filtered to installed ones."""
    specs = [get_algorithm(d.name) for d in _REGISTRY.values()]
    if only_available:
        specs = [s for s in specs if all_modules_available(s.required_modules)]
    return specs


def coevolution_optimizer_for(name: str) -> Optional[str]:
    """Return the ``Side.optimizer`` token for ``name`` (or ``None``)."""
    canonical = canonical_name(name)
    if canonical is None:
        return None
    return _BY_NAME[canonical].coevolution_optimizer


def clear_cache() -> None:
    """Drop the memoised :func:`get_algorithm` results (used by tests)."""
    get_algorithm.cache_clear()


# Open registration API


def _normalize_binding(obj: Union[str, type], *, kind: str) -> str:
    """Normalise a class or dotted string into a ``"module:QualName"`` binding.

    A class is rendered as ``f"{cls.__module__}:{cls.__qualname__}"`` so worker
    subprocesses can re-import it (honouring the pickle-for-workers invariant).
    A string may already be ``"module:QualName"`` or the ``"module.QualName"``
    dotted form.
    """
    if isinstance(obj, type):
        return f"{obj.__module__}:{obj.__qualname__}"
    if isinstance(obj, str):
        if ":" in obj:
            return obj
        module_path, _, qualname = obj.rpartition(".")
        if not module_path or not qualname:
            raise ValueError(
                f"{kind} string {obj!r} must be 'module:QualName' "
                "(or 'module.QualName')."
            )
        return f"{module_path}:{qualname}"
    raise TypeError(
        f"{kind} must be a class or a dotted 'module:QualName' string; got {obj!r}."
    )


def _split_binding(dotted: str) -> Tuple[str, str]:
    """Split a ``"module:QualName"`` binding into ``(module, class_name)``."""
    module_path, _, class_name = dotted.partition(":")
    return module_path, class_name


def _insert_definition(defn: _AlgorithmDefinition, *, overwrite: bool) -> None:
    """Insert ``defn`` into :data:`_REGISTRY`, guarding name/alias collisions."""
    if not overwrite:
        if defn.name in _REGISTRY:
            raise ValueError(
                f"Algorithm {defn.name!r} is already registered. Pass "
                "overwrite=True to replace."
            )
        clashing = [
            tok
            for tok in (defn.name, *defn.aliases)
            if tok in _ALIAS_TO_CANONICAL and _ALIAS_TO_CANONICAL[tok] != defn.name
        ]
        if clashing:
            raise ValueError(
                f"Algorithm {defn.name!r} declares name/alias(es) {clashing} "
                "that already resolve to a different algorithm. Pass "
                "overwrite=True or choose distinct names."
            )
    _REGISTRY[defn.name] = defn
    _rebuild_lookups()


def register_algorithm(
    name: str,
    *,
    label: str,
    family: str,
    trainer: Union[str, type],
    capabilities: Mapping[str, bool],
    aliases: Sequence[str] = (),
    coevolution_optimizer: Optional[str] = None,
    side_state: Union[str, type, None] = None,
    required_modules: Sequence[str] = (),
    required_extras: Sequence[str] = (),
    supported_budget_modes: Sequence[str] = _BUDGET_MODES,
    supported_aggregations: Sequence[str] = _POP_AGGREGATIONS,
    description: str = "",
    overwrite: bool = False,
) -> None:
    """Register a third-party algorithm into the open catalog.

    This is the single seam for adding a trainable algorithm without editing
    the built-in table. The new algorithm becomes resolvable through the whole
    stack — :func:`get_algorithm`, :func:`list_algorithm_names`,
    :func:`catalog.to_json`, and (via the self-healing trainer registry)
    :func:`deap_pettingzoo.baselines.comparison.get_trainer`.

    Args:
        name: Canonical algorithm identifier.
        label: Human-readable display name.
        family: Coarse grouping (``"evolutionary"`` / ``"marl"`` / custom).
        trainer: The trainer class, or a dotted ``"module:QualName"`` string.
            A class is normalised to its dotted form so worker subprocesses can
            re-import it.
        capabilities: The trainer's ``get_capabilities()`` dict.
        aliases: Alternative names that resolve to ``name``.
        coevolution_optimizer: The ``Side.optimizer`` token this algorithm uses
            as a co-evolution side (or ``None`` if it cannot co-evolve). Also
            the key ``side_state`` is registered under.
        side_state: Optional :class:`SideState` subclass (or dotted string) to
            register as the co-evolution side for ``coevolution_optimizer`` —
            lets a single call feed both the standalone catalog and the side factory.
        required_modules: Importable module names gating availability.
        required_extras: pip extras that provide them (for install hints).
        supported_budget_modes: Budget modes the trainer accepts.
        supported_aggregations: Fitness aggregations the trainer supports.
        description: One-line summary.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ValueError: On a name/alias collision when ``overwrite`` is ``False``,
            or if ``side_state`` is given without ``coevolution_optimizer``,
            or if ``side_state`` is given and ``coevolution_optimizer`` already
            exists in ``_SIDE_STATE_FACTORIES`` (overwrite must be ``True`` to
            replace).

    .. note::
        A programmatic ``register_algorithm`` runs only in the process that
        calls it; like a custom :func:`register_env`, it does **not** auto-
        replay into spawned workers. For parallel/worker runs, ship the
        registration as an entry-point plugin (discovered in every process at
        catalog import) instead.
    """
    if side_state is not None and coevolution_optimizer is None:
        raise ValueError(
            f"register_algorithm({name!r}): side_state was given without a "
            "coevolution_optimizer token to register it under."
        )

    trainer_module, trainer_class = _split_binding(
        _normalize_binding(trainer, kind="trainer")
    )
    side_binding = (
        _normalize_binding(side_state, kind="side_state")
        if side_state is not None
        else None
    )

    defn = _AlgorithmDefinition(
        name=name,
        label=label,
        family=family,
        trainer_module=trainer_module,
        trainer_class=trainer_class,
        capabilities=dict(capabilities),
        aliases=tuple(aliases),
        coevolution_optimizer=coevolution_optimizer,
        required_modules=tuple(required_modules),
        required_extras=tuple(required_extras),
        supported_budget_modes=tuple(supported_budget_modes),
        supported_aggregations=tuple(supported_aggregations),
        description=description,
        side_state=side_binding,
    )
    # Validate the co-evolution side-token collision *before* committing the
    # algorithm, so a rejected side registration can never leave the algorithm
    # half-registered (present in the catalog with a non-working side). Deferred
    # imports keep the catalog import-light and avoid a cycle (registration is a
    # runtime action, not an import concern).
    if side_binding is not None and not overwrite:
        from deap_pettingzoo.baselines.coevolution.sides import (
            _SIDE_STATE_FACTORIES,
        )

        if coevolution_optimizer in _SIDE_STATE_FACTORIES:
            raise ValueError(
                f"register_algorithm({name!r}): co-evolution side optimizer "
                f"{coevolution_optimizer!r} is already registered. Pass "
                "overwrite=True to replace."
            )

    _insert_definition(defn, overwrite=overwrite)
    clear_cache()

    # The side-token collision was pre-validated, so this forward cannot fail.
    if side_binding is not None:
        from deap_pettingzoo.baselines.coevolution.sides import (
            register_side_state,
        )

        register_side_state(coevolution_optimizer, side_binding, overwrite=overwrite)


def _discover_entry_point_plugins() -> None:
    """Invoke every registered ``deap_pettingzoo`` plugin entry point once.

    A broken plugin must never break catalog import: each entry point is loaded
    and called under ``try/except`` and any failure is downgraded to a
    :class:`RuntimeWarning` naming the offending entry point.
    """
    global _plugins_discovered
    if _plugins_discovered:
        return
    _plugins_discovered = True

    from importlib.metadata import entry_points

    for group in (_ALGORITHM_ENTRY_POINT_GROUP, _SIDE_ENTRY_POINT_GROUP):
        try:
            eps = entry_points(group=group)
        except Exception:  # pragma: no cover - importlib.metadata quirks
            continue
        for ep in eps:
            try:
                register = ep.load()
                register()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"deap_pettingzoo plugin entry point {ep.name!r} in group "
                    f"{group!r} failed to load and was skipped: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _register_builtins() -> None:
    """Seed :data:`_REGISTRY` with the built-ins, then discover plugins."""
    for defn in _BUILTIN_DEFINITIONS:
        _REGISTRY[defn.name] = defn
    _rebuild_lookups()
    _discover_entry_point_plugins()


_register_builtins()
