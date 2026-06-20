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
from typing import Dict, List, Mapping, Optional, Tuple

from deap_pettingzoo.catalog.availability import all_modules_available
from deap_pettingzoo.catalog.introspection import collect_defaults, refresh_defaults
from deap_pettingzoo.catalog.schema import AlgorithmSpec, ParamSpec

# Aggregation sets reused below.
_POP_AGGREGATIONS: Tuple[str, ...] = ("sum", "mean", "min", "max", "team")
_MARL_AGGREGATIONS: Tuple[str, ...] = ()  # MARL uses env reward directly.
_BUDGET_MODES: Tuple[str, ...] = ("env_steps", "eval_episodes")


@dataclass(frozen=True)
class _AlgorithmDefinition:
    """Static, dependency-free metadata for one algorithm.

    Holds everything needed to build an :class:`AlgorithmSpec` *except* the
    hyperparameter overlay, which is pulled from the trainer class lazily.
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


# ---------------------------------------------------------------------------
# The authoritative algorithm table
# ---------------------------------------------------------------------------

_ALGORITHM_DEFINITIONS: Tuple[_AlgorithmDefinition, ...] = (
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
)


# Canonical-name and alias lookup tables (built once).
_BY_NAME: Dict[str, _AlgorithmDefinition] = {d.name: d for d in _ALGORITHM_DEFINITIONS}
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _d in _ALGORITHM_DEFINITIONS:
    _ALIAS_TO_CANONICAL[_d.name] = _d.name
    for _alias in _d.aliases:
        _ALIAS_TO_CANONICAL[_alias] = _d.name


def canonical_name(name: str) -> Optional[str]:
    """Resolve an algorithm name or alias to its canonical form.

    ``"cma"`` → ``"cma-es"``; ``"ea"`` → ``"ea"``; unknown → ``None``.
    """
    return _ALIAS_TO_CANONICAL.get(name)


def list_algorithm_names() -> List[str]:
    """Return the canonical names of every catalog algorithm (declared order)."""
    return [d.name for d in _ALGORITHM_DEFINITIONS]


def iter_trainer_bindings() -> List[Tuple[str, str, str]]:
    """Return ``(name, module_path, class_name)`` for every algorithm.

    Consumed by :func:`deap_pettingzoo.baselines.comparison._register_builtins`
    so the trainer registry and the catalog never drift apart. Does **not**
    import the trainer modules.
    """
    return [
        (d.name, d.trainer_module, d.trainer_class) for d in _ALGORITHM_DEFINITIONS
    ]


def _load_hyperparameters(defn: _AlgorithmDefinition) -> Tuple[ParamSpec, ...]:
    """Import the trainer lazily and build its refreshed hyperparameter tuple.

    Returns an empty tuple (with a warning) if the trainer module cannot be
    imported or does not expose ``describe_hyperparameters`` — the algorithm is
    still listed (so the UI can show it as unavailable) but carries no form.
    """
    try:
        module = import_module(defn.trainer_module)
        trainer_cls = getattr(module, defn.trainer_class)
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
    specs = [get_algorithm(d.name) for d in _ALGORITHM_DEFINITIONS]
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
