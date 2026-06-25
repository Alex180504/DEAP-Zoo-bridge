"""Unified catalog — single source of truth for the experiment-setup UI.

The catalog answers the questions the experiment-setup UI needs to build
dynamic forms and enforce compatibility, **without** the frontend importing any
heavy training dependency:

* *Which algorithms exist, and what are their hyperparameters?*
  :func:`list_algorithms` / :func:`get_algorithm`.
* *Which environments exist, what are their config fields, and are they
  multi-team (co-evolution-capable)?* :func:`list_environments` /
  :func:`get_environment`.
* *What co-evolution modes / schedules / optimizers are valid?*
  :func:`get_coevolution_options`.
* *Which algorithms are compatible with a given environment?*
  :func:`compatible_algorithms` / :func:`compatibility_matrix`.
* *What does the whole thing look like as one JSON document the backend can
  serve?* :func:`to_json`.

Design note — **lazy imports**. The pure schema primitives
(:class:`ParamSpec`, :class:`AlgorithmSpec`, :class:`EnvironmentSpec`, ...) are
re-exported eagerly because they have no dependencies. Everything that touches
the environment registry, trainers or compatibility checker is imported *inside*
the facade functions. This keeps ``import deap_pettingzoo.catalog`` cheap and,
critically, lets :mod:`deap_pettingzoo.environments.registry` import
:class:`ParamSpec` from :mod:`deap_pettingzoo.catalog.schema` without creating
an import cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Eager (pure, dependency-free) re-exports.
from deap_pettingzoo.catalog.schema import (
    CATALOG_SCHEMA_VERSION,
    AlgorithmSpec,
    EnvironmentSpec,
    ParamSpec,
    ParamTarget,
    ParamType,
    TeamSpec,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "AlgorithmSpec",
    "EnvironmentSpec",
    "ParamSpec",
    "ParamTarget",
    "ParamType",
    "TeamSpec",
    "ValidationIssue",
    "ValidationReport",
    "list_algorithms",
    "get_algorithm",
    "canonical_algorithm_name",
    "is_algorithm_available",
    "list_environments",
    "get_environment",
    "is_environment_available",
    "get_coevolution_options",
    "compatible_algorithms",
    "compatibility_matrix",
    "to_json",
]


# -- algorithms -----------------------------------------------------------


def list_algorithms(*, only_available: bool = False) -> List[AlgorithmSpec]:
    """Return every algorithm spec (see :mod:`.algorithms`)."""
    from deap_pettingzoo.catalog import algorithms

    return algorithms.list_algorithms(only_available=only_available)


def get_algorithm(name: str) -> AlgorithmSpec:
    """Return the :class:`AlgorithmSpec` for ``name`` (alias-aware)."""
    from deap_pettingzoo.catalog import algorithms

    return algorithms.get_algorithm(name)


def canonical_algorithm_name(name: str) -> Optional[str]:
    """Resolve an algorithm name/alias to its canonical form, or ``None``."""
    from deap_pettingzoo.catalog import algorithms

    return algorithms.canonical_name(name)


def is_algorithm_available(name: str) -> bool:
    """Return ``True`` if the algorithm's optional dependencies are installed."""
    from deap_pettingzoo.catalog import algorithms

    return algorithms.is_algorithm_available(name)


# -- environments ---------------------------------------------------------


def list_environments(*, only_available: bool = False) -> List[EnvironmentSpec]:
    """Return every registered environment as an :class:`EnvironmentSpec`."""
    from deap_pettingzoo.catalog import environments

    return environments.list_environments(only_available=only_available)


def get_environment(name: str) -> EnvironmentSpec:
    """Return the :class:`EnvironmentSpec` for a registered environment."""
    from deap_pettingzoo.catalog import environments

    return environments.get_environment(name)


def is_environment_available(name: str) -> bool:
    """Return ``True`` if the environment's optional dependencies are installed."""
    from deap_pettingzoo.catalog import environments

    return environments.is_environment_available(name)


# -- co-evolution ---------------------------------------------------------


def get_coevolution_options() -> Dict[str, Any]:
    """Return the co-evolution modes / schedules / samplers / optimizers."""
    from deap_pettingzoo.catalog import coevolution

    return coevolution.get_coevolution_options()


# -- compatibility --------------------------------------------------------


def compatible_algorithms(env_name: str) -> List[str]:
    """Return canonical names of algorithms compatible with ``env_name``."""
    from deap_pettingzoo.catalog import compatibility

    return compatibility.compatible_algorithms(env_name)


def compatibility_matrix(
    env_name: str,
    algorithm_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a JSON-able compatibility matrix for ``env_name``."""
    from deap_pettingzoo.catalog import compatibility

    return compatibility.compatibility_matrix(env_name, algorithm_names)


# -- full snapshot --------------------------------------------------------


def to_json(*, only_available: bool = False) -> Dict[str, Any]:
    """Return the entire catalog as one JSON-serialisable document.

    This is what the FastAPI service serves at ``GET /catalog`` and what the
    Streamlit frontend renders forms from. Each algorithm / environment entry
    carries an ``available`` flag so the UI can grey out options whose optional
    extras are missing.
    """
    from deap_pettingzoo.catalog import algorithms, coevolution, environments

    algo_specs = algorithms.list_algorithms()
    env_specs = environments.list_environments()
    return {
        "catalog_version": CATALOG_SCHEMA_VERSION,
        "algorithms": [
            spec.to_dict(available=algorithms.is_algorithm_available(spec.name))
            for spec in algo_specs
            if (not only_available) or algorithms.is_algorithm_available(spec.name)
        ],
        "environments": [
            spec.to_dict(available=environments.is_environment_available(spec.name))
            for spec in env_specs
            if (not only_available)
            or environments.is_environment_available(spec.name)
        ],
        "coevolution": coevolution.get_coevolution_options(),
        "budget_modes": [
            {"name": "env_steps", "label": "Environment steps"},
            {"name": "eval_episodes", "label": "Evaluation episodes"},
        ],
    }
