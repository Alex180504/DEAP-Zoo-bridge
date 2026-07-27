"""Catalog-level compatibility queries.

This module sources capabilities from the **catalog** — the
:class:`~deap_pettingzoo.catalog.schema.AlgorithmSpec.capabilities` dict, and
feeds them to the existing rule engine via a tiny capability shim.
That keeps one implementation of the compatibility rules while letting the
catalog answer "which algorithms work with this env?" without importing torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


class _CapabilityShim:
    """Minimal object exposing ``get_capabilities()`` for the rule engine.

    :func:`deap_pettingzoo.baselines.compatibility.check_compatibility` only
    calls ``get_capabilities()`` on the trainer it is given, so this shim lets
    us evaluate compatibility from static catalog capabilities without
    constructing a real (possibly uninstallable) trainer.
    """

    def __init__(self, capabilities: Mapping[str, bool]) -> None:
        self._capabilities = dict(capabilities)

    def get_capabilities(self) -> Dict[str, bool]:
        return dict(self._capabilities)


def compatibility_matrix(
    env_name: str,
    algorithm_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a JSON-able compatibility report for ``env_name``.

    Args:
        env_name: Registered environment name.
        algorithm_names: Subset of algorithms to check; defaults to every
            catalog algorithm.

    Returns:
        A JSON-able report of the shape::

            {"env_name", "results": {algo: {compatible, issues, warnings,
             available}}, "compatible": [algo, ...]}
    """
    from deap_pettingzoo.baselines.compatibility import (
        check_compatibility,
        inspect_environment,
    )
    from deap_pettingzoo.catalog import algorithms

    if algorithm_names is None:
        algorithm_names = algorithms.list_algorithm_names()

    env_info = inspect_environment(env_name)

    results: Dict[str, Any] = {}
    compatible: List[str] = []
    for name in algorithm_names:
        canonical = algorithms.canonical_name(name)
        if canonical is None:
            results[name] = {
                "compatible": False,
                "algorithm": name,
                "env_name": env_name,
                "issues": [f"Unknown algorithm {name!r}."],
                "warnings": [],
                "available": False,
            }
            continue
        spec = algorithms.get_algorithm(canonical)
        available = algorithms.is_algorithm_available(canonical)
        shim = _CapabilityShim(spec.capabilities)
        result = check_compatibility(shim, env_info, canonical)
        entry = result.to_dict()
        entry["available"] = available
        results[canonical] = entry
        if result.compatible and available:
            compatible.append(canonical)

    return {
        "env_name": env_name,
        "results": results,
        "compatible": sorted(compatible),
    }


def compatible_algorithms(env_name: str) -> List[str]:
    """Return canonical names of installed algorithms compatible with ``env_name``."""
    return compatibility_matrix(env_name)["compatible"]
