"""Optional-dependency availability probing for the catalog.

Algorithms and environments in deap-bridge are gated behind optional pip
extras (``rl-baselines`` → ``agilerl``/``torch``, ``aquarium`` →
``marl_aquarium``, ``jax``/``qdax`` → ``jax``/``qdax`` ...). The catalog must
report *which* options are actually usable on the current install so the UI
can grey out the rest instead of letting a user submit an experiment that
fails at import time.

This module performs **pure metadata probing** — it uses
:func:`importlib.util.find_spec`, which checks whether a module *could* be
imported without actually importing it. That keeps the catalog cheap and free
of heavy side effects (no ``torch`` CUDA init, no ``jax`` device probing).
Results are cached for the process lifetime.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Dict, Iterable, Sequence


@lru_cache(maxsize=None)
def module_available(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be located on ``sys.path``.

    Uses :func:`importlib.util.find_spec` so the module is **not** imported.
    A defensive ``except`` guards against packages whose parent raises during
    spec resolution (some namespace packages misbehave).
    """

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def all_modules_available(module_names: Iterable[str]) -> bool:
    """Return ``True`` when **every** named module is importable.

    An empty iterable means "no optional dependency" and returns ``True``
    (core functionality is always available).
    """

    return all(module_available(m) for m in module_names)


def availability_map(module_names: Sequence[str]) -> Dict[str, bool]:
    """Return a ``{module: available}`` map for a collection of modules."""

    return {m: module_available(m) for m in dict.fromkeys(module_names)}


def clear_cache() -> None:
    """Reset the memoised availability results.

    Primarily for tests that monkeypatch :func:`importlib.util.find_spec`
    to simulate a missing optional extra.
    """

    module_available.cache_clear()
