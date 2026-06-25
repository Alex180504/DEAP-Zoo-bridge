"""Environment catalog — wraps the env registry with UI-facing metadata.

The low-level :class:`~deap_pettingzoo.environments.registry.EnvSpec` now
carries a declarative ``params`` schema and ``teams`` metadata (populated for
the built-ins in :mod:`deap_pettingzoo.environments.builtin`). This module
projects each registered env into an
:class:`~deap_pettingzoo.catalog.schema.EnvironmentSpec` — the shape the UI and
the FastAPI ``/catalog/environments`` endpoint consume — and derives
optional-dependency gating from the env's registry tags.

Only **registered** environments are listed. Because the built-ins
auto-register lazily (an env only registers when its package is importable),
the catalog naturally reflects what is installed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from deap_pettingzoo.catalog.availability import all_modules_available
from deap_pettingzoo.catalog.schema import EnvironmentSpec, humanize

# Map a registry tag onto the modules / pip extras that provide the env, so the
# emitted JSON can tell the UI which install unlocks a greyed-out option.
_TAG_REQUIREMENTS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # tag: (required_modules, required_extras)
    "mpe2": (("mpe2",), ()),
    "aquarium": (("marl_aquarium",), ("aquarium",)),
    "openspiel": (("pyspiel",), ("shimmy",)),
}


def _requirements_for_tags(tags) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Union the module / extra requirements implied by an env's tags."""
    modules: List[str] = []
    extras: List[str] = []
    for tag in tags:
        req = _TAG_REQUIREMENTS.get(tag)
        if req is None:
            continue
        modules.extend(req[0])
        extras.extend(req[1])
    # De-duplicate, preserve order.
    return tuple(dict.fromkeys(modules)), tuple(dict.fromkeys(extras))


def _build_spec(env_name: str) -> EnvironmentSpec:
    from deap_pettingzoo.environments.registry import get_env_spec

    env_spec = get_env_spec(env_name)
    modules, extras = _requirements_for_tags(env_spec.tags)
    return EnvironmentSpec(
        name=env_spec.name,
        label=humanize(env_spec.name),
        env_config=tuple(env_spec.params),
        teams=tuple(env_spec.teams),
        api=str(env_spec.api),
        continuous_default=bool(env_spec.continuous_default),
        tags=tuple(sorted(env_spec.tags)),
        default_episode_len=int(env_spec.default_episode_len),
        notes=env_spec.notes,
        required_modules=modules,
        required_extras=extras,
    )


@lru_cache(maxsize=None)
def get_environment(name: str) -> EnvironmentSpec:
    """Return the :class:`EnvironmentSpec` for a registered environment.

    Raises:
        ValueError: If ``name`` is not registered (propagated from
            :func:`~deap_pettingzoo.environments.registry.get_env_spec`).
    """
    return _build_spec(name)


def is_environment_available(name: str) -> bool:
    """Return ``True`` if the env's optional dependencies are importable.

    Registered built-ins are by construction available (they only register
    when importable); this still returns the honest answer for envs whose
    tags imply an extra.
    """
    try:
        spec = get_environment(name)
    except ValueError:
        return False
    return all_modules_available(spec.required_modules)


def list_environments(*, only_available: bool = False) -> List[EnvironmentSpec]:
    """Return an :class:`EnvironmentSpec` for every registered environment."""
    from deap_pettingzoo.environments.registry import list_envs

    specs = [get_environment(name) for name in list_envs()]
    if only_available:
        specs = [s for s in specs if all_modules_available(s.required_modules)]
    return specs


def clear_cache() -> None:
    """Drop memoised env specs (used by tests that register/unregister envs)."""
    get_environment.cache_clear()
