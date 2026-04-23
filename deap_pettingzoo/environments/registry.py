"""Open environment registry for deap-bridge.

A single source of truth mapping a string ``env_name`` to the factory used
by :class:`~deap_pettingzoo.environments.PettingZooWrapper` and
:func:`~deap_pettingzoo.baselines.agilerl_adapter.build_parallel_env`.

The registry is **open**: any PettingZoo parallel-API environment can be
registered by calling :func:`register_env` before the first wrapper/trainer
instantiation.  MPE2 and ``marl-aquarium`` environments are auto-registered
on package import if their respective packages are installed (see
``deap_pettingzoo/environments/builtin.py``).

Typical third-party use::

    from my_pkg import my_env_module  # exposes .parallel_env(**cfg)
    from deap_pettingzoo.environments.registry import register_env

    register_env("my_env_v0", my_env_module, default_episode_len=50)

    wrapper = PettingZooWrapper("my_env_v0")

A minimal env *module* is any object exposing a callable ``parallel_env``
attribute that returns a PettingZoo parallel environment.  Alternatively
``factory`` can itself be a plain callable (``factory(render_mode=None,
**cfg) -> ParallelEnv``); both forms are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# A ``factory`` is either:
#   - a module exposing ``.parallel_env(render_mode=None, **cfg)``, or
#   - a callable ``(render_mode=None, **cfg) -> ParallelEnv``.
EnvFactory = Any


@dataclass
class EnvSpec:
    """Metadata + factory for a registered environment.

    Attributes:
        name: Canonical environment identifier (e.g. ``'simple_tag_v3'``).
        factory: Module or callable used to build the parallel environment.
        default_episode_len: Expected episode length, used when translating
            ``BudgetMode.EVAL_EPISODES`` to environment steps.  Fallback to
            :data:`DEFAULT_EPISODE_LEN_FALLBACK` if unknown.
        continuous_default: Whether the env defaults to continuous (Box)
            actions.  Informational only.
        notes: Free-form compatibility notes (e.g. ``"MultiDiscrete actions
            — FeedforwardNeuralPolicy unsupported; use AgileRL trainers"``).
        tags: Arbitrary labels for grouping / filtering (e.g. ``{'mpe2',
            'cooperative'}``).
    """

    name: str
    factory: EnvFactory
    default_episode_len: int = 25
    continuous_default: bool = False
    notes: str = ""
    tags: frozenset = field(default_factory=frozenset)


#: Fallback episode length when an :class:`EnvSpec` does not specify one.
DEFAULT_EPISODE_LEN_FALLBACK: int = 25


_REGISTRY: Dict[str, EnvSpec] = {}


def register_env(
    name: str,
    factory: EnvFactory,
    *,
    default_episode_len: int = DEFAULT_EPISODE_LEN_FALLBACK,
    continuous_default: bool = False,
    notes: str = "",
    tags: Optional[set] = None,
    overwrite: bool = False,
) -> EnvSpec:
    """Register a PettingZoo parallel-API environment.

    Args:
        name: Canonical environment name.
        factory: Module exposing ``parallel_env(**cfg)`` or a callable with
            the same signature.
        default_episode_len: Typical episode length for budget translation.
        continuous_default: Whether the environment's actions are Box by
            default (informational).
        notes: Free-form compatibility notes surfaced in error messages and
            docs.
        tags: Optional set of string tags (e.g. ``{'mpe2', 'competitive'}``).
        overwrite: If ``False`` (default), re-registering an existing name
            raises :class:`ValueError`.  Set ``True`` to replace.

    Returns:
        The newly-registered :class:`EnvSpec`.

    Raises:
        ValueError: If ``name`` is already registered and ``overwrite`` is
            ``False``.
        TypeError: If ``factory`` exposes neither a ``parallel_env``
            attribute nor is callable.
    """
    if not overwrite and name in _REGISTRY:
        raise ValueError(
            f"Environment '{name}' is already registered. "
            "Pass overwrite=True to replace."
        )
    if not callable(factory) and not callable(getattr(factory, "parallel_env", None)):
        raise TypeError(
            f"Factory for '{name}' must be callable or expose a callable "
            "'parallel_env' attribute."
        )
    spec = EnvSpec(
        name=name,
        factory=factory,
        default_episode_len=int(default_episode_len),
        continuous_default=bool(continuous_default),
        notes=notes,
        tags=frozenset(tags or ()),
    )
    _REGISTRY[name] = spec
    return spec


def unregister_env(name: str) -> None:
    """Remove ``name`` from the registry.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    del _REGISTRY[name]


def get_env_spec(name: str) -> EnvSpec:
    """Return the :class:`EnvSpec` for ``name``.

    Raises:
        ValueError: If ``name`` is not registered.  The error message lists
            all currently-registered names and suggests installing the
            MPE2 / aquarium extras or calling :func:`register_env`.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unsupported / unregistered environment '{name}'. "
            f"Known: {known}. "
            "Register a third-party env with "
            "`deap_pettingzoo.environments.registry.register_env(...)` or "
            "install the required extra (e.g. `pip install mpe2` or "
            "`pip install marl-aquarium`)."
        ) from None


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is in the registry."""
    return name in _REGISTRY


def list_envs(tag: Optional[str] = None) -> List[str]:
    """Return a sorted list of registered env names.

    Args:
        tag: Optional tag filter; only envs whose :attr:`EnvSpec.tags`
            contain ``tag`` are returned.
    """
    if tag is None:
        return sorted(_REGISTRY.keys())
    return sorted(n for n, s in _REGISTRY.items() if tag in s.tags)


def build_env(
    name: str,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Build a parallel-API PettingZoo environment by registered name.

    Args:
        name: Registered environment name.
        env_config: Optional kwargs forwarded to ``parallel_env``.
        render_mode: Optional render mode.

    Returns:
        A PettingZoo parallel environment.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    spec = get_env_spec(name)
    factory = spec.factory
    kwargs = dict(env_config or {})
    if callable(getattr(factory, "parallel_env", None)):
        return factory.parallel_env(render_mode=render_mode, **kwargs)
    return factory(render_mode=render_mode, **kwargs)


def clear_registry() -> None:
    """Remove every registered environment.

    Intended for tests that want a clean slate.  Production code should not
    need this.
    """
    _REGISTRY.clear()


__all__ = [
    "EnvSpec",
    "DEFAULT_EPISODE_LEN_FALLBACK",
    "register_env",
    "unregister_env",
    "get_env_spec",
    "is_registered",
    "list_envs",
    "build_env",
    "clear_registry",
]
