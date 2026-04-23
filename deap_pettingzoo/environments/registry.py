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

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


# A ``factory`` is either:
#   - a module exposing ``.parallel_env(render_mode=None, **cfg)`` for
#     ``api='parallel'`` (or ``.env``/``.raw_env`` for ``api='aec'``), or
#   - a callable ``(render_mode=None, **cfg) -> ParallelEnv | AECEnv``.
EnvFactory = Any

#: Declared API of a registered env. ``'parallel'`` keeps the legacy
#: fast path; ``'aec'`` routes the factory output through
#: :func:`pettingzoo.utils.conversions.aec_to_parallel` at build time
#: (requires simultaneous-step semantics); ``'auto'`` inspects the
#: constructed env against PettingZoo base classes and converts as
#: needed.
EnvApi = Literal["parallel", "aec", "auto"]


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
        api: Declared PettingZoo API of the factory output. ``'parallel'``
            (default) means the factory returns a ``ParallelEnv``;
            ``'aec'`` means it returns an ``AECEnv`` that will be wrapped
            with ``aec_to_parallel`` at build time; ``'auto'`` inspects
            the constructed env and converts only if needed. ``'aec'``
            requires **simultaneous-step** semantics (homogeneous
            agent set across steps, no order-dependent rewards,
            no mid-episode agent revival) — turn-based AEC envs
            (e.g. classic board games) are not supported.
    """

    name: str
    factory: EnvFactory
    default_episode_len: int = 25
    continuous_default: bool = False
    notes: str = ""
    tags: frozenset = field(default_factory=frozenset)
    api: EnvApi = "parallel"


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
    api: EnvApi = "parallel",
    overwrite: bool = False,
) -> EnvSpec:
    """Register a PettingZoo environment.

    Args:
        name: Canonical environment name.
        factory: Module exposing ``parallel_env(**cfg)`` (parallel envs),
            ``env(**cfg)`` / ``raw_env(**cfg)`` (AEC envs), or a callable
            with the same signature.
        default_episode_len: Typical episode length for budget translation.
        continuous_default: Whether the environment's actions are Box by
            default (informational).
        notes: Free-form compatibility notes surfaced in error messages and
            docs.
        tags: Optional set of string tags (e.g. ``{'mpe2', 'competitive'}``).
        api: Declared PettingZoo API of the factory output — ``'parallel'``
            (default), ``'aec'`` (converted via ``aec_to_parallel`` at
            build time; requires simultaneous-step semantics), or
            ``'auto'`` (inspect constructed env and convert if needed).
        overwrite: If ``False`` (default), re-registering an existing name
            raises :class:`ValueError`.  Set ``True`` to replace.

    Returns:
        The newly-registered :class:`EnvSpec`.

    Raises:
        ValueError: If ``name`` is already registered and ``overwrite`` is
            ``False``, or if ``api`` is not one of the supported values.
        TypeError: If ``factory`` is not callable and exposes none of
            ``parallel_env`` / ``env`` / ``raw_env``.
    """
    if api not in ("parallel", "aec", "auto"):
        raise ValueError(
            f"Unsupported api={api!r} for env '{name}'. "
            "Expected 'parallel', 'aec', or 'auto'."
        )
    if not overwrite and name in _REGISTRY:
        raise ValueError(
            f"Environment '{name}' is already registered. "
            "Pass overwrite=True to replace."
        )
    if not callable(factory) and not any(
        callable(getattr(factory, attr, None))
        for attr in ("parallel_env", "env", "raw_env")
    ):
        raise TypeError(
            f"Factory for '{name}' must be callable or expose a callable "
            "'parallel_env', 'env', or 'raw_env' attribute."
        )
    spec = EnvSpec(
        name=name,
        factory=factory,
        default_episode_len=int(default_episode_len),
        continuous_default=bool(continuous_default),
        notes=notes,
        tags=frozenset(tags or ()),
        api=api,
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


def list_envs(tag: Optional[str] = None, api: Optional[EnvApi] = None) -> List[str]:
    """Return a sorted list of registered env names.

    Args:
        tag: Optional tag filter; only envs whose :attr:`EnvSpec.tags`
            contain ``tag`` are returned.
        api: Optional API filter (``'parallel'`` / ``'aec'`` / ``'auto'``);
            only envs whose :attr:`EnvSpec.api` matches are returned.
    """
    items = _REGISTRY.items()
    if tag is not None:
        items = [(n, s) for n, s in items if tag in s.tags]
    if api is not None:
        items = [(n, s) for n, s in items if s.api == api]
    return sorted(n for n, _ in items)


def _pettingzoo_bases():
    """Lazily import PettingZoo base classes + conversion helper."""
    from pettingzoo.utils.conversions import aec_to_parallel
    from pettingzoo.utils.env import AECEnv, ParallelEnv

    return ParallelEnv, AECEnv, aec_to_parallel


def _instantiate_factory(factory, render_mode, kwargs, preferred: str):
    """Invoke ``factory`` and return the raw env.

    Resolution order:
      1. If ``factory`` exposes ``<preferred>``, call that.
      2. Else fall back in order: ``parallel_env`` → ``env`` → ``raw_env``.
      3. Else treat ``factory`` itself as callable.
    """
    tried = []
    for attr in (preferred, "parallel_env", "env", "raw_env"):
        if attr in tried:
            continue
        tried.append(attr)
        fn = getattr(factory, attr, None)
        if callable(fn):
            return fn(render_mode=render_mode, **kwargs)
    if callable(factory):
        return factory(render_mode=render_mode, **kwargs)
    raise TypeError(  # pragma: no cover - guarded at register_env time
        f"Factory {factory!r} is not callable and exposes no "
        "parallel_env/env/raw_env attribute."
    )


def _attach_simultaneity_guard(env, env_name: str):
    """Wrap ``env.reset``/``env.step`` to warn/fail on non-simultaneous AEC.

    Trust-but-verify: ``api='aec'`` callers assert the env has
    simultaneous-step semantics; we warn once if the first reset shows
    ``set(env.agents) != set(env.possible_agents)`` and raise
    ``ValueError`` if a later step drops a previously-active agent
    without a terminal flag (the parallel contract forbids this).
    """
    original_reset = env.reset
    original_step = env.step
    state = {"warned": False, "expected": None}

    def _reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        active = set(getattr(env, "agents", ()) or ())
        possible = set(getattr(env, "possible_agents", ()) or ())
        if not state["warned"] and possible and active and active != possible:
            warnings.warn(
                f"AEC-converted env '{env_name}' reset with active agents "
                f"{sorted(active)} != possible agents {sorted(possible)}. "
                "The parallel contract assumes simultaneous-step semantics; "
                "continuing but results may be inconsistent.",
                UserWarning,
                stacklevel=2,
            )
            state["warned"] = True
        state["expected"] = active or possible
        return result

    def _step(actions, *args, **kwargs):
        result = original_step(actions, *args, **kwargs)
        if state["expected"] is None or not isinstance(result, tuple) or len(result) < 5:
            return result
        observations = result[0] if isinstance(result[0], dict) else {}
        terminations = result[2] if isinstance(result[2], dict) else {}
        truncations = result[3] if isinstance(result[3], dict) else {}
        returned = set(observations.keys())
        terminal = {
            a
            for a in state["expected"]
            if terminations.get(a, False) or truncations.get(a, False)
        }
        missing = (state["expected"] - returned) - terminal
        if missing:
            raise ValueError(
                f"AEC-converted env '{env_name}' dropped agents "
                f"{sorted(missing)} from step output without a terminal "
                "flag — this env is not parallel-compatible. Register "
                "with api='parallel' and a custom adapter instead."
            )
        state["expected"] = returned
        return result

    env.reset = _reset
    env.step = _step
    return env


def _coerce_to_parallel(env, name: str, *, declared: str):
    ParallelEnv, AECEnv, aec_to_parallel = _pettingzoo_bases()
    if isinstance(env, ParallelEnv):
        return env
    if isinstance(env, AECEnv):
        converted = aec_to_parallel(env)
        return _attach_simultaneity_guard(converted, name)
    raise TypeError(
        f"Env '{name}' factory returned {type(env).__name__}, which is "
        "neither ParallelEnv nor AECEnv. Check the factory signature "
        f"and the api={declared!r} registration."
    )


def _reject_if_aec(env, name: str, *, declared: str) -> None:
    """Raise ``TypeError`` when a parallel-declared env is actually AEC."""
    ParallelEnv, AECEnv, _ = _pettingzoo_bases()
    if isinstance(env, AECEnv) and not isinstance(env, ParallelEnv):
        raise TypeError(
            f"Env '{name}' declared api={declared!r} but factory returned "
            f"an AECEnv ({type(env).__name__}). Re-register with "
            "api='aec' (or api='auto') to enable automatic conversion."
        )


def build_env(
    name: str,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Build a parallel-API PettingZoo environment by registered name.

    Args:
        name: Registered environment name.
        env_config: Optional kwargs forwarded to the factory.
        render_mode: Optional render mode.

    Returns:
        A PettingZoo parallel environment. For ``api='aec'`` or
        ``api='auto'`` registrations whose factory returns an
        ``AECEnv``, the env is wrapped with
        :func:`pettingzoo.utils.conversions.aec_to_parallel` and guarded
        by a simultaneity check on the first reset/step.

    Raises:
        ValueError: If ``name`` is not registered.
        TypeError: If the constructed env doesn't match the declared
            ``api`` (e.g. ``api='parallel'`` but the factory returns an
            ``AECEnv``).
    """
    spec = get_env_spec(name)
    factory = spec.factory
    kwargs = dict(env_config or {})

    if spec.api == "parallel":
        # Legacy fast path: factory must supply a parallel env.
        if callable(getattr(factory, "parallel_env", None)):
            env = factory.parallel_env(render_mode=render_mode, **kwargs)
        elif callable(getattr(factory, "env", None)) or callable(
            getattr(factory, "raw_env", None)
        ):
            raise TypeError(
                f"Env '{name}' declared api='parallel' but factory exposes "
                "'env'/'raw_env' (AEC shape) without a 'parallel_env' "
                "attribute. Re-register with api='aec' to enable automatic "
                "conversion."
            )
        elif callable(factory):
            env = factory(render_mode=render_mode, **kwargs)
        else:
            raise TypeError(
                f"Env '{name}' registered with api='parallel' but factory "
                "exposes no callable parallel_env and isn't callable."
            )
        _reject_if_aec(env, name, declared="parallel")
        return env

    # AEC or auto: construct then convert if needed.
    preferred = "env" if spec.api == "aec" else "parallel_env"
    raw = _instantiate_factory(factory, render_mode, kwargs, preferred=preferred)
    return _coerce_to_parallel(raw, name, declared=spec.api)


def clear_registry() -> None:
    """Remove every registered environment.

    Intended for tests that want a clean slate.  Production code should not
    need this.
    """
    _REGISTRY.clear()


__all__ = [
    "EnvSpec",
    "EnvApi",
    "DEFAULT_EPISODE_LEN_FALLBACK",
    "register_env",
    "unregister_env",
    "get_env_spec",
    "is_registered",
    "list_envs",
    "build_env",
    "clear_registry",
]
