"""Open environment registry for deap-bridge.

A single source-of-truth mapping a string ``env_name`` to the factory used
by :class:`~deap_pettingzoo.environments.PettingZooWrapper` and
:func:`~deap_pettingzoo.baselines.agilerl_adapter.build_parallel_env`.

The registry is **open**: any PettingZoo parallel-API environment can be
registered by calling :func:`register_env` before the first wrapper/trainer
instantiation.  MPE2 and ``marl-aquarium`` environments are auto-registered
on package import if their respective packages are installed (see
``deap_pettingzoo/environments/builtin.py``).

Example third-party use::

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
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from deap_pettingzoo.catalog.schema import ParamSpec, TeamSpec


class EnvFactory:
    """Factory for constructing PettingZoo environments.

    Can be either:
      - A module exposing ``.parallel_env(render_mode=None, **cfg)`` for
        ``api='parallel'`` (or ``.env``/``.raw_env`` for ``api='aec'``).
      - A callable ``(render_mode=None, **cfg) -> ParallelEnv | AECEnv``.
    """


EnvApi = Literal[
    "parallel",
    "aec",
    "auto",
    "aec_masked",
    "aec_sequential",
    "jax_parallel",
]
"""Declared API of a registered environment.

Attributes:
    parallel: Factory returns a ``ParallelEnv`` directly (legacy fast path).
    aec: Factory returns an ``AECEnv``, converted via
        :func:`pettingzoo.utils.conversions.aec_to_parallel` at build time.
        Requires simultaneous-step semantics
        (``metadata['is_parallelizable'] = True``).
    auto: Inspects constructed env against PettingZoo base classes and
        converts as needed.
    aec_masked: Wraps strictly turn-based AEC envs via
        :class:`~deap_pettingzoo.environments.turn_based.MaskedSimultaneousAdapter`.
        Presents a parallel-shaped facade where every agent appears each
        step with only the current player's action executed (EA-friendly;
        wastes N−1 inferences/step but needs no evaluator changes).
    aec_sequential: Wraps strictly turn-based AEC envs via
        :class:`~deap_pettingzoo.environments.turn_based.SequentialTurnAdapter`.
        Returns single-agent dicts per step, preserving true turn semantics.
    jax_parallel: Routes a JAX-native env (e.g. JaxMARL) through
        :class:`~deap_pettingzoo.environments.jax_parallel.JaxEnvToPettingZooAdapter`,
        giving a numpy-facing parallel facade. The underlying env is not
        ``jit``'d; use :mod:`deap_pettingzoo.contrib.qdax.scoring_jax` for
        the fast JAX path.
"""


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
    """Ordered tuple of callables applied to the constructed env after API
    coercion (turn-based adapter / ``aec_to_parallel`` / identity). Each
    callable takes the current env and returns a wrapped env, matching the
    :mod:`supersuit` wrapper signature. The chain runs left-to-right.
    Default is an empty tuple (no wrapping).
    """
    wrappers: Tuple[Callable[[Any], Any], ...] = ()
    #: Declarative schema for the environment-constructor kwargs, consumed by
    #: the catalog / experiment-setup UI to auto-generate a typed config form.
    #: Empty ⇒ no introspectable params (the env takes no kwargs, or none are
    #: surfaced). MPE2 factories use ``**kwargs`` passthrough so this metadata
    #: is hand-authored in :mod:`deap_pettingzoo.environments.builtin`.
    params: Tuple[ParamSpec, ...] = ()
    #: Declared agent teams (groups of agent-name prefixes). ``len(teams) >= 2``
    #: marks the env as multi-team and therefore co-evolution-capable. Empty ⇒
    #: single-team / cooperative (the UI hides co-evolution for it unless the
    #: user explicitly opts in).
    teams: Tuple[TeamSpec, ...] = ()


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
    wrappers: Sequence[Callable[[Any], Any]] = (),
    params: Sequence[ParamSpec] = (),
    teams: Sequence[TeamSpec] = (),
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
        wrappers: Optional ordered sequence of callables applied to the
            constructed env after api coercion. Typical use is composing
            :mod:`supersuit` wrappers (``pad_observations_v0``,
            ``pad_action_space_v0``, ``agent_indicator_v0``,
            ``normalize_obs_v0``, ``black_death_v3``, ...). Each entry
            must be callable; they run left-to-right.
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
    if api not in (
        "parallel",
        "aec",
        "auto",
        "aec_masked",
        "aec_sequential",
        "jax_parallel",
    ):
        raise ValueError(
            f"Unsupported api={api!r} for env '{name}'. "
            "Expected 'parallel', 'aec', 'auto', 'aec_masked', "
            "'aec_sequential', or 'jax_parallel'."
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
    wrapper_tuple = tuple(wrappers)
    for idx, w in enumerate(wrapper_tuple):
        if not callable(w):
            raise TypeError(
                f"wrappers[{idx}] for env '{name}' is not callable: "
                f"got {w!r} (type {type(w).__name__})."
            )
    spec = EnvSpec(
        name=name,
        factory=factory,
        default_episode_len=int(default_episode_len),
        continuous_default=bool(continuous_default),
        notes=notes,
        tags=frozenset(tags or ()),
        api=api,
        wrappers=wrapper_tuple,
        params=tuple(params),
        teams=tuple(teams),
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
        if (
            state["expected"] is None
            or not isinstance(result, tuple)
            or len(result) < 5
        ):
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
        try:
            converted = aec_to_parallel(env)
        except AssertionError as exc:
            raise TypeError(
                f"Env '{name}' is an AECEnv without "
                "metadata['is_parallelizable']=True, so aec_to_parallel "
                "refused it. For strictly turn-based games register with "
                "api='aec_masked' (EA-friendly parallel-shaped facade) "
                "or api='aec_sequential' (single-agent-per-step, true "
                "turn semantics) instead of api='aec'."
            ) from exc
        return _attach_simultaneity_guard(converted, name)
    raise TypeError(
        f"Env '{name}' factory returned {type(env).__name__}, which is "
        "neither ParallelEnv nor AECEnv. Check the factory signature "
        f"and the api={declared!r} registration."
    )


def _coerce_turn_based(env, name: str, *, mode: str):
    """Wrap a turn-based AEC env in the appropriate adapter.

    ``mode='aec_masked'`` → :class:`MaskedSimultaneousAdapter`;
    ``mode='aec_sequential'`` → :class:`SequentialTurnAdapter`.
    Both are imported lazily from
    :mod:`deap_pettingzoo.environments.turn_based`.
    """
    _, AECEnv, _ = _pettingzoo_bases()
    if not isinstance(env, AECEnv):
        raise TypeError(
            f"Env '{name}' declared api={mode!r} but factory returned "
            f"{type(env).__name__}, not an AECEnv. Turn-based adapters "
            "only wrap AEC envs."
        )
    from deap_pettingzoo.environments.turn_based import (
        MaskedSimultaneousAdapter,
        SequentialTurnAdapter,
    )

    if mode == "aec_masked":
        return MaskedSimultaneousAdapter(env, env_name=name)
    if mode == "aec_sequential":
        return SequentialTurnAdapter(env, env_name=name)
    raise ValueError(f"Unknown turn-based mode {mode!r} for env '{name}'.")


def _reject_if_aec(env, name: str, *, declared: str) -> None:
    """Raise ``TypeError`` when a parallel-declared env is actually AEC."""
    ParallelEnv, AECEnv, _ = _pettingzoo_bases()
    if isinstance(env, AECEnv) and not isinstance(env, ParallelEnv):
        raise TypeError(
            f"Env '{name}' declared api={declared!r} but factory returned "
            f"an AECEnv ({type(env).__name__}). Re-register with "
            "api='aec' (or api='auto') to enable automatic conversion."
        )


def build_env_from_spec(
    spec: "EnvSpec",
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Build a parallel-API environment directly from a resolved :class:`EnvSpec`.

    Identical to :func:`build_env` but takes a pre-resolved ``spec``
    instead of looking up a name in the main-process registry.  This
    makes it safe to call from worker subprocesses (e.g.
    ``AsyncPettingZooVecEnv`` workers) that do not inherit the
    main-process registry state: the caller resolves the spec once in
    the main process, pickles it into the factory closure, and each
    worker calls this function directly.

    Args:
        spec: Pre-resolved :class:`EnvSpec` (picklable).
        env_config: Optional kwargs forwarded to the factory.
        render_mode: Optional render mode.

    Returns:
        A PettingZoo parallel environment with all wrapper chain applied.

    Raises:
        TypeError: If the constructed env doesn't match the declared
            ``api``.
    """
    name = spec.name
    factory = spec.factory
    kwargs = dict(env_config or {})

    if spec.api == "parallel":
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
    elif spec.api in ("aec_masked", "aec_sequential"):
        raw = _instantiate_factory(factory, render_mode, kwargs, preferred="env")
        env = _coerce_turn_based(raw, name, mode=spec.api)
    elif spec.api == "jax_parallel":
        if callable(factory):
            raw = factory(**kwargs)
        else:
            raise TypeError(
                f"Env '{name}' registered with api='jax_parallel' but "
                "factory is not callable."
            )
        from deap_pettingzoo.environments.jax_parallel import (
            JaxEnvToPettingZooAdapter,
        )

        env = JaxEnvToPettingZooAdapter(raw, env_name=name, render_mode=render_mode)
    else:
        preferred = "env" if spec.api == "aec" else "parallel_env"
        raw = _instantiate_factory(factory, render_mode, kwargs, preferred=preferred)
        env = _coerce_to_parallel(raw, name, declared=spec.api)

    for wrapper in spec.wrappers:
        env = wrapper(env)
    return env


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
    return build_env_from_spec(spec, env_config=env_config, render_mode=render_mode)


def clear_registry() -> None:
    """Remove every registered environment. Intended for tests."""
    _REGISTRY.clear()


def get_registry_snapshot() -> List[Tuple[str, "EnvSpec"]]:
    """Return a picklable snapshot of all currently registered environments.

    The snapshot is a plain list of ``(name, EnvSpec)`` tuples; both
    strings and :class:`EnvSpec` dataclass instances are picklable as
    long as the ``factory`` attribute is picklable (i.e. it is a named
    module, a top-level callable, or a picklable class instance — *not*
    a lambda or a locally-defined closure).

    Intended use: call this in the **main process** immediately before
    creating a :class:`~concurrent.futures.ProcessPoolExecutor`, then
    pass the result to :func:`replay_registry_snapshot` as the worker
    ``initializer`` argument.  This propagates custom
    :func:`register_env` calls to each spawned worker, which would
    otherwise start with only the built-in registrations::

        from deap_pettingzoo.environments.registry import (
            get_registry_snapshot,
            replay_registry_snapshot,
        )
        snapshot = get_registry_snapshot()
        with make_executor(
            n_workers,
            backend="process",
            initializer=replay_registry_snapshot,
            initargs=(snapshot,),
        ) as executor:
            ...

    .. note::
        The Dask backend does not expose a per-worker
        ``initializer`` hook.  :func:`make_executor` accepts but
        silently ignores ``initializer``/``initargs`` for that
        backend.  To propagate custom environments there, the snapshot
        is instead carried on each picklable worker spec/payload and
        replayed lazily in the worker via
        :func:`replay_registry_snapshot_once`.

    .. note::
        Entries whose ``factory`` is not picklable are omitted from the
        snapshot.  Built-in environments (MPE2, aquarium) are registered
        with a bare **module** object (also not picklable) but are
        auto-registered in every worker process at import time via
        :mod:`deap_pettingzoo.environments.builtin`, so they do not need
        to be propagated through the snapshot and their omission is
        silent.  Any *other* non-picklable factory — a lambda, a closure,
        or a locally-defined class — is a custom :func:`register_env` that
        will **not** reach ``num_workers>1`` workers (surfacing later as a
        cryptic "unregistered env"): each such drop emits a
        :class:`RuntimeWarning` naming the env so the cause is visible at
        snapshot time rather than deep inside a worker. Register custom
        envs with a module-level factory function/class to make them
        propagate.
    """
    import inspect
    import pickle

    result = []
    for name, spec in _REGISTRY.items():
        try:
            pickle.dumps(spec)
        except Exception:  # noqa: BLE001
            # Module-object factories are re-registered in every worker at import time;
            # their omission is expected and silent. Warn only for the surprising cases.
            if not inspect.ismodule(spec.factory):
                # Pinpoint the source: the factory itself, or something in the
                # wrapper chain / params / teams (so the message doesn't blame
                # a factory that actually pickles fine).
                try:
                    pickle.dumps(spec.factory)
                    culprit = (
                        "non-picklable metadata (a wrapper / param / team — "
                        "the factory itself pickles fine)"
                    )
                except Exception:  # noqa: BLE001
                    culprit = (
                        "a non-picklable factory " f"({type(spec.factory).__name__})"
                    )
                warnings.warn(
                    f"Environment {name!r} has {culprit} and will not propagate "
                    "to parallel workers (num_workers>1); it must be "
                    "re-registered there. Use a module-level factory function "
                    "or class (and picklable wrappers) instead of a "
                    "lambda/closure so it can be pickled into the worker "
                    "snapshot.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        result.append((name, spec))
    return result


def replay_registry_snapshot(snapshot: List[Tuple[str, "EnvSpec"]]) -> None:
    """Re-register environments in a worker process from a snapshot.

    This function is designed to be used as the ``initializer`` argument
    to :class:`~concurrent.futures.ProcessPoolExecutor` (via
    :func:`~deap_pettingzoo.utils.parallel.make_executor`).  It is
    called **once per worker process** at startup and re-populates
    ``_REGISTRY`` with every entry from ``snapshot``.

    Entries that the worker already knows about (e.g. built-ins
    auto-registered at import time) are overwritten safely using
    ``overwrite=True``, so the function is fully idempotent.

    Args:
        snapshot: The list returned by :func:`get_registry_snapshot`
            in the main process.
    """
    for name, spec in snapshot:
        register_env(
            name,
            spec.factory,
            default_episode_len=spec.default_episode_len,
            continuous_default=spec.continuous_default,
            notes=spec.notes,
            tags=set(spec.tags),
            api=spec.api,
            wrappers=spec.wrappers,
            params=spec.params,
            teams=spec.teams,
            overwrite=True,
        )


# Per-process guard for :func:`replay_registry_snapshot_once`. Keyed on the set
# of env names a snapshot carries, which is stable across pickled copies (each
# task ships its own copy of the snapshot), so a worker replays a given snapshot
# at most once even when it has no evaluator cache to gate on.
_REPLAYED_SNAPSHOT_KEYS: set = set()


def replay_registry_snapshot_once(
    snapshot: Optional[List[Tuple[str, "EnvSpec"]]],
) -> None:
    """Replay ``snapshot`` at most once per worker process.

    Worker payloads carry a :func:`get_registry_snapshot` result so
    Dask workers (which have no per-worker ``initializer`` hook) can
    re-register custom environments before building any env. This wrapper
    makes that cheap to call on every task: the first call for a given set of
    env names replays the snapshot; later calls are no-ops.

    ``replay_registry_snapshot`` is itself idempotent (``overwrite=True``), so
    the guard is a performance optimisation rather than a correctness
    requirement. A ``None`` or empty snapshot is ignored.

    Args:
        snapshot: A :func:`get_registry_snapshot` result, or ``None`` when the
            caller ran serially / did not populate one.
    """
    if not snapshot:
        return
    key = tuple(sorted(name for name, _ in snapshot))
    if key in _REPLAYED_SNAPSHOT_KEYS:
        return
    replay_registry_snapshot(snapshot)
    _REPLAYED_SNAPSHOT_KEYS.add(key)


__all__ = [
    "EnvSpec",
    "build_env_from_spec",
    "EnvApi",
    "DEFAULT_EPISODE_LEN_FALLBACK",
    "register_env",
    "unregister_env",
    "get_env_spec",
    "is_registered",
    "list_envs",
    "build_env",
    "clear_registry",
    "get_registry_snapshot",
    "replay_registry_snapshot",
    "replay_registry_snapshot_once",
]
