"""Opt-in JaxMARL bridge for deap-bridge.

`JaxMARL <https://github.com/FLAIROx/JaxMARL>`_ is a JAX-native
multi-agent RL env suite. Its envs run end-to-end on GPU/TPU under
``jit``/``vmap`` and are the intended substrate for QDax scoring
functions (see :mod:`deap_pettingzoo.contrib.qdax`).

This module is opt-in: nothing is registered on import. Call
:func:`register_jaxmarl_env` explicitly to add an env to the registry
under ``api='jax_parallel'`` (which routes through
:class:`~deap_pettingzoo.environments.jax_parallel.JaxEnvToPettingZooAdapter`
to expose a numpy-facing parallel env for non-JAX trainers).

Install the extra with::

    pip install 'deap-pettingzoo[jaxmarl]'
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from deap_pettingzoo.environments.registry import register_env


_INSTALL_HINT = (
    "JaxMARL integration requires the 'jaxmarl' optional extra. "
    "Install with: pip install 'deap-pettingzoo[jaxmarl]' "
    "(or: pip install jaxmarl)."
)


def _load_jaxmarl_make():
    try:
        from jaxmarl import make  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(_INSTALL_HINT) from exc
    return make


class JaxmarlEnvFactory:
    """Module-level, picklable factory for a JaxMARL env.

    Captures the JaxMARL ``env_id`` and its constructor kwargs as plain
    attributes (no closure), so an :class:`~deap_pettingzoo.environments.
    registry.EnvSpec` built from it survives
    :func:`~deap_pettingzoo.environments.registry.get_registry_snapshot`
    and reaches ``num_workers>1`` workers. ``jaxmarl.make`` is (re)imported
    lazily in :meth:`__call__` so the factory pickles even where the caller
    resolved it once in the main process.
    """

    def __init__(self, env_id: str, jaxmarl_kwargs: Optional[dict] = None) -> None:
        self.env_id = env_id
        self.jaxmarl_kwargs = dict(jaxmarl_kwargs or {})

    def __call__(self, **cfg: Any):
        make = _load_jaxmarl_make()
        merged = {**self.jaxmarl_kwargs, **cfg}
        return make(self.env_id, **merged)


def register_jaxmarl_env(
    name: str,
    *,
    env_id: str,
    default_episode_len: int = 25,
    continuous_default: bool = False,
    notes: str = "",
    tags: Optional[set] = None,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    overwrite: bool = False,
    **jaxmarl_kwargs: Any,
):
    """Register a JaxMARL env under the deap-bridge env registry.

    Args:
        name: Canonical registry name (e.g. ``"jaxmarl_simple_tag_v3"``).
        env_id: JaxMARL env identifier passed to ``jaxmarl.make`` (e.g.
            ``"MPE_simple_tag_v3"``).
        default_episode_len: Episode-length hint (typically the env's
            ``max_steps``) for budget translation.
        continuous_default: Whether the env defaults to continuous
            actions (informational; JaxMARL MPE envs are discrete).
        notes: Free-form compatibility notes.
        tags: Additional tags; ``{"jaxmarl", "jax", "parallel"}`` are
            always added.
        wrappers: Optional ordered sequence of PettingZoo-style
            wrappers applied after the JAX->numpy adapter.
        overwrite: Forward to :func:`register_env`.
        **jaxmarl_kwargs: Forwarded to ``jaxmarl.make(env_id, ...)``.

    Raises:
        ImportError: If ``jaxmarl`` / ``jax`` are not installed.

    Example:
        >>> from deap_pettingzoo.contrib.jaxmarl import register_jaxmarl_env
        >>> register_jaxmarl_env(
        ...     "jaxmarl_simple_tag_v3",
        ...     env_id="MPE_simple_tag_v3",
        ...     default_episode_len=25,
        ... )
    """
    # Fail if the extra is missing; the factory re-imports lazily.
    _load_jaxmarl_make()
    _factory = JaxmarlEnvFactory(env_id, jaxmarl_kwargs)

    merged_tags = set(tags or ()) | {"jaxmarl", "jax", "parallel"}
    suffix = f"JaxMARL env '{env_id}' (numpy-boundary via JaxEnvToPettingZooAdapter)."
    combined_notes = f"{notes} {suffix}".strip() if notes else suffix

    return register_env(
        name,
        _factory,
        default_episode_len=default_episode_len,
        continuous_default=continuous_default,
        notes=combined_notes,
        tags=merged_tags,
        api="jax_parallel",
        wrappers=tuple(wrappers),
        overwrite=overwrite,
    )


__all__ = ["register_jaxmarl_env", "JaxmarlEnvFactory"]
