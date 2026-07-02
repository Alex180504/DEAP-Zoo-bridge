"""OpenSpiel turn-based games via Shimmy.

Optional integration with `Farama Shimmy`_'s
:class:`~shimmy.OpenSpielCompatibilityV0`, which exposes OpenSpiel's
70+ board / card games under the PettingZoo AEC API. The games are
strictly turn-based (``metadata['is_parallelizable'] = False``) and
surface legal-action masks in ``info['action_mask']``, so they slot
straight into :mod:`deap_pettingzoo`'s ``api='aec_masked'`` adapter
and :class:`~deap_pettingzoo.evaluators.SelfPlayEvaluator`.

This module is **opt-in**: nothing is registered on import. Call
:func:`register_openspiel_game` explicitly to add a game to the env
registry. Doing so requires the ``shimmy`` (and transitive
``pyspiel``) optional extra — install with::

    pip install deap-pettingzoo[shimmy]

Small-action-space games (``tic_tac_toe``, ``connect_four``,
``kuhn_poker``, ``matrix_rps``, ``matrix_pd``) are a good fit for
:class:`FeedforwardNeuralPolicy` + EA / CMA-ES. Large-action games
like ``chess`` or ``go`` produce huge output layers and are better
driven by the AgileRL trainers (IPPO).

.. _Farama Shimmy: https://shimmy.farama.org/
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from deap_pettingzoo.environments.registry import register_env

_INSTALL_HINT = (
    "OpenSpiel integration requires the 'shimmy' optional extra. "
    "Install with: pip install 'deap-pettingzoo[shimmy]' "
    "(or: pip install 'shimmy[openspiel]')."
)


def _load_compat_cls():
    """Import :class:`shimmy.OpenSpielCompatibilityV0` or raise cleanly."""
    try:
        from shimmy import OpenSpielCompatibilityV0  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(_INSTALL_HINT) from exc
    return OpenSpielCompatibilityV0


class OpenSpielGameFactory:
    """Module-level, picklable factory for an OpenSpiel game.

    Captures the OpenSpiel ``game_name`` as a plain attribute (no closure),
    so an :class:`~deap_pettingzoo.environments.registry.EnvSpec` built from
    it survives
    :func:`~deap_pettingzoo.environments.registry.get_registry_snapshot` and
    reaches ``num_workers>1`` workers.
    :class:`shimmy.OpenSpielCompatibilityV0` is imported lazily in
    :meth:`__call__`.
    """

    def __init__(self, game_name: str) -> None:
        self.game_name = game_name

    def __call__(self, render_mode: Optional[str] = None, **cfg: Any):
        compat_cls = _load_compat_cls()
        # Merge any caller-supplied overrides on top of the fixed game_name.
        cfg.setdefault("game_name", self.game_name)
        return compat_cls(render_mode=render_mode, **cfg)


def register_openspiel_game(
    name: str,
    *,
    game_name: str,
    api: str = "aec_masked",
    default_episode_len: int = 50,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    overwrite: bool = False,
    notes: str = "",
    tags: Optional[set] = None,
    **extra_reg_kwargs: Any,
):
    """Register an OpenSpiel game under the deap-bridge env registry.

    Args:
        name: Canonical registry name (e.g. ``"openspiel_connect_four"``).
        game_name: OpenSpiel short name passed to
            ``OpenSpielCompatibilityV0`` (e.g. ``"tic_tac_toe"``,
            ``"connect_four"``, ``"kuhn_poker"``).
        api: One of ``"aec_masked"`` (default, EA-friendly parallel
            facade) or ``"aec_sequential"`` (single-agent-per-step).
            The ``"parallel"`` / ``"aec"`` / ``"auto"`` modes are not
            supported: OpenSpiel games are strictly turn-based.
        default_episode_len: Episode-length hint used for budget
            translation. Tune to the specific game; 50 is a sensible
            default for small classic games.
        wrappers: Optional ordered sequence of PettingZoo / SuperSuit
            wrappers applied after the turn-based adapter. Passed
            through to :func:`register_env`.
        overwrite: Forward to :func:`register_env`.
        notes: Free-form compatibility notes (merged with a Shimmy
            attribution suffix).
        tags: Additional tags; ``{"shimmy", "openspiel", "turn_based"}``
            are always added.
        **extra_reg_kwargs: Forwarded to :func:`register_env`.

    Raises:
        ImportError: If ``shimmy`` (or its ``openspiel`` extra) is not
            installed.
        ValueError: If ``api`` is not one of the two turn-based modes.

    Example:
        >>> from deap_pettingzoo.contrib.openspiel import (
        ...     register_openspiel_game,
        ... )
        >>> register_openspiel_game(
        ...     "openspiel_connect_four",
        ...     game_name="connect_four",
        ...     default_episode_len=50,
        ... )
    """
    if api not in ("aec_masked", "aec_sequential"):
        raise ValueError(
            f"register_openspiel_game supports only api='aec_masked' or "
            f"'aec_sequential' (OpenSpiel games are strictly turn-based); "
            f"got api={api!r}."
        )

    # Fail fast if the extra is missing; the factory re-imports lazily.
    _load_compat_cls()
    _factory = OpenSpielGameFactory(game_name)

    merged_tags = set(tags or ()) | {"shimmy", "openspiel", "turn_based"}
    combined_notes = notes
    suffix = f"OpenSpiel game '{game_name}' via Shimmy."
    if combined_notes:
        combined_notes = f"{combined_notes} {suffix}"
    else:
        combined_notes = suffix

    return register_env(
        name,
        _factory,
        default_episode_len=default_episode_len,
        continuous_default=False,
        notes=combined_notes,
        tags=merged_tags,
        api=api,
        wrappers=tuple(wrappers),
        overwrite=overwrite,
        **extra_reg_kwargs,
    )


__all__ = ["register_openspiel_game", "OpenSpielGameFactory"]
