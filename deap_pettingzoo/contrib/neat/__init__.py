"""NEAT integration via ``neat-python``.

Provides a :class:`NeatPolicy` that satisfies
:class:`~deap_pettingzoo.baselines.base.PolicyProtocol`, a
:class:`~deap_pettingzoo.baselines.base.BaselineTrainer`-conforming
:class:`NEATTrainer`, and a :class:`NeatSideState` for co-evolution.

All public objects are available as direct imports from this package,
but they raise a helpful :exc:`ImportError` unless ``neat-python`` is
installed (the ``neat`` extra of ``deap-pettingzoo``).

Quick start::

    from deap_pettingzoo.contrib.neat import NEATTrainer
    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig

    trainer = NEATTrainer()
    result = trainer.train(TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=50_000,
        algorithm_params={"pop_size": 50},
    ))
"""

from __future__ import annotations

__all__ = [
    "NEATTrainer",
    "NeatPolicy",
    "NeatSideState",
    "HyperNEATTrainer",
    "ESHyperNEATTrainer",
    "HyperNeatPolicy",
    "ESHyperNeatPolicy",
    "HyperNeatSideState",
    "ESHyperNeatSideState",
    "build_neat_config",
    "build_cppn_config",
    "Substrate",
    "is_available",
    "_require_neat",
]


def is_available() -> bool:
    """Return ``True`` if ``neat`` is importable."""
    try:
        import neat  # noqa: F401

        return True
    except ImportError:
        return False


def _require_neat() -> None:
    """Raise :exc:`ImportError` with a helpful message if ``neat`` is missing."""
    if not is_available():
        raise ImportError(
            "The 'neat' extra is required for NEAT support. "
            "Install it with:  pip install deap-pettingzoo[neat]"
        )


# PEP 562 lazy-access shim — imports are deferred until first attribute
# access so that ``import deap_pettingzoo.contrib.neat`` does not fail
# when neat-python is absent.
def __getattr__(name: str):
    if name == "build_neat_config":
        _require_neat()
        from .config import build_neat_config as _fn

        return _fn
    if name == "build_cppn_config":
        _require_neat()
        from .config import build_cppn_config as _fn

        return _fn
    if name == "Substrate":
        _require_neat()
        from ._pureples.substrate import Substrate as _cls

        return _cls
    if name == "NeatPolicy":
        _require_neat()
        from .policy import NeatPolicy as _cls

        return _cls
    if name == "HyperNeatPolicy":
        _require_neat()
        from .hyperneat import HyperNeatPolicy as _cls

        return _cls
    if name == "ESHyperNeatPolicy":
        _require_neat()
        from .hyperneat import ESHyperNeatPolicy as _cls

        return _cls
    if name == "NEATTrainer":
        _require_neat()
        from .trainer import NEATTrainer as _cls

        return _cls
    if name == "HyperNEATTrainer":
        _require_neat()
        from .hyperneat_trainer import HyperNEATTrainer as _cls

        return _cls
    if name == "ESHyperNEATTrainer":
        _require_neat()
        from .hyperneat_trainer import ESHyperNEATTrainer as _cls

        return _cls
    if name == "NeatSideState":
        _require_neat()
        from .coevolution import NeatSideState as _cls

        return _cls
    if name == "HyperNeatSideState":
        _require_neat()
        from .hyperneat_coevolution import HyperNeatSideState as _cls

        return _cls
    if name == "ESHyperNeatSideState":
        _require_neat()
        from .hyperneat_coevolution import ESHyperNeatSideState as _cls

        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
