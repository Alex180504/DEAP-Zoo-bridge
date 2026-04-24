"""Environment wrappers and utilities for PettingZoo.

The :mod:`.registry` module is the single source of truth for env names;
:mod:`.builtin` auto-registers MPE2 envs and ``aquarium_v0`` on import.

Legacy ``PursuitEvasionEnv`` / ``PursuitEvasionConfig`` / ``ObsLayout``
symbols were previously re-exported from this package.  They now live in
:mod:`deap_pettingzoo.contrib.pursuit_evasion`; the shim module
``deap_pettingzoo.environments.pursuit_evasion`` forwards them with a
``DeprecationWarning`` for one release cycle.  New code should import
from the contrib subpackage directly.
"""

# Side-effect import: populate the open env registry.
from . import builtin  # noqa: F401

from .pettingzoo_wrapper import PettingZooWrapper
from .registry import (
    EnvSpec,
    build_env,
    get_env_spec,
    is_registered,
    list_envs,
    register_env,
    unregister_env,
)
from .turn_based import MaskedSimultaneousAdapter, SequentialTurnAdapter
from .jax_parallel import JaxEnvProtocol, JaxEnvToPettingZooAdapter

__all__ = [
    'PettingZooWrapper',
    'EnvSpec',
    'register_env',
    'unregister_env',
    'get_env_spec',
    'is_registered',
    'list_envs',
    'build_env',
    'MaskedSimultaneousAdapter',
    'SequentialTurnAdapter',
    'JaxEnvProtocol',
    'JaxEnvToPettingZooAdapter',
]


# PEP 562: lazy attribute forwarding for legacy names (one-release shim).
def __getattr__(name):  # pragma: no cover - trivial forwarder
    if name in ("PursuitEvasionEnv", "PursuitEvasionConfig", "ObsLayout"):
        import warnings
        warnings.warn(
            f"{name} moved to deap_pettingzoo.contrib.pursuit_evasion; "
            "update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        from deap_pettingzoo.contrib.pursuit_evasion.env import (
            PursuitEvasionEnv,
            PursuitEvasionConfig,
            ObsLayout,
        )
        return {
            "PursuitEvasionEnv": PursuitEvasionEnv,
            "PursuitEvasionConfig": PursuitEvasionConfig,
            "ObsLayout": ObsLayout,
        }[name]
    raise AttributeError(f"module 'deap_pettingzoo.environments' has no attribute {name!r}")
