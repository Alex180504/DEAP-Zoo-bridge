"""Environment wrappers and utilities for PettingZoo.

The :mod:`.registry` module is the single source of truth for env names;
:mod:`.builtin` auto-registers MPE2 envs and ``aquarium_v0`` on import.
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
