"""Aquarium pursuit-evasion helpers (APF prey policy + scripted-opponent wrapper).

Used by ``examples/pursuit_evasion/aquarium_ea_cma_mappo.py``.
"""

from .apf_policy import (
    AquariumAPFConfig,
    AquariumAPFEvaderPolicy,
    aquarium_obs_layout,
)
from .qdax_position_extractor import make_aquarium_extractor
from .scripted_opponent_env import (
    ScriptedOpponentEnvWrapper,
    register_scripted_aquarium,
)

__all__ = [
    "AquariumAPFConfig",
    "AquariumAPFEvaderPolicy",
    "aquarium_obs_layout",
    "make_aquarium_extractor",
    "ScriptedOpponentEnvWrapper",
    "register_scripted_aquarium",
]
