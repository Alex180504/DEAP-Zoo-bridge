"""Aquarium pursuit-evasion helpers.

Shared building blocks for the aquarium experiment suite
(``examples/pursuit_evasion/aquarium_*.py``): scripted APF policies for
both roles, scripted-opponent env wrappers and registration helpers,
observation featurization, EA/CMA action heads, QDax extractors, and the
hyperparameter configuration (:mod:`.experiment_config`).
"""

from .apf_policy import (
    AquariumAPFConfig,
    AquariumAPFEvaderPolicy,
    AquariumAPFPursuerPolicy,
    aquarium_obs_layout,
    continuous_action_for_heading,
)
from .qdax_position_extractor import make_aquarium_extractor
from .qdax_state_extractor import AquariumStateExtractor
from .scripted_opponent_env import (
    AquariumScriptedPredatorWrapper,
    AquariumScriptedPreyWrapper,
    register_scripted_aquarium,
    register_scripted_predator_aquarium,
)

__all__ = [
    "AquariumAPFConfig",
    "AquariumAPFEvaderPolicy",
    "AquariumAPFPursuerPolicy",
    "aquarium_obs_layout",
    "continuous_action_for_heading",
    "make_aquarium_extractor",
    "AquariumStateExtractor",
    "AquariumScriptedPredatorWrapper",
    "AquariumScriptedPreyWrapper",
    "register_scripted_aquarium",
    "register_scripted_predator_aquarium",
]
