"""
DEAP-PettingZoo Bridge Library

A bridge library enabling evolutionary algorithms from DEAP to work with 
multi-agent reinforcement learning environments from PettingZoo.
"""

__version__ = "0.1.0"
__author__ = "Alex"

from deap_pettingzoo import genomes, policies, evaluators, environments, utils

__all__ = [
    "genomes",
    "policies",
    "evaluators",
    "environments",
    "utils",
]
