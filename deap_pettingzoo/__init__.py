"""
DEAP-PettingZoo Bridge Library

A bridge library enabling evolutionary algorithms from DEAP to work with
multi-agent reinforcement learning environments from PettingZoo.
"""

__version__ = "0.1.0"
__author__ = "Alex"

# Windows-only: importing pyarrow *after* torch in the same process hard
# crashes (access violation in pyarrow.pandas_compat.convert_column) the
# first time pandas touches parquet I/O. Importing pyarrow here, before any downstream module gets a
# chance to lazily import torch, fixes the DLL init order for the rest of
# the process. No-op (and harmless) if pyarrow is not installed.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

from deap_pettingzoo import environments, evaluators, genomes, policies, utils
from deap_pettingzoo.catalog import register_algorithm

__all__ = [
    "genomes",
    "policies",
    "evaluators",
    "environments",
    "utils",
    "register_algorithm",
]
