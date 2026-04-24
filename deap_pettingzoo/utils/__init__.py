"""Utility functions for logging, visualization, and analysis.

TODO: Implement utility functions for experiment support.
Planned utilities:
- Logging/experiment tracking
- Fitness visualization
- Evolution statistics
- Performance profiling
"""

from deap_pettingzoo.utils.parallel import (
    SUPPORTED_BACKENDS,
    make_executor,
    parallel_map,
    resolve_num_workers,
)

__all__ = [
    "SUPPORTED_BACKENDS",
    "make_executor",
    "parallel_map",
    "resolve_num_workers",
]
