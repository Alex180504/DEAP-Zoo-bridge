"""Utility functions for logging, parallel execution, and analysis tooling.

Subpackages:

- :mod:`deap_pettingzoo.utils.parallel` — process/Dask backends
  for trainer fitness evaluation (already established).
- :mod:`deap_pettingzoo.utils.logging` — opt-in result logging layer
  (manifest + CSV/JSON/Parquet/TensorBoard sinks + replay frame
  recording). The backend remains fully functional with the logging
  subsystem unused; trainers fall through to a :class:`NoOpLogger`
  whenever ``TrainingConfig.output_dir`` is ``None``.
"""

from deap_pettingzoo.utils.parallel import (
    DASK_CONNECT_TIMEOUT_ENV_VAR,
    DASK_SCHEDULER_ENV_VAR,
    SUPPORTED_BACKENDS,
    make_executor,
    parallel_map,
    resolve_num_workers,
    resolve_scheduler_address,
)
from deap_pettingzoo.utils.rng import seed_everything

__all__ = [
    "DASK_CONNECT_TIMEOUT_ENV_VAR",
    "DASK_SCHEDULER_ENV_VAR",
    "SUPPORTED_BACKENDS",
    "make_executor",
    "parallel_map",
    "resolve_num_workers",
    "resolve_scheduler_address",
    "seed_everything",
]
