"""Utility functions for logging, parallel execution, and analysis support.

Subpackages:

- :mod:`deap_pettingzoo.utils.parallel` — process/SCOOP/Dask backends
  for trainer fitness evaluation (already established).
- :mod:`deap_pettingzoo.utils.logging` — opt-in result logging layer
  (manifest + CSV/JSON/Parquet/TensorBoard sinks + replay frame
  recording). The backend remains fully functional with the logging
  subsystem unused; trainers fall through to a :class:`NoOpLogger`
  whenever ``TrainingConfig.output_dir`` is ``None``.
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
