"""On-disk schema constants for the logging subsystem.

Bumping :data:`SCHEMA_VERSION` is required whenever the on-disk layout
or any required column changes. The analysis layer
(:mod:`deap_pettingzoo.analysis.loaders`) reads ``schema_version`` from
``manifest.json`` and refuses to load incompatible runs.
"""

from __future__ import annotations

SCHEMA_VERSION: int = 2

#: Required columns on every row written to ``metrics.csv``.
#: Trainers may add arbitrary extra columns; loaders treat unknown
#: columns as ``object`` dtype.
METRICS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "iteration",  # generation/epoch/iteration index (int, 0-based)
    "env_steps",  # cumulative env steps *actually* consumed so far (int)
    "nominal_env_steps",  # nominal ceiling (gen × pop × max_steps); may equal env_steps
    "wall_time",  # seconds since run start (float)
)

#: Required columns on every row written to ``episodes.csv``.
EPISODES_COLUMNS: tuple[str, ...] = (
    "episode_idx",  # int
    "seed",         # int or empty
    "reward",       # float (team/aggregate reward)
    "length",       # int
    "success",      # 0/1, env-defined; -1 if unknown
)

#: Required columns on every row of ``trajectories.parquet``.
TRAJECTORIES_COLUMNS: tuple[str, ...] = (
    "episode_idx",
    "step",
    "agent",
    "reward",
    "done",
)
