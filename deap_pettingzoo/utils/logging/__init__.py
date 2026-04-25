"""Result logging for DEAP-bridge trainers.

This subpackage provides a non-invasive, opt-in result-logging layer.
Trainers accept an :class:`ExperimentLogger` (or build one from
``TrainingConfig.output_dir``) and call :meth:`ExperimentLogger.row`,
:meth:`ExperimentLogger.scalar`, :meth:`ExperimentLogger.artifact`
at the existing places where ``metrics`` dicts are built.

The core design contract: **the backend remains fully usable without
any logging configured**. ``ExperimentLogger.from_config(config)``
returns a no-op logger when ``config.output_dir is None``.

On-disk schema (per run directory):

::

    <run_dir>/
        manifest.json          # RunManifest (always)
        metrics.csv            # per-generation/per-iteration scalars
        episodes.csv           # per-episode evaluation rows
        trajectories.parquet   # opt-in, per-step rollout data
        frames/episode_*.mp4   # opt-in, recorded rgb_array frames
        tb/                    # opt-in, TensorBoard event files
        checkpoints/           # opt-in, policy snapshots

See ``docs/api/api_specification.md`` (section "Logging") for the full
contract.
"""

from deap_pettingzoo.utils.logging.experiment_logger import (
    ExperimentLogger,
    NoOpLogger,
)
from deap_pettingzoo.utils.logging.manifest import (
    RunManifest,
    capture_environment_metadata,
)
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.utils.logging.schema import (
    SCHEMA_VERSION,
    EPISODES_COLUMNS,
    METRICS_REQUIRED_COLUMNS,
)
from deap_pettingzoo.utils.logging.sinks import (
    CsvScalarSink,
    JsonManifestSink,
    MultiSink,
    NoOpSink,
    ParquetTrajectorySink,
    Sink,
    TensorBoardSink,
)

__all__ = [
    "ExperimentLogger",
    "NoOpLogger",
    "RunManifest",
    "capture_environment_metadata",
    "FrameRecorder",
    "Sink",
    "NoOpSink",
    "MultiSink",
    "JsonManifestSink",
    "CsvScalarSink",
    "ParquetTrajectorySink",
    "TensorBoardSink",
    "SCHEMA_VERSION",
    "EPISODES_COLUMNS",
    "METRICS_REQUIRED_COLUMNS",
]
