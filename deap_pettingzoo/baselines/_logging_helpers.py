"""Shared helpers for wiring trainers to :class:`ExperimentLogger`.

Keeps trainer-side integration minimal: each trainer calls
:func:`build_logger` near the top of ``train()`` and
:func:`dump_fitness_history` (or the equivalent loop-internal calls)
near the end. The helpers gracefully degrade when the logger is a
:class:`~deap_pettingzoo.utils.logging.NoOpLogger`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from deap_pettingzoo.utils.logging import (
    ExperimentLogger,
    NoOpLogger,
)


def build_logger(
    config: Any,
    *,
    algorithm: str,
    notes: str = "",
) -> ExperimentLogger:
    """Construct an :class:`ExperimentLogger` from a ``TrainingConfig``-shaped object.

    Returns a :class:`NoOpLogger` when ``config.output_dir is None``,
    so call sites can treat the result uniformly.
    """
    return ExperimentLogger.from_config(config, algorithm=algorithm, notes=notes)


def dump_fitness_history(
    logger: ExperimentLogger,
    fitness_history: Iterable[Mapping[str, Any]],
    *,
    env_steps_per_iteration: int,
    iteration_key: str = "generation",
    extra_columns_per_row: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a generation-keyed history list to the ``metrics`` table.

    Each entry of ``fitness_history`` becomes one row of
    ``<run_dir>/metrics.csv`` with the canonical schema
    ``(iteration, env_steps, wall_time, ...)``. Existing entry keys
    other than ``iteration_key`` flow through unchanged.

    No-op when ``logger`` is a :class:`NoOpLogger`.
    """
    if isinstance(logger, NoOpLogger):
        return
    extras = dict(extra_columns_per_row or {})
    for entry in fitness_history:
        gen = int(entry.get(iteration_key, 0))
        row = {
            "iteration": gen,
            "env_steps": gen * int(env_steps_per_iteration),
            "wall_time": float(entry.get("wall_time", 0.0)),
        }
        for k, v in entry.items():
            if k == iteration_key or k == "wall_time":
                continue
            row[k] = v
        row.update(extras)
        logger.row("metrics", row)


def write_summary(
    logger: ExperimentLogger,
    summary: Mapping[str, Any],
    *,
    training_time_s: Optional[float] = None,
) -> None:
    """Persist the final summary block.

    The ``training_time_s`` argument is folded into the summary if
    provided so the on-disk record matches ``TrainingResult.training_time``.
    """
    payload = dict(summary)
    if training_time_s is not None and "training_time_s" not in payload:
        payload["training_time_s"] = float(training_time_s)
    # Override the manifest's wall_time_s to reflect *training* duration
    # rather than the time spent inside the logging context manager.
    if training_time_s is not None and not isinstance(logger, NoOpLogger):
        try:
            logger.manifest.wall_time_s = float(training_time_s)
            logger._t0 = None  # prevent close() from overwriting the value
        except AttributeError:  # pragma: no cover - defensive
            pass
    logger.summary(payload)
