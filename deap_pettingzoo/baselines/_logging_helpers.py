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
    gen_offset: int = 0,
    nominal_env_steps_per_iteration: Optional[int] = None,
) -> None:
    """Write a generation-keyed history list to the ``metrics`` table.

    Each entry of ``fitness_history`` becomes one row of
    ``<run_dir>/metrics.csv`` with the canonical schema
    ``(iteration, env_steps, nominal_env_steps, wall_time, ...)``.
    Existing entry keys other than ``iteration_key`` and ``wall_time``
    flow through unchanged.

    **Real vs nominal env steps:**

    * If a fitness-history entry carries an ``"env_steps"`` key, that value
      (the real consumed steps) is used as the ``env_steps`` column.
      Otherwise ``env_steps`` falls back to the nominal formula
      ``(gen + gen_offset) * env_steps_per_iteration``.
    * ``nominal_env_steps`` is computed as a **cumulative** ceiling:
      ``max(1, gen + 1 + gen_offset) * nominal_env_steps_per_iteration``
      when the ``nominal_env_steps_per_iteration`` kwarg is supplied;
      otherwise it equals ``env_steps``.  The ``+1`` term charges the
      initial-population evaluation regardless of indexing scheme.

    The ``gen_offset`` parameter aligns the x-axis across indexing schemes:

    * EA / NEAT are 0-indexed (gen 0 = initial population) — use
      ``gen_offset=0`` (default) so gen 0 → env_steps 0, nominal = 1×N.
    * CMA-ES / PSO are 1-indexed (gen 1 = first update) — use
      ``gen_offset=-1`` so gen 1 → env_steps 0, nominal = 1×N.

    No-op when ``logger`` is a :class:`NoOpLogger`.
    """
    if isinstance(logger, NoOpLogger):
        return
    extras = dict(extra_columns_per_row or {})
    for entry in fitness_history:
        gen = int(entry.get(iteration_key, 0))
        # Real consumed steps: prefer the entry's own value, fall back to nominal.
        if "env_steps" in entry:
            real_env_steps = int(entry["env_steps"])
        else:
            real_env_steps = max(0, (gen + gen_offset)) * int(env_steps_per_iteration)
        # Nominal ceiling: cumulative — gen + 1 charges the initial population
        # regardless of indexing scheme, gen_offset aligns the origin.
        if nominal_env_steps_per_iteration is not None:
            nominal = max(1, gen + 1 + gen_offset) * int(
                nominal_env_steps_per_iteration
            )
        else:
            nominal = real_env_steps
        row = {
            "iteration": gen,
            "env_steps": real_env_steps,
            "nominal_env_steps": nominal,
            "wall_time": float(entry.get("wall_time", 0.0)),
        }
        for k, v in entry.items():
            if k in (iteration_key, "env_steps", "wall_time"):
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
    # Override the manifest's wall_time_s to reflect *training* duration rather than the time spent inside the logging context manager.
    if training_time_s is not None and not isinstance(logger, NoOpLogger):
        try:
            logger.manifest.wall_time_s = float(training_time_s)
            logger._t0 = None  # prevent close() from overwriting the value
        except AttributeError:  # pragma: no cover - defensive
            pass
    logger.summary(payload)
