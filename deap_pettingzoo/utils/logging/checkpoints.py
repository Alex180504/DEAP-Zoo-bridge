"""Policy checkpoint persistence for the experiment-logging foundation.

Trainers call :func:`save_policy_checkpoint` at the end of ``train()``
to drop a kind-tagged file under ``<run_dir>/checkpoints/`` and register
it in ``manifest.artifacts["policy"]`` as

.. code-block:: python

    {"kind": "ea_genome_v1", "path": "checkpoints/policy.json"}

The matching loader is :func:`deap_pettingzoo.analysis.policies.load_policy`
which dispatches on ``kind`` to rebuild a :class:`PolicyProtocol`-compatible
object from the run directory.

Kept deliberately small:

* No new optional dependency for the EA / CMA paths (JSON-only).
* AgileRL uses its own ``agent.save_checkpoint(path)`` API.
* QDax registers the existing ``repertoire.pkl`` produced by
  :func:`deap_pettingzoo.contrib.qdax.trainer._save_repertoire`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from deap_pettingzoo.utils.logging.experiment_logger import ExperimentLogger


# Kind enum (string-valued so the manifest stays json-serialisable).
KIND_EA_GENOME = "ea_genome_v1"
KIND_CMA_GENOME = "cma_genome_v1"
KIND_AGILERL = "agilerl_v1"
KIND_QDAX_REPERTOIRE = "qdax_repertoire_v1"
KIND_COEVOLUTION = "coevolution_v1"

KNOWN_KINDS = frozenset(
    {
        KIND_EA_GENOME,
        KIND_CMA_GENOME,
        KIND_AGILERL,
        KIND_QDAX_REPERTOIRE,
        KIND_COEVOLUTION,
    }
)


@dataclass(frozen=True)
class PolicyCheckpointEntry:
    """In-memory view of ``manifest.artifacts["policy"]``.

    The on-disk representation is a plain dict to keep the manifest
    self-describing; this dataclass is just a typed accessor used by
    the analysis side.
    """

    kind: str
    path: str
    extra: Mapping[str, Any]

    @classmethod
    def from_manifest_entry(cls, entry: Mapping[str, Any]) -> "PolicyCheckpointEntry":
        if not isinstance(entry, Mapping):
            raise TypeError(
                "manifest.artifacts['policy'] must be a mapping; "
                f"got {type(entry).__name__}"
            )
        kind = entry.get("kind")
        path = entry.get("path")
        if not isinstance(kind, str) or not isinstance(path, str):
            raise ValueError(
                "manifest.artifacts['policy'] must contain string 'kind' "
                f"and 'path' fields; got {dict(entry)!r}"
            )
        extra = {k: v for k, v in entry.items() if k not in {"kind", "path"}}
        return cls(kind=kind, path=path, extra=extra)


def _checkpoints_dir(run_dir: Path) -> Path:
    """Ensure ``<run_dir>/checkpoints/`` exists and return it."""
    target = run_dir / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _register(
    logger: "ExperimentLogger",
    *,
    kind: str,
    path: Path,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Register a policy artefact in ``manifest.artifacts['policy']``.

    Stores the path **relative to the run dir** so the manifest stays
    portable. Returns the absolute path that was registered.
    """
    if kind not in KNOWN_KINDS:
        raise ValueError(
            f"Unknown checkpoint kind {kind!r}; expected one of {sorted(KNOWN_KINDS)}"
        )
    abs_path = Path(path).resolve()
    try:
        rel = abs_path.relative_to(Path(logger.run_dir).resolve())
    except ValueError:
        rel = abs_path  # path lives outside run_dir; keep absolute
    entry: dict[str, Any] = {"kind": kind, "path": str(rel).replace("\\", "/")}
    if extra:
        # Reserved keys win.
        for k, v in extra.items():
            if k in {"kind", "path"}:
                continue
            entry[k] = v
    logger.manifest.artifacts["policy"] = entry
    return abs_path


def save_ea_genome(
    logger: "ExperimentLogger",
    genome: Any,
    *,
    filename: str = "policy.json",
    kind: str = KIND_EA_GENOME,
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Persist a ``FixedTopologyGenome`` (or any genome with ``to_dict``)
    as JSON.

    Returns the absolute path on success, ``None`` for a no-op logger
    (so trainers can call this unconditionally).
    """
    if not _is_real_logger(logger):
        return None
    target = _checkpoints_dir(logger.run_dir) / filename
    payload = genome.to_dict() if hasattr(genome, "to_dict") else dict(genome)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return _register(logger, kind=kind, path=target, extra=extra)


def save_agilerl_agent(
    logger: "ExperimentLogger",
    agent: Any,
    *,
    filename: str = "agent.pt",
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Persist an AgileRL agent via its ``save_checkpoint(path)`` API."""
    if not _is_real_logger(logger):
        return None
    if not hasattr(agent, "save_checkpoint"):
        # Some AgileRL versions / wrappers expose ``save`` instead.
        save_fn = getattr(agent, "save", None)
        if save_fn is None:
            return None
    else:
        save_fn = agent.save_checkpoint
    target = _checkpoints_dir(logger.run_dir) / filename
    save_fn(str(target))
    return _register(logger, kind=KIND_AGILERL, path=target, extra=extra)


def register_existing_artifact(
    logger: "ExperimentLogger",
    *,
    kind: str,
    path: Path,
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Register an artefact written by something other than this module
    (e.g. QDax's ``repertoire.pkl``)."""
    if not _is_real_logger(logger):
        return None
    if not Path(path).is_file():
        return None
    return _register(logger, kind=kind, path=Path(path), extra=extra)


def _is_real_logger(logger: Any) -> bool:
    """True iff ``logger`` writes to disk (i.e. not :class:`NoOpLogger`)."""
    from deap_pettingzoo.utils.logging.experiment_logger import (
        NoOpLogger,
    )  # lazy import to avoid circular dependency

    return logger is not None and not isinstance(logger, NoOpLogger)
