"""Reconstruct a runnable policy from a logged run.

The UI's *Live Rollout* page calls
:func:`load_policy` to turn a :class:`~deap_pettingzoo.analysis.loaders.RunRecord`
back into a :class:`~deap_pettingzoo.policies.base.PolicyProtocol` it can
roll out against an env factory.

The loader is read-only (no env construction) so the UI can
preview metadata before deciding whether to spin up a rollout env.
"""

# TODO: Finish the live rollout functionality
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from deap_pettingzoo.analysis.loaders import RunRecord


class PolicyReloadUnavailable(RuntimeError):
    """Raised when a checkpoint kind is known but its loader is not yet
    implemented (or its optional dependency is missing)."""


def _resolve_artifact_path(run_dir: Path, raw_path: str) -> Path:
    """Resolve a manifest-stored path against ``run_dir``.

    The save side stores paths relative to the run dir when the file
    lives inside it, otherwise absolute. Match that contract here.
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (run_dir / p).resolve()


def _read_manifest_entry(run_record: "RunRecord") -> Dict[str, Any]:
    artifacts = run_record.manifest.get("artifacts") or {}
    entry = artifacts.get("policy")
    if entry is None:
        raise FileNotFoundError(
            f"Run {run_record.run_dir} has no policy artefact registered "
            "in manifest.artifacts['policy']."
        )
    if not isinstance(entry, Mapping):
        raise TypeError(
            "manifest.artifacts['policy'] must be a mapping; "
            f"got {type(entry).__name__}"
        )
    return dict(entry)


@dataclass
class CoevolutionPolicyBundle:
    """Container for the per-side payloads in a ``coevolution_v1`` policy
    artefact. The UI uses :meth:`get_side` to pick which side it wants
    to roll out (typically ``manifest.artifacts['policy']['reporting_side']``).
    """

    reporting_side: Optional[str]
    sides: Dict[str, Dict[str, Any]]
    env_name: str

    def side_names(self) -> list[str]:
        return sorted(self.sides.keys())

    def get_side(self, name: str, *, action_space: Any = None) -> Any:
        """Reconstruct the named side as a :class:`GenomePolicyWrapper`."""
        if name not in self.sides:
            raise KeyError(
                f"Side {name!r} not in coevolution policy bundle; "
                f"available: {self.side_names()}"
            )
        bundle = self.sides[name]
        return _build_genome_policy(
            bundle["genome"], env_name=self.env_name, action_space=action_space
        )


def _build_genome_policy(
    genome_dict: Mapping[str, Any],
    *,
    env_name: str,
    action_space: Any = None,
) -> Any:
    """Rehydrate a ``FixedTopologyGenome`` and wrap it in a policy."""
    # Imported lazily so analysis-only consumers don't pay the cost of
    # importing the trainer module just to read manifests.
    from deap_pettingzoo.baselines.ea_trainer import GenomePolicyWrapper
    from deap_pettingzoo.genomes.fixed_topology import FixedTopologyGenome

    genome = FixedTopologyGenome.from_dict(dict(genome_dict))
    return GenomePolicyWrapper(genome, env_name, action_space=action_space)


def load_policy(run_record: "RunRecord", *, action_space: Any = None) -> Any:
    """Rehydrate the policy artefact for ``run_record``.

    Parameters
    ----------
    run_record:
        A :class:`~deap_pettingzoo.analysis.loaders.RunRecord` produced
        by :func:`deap_pettingzoo.analysis.loaders.load_run`.
    action_space:
        Optional env action space, forwarded to
        :class:`GenomePolicyWrapper`. Required for environments where
        raw network outputs need to be decoded (Box / Discrete).

    Returns
    -------
    Any
        For ``ea_genome_v1`` / ``cma_genome_v1`` / ``pso_genome_v1``: a
        :class:`GenomePolicyWrapper`.
        For ``coevolution_v1``: a :class:`CoevolutionPolicyBundle`
        (call :meth:`CoevolutionPolicyBundle.get_side` to pick a side).

    Raises
    ------
    FileNotFoundError
        If the run has no policy artefact registered, or the file is
        missing on disk.
    PolicyReloadUnavailable
        If the kind is recognised but its loader is not yet implemented
        (currently ``agilerl_v1`` and ``qdax_repertoire_v1``).
    ValueError
        If ``manifest.artifacts['policy']['kind']`` is unknown.
    """
    entry = _read_manifest_entry(run_record)
    kind = entry.get("kind")
    raw_path = entry.get("path")
    if not isinstance(kind, str) or not isinstance(raw_path, str):
        raise ValueError(
            f"Malformed policy artefact entry in {run_record.run_dir}: {entry!r}"
        )
    abs_path = _resolve_artifact_path(Path(run_record.run_dir), raw_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Policy artefact for run {run_record.run_dir} not found at "
            f"{abs_path} (manifest references {raw_path!r})."
        )

    env_name = run_record.env_name

    if kind in {"ea_genome_v1", "cma_genome_v1", "pso_genome_v1"}:
        # PSO shares the FixedTopologyGenome representation with EA/CMA, so
        # its checkpoint rehydrates through the same genome-policy builder.
        data = json.loads(abs_path.read_text(encoding="utf-8"))
        return _build_genome_policy(data, env_name=env_name, action_space=action_space)

    if kind == "coevolution_v1":
        data = json.loads(abs_path.read_text(encoding="utf-8"))
        sides = data.get("sides") or {}
        if not isinstance(sides, Mapping):
            raise ValueError(
                f"Malformed coevolution_v1 payload at {abs_path}: 'sides' must be a mapping."
            )
        return CoevolutionPolicyBundle(
            reporting_side=data.get("reporting_side"),
            sides={str(k): dict(v) for k, v in sides.items()},
            env_name=env_name,
        )

    if kind in {"agilerl_v1", "qdax_repertoire_v1"}:
        raise PolicyReloadUnavailable(
            f"load_policy for kind={kind!r} is not implemented yet "
        )

    raise ValueError(
        f"Unknown policy artefact kind {kind!r} for run {run_record.run_dir}; "
        "expected one of: ea_genome_v1, cma_genome_v1, pso_genome_v1, "
        "coevolution_v1, agilerl_v1, qdax_repertoire_v1."
    )


__all__ = [
    "CoevolutionPolicyBundle",
    "PolicyReloadUnavailable",
    "load_policy",
]
