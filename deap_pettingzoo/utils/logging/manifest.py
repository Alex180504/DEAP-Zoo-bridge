"""Run manifest: the always-present JSON header for a run directory."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from deap_pettingzoo.utils.logging.schema import SCHEMA_VERSION


def _safe_git_sha() -> Optional[str]:
    """Return the current git commit SHA, or ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    sha = out.strip()
    return sha or None


def capture_environment_metadata() -> Dict[str, Any]:
    """Return a dict of host/runtime metadata for the run manifest.

    Best-effort: missing tools (e.g. ``git``) degrade silently.
    """
    from importlib.metadata import PackageNotFoundError, version

    pkg_versions: Dict[str, Optional[str]] = {}
    for pkg in (
        "deap-pettingzoo",
        "deap",
        "pettingzoo",
        "mpe2",
        "numpy",
        "gymnasium",
        "agilerl",
        "torch",
        "jax",
        "qdax",
        "rliable",
    ):
        try:
            pkg_versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pkg_versions[pkg] = None

    return {
        "git_sha": _safe_git_sha(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "package_versions": pkg_versions,
    }


@dataclass
class RunManifest:
    """Self-describing header for one training run.

    Persisted as ``<run_dir>/manifest.json``. Stable across schema
    versions: never remove a field; bump :data:`SCHEMA_VERSION` and
    add a migration in
    :func:`deap_pettingzoo.analysis.loaders.load_run` instead.

    Attributes:
        run_id: Unique identifier; recommended format
            ``{algo}_{env}_{seed}_{ts}_{shortuuid}``.
        algorithm: Algorithm short name (``"ea"``, ``"cma-es"``,
            ``"maddpg"``, ...).
        env_name: Registered PettingZoo env name.
        env_config: Env config dict actually used.
        seed: Master seed for the run, or ``None`` if not seeded.
        budget_mode: ``"env_steps" | "eval_episodes" | "generations"``.
        budget_value: Amount in units specified by ``budget_mode``.
        hyperparams: Trainer-specific algorithm parameters.
        notes: Free-form notes (e.g. experiment label).
        schema_version: Bumped when on-disk layout changes.
        started_at: ISO-8601 UTC timestamp.
        finished_at: ISO-8601 UTC timestamp (set by ``ExperimentLogger.close``).
        wall_time_s: Total wall-clock duration in seconds.
        environment: Output of :func:`capture_environment_metadata`.
        artifacts: Paths (relative to ``run_dir``) of named artefacts.
        final_summary: Final scalar summary (best fitness, eval mean, ...).
    """

    run_id: str
    algorithm: str
    env_name: str
    env_config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    budget_mode: Optional[str] = None
    budget_value: Optional[int] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    wall_time_s: Optional[float] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    final_summary: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + replace.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path) -> "RunManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Tolerate unknown extra keys for forward-compat.
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        kept = {k: v for k, v in data.items() if k in known}
        return cls(**kept)
