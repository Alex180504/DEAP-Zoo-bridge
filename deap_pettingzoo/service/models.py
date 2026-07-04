"""Pydantic models for the FastAPI service.

These models define the request/response JSON contracts for every endpoint.
Validation is handled by Pydantic at the HTTP boundary; deeper semantic
validation (catalog compatibility, budget-mode support by algorithm) is
delegated to :func:`~deap_pettingzoo.baselines.experiment_spec.validate_spec`
as a second layer (defense in depth).

Design notes:
- The API-layer :class:`BudgetSpec`, :class:`FitnessSettings` etc. mirror the
  dataclasses in :mod:`deap_pettingzoo.baselines.experiment_spec` but are
  **pydantic** so they participate in FastAPI's automatic OpenAPI schema
  generation and request-body parsing.
- Conversion to the dataclass form happens in the endpoint handlers via
  :meth:`ExperimentSpecModel.to_dataclass`.
- The ``model_`` prefix on the Pydantic variants avoids shadowing the
  dataclass names in the service's namespace.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:  # pragma: no cover
    raise ImportError(
        "The service package requires pydantic>=2. "
        "Install with: pip install deap-pettingzoo[service]"
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Spec sub-models (API layer — mirror experiment_spec dataclasses)
# ---------------------------------------------------------------------------


class BudgetSpecModel(BaseModel):
    mode: str = "env_steps"
    value: int = Field(default=50_000, gt=0)
    num_seeds: int = Field(default=1, ge=1)
    budget_on_consumed: bool = False

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import BudgetSpec

        return BudgetSpec(
            mode=self.mode,
            value=self.value,
            num_seeds=self.num_seeds,
            budget_on_consumed=self.budget_on_consumed,
        )


class FitnessSettingsModel(BaseModel):
    aggregation_method: str = "sum"
    team_mapping: Optional[Dict[str, str]] = None

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import FitnessSettings

        return FitnessSettings(
            aggregation_method=self.aggregation_method,
            team_mapping=self.team_mapping,
        )


class RecordingSettingsModel(BaseModel):
    enabled: bool = False
    fps: int = Field(default=20, gt=0)
    fmt: str = "mp4"
    mode: str = "count"
    every_n_steps: Optional[int] = Field(default=None, ge=1)
    max_recordings: Optional[int] = Field(default=None, ge=1)
    max_dim: Optional[int] = None

    @field_validator("fmt")
    @classmethod
    def _check_fmt(cls, v):
        if v not in ("mp4", "gif"):
            raise ValueError("fmt must be 'mp4' or 'gif'")
        return v

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v):
        valid = {"count", "every", "interval", "final"}
        if v not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")
        return v

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import RecordingSettings

        return RecordingSettings(
            enabled=self.enabled,
            fps=self.fps,
            fmt=self.fmt,
            mode=self.mode,
            every_n_steps=self.every_n_steps,
            max_recordings=self.max_recordings,
            max_dim=self.max_dim,
        )


class ParallelismSettingsModel(BaseModel):
    num_workers: int = Field(default=1, ge=0)
    parallel_backend: str = "process"

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import ParallelismSettings

        return ParallelismSettings(
            num_workers=self.num_workers,
            parallel_backend=self.parallel_backend,
        )


class CoevolutionSideSpecModel(BaseModel):
    name: str
    algorithm: str
    role_prefixes: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import CoevolutionSideSpec

        return CoevolutionSideSpec(
            name=self.name,
            algorithm=self.algorithm,
            role_prefixes=tuple(self.role_prefixes),
            hyperparameters=self.hyperparameters,
        )


class CoevolutionSettingsModel(BaseModel):
    enabled: bool = False
    mode: str = "competitive_2pop"
    schedule: str = "synchronous"
    k_opponents: int = Field(default=3, ge=1)
    aggregation: str = "mean"
    alternating_generations: int = Field(default=1, ge=1)
    reporting_side: Optional[str] = None
    sides: List[CoevolutionSideSpecModel] = Field(default_factory=list)

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import CoevolutionSettings

        return CoevolutionSettings(
            enabled=self.enabled,
            mode=self.mode,
            schedule=self.schedule,
            k_opponents=self.k_opponents,
            aggregation=self.aggregation,
            alternating_generations=self.alternating_generations,
            reporting_side=self.reporting_side,
            sides=[s.to_dataclass() for s in self.sides],
        )


# ---------------------------------------------------------------------------
# Top-level request model
# ---------------------------------------------------------------------------


class ExperimentSpecModel(BaseModel):
    """The JSON payload accepted at ``POST /experiments`` and
    ``POST /experiments/validate``."""

    environment: str
    algorithms: List[str] = Field(default_factory=list)
    env_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    budget: BudgetSpecModel = Field(default_factory=BudgetSpecModel)
    fitness: FitnessSettingsModel = Field(default_factory=FitnessSettingsModel)
    recording: RecordingSettingsModel = Field(default_factory=RecordingSettingsModel)
    parallelism: ParallelismSettingsModel = Field(
        default_factory=ParallelismSettingsModel
    )
    coevolution: CoevolutionSettingsModel = Field(
        default_factory=CoevolutionSettingsModel
    )
    raw_config: Dict[str, Any] = Field(default_factory=dict)
    label: str = ""

    @field_validator("label")
    @classmethod
    def _sanitise_label(cls, v):
        return re.sub(r"[^A-Za-z0-9 _\-,.:]", "_", str(v))[:200]

    def to_dataclass(self):
        from deap_pettingzoo.baselines.experiment_spec import (
            ExperimentSpec,
        )

        return ExperimentSpec(
            environment=self.environment,
            algorithms=list(self.algorithms),
            env_config=dict(self.env_config),
            hyperparameters={k: dict(v) for k, v in self.hyperparameters.items()},
            budget=self.budget.to_dataclass(),
            fitness=self.fitness.to_dataclass(),
            recording=self.recording.to_dataclass(),
            parallelism=self.parallelism.to_dataclass(),
            coevolution=self.coevolution.to_dataclass(),
            raw_config=dict(self.raw_config),
            label=self.label,
        )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ValidationIssueModel(BaseModel):
    path: str
    code: str
    message: str
    severity: str = "error"


class ValidationReportModel(BaseModel):
    ok: bool
    issues: List[ValidationIssueModel] = Field(default_factory=list)


class SubmitResponse(BaseModel):
    id: str
    status: str
    position: int


class ProgressSnapshot(BaseModel):
    unit: str = "generation"
    current: int = 0
    total: int = 0
    fraction: float = 0.0
    best_fitness: Optional[float] = None
    mean_fitness: Optional[float] = None
    wall_time_s: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    id: str
    label: str
    status: str
    spec: ExperimentSpecModel
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_dirs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    position: int = -1
    progress: Optional[ProgressSnapshot] = None
    origin: str = "ui"


class QueueState(BaseModel):
    paused: bool
    running_id: Optional[str] = None
    queued_ids: List[str] = Field(default_factory=list)
    worker_limit: int = 1


class StdoutChunk(BaseModel):
    text: str
    next_offset: int
    eof: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    catalog_version: str = "1"
    advanced_enabled: bool = True
    """Whether the advanced (install panel + custom-script) surface is enabled.
    Driven by ``DEAP_ENABLE_ADVANCED``; the UI reads this to show/hide the
    Setup → Advanced section."""


# ---------------------------------------------------------------------------
# Advanced surface — dependency install + custom-script submission
# ---------------------------------------------------------------------------


class InstallRequest(BaseModel):
    """Request body for ``POST /env/install``.

    Only a strict allowlist of operations is accepted (validated server-side in
    :mod:`deap_pettingzoo.service.install_jobs`): ``pip install``,
    ``pip uninstall``, ``git clone``. Commands are spawned with an explicit
    argument list (never a shell), and args containing shell metacharacters are
    rejected.
    """

    manager: str = "pip"  # one of: pip, git
    action: str = "install"  # pip -> install or uninstall; git -> clone
    args: List[str] = Field(default_factory=list)


class InstallStartResponse(BaseModel):
    job_id: str
    status: str = "running"


class InstallStatusResponse(BaseModel):
    job_id: str
    status: str  # running | completed | failed
    returncode: Optional[int] = None
    command: str = ""
    text: str = ""  # log tail
    next_offset: int = 0
    eof: bool = False


class ScriptSubmitModel(BaseModel):
    """Request body for ``POST /experiments/script`` and ``.../validate``.

    A full hand-written experiment script (e.g. an ``examples/pursuit_evasion``
    module) is submitted as text; the dispatcher runs it as ``python
    user_script.py`` in the experiment working directory. See the dispatcher's
    ``kind == "script"`` branch.
    """

    script: str
    label: str = ""
    entry: str = "script"  # "script" (run file) — reserved: "module"

    @field_validator("label")
    @classmethod
    def _sanitise_label(cls, v):
        return re.sub(r"[^A-Za-z0-9 _\-,.:]", "_", str(v))[:200]
