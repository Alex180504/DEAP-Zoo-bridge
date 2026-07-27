"""Serializable experiment specification + config builder + validator.

This module provides a bridge between the experiment setup UI and the
unified training API. The UI produces an :class:`ExperimentSpec` (pure data,
JSON round-trippable); the subprocess runner calls
:func:`validate_spec` then :func:`build_configs` to turn it into the
``TrainingConfig`` / ``CoEvolutionConfig`` objects the trainers consume.

Three-layer configuration merge
-------------------------------
The experiment configuration resolves in three layers with increasing precedence:

1. **Catalog defaults** — every ``ParamSpec.default`` from the catalog.
2. **Raw config overlay** — a free-form YAML/JSON dict the advanced user
   supplies (``ExperimentSpec.raw_config``), deep-merged on top.
3. **Explicit form values** — only the fields the user actually touched
   (``ExperimentSpec.env_config`` / ``ExperimentSpec.hyperparameters``),
   merged last so a single tweaked slider wins over the raw config while the
   rest of the raw config stays intact.

The contract is therefore: the form submits only explicitly-set values, while the
catalog fills in everything else.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    FitnessConfig,
    RecordingConfig,
    RecordingMode,
    TrainingConfig,
)

# Helpers


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins).

    Nested dicts are merged key-by-key; any non-dict value (or type mismatch)
    in ``overlay`` replaces the corresponding ``base`` value outright. Neither
    input is mutated.
    """
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Spec dataclasses


@dataclass
class BudgetSpec:
    """Budget + seed count for an experiment."""

    mode: str = "env_steps"
    value: int = 50_000
    num_seeds: int = 1
    budget_on_consumed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "value": self.value,
            "num_seeds": self.num_seeds,
            "budget_on_consumed": self.budget_on_consumed,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "BudgetSpec":
        data = data or {}
        return cls(
            mode=str(data.get("mode", "env_steps")),
            value=int(data.get("value", 50_000)),
            num_seeds=int(data.get("num_seeds", 1)),
            budget_on_consumed=bool(data.get("budget_on_consumed", False)),
        )


@dataclass
class FitnessSettings:
    """Fitness aggregation choice (population-based trainers)."""

    aggregation_method: str = "sum"
    team_mapping: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_method": self.aggregation_method,
            "team_mapping": self.team_mapping,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "FitnessSettings":
        data = data or {}
        tm = data.get("team_mapping")
        return cls(
            aggregation_method=str(data.get("aggregation_method", "sum")),
            team_mapping=dict(tm) if tm else None,
        )


@dataclass
class RecordingSettings:
    """Frame-recording options for the experiment."""

    enabled: bool = False
    fps: int = 20
    fmt: str = "mp4"
    mode: str = RecordingMode.COUNT.value
    every_n_steps: Optional[int] = None
    max_recordings: Optional[int] = None
    max_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fps": self.fps,
            "fmt": self.fmt,
            "mode": str(self.mode),
            "every_n_steps": self.every_n_steps,
            "max_recordings": self.max_recordings,
            "max_dim": self.max_dim,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RecordingSettings":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            fps=int(data.get("fps", 20)),
            fmt=str(data.get("fmt", "mp4")),
            mode=str(data.get("mode", RecordingMode.COUNT.value)),
            every_n_steps=data.get("every_n_steps"),
            max_recordings=data.get("max_recordings"),
            max_dim=data.get("max_dim"),
        )

    def to_recording_config(self) -> RecordingConfig:
        """Project onto the trainer-facing :class:`RecordingConfig`."""
        return RecordingConfig(
            enabled=self.enabled,
            fps=self.fps,
            fmt=self.fmt,
            mode=RecordingMode(str(self.mode)),
            every_n_steps=self.every_n_steps,
            max_recordings=self.max_recordings,
            max_dim=self.max_dim,
        )


@dataclass
class ParallelismSettings:
    """Parallel evaluation options (population-based trainers).

    ``scheduler_address`` targets a remote Dask scheduler (a ``tcp://...``
    URL or a scheduler-file path) and requires ``parallel_backend ==
    "dask"``. Empty strings are normalised to ``None`` (unset).
    """

    num_workers: int = 1
    parallel_backend: str = "process"
    scheduler_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "parallel_backend": self.parallel_backend,
            "scheduler_address": self.scheduler_address,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ParallelismSettings":
        data = data or {}
        raw_address = data.get("scheduler_address")
        return cls(
            num_workers=int(data.get("num_workers", 1)),
            parallel_backend=str(data.get("parallel_backend", "process")),
            scheduler_address=(
                str(raw_address).strip() or None if raw_address is not None else None
            ),
        )


@dataclass
class CoevolutionSideSpec:
    """One co-evolution side in the spec (UI-facing)."""

    name: str
    algorithm: str  # catalog algorithm name, or "frozen"
    role_prefixes: Tuple[str, ...] = ()
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "role_prefixes": list(self.role_prefixes),
            "hyperparameters": dict(self.hyperparameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoevolutionSideSpec":
        return cls(
            name=str(data["name"]),
            algorithm=str(data["algorithm"]),
            role_prefixes=tuple(data.get("role_prefixes", ())),
            hyperparameters=dict(data.get("hyperparameters", {})),
        )


@dataclass
class CoevolutionSettings:
    """Co-evolution configuration block."""

    enabled: bool = False
    mode: str = "competitive_2pop"
    schedule: str = "synchronous"
    k_opponents: int = 3
    aggregation: str = "mean"
    alternating_generations: int = 1
    reporting_side: Optional[str] = None
    sides: List[CoevolutionSideSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "schedule": self.schedule,
            "k_opponents": self.k_opponents,
            "aggregation": self.aggregation,
            "alternating_generations": self.alternating_generations,
            "reporting_side": self.reporting_side,
            "sides": [s.to_dict() for s in self.sides],
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "CoevolutionSettings":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            mode=str(data.get("mode", "competitive_2pop")),
            schedule=str(data.get("schedule", "synchronous")),
            k_opponents=int(data.get("k_opponents", 3)),
            aggregation=str(data.get("aggregation", "mean")),
            alternating_generations=int(data.get("alternating_generations", 1)),
            reporting_side=data.get("reporting_side"),
            sides=[CoevolutionSideSpec.from_dict(s) for s in data.get("sides", [])],
        )


@dataclass
class ExperimentSpec:
    """A complete, serializable experiment configuration.

    ``env_config`` and ``hyperparameters`` should carry only the values the
    user explicitly set; the catalog supplies all other defaults during
    :func:`build_configs` (see the module docstring).
    """

    environment: str
    algorithms: List[str] = field(default_factory=list)
    env_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    fitness: FitnessSettings = field(default_factory=FitnessSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    parallelism: ParallelismSettings = field(default_factory=ParallelismSettings)
    coevolution: CoevolutionSettings = field(default_factory=CoevolutionSettings)
    raw_config: Dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "algorithms": list(self.algorithms),
            "env_config": dict(self.env_config),
            "hyperparameters": {
                algo: dict(params) for algo, params in self.hyperparameters.items()
            },
            "budget": self.budget.to_dict(),
            "fitness": self.fitness.to_dict(),
            "recording": self.recording.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "coevolution": self.coevolution.to_dict(),
            "raw_config": copy.deepcopy(self.raw_config),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        if "environment" not in data:
            raise ValueError("ExperimentSpec requires an 'environment' field")
        return cls(
            environment=str(data["environment"]),
            algorithms=list(data.get("algorithms", [])),
            env_config=dict(data.get("env_config", {})),
            hyperparameters={
                str(algo): dict(params)
                for algo, params in (data.get("hyperparameters", {}) or {}).items()
            },
            budget=BudgetSpec.from_dict(data.get("budget")),
            fitness=FitnessSettings.from_dict(data.get("fitness")),
            recording=RecordingSettings.from_dict(data.get("recording")),
            parallelism=ParallelismSettings.from_dict(data.get("parallelism")),
            coevolution=CoevolutionSettings.from_dict(data.get("coevolution")),
            raw_config=dict(data.get("raw_config", {})),
            label=str(data.get("label", "")),
        )


# Built plan (output of build_configs)


@dataclass
class SubRun:
    """One concrete training invocation derived from an :class:`ExperimentSpec`.

    The runner executes a sub-run as::

        trainer = get_trainer(sub.algorithm, **sub.trainer_params)
        trainer.train(sub.config, progress_callback=...)
    """

    algorithm: str
    seed: int
    trainer_params: Dict[str, Any]
    config: TrainingConfig
    run_index: int
    total_runs: int


@dataclass
class BuiltPlan:
    """Ordered list of sub-runs plus the high-level experiment mode."""

    mode: str  # "standard" | "coevolution"
    sub_runs: List[SubRun] = field(default_factory=list)


# Effective-value resolution (three-layer merge)


def _catalog_env_defaults(env_spec) -> Dict[str, Any]:
    """Non-null env-config defaults from the catalog."""
    return {p.name: p.default for p in env_spec.env_config if p.default is not None}


def _catalog_algo_defaults(algo_spec) -> Dict[str, Any]:
    """Non-null hyperparameter defaults from the catalog."""
    return {
        p.name: p.default for p in algo_spec.hyperparameters if p.default is not None
    }


def _effective_env_config(spec: ExperimentSpec, env_spec) -> Dict[str, Any]:
    defaults = _catalog_env_defaults(env_spec)
    raw = dict(spec.raw_config.get("environment", {}).get("env_config", {}))
    merged = deep_merge(defaults, raw)
    merged = deep_merge(merged, spec.env_config)
    return merged


def _effective_hyperparameters(
    spec: ExperimentSpec, algo: str, algo_spec
) -> Dict[str, Any]:
    defaults = _catalog_algo_defaults(algo_spec)
    raw = dict(
        spec.raw_config.get("algorithms", {}).get(algo, {}).get("hyperparameters", {})
    )
    merged = deep_merge(defaults, raw)
    merged = deep_merge(merged, spec.hyperparameters.get(algo, {}))
    return merged


def _split_hyperparameters(
    values: Mapping[str, Any], algo_spec
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Partition + coerce hyperparameters into (trainer, algorithm, run_config).

    Routing follows each :class:`ParamSpec.target`. Values are coerced to their
    declared types; nullable params left as ``None`` are dropped so the trainer
    falls back to its own default. ``run_config`` holds universal per-run knobs
    (``seed`` / ``max_steps_per_episode``) destined for the matching
    :class:`~deap_pettingzoo.baselines.base.TrainingConfig` attribute.
    """
    from deap_pettingzoo.catalog.schema import ParamTarget

    trainer_params: Dict[str, Any] = {}
    algorithm_params: Dict[str, Any] = {}
    run_config: Dict[str, Any] = {}
    for name, value in values.items():
        param = algo_spec.hyperparameter(name)
        if param is None:
            # Unknown key (raw-config extension) -> default to algorithm_params.
            algorithm_params[name] = value
            continue
        try:
            coerced = param.coerce(value)
        except ValueError:
            coerced = value
        if coerced is None:
            continue
        if param.target is ParamTarget.TRAINER:
            trainer_params[name] = coerced
        elif param.target is ParamTarget.RUN_CONFIG:
            run_config[name] = coerced
        else:
            algorithm_params[name] = coerced
    return trainer_params, algorithm_params, run_config


# Auto-padding for heterogeneous-obs environments

_AUTO_PADDED_SUFFIX = "__deap_auto_padded"


def _maybe_auto_pad_env(
    spec: "ExperimentSpec",
    env_spec,
) -> str:
    """Auto-register a SuperSuit-padded variant of *env_spec* when an
    evolutionary algorithm is used against a heterogeneous-observation
    environment.

    Returns the (possibly new) environment name to use in
    :class:`TrainingConfig`.  The original name is returned unchanged
    when no padding is needed.
    """
    from deap_pettingzoo.catalog import algorithms as cat_algorithms

    # Only fire when at least one evolutionary algorithm is selected.
    has_evolutionary = any(
        cat_algorithms.get_algorithm(a).family == "evolutionary"
        for a in spec.algorithms
    )
    if not has_evolutionary:
        return spec.environment

    # Detect heterogeneous observations.
    try:
        from deap_pettingzoo.baselines.compatibility import inspect_environment

        info = inspect_environment(spec.environment)
        if info.obs_dim is not None or info.num_agents <= 1:
            return spec.environment  # homogeneous — nothing to do
    except Exception:
        return spec.environment  # can't inspect — trust the user

    # Check if SuperSuit is installed.
    try:
        import supersuit  # noqa: F401
    except ImportError:
        return spec.environment  # can't pad — let it fail with a clear error

    # Determine whether a padded variant is already registered; if not, register one.
    padded_name = f"{spec.environment}{_AUTO_PADDED_SUFFIX}"
    from deap_pettingzoo.environments.registry import (
        get_env_spec,
        is_registered,
        register_env,
    )

    if is_registered(padded_name):
        return padded_name

    # Re-use the original env's factory so the padded variant is
    # environment-identical apart from the SuperSuit wrapper chain.
    original_spec = get_env_spec(spec.environment)

    try:
        register_env(
            padded_name,
            original_spec.factory,
            default_episode_len=env_spec.default_episode_len,
            api=env_spec.api,
            continuous_default=env_spec.continuous_default,
            tags=set(env_spec.tags) | {"auto_padded"},
            wrappers=(
                supersuit.pad_observations_v0,
                supersuit.pad_action_space_v0,
                supersuit.agent_indicator_v0,
            ),
            params=original_spec.params,
            teams=original_spec.teams,
            overwrite=True,
        )
    except Exception:
        return spec.environment

    return padded_name


# Build configs


def build_configs(
    spec: ExperimentSpec,
    output_root: Optional[Path] = None,
) -> BuiltPlan:
    """Resolve an :class:`ExperimentSpec` into an ordered :class:`BuiltPlan`.

    Args:
        spec: The (validated) experiment spec.
        output_root: Directory under which per-run output dirs are created
            (``<root>/<algorithm>/seed_<i>``). ``None`` disables disk output.

    Returns:
        A :class:`BuiltPlan` whose ``sub_runs`` the runner executes in order.
    """
    from deap_pettingzoo.catalog import algorithms as cat_algorithms
    from deap_pettingzoo.catalog import environments as cat_environments

    env_spec = cat_environments.get_environment(spec.environment)
    env_config = _effective_env_config(spec, env_spec)
    budget_mode = BudgetMode(spec.budget.mode)
    recording_config = spec.recording.to_recording_config()
    fitness_config = FitnessConfig(
        aggregation_method=spec.fitness.aggregation_method,
        team_mapping=spec.fitness.team_mapping,
    )

    # Auto-apply SuperSuit padding when an evolutionary algorithm is used
    # on a heterogeneous-observation environment. Python API experiments that
    # construct TrainingConfig directly are unaffected as this only fires
    # through the ExperimentSpec -> build_configs path (UI + programmatic).
    effective_env_name = _maybe_auto_pad_env(spec, env_spec)

    padded = (
        f" (auto-padded -> {effective_env_name})"
        if effective_env_name != spec.environment
        else ""
    )
    logger.info(
        "Building experiment: env=%s%s, algorithms=%s, budget=%s=%s, seeds=%s, "
        "parallelism=%sx%s, recording=%s, coevolution=%s",
        spec.environment,
        padded,
        spec.algorithms or "(coevolution)",
        spec.budget.mode,
        spec.budget.value,
        spec.budget.num_seeds,
        spec.parallelism.num_workers,
        spec.parallelism.parallel_backend,
        spec.recording.enabled,
        spec.coevolution.mode if spec.coevolution.enabled else False,
    )

    if spec.coevolution.enabled:
        return _build_coevolution_plan(
            spec, env_config, budget_mode, recording_config, output_root
        )

    num_seeds = max(1, spec.budget.num_seeds)
    total_runs = len(spec.algorithms) * num_seeds
    sub_runs: List[SubRun] = []
    run_index = 0
    for algo in spec.algorithms:
        algo_spec = cat_algorithms.get_algorithm(algo)
        effective = _effective_hyperparameters(spec, algo, algo_spec)
        trainer_params, base_algo_params, base_run_config = _split_hyperparameters(
            effective, algo_spec
        )
        logger.debug(
            "Resolved hyperparameters for %s: trainer=%s, algo=%s, run=%s",
            algo,
            trainer_params,
            base_algo_params,
            base_run_config,
        )
        # Population-based trainers consume parallelism via TrainingConfig fields;
        # MARL trainers ignore them (they parallelise via num_envs), so leave those at the serial defaults.
        if algo_spec.family == "evolutionary":
            cfg_num_workers = spec.parallelism.num_workers
            cfg_parallel_backend = spec.parallelism.parallel_backend
            cfg_scheduler_address = spec.parallelism.scheduler_address
        else:
            cfg_num_workers = 1
            cfg_parallel_backend = "process"
            cfg_scheduler_address = None
        for seed_idx in range(num_seeds):
            algo_params = dict(base_algo_params)
            run_config = dict(base_run_config)
            run_config["seed"] = seed_idx
            out_dir = (
                output_root / cat_algorithms.canonical_name(algo) / f"seed_{seed_idx}"
                if output_root is not None
                else None
            )
            config = TrainingConfig(
                env_name=effective_env_name,
                budget_mode=budget_mode,
                budget_value=spec.budget.value,
                num_seeds=1,
                seed=run_config.get("seed"),
                max_steps_per_episode=run_config.get("max_steps_per_episode"),
                algorithm_params=algo_params,
                env_config=dict(env_config),
                output_dir=out_dir,
                num_workers=cfg_num_workers,
                parallel_backend=cfg_parallel_backend,
                scheduler_address=cfg_scheduler_address,
                recording=recording_config,
                fitness=fitness_config,
                budget_on_consumed=spec.budget.budget_on_consumed,
            )
            sub_runs.append(
                SubRun(
                    algorithm=cat_algorithms.canonical_name(algo),
                    seed=seed_idx,
                    trainer_params=trainer_params,
                    config=config,
                    run_index=run_index,
                    total_runs=total_runs,
                )
            )
            run_index += 1
    return BuiltPlan(mode="standard", sub_runs=sub_runs)


def _build_coevolution_plan(
    spec: ExperimentSpec,
    env_config: Dict[str, Any],
    budget_mode: BudgetMode,
    recording_config: RecordingConfig,
    output_root: Optional[Path],
) -> BuiltPlan:
    """Build a co-evolution :class:`BuiltPlan` (one sub-run per seed)."""
    from deap_pettingzoo.baselines.coevolution import CoEvolutionConfig, Side
    from deap_pettingzoo.catalog import algorithms as cat_algorithms

    coev = spec.coevolution
    num_seeds = max(1, spec.budget.num_seeds)
    sub_runs: List[SubRun] = []
    for seed_idx in range(num_seeds):
        sides: List[Side] = []
        for side_spec in coev.sides:
            optimizer_config = dict(side_spec.hyperparameters)
            if side_spec.algorithm == "frozen":
                optimizer = "frozen"
            else:
                optimizer = (
                    cat_algorithms.coevolution_optimizer_for(side_spec.algorithm)
                    or side_spec.algorithm
                )
                # RL / RL-HPO sides resolve their concrete AgileRL algorithm
                # from ``optimizer_config['algorithm']``;
                # inject the catalog name the user picked (the UI sends empty per-side hyperparameters).
                optimizer_config.setdefault("algorithm", side_spec.algorithm)
            sides.append(
                Side(
                    name=side_spec.name,
                    role_prefixes=tuple(side_spec.role_prefixes),
                    optimizer=optimizer,
                    optimizer_config=optimizer_config,
                )
            )
        coev_config = CoEvolutionConfig(
            mode=coev.mode,
            sides=sides,
            schedule=coev.schedule,
            alternating_generations=coev.alternating_generations,
            k_opponents=coev.k_opponents,
            aggregation=coev.aggregation,
            reporting_side=coev.reporting_side,
            seed=seed_idx,
            num_workers=spec.parallelism.num_workers,
            parallel_backend=spec.parallelism.parallel_backend,
            scheduler_address=spec.parallelism.scheduler_address,
        )
        out_dir = (
            output_root / "coevolution" / f"seed_{seed_idx}"
            if output_root is not None
            else None
        )
        config = TrainingConfig(
            env_name=spec.environment,
            budget_mode=budget_mode,
            budget_value=spec.budget.value,
            num_seeds=1,
            seed=seed_idx,
            coevolution_config=coev_config,
            env_config=dict(env_config),
            output_dir=out_dir,
            recording=recording_config,
            budget_on_consumed=spec.budget.budget_on_consumed,
        )
        sub_runs.append(
            SubRun(
                algorithm="coevolution",
                seed=seed_idx,
                trainer_params={},
                config=config,
                run_index=seed_idx,
                total_runs=num_seeds,
            )
        )
    return BuiltPlan(mode="coevolution", sub_runs=sub_runs)


# Validate spec

_VALID_BUDGET_MODES = ("env_steps", "eval_episodes", "generations")
_SUPPORTED_BACKENDS = ("process", "dask")


def validate_spec(spec: ExperimentSpec) -> "ValidationReport":
    """Validate an :class:`ExperimentSpec` against the catalog.

    Produces a :class:`~deap_pettingzoo.catalog.schema.ValidationReport` with
    per-field machine-coded issues. This is the single validation entry
    point: the Setup page calls it client-side and the FastAPI service calls it
    again server-side on submit (defense in depth).

    Checks (non-exhaustive): environment exists + installed; env-config types /
    ranges; each algorithm exists + installed; budget mode supported by every
    selected algorithm (e.g. ``generations`` is rejected for the step-based
    AgileRL trainers); aggregation supported;
    co-evolution sides cover every team with eligible optimizers; numeric
    guards on budget / seeds / recording / parallelism.
    """
    from deap_pettingzoo.catalog import algorithms as cat_algorithms
    from deap_pettingzoo.catalog import environments as cat_environments
    from deap_pettingzoo.catalog.schema import ValidationReport

    report = ValidationReport()

    env_spec = _validate_environment(spec, report, cat_environments)
    _validate_budget(spec, report)
    _validate_recording(spec, report)
    _validate_parallelism(spec, report)

    if spec.coevolution.enabled:
        _validate_coevolution(spec, report, env_spec, cat_algorithms)
    else:
        _validate_standard_algorithms(spec, report, env_spec, cat_algorithms)

    return report


def _validate_environment(spec, report, cat_environments):
    """Validate the environment + its config; return the EnvironmentSpec or None."""
    if not spec.environment:
        report.add("environment", "missing_env", "No environment selected.")
        return None
    try:
        env_spec = cat_environments.get_environment(spec.environment)
    except ValueError:
        report.add(
            "environment",
            "unknown_env",
            f"Environment {spec.environment!r} is not registered.",
        )
        return None
    if not cat_environments.is_environment_available(spec.environment):
        report.add(
            "environment",
            "env_unavailable",
            f"Environment {spec.environment!r} requires an uninstalled extra "
            f"({', '.join(env_spec.required_extras) or 'unknown'}).",
        )
    # Validate explicitly-set env-config values against the schema.
    for name, value in spec.env_config.items():
        param = env_spec.env_param(name)
        if param is None:
            report.add(
                f"env_config.{name}",
                "unknown_param",
                f"Unknown env-config parameter {name!r} for " f"{spec.environment!r}.",
                severity="warning",
            )
            continue
        report.extend(param.validate(value, path=f"env_config.{name}"))
    return env_spec


def _validate_budget(spec, report):
    budget = spec.budget
    if budget.mode not in _VALID_BUDGET_MODES:
        report.add(
            "budget.mode",
            "invalid_budget_mode",
            f"Budget mode {budget.mode!r} is not supported. Choose one of "
            f"{list(_VALID_BUDGET_MODES)}.",
        )
    if budget.value <= 0:
        report.add("budget.value", "out_of_range", "Budget value must be positive.")
    if budget.num_seeds < 1:
        report.add("budget.num_seeds", "out_of_range", "Number of seeds must be >= 1.")
    if budget.budget_on_consumed and budget.mode == "env_steps":
        for algo in spec.algorithms:
            canonical = algo.lower().strip()
            if canonical in ("ea", "genetic", "ga"):
                strategy = (spec.hyperparameters.get(algo) or {}).get(
                    "strategy", "simple"
                )
                if strategy not in ("simple", None, ""):
                    report.add(
                        "budget.budget_on_consumed",
                        "unsupported_trainer",
                        f"budget_on_consumed=True is only supported by the EA "
                        f"'simple' strategy; strategy={strategy!r} runs a DEAP "
                        f"built-in with a fixed generation count. Use "
                        f"budget_on_consumed=False or BudgetMode.GENERATIONS.",
                    )


def _validate_recording(spec, report):
    rec = spec.recording
    if rec.fps <= 0:
        report.add("recording.fps", "out_of_range", "FPS must be positive.")
    if rec.fmt not in ("mp4", "gif"):
        report.add("recording.fmt", "invalid_format", "Format must be 'mp4' or 'gif'.")
    if rec.max_recordings is not None and rec.max_recordings < 1:
        report.add(
            "recording.max_recordings",
            "out_of_range",
            "Max recordings must be >= 1.",
        )
    valid_modes = {m.value for m in RecordingMode}
    if str(rec.mode) not in valid_modes:
        report.add(
            "recording.mode",
            "invalid_mode",
            f"Mode must be one of {sorted(valid_modes)}.",
        )
    if rec.every_n_steps is not None and rec.every_n_steps < 1:
        report.add(
            "recording.every_n_steps",
            "out_of_range",
            "Recording env-step interval must be >= 1.",
        )
    if str(rec.mode) == RecordingMode.INTERVAL.value and rec.every_n_steps is None:
        report.add(
            "recording.every_n_steps",
            "missing",
            "mode='interval' requires every_n_steps.",
        )


def _validate_parallelism(spec, report):
    par = spec.parallelism
    if par.num_workers < 0:
        report.add(
            "parallelism.num_workers",
            "out_of_range",
            "Number of workers must be >= 0.",
        )
    if par.parallel_backend not in _SUPPORTED_BACKENDS:
        report.add(
            "parallelism.parallel_backend",
            "invalid_backend",
            f"Backend {par.parallel_backend!r} is not supported. Choose one "
            f"of {list(_SUPPORTED_BACKENDS)}.",
        )
        return
    scheduler_address = getattr(par, "scheduler_address", None)
    if scheduler_address and par.parallel_backend != "dask":
        report.add(
            "parallelism.scheduler_address",
            "backend_mismatch",
            f"scheduler_address={scheduler_address!r} requires "
            "parallel_backend='dask'.",
        )
        return
    # Warn (non-blocking) when a real parallel backend is selected but its
    # optional package is not installed — it would otherwise only fail at runtime
    # with an ImportError.
    if par.num_workers and par.num_workers > 1:
        from deap_pettingzoo.catalog.availability import module_available

        _backend_requirement = {
            "dask": ("dask.distributed", "deap-pettingzoo[dask]"),
        }
        req = _backend_requirement.get(par.parallel_backend)
        if req is not None and not module_available(req[0]):
            module_name, extra = req
            report.add(
                "parallelism.parallel_backend",
                "backend_not_installed",
                f"Backend {par.parallel_backend!r} is selected but the "
                f"{module_name!r} package is not installed; the run will fail "
                f"at start. Install it with `pip install '{extra}'`.",
                severity="warning",
            )


def _validate_hyperparameters(spec, report, algo, algo_spec):
    """Validate explicitly-set hyperparameters for one algorithm."""
    for name, value in spec.hyperparameters.get(algo, {}).items():
        param = algo_spec.hyperparameter(name)
        path = f"hyperparameters.{algo}.{name}"
        if param is None:
            report.add(
                path,
                "unknown_param",
                f"Unknown hyperparameter {name!r} for {algo!r}.",
                severity="warning",
            )
            continue
        report.extend(param.validate(value, path=path))


def _validate_aggregation(spec, report, algo_spec):
    method = spec.fitness.aggregation_method
    if not algo_spec.supported_aggregations:
        # MARL trainers ignore aggregation entirely.
        if method != "sum":
            report.add(
                "fitness.aggregation_method",
                "aggregation_ignored",
                f"{algo_spec.name!r} ignores fitness aggregation "
                f"(reward comes from the environment).",
                severity="warning",
            )
        return
    if method not in algo_spec.supported_aggregations:
        report.add(
            "fitness.aggregation_method",
            "invalid_aggregation",
            f"{algo_spec.name!r} does not support aggregation {method!r}. "
            f"Supported: {list(algo_spec.supported_aggregations)}.",
        )
    if method == "team" and not spec.fitness.team_mapping:
        report.add(
            "fitness.team_mapping",
            "missing_team_mapping",
            "'team' aggregation requires a team_mapping (agent -> team).",
        )


def _validate_standard_algorithms(spec, report, env_spec, cat_algorithms):
    if not spec.algorithms:
        report.add("algorithms", "no_algorithms", "Select at least one algorithm.")
        return

    compatible: List[str] = []
    env_has_heterogeneous_obs = False
    if env_spec is not None:
        try:
            from deap_pettingzoo.catalog import compatibility as cat_compat

            compatible = cat_compat.compatible_algorithms(env_spec.name)
        except Exception:  # pragma: no cover - env build failure
            compatible = []

        # Detect heterogeneous observation sizes (EA/CMA/PSO require homogeneous).
        try:
            from deap_pettingzoo.baselines.compatibility import inspect_environment

            env_info = inspect_environment(env_spec.name)
            env_has_heterogeneous_obs = (
                env_info.obs_dim is None and env_info.num_agents > 1
            )
        except Exception:
            pass

    for algo in spec.algorithms:
        canonical = cat_algorithms.canonical_name(algo)
        if canonical is None:
            report.add(
                f"algorithms.{algo}",
                "unknown_algorithm",
                f"Unknown algorithm {algo!r}.",
            )
            continue
        algo_spec = cat_algorithms.get_algorithm(canonical)
        if not cat_algorithms.is_algorithm_available(canonical):
            report.add(
                f"algorithms.{canonical}",
                "algorithm_unavailable",
                f"{canonical!r} requires an uninstalled extra "
                f"({', '.join(algo_spec.required_extras) or 'unknown'}).",
            )
        if spec.budget.mode not in algo_spec.supported_budget_modes:
            report.add(
                "budget.mode",
                "budget_mode_unsupported",
                f"{canonical!r} does not support budget mode " f"{spec.budget.mode!r}.",
            )
        _validate_hyperparameters(spec, report, algo, algo_spec)
        _validate_aggregation(spec, report, algo_spec)

        # EA / CMA-ES / PSO require homogeneous observation sizes.
        # When SuperSuit is available, build_configs auto-pads — warn instead of blocking.
        # Without SuperSuit, this is a hard error.
        if env_has_heterogeneous_obs and algo_spec.family == "evolutionary":
            try:
                import supersuit  # noqa: F401

                severity = "warning"
                hint = (
                    "Observations will be auto-padded with SuperSuit "
                    "(pad_observations_v0, pad_action_space_v0, "
                    "agent_indicator_v0)."
                )
            except ImportError:
                severity = "error"
                hint = (
                    "Use AgileRL trainers (MADDPG/MATD3/IPPO) which build "
                    "per-agent networks, or install supersuit for automatic "
                    "padding (pip install deap-pettingzoo[supersuit])."
                )
            report.add(
                f"algorithms.{canonical}",
                "heterogeneous_obs",
                f"{canonical!r} requires all agents to have identical "
                f"observation sizes, but {env_spec.name!r} has per-role "
                f"observations. {hint}",
                severity=severity,
            )
        if env_spec is not None and compatible and canonical not in compatible:
            report.add(
                f"algorithms.{canonical}",
                "incompatible",
                f"{canonical!r} is not compatible with environment "
                f"{env_spec.name!r}.",
            )


def _validate_coevolution(spec, report, env_spec, cat_algorithms):
    from deap_pettingzoo.catalog import coevolution as cat_coevolution

    coev = spec.coevolution
    if coev.mode not in cat_coevolution.COEV_MODES:
        report.add(
            "coevolution.mode",
            "invalid_mode",
            f"Invalid co-evolution mode {coev.mode!r}. Choose one of "
            f"{list(cat_coevolution.COEV_MODES)}.",
        )
    if coev.schedule not in cat_coevolution.COEV_SCHEDULES:
        report.add(
            "coevolution.schedule",
            "invalid_schedule",
            f"Invalid schedule {coev.schedule!r}. Choose one of "
            f"{list(cat_coevolution.COEV_SCHEDULES)}.",
        )
    if coev.aggregation not in cat_coevolution.COEV_AGGREGATIONS:
        report.add(
            "coevolution.aggregation",
            "invalid_aggregation",
            f"Invalid co-evolution aggregation {coev.aggregation!r}.",
        )
    if len(coev.sides) < 2:
        report.add(
            "coevolution.sides",
            "too_few_sides",
            "Co-evolution requires at least two sides.",
        )

    eligible = {
        entry["optimizer"] for entry in cat_coevolution.eligible_side_optimizers()
    }
    side_names = [s.name for s in coev.sides]
    if len(set(side_names)) != len(side_names):
        report.add("coevolution.sides", "duplicate_side", "Side names must be unique.")

    covered_prefixes: List[str] = []
    for idx, side in enumerate(coev.sides):
        path = f"coevolution.sides.{idx}"
        if not side.role_prefixes:
            report.add(
                f"{path}.role_prefixes",
                "missing_prefixes",
                f"Side {side.name!r} must control at least one agent prefix.",
            )
        covered_prefixes.extend(side.role_prefixes)
        if side.algorithm == "frozen":
            optimizer = "frozen"
        else:
            optimizer = cat_algorithms.coevolution_optimizer_for(side.algorithm)
        if optimizer is None or optimizer not in eligible:
            report.add(
                f"{path}.algorithm",
                "ineligible_optimizer",
                f"Algorithm {side.algorithm!r} cannot be used as a "
                f"co-evolution side.",
            )

    # Every declared team prefix should be controlled by some side.
    if env_spec is not None and env_spec.teams:
        for team in env_spec.teams:
            if not any(prefix in covered_prefixes for prefix in team.agent_prefixes):
                report.add(
                    "coevolution.sides",
                    "uncovered_team",
                    f"No side controls team {team.name!r} "
                    f"(prefixes {list(team.agent_prefixes)}).",
                )

    if coev.reporting_side is not None and coev.reporting_side not in side_names:
        report.add(
            "coevolution.reporting_side",
            "unknown_reporting_side",
            f"Reporting side {coev.reporting_side!r} is not a declared side.",
        )

    # QDax co-evolution sides are allowed under budget_on_consumed: they
    # evaluate a full batch per generation, so the shared consumed pool's
    # overshoot bound is one generation's pooled cost *including* that
    # batch (batch_size × k_opponents × max_steps ticks) — the same
    # finish-generation-then-stop bound every other side type has, and
    # negligible at realistic ENV_STEPS budgets. The QDax side reports
    # real per-generation env ticks into GenerationStats.env_steps, so
    # the aggregate the trainer stops on is exact.
