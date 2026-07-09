"""Dependency-free schema primitives for the unified catalog.

This module is the **foundation layer** of :mod:`deap_pettingzoo.catalog`.
It is self-contained — it imports only the Python standard
library, never :mod:`deap_pettingzoo.baselines`, :mod:`...environments`,
``torch``, ``agilerl`` or ``jax``. That guarantees two things:

1. The environment registry (:mod:`deap_pettingzoo.environments.registry`)
   can import :class:`ParamSpec` / :class:`TeamSpec` to attach declarative
   metadata to an :class:`~deap_pettingzoo.environments.registry.EnvSpec`
   without creating an import cycle.
2. The Streamlit frontend (and the FastAPI service) can build forms from the
   JSON produced by :meth:`ParamSpec.to_dict` without importing any heavy
   training dependency.

The types here describe *what a parameter is* (name, type, default, range,
widget grouping) and *how to coerce / validate a user-supplied value*. The
higher-level modules (``algorithms``, ``environments``, ``coevolution``) build
:class:`AlgorithmSpec` / :class:`EnvironmentSpec` objects out of these
primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

CATALOG_SCHEMA_VERSION = "1"
"""Bumped whenever the JSON contract emitted by ``to_dict`` changes shape.

The frontend stores this alongside cached catalog data so a backend upgrade
that changes the schema can be detected (the UI simply refetches).
"""


class ParamType(str, Enum):
    """The value type of a single configurable parameter.

    Inherits from :class:`str` so instances serialise directly to JSON
    (``ParamType.INT`` → ``"int"``) and compare equal to their string form.
    """

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    ENUM = "enum"
    LIST = "list"


class ParamTarget(str, Enum):
    """Where a parameter's value is routed when building backend configs.

    * :attr:`TRAINER` — architecture / operator choices consumed by the
      trainer **constructor** (``ComparisonConfig.algorithm_configs[name]
      ["trainer_params"]``).
    * :attr:`ALGORITHM_PARAMS` — per-run knobs merged into
      ``TrainingConfig.algorithm_params`` (e.g. ``num_envs``, AgileRL
      passthroughs) — algorithm-specific hyperparameters only.
    * :attr:`ENV_CONFIG` — keyword arguments forwarded to the environment
      factory (``TrainingConfig.env_config``).
    * :attr:`RUN_CONFIG` — universal per-run knobs (``seed``,
      ``max_steps_per_episode``) routed onto the
      :class:`~deap_pettingzoo.baselines.base.TrainingConfig` attribute of
      the same name.
    """

    TRAINER = "trainer"
    ALGORITHM_PARAMS = "algorithm_params"
    ENV_CONFIG = "env_config"
    RUN_CONFIG = "run_config"


def humanize(name: str) -> str:
    """Turn a snake_case identifier into a Title Case label.

    ``"population_size"`` → ``"Population Size"``; ``"lr_actor"`` →
    ``"Lr Actor"``. Used as the fallback display label when a
    :class:`ParamSpec` does not set one explicitly.
    """

    return name.replace("_", " ").strip().title()


@dataclass(frozen=True)
class ValidationIssue:
    """A single problem (or note) discovered while validating a spec.

    Attributes:
        path: Dotted location of the offending value, e.g.
            ``"algorithms.ea.hyperparameters.population_size"`` or
            ``"environment.env_config.max_cycles"``.
        code: Stable machine-readable code (``"out_of_range"``,
            ``"unknown_env"``, ...) so the frontend can branch / i18n.
        message: Human-readable explanation suitable for direct display.
        severity: ``"error"`` (blocks submission) or ``"warning"``
            (surfaced but non-blocking).
    """

    path: str
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ValidationReport:
    """Accumulates :class:`ValidationIssue` objects from validation passes.

    The report is :attr:`ok` only when it contains no ``error``-severity
    issues; warnings never block submission.
    """

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when there are no error-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(
        self,
        path: str,
        code: str,
        message: str,
        severity: str = "error",
    ) -> None:
        """Append a new issue described by its parts."""
        self.issues.append(ValidationIssue(path, code, message, severity))

    def extend(self, issues: Sequence[ValidationIssue]) -> None:
        """Append a batch of pre-built issues."""
        self.issues.extend(issues)

    def merge(self, other: "ValidationReport") -> None:
        """Absorb every issue from ``other`` into this report."""
        self.issues.extend(other.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
        }

    def raise_if_invalid(self) -> None:
        """Raise :class:`ValueError` if any error-severity issue is present."""
        if not self.ok:
            lines = "\n".join(f"  - [{i.path}] {i.message}" for i in self.errors)
            raise ValueError(f"Invalid experiment specification:\n{lines}")


def _is_bool(value: Any) -> bool:
    # ``bool`` is a subclass of ``int``; guard against silently accepting
    # ``True`` where an integer is required and vice versa.
    return isinstance(value, bool)


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one configurable parameter.

    A :class:`ParamSpec` is the atom the dynamic-form UI is built from and
    the unit :func:`validate` / :func:`coerce` operate on. It is immutable
    (``frozen=True``) so specs can be shared freely and cached.

    Attributes:
        name: Parameter identifier (the kwarg name passed to the backend).
        type: One of :class:`ParamType`.
        default: Default value (already in native Python type).
        label: Display label; falls back to :func:`humanize` of ``name``.
        help: Tooltip / helper text.
        minimum: Inclusive lower bound (numeric types only).
        maximum: Inclusive upper bound (numeric types only).
        step: Suggested UI increment (numeric types only).
        choices: Allowed values for :attr:`ParamType.ENUM`.
        group: Tab / section grouping hint for the form.
        advanced: When ``True`` the UI hides the field behind an
            "Advanced" expander.
        target: Routing channel — see :class:`ParamTarget`.
        item_type: Element type for :attr:`ParamType.LIST`.
        nullable: When ``True`` an empty / ``None`` value is accepted and
            means "leave unset" (the key is omitted from the built config).
        unavailable_choices: Subset of :attr:`choices` (:attr:`ParamType.ENUM`
            only) that are declared but currently ungated — e.g. ``"cuda"``
            on a CPU-only torch build. The UI renders these disabled /
            annotated; :meth:`coerce` and :meth:`validate` reject them so a
            stale form submission can never submit one.
    """

    name: str
    type: ParamType
    default: Any = None
    label: str = ""
    help: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    group: str = "General"
    advanced: bool = False
    target: ParamTarget = ParamTarget.ALGORITHM_PARAMS
    item_type: Optional[ParamType] = None
    nullable: bool = False
    unavailable_choices: Tuple[Any, ...] = ()

    # Presentation

    @property
    def display_label(self) -> str:
        """The label to show, computed from :attr:`name` when unset."""
        return self.label or humanize(self.name)

    @property
    def available_choices(self) -> Tuple[Any, ...]:
        """:attr:`choices` with :attr:`unavailable_choices` removed."""
        if not self.choices:
            return ()
        unavailable = set(self.unavailable_choices)
        return tuple(c for c in self.choices if c not in unavailable)

    # Coercion

    def coerce(self, value: Any) -> Any:
        """Convert a loosely-typed input into this parameter's native type.

        Forms and JSON frequently deliver numbers as strings (``"50"``) or
        integers as floats (``50.0``). Coercion normalises these so range
        checks and the eventual backend call receive correctly-typed values.

        Args:
            value: The raw user-supplied value.

        Returns:
            The value converted to the native Python type, or ``None`` when
            ``value is None`` and the spec is :attr:`nullable`.

        Raises:
            ValueError: If the value cannot be represented as this type.
        """
        if value is None:
            if self.nullable:
                return None
            raise ValueError(f"{self.name!r} does not accept a null value")

        if self.type is ParamType.INT:
            if _is_bool(value):
                raise ValueError(f"{self.name!r} expects an integer, got bool")
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"{self.name!r} expects an integer, got {value!r}")
                return int(value)
            if isinstance(value, str):
                return int(value.strip())
            if isinstance(value, int):
                return int(value)
            raise ValueError(f"{self.name!r} expects an integer, got {value!r}")

        if self.type is ParamType.FLOAT:
            if _is_bool(value):
                raise ValueError(f"{self.name!r} expects a number, got bool")
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            raise ValueError(f"{self.name!r} expects a number, got {value!r}")

        if self.type is ParamType.BOOL:
            if _is_bool(value):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "1", "yes", "on"}:
                    return True
                if low in {"false", "0", "no", "off"}:
                    return False
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            raise ValueError(f"{self.name!r} expects a boolean, got {value!r}")

        if self.type is ParamType.ENUM:
            # Compare as strings so JSON round-trips of non-string choices work.
            choices = self.choices or ()
            unavailable = set(self.unavailable_choices)
            for choice in choices:
                if value == choice or str(value) == str(choice):
                    if choice in unavailable:
                        raise ValueError(
                            f"{self.name!r} choice {choice!r} is currently "
                            "unavailable in this deployment"
                        )
                    return choice
            raise ValueError(
                f"{self.name!r} must be one of {list(choices)!r}, got {value!r}"
            )

        if self.type is ParamType.STR:
            return str(value)

        if self.type is ParamType.LIST:
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{self.name!r} expects a list, got a string")
            try:
                items = list(value)
            except TypeError as exc:  # not iterable
                raise ValueError(
                    f"{self.name!r} expects a list, got {value!r}"
                ) from exc
            if self.item_type is not None:
                element = ParamSpec(name=f"{self.name}[]", type=self.item_type)
                return [element.coerce(v) for v in items]
            return items

        raise ValueError(f"Unsupported ParamType {self.type!r}")  # pragma: no cover

    # Validation

    def validate(self, value: Any, path: Optional[str] = None) -> List[ValidationIssue]:
        """Validate ``value`` against this spec, returning any issues.

        Performs coercion first (so type errors are reported), then range
        and choice checks for numeric / enum types. Returns an empty list
        when the value is acceptable.

        Args:
            value: Raw user-supplied value.
            path: Dotted path used in emitted :class:`ValidationIssue`
                objects; defaults to :attr:`name`.

        Returns:
            A list of :class:`ValidationIssue` (empty when valid).
        """
        loc = path or self.name
        issues: List[ValidationIssue] = []

        if value is None and self.nullable:
            return issues

        try:
            coerced = self.coerce(value)
        except ValueError as exc:
            issues.append(ValidationIssue(loc, "type_error", str(exc)))
            return issues

        if self.type in (ParamType.INT, ParamType.FLOAT):
            if self.minimum is not None and coerced < self.minimum:
                issues.append(
                    ValidationIssue(
                        loc,
                        "out_of_range",
                        f"{self.display_label} must be >= {self.minimum} "
                        f"(got {coerced}).",
                    )
                )
            if self.maximum is not None and coerced > self.maximum:
                issues.append(
                    ValidationIssue(
                        loc,
                        "out_of_range",
                        f"{self.display_label} must be <= {self.maximum} "
                        f"(got {coerced}).",
                    )
                )

        return issues

    # Serialization

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description for the frontend."""
        return {
            "name": self.name,
            "type": self.type.value,
            "default": self.default,
            "label": self.display_label,
            "help": self.help,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices) if self.choices is not None else None,
            "group": self.group,
            "advanced": self.advanced,
            "target": self.target.value,
            "item_type": self.item_type.value if self.item_type else None,
            "nullable": self.nullable,
            "unavailable_choices": list(self.unavailable_choices),
        }


@dataclass(frozen=True)
class TeamSpec:
    """A named group of agents within a multi-team environment.

    Used both for co-evolution side assignment (which agents a side
    controls) and for the UI to know whether an environment is multi-team
    (and therefore co-evolution-capable).

    Attributes:
        name: Display name (e.g. ``"predators"``).
        agent_prefixes: Agent-name prefixes belonging to this team. An
            agent ``a`` is on this team when ``a.startswith(prefix)`` for
            some prefix here (matching the co-evolution evaluator's logic).
        role: Optional semantic role hint (``"predator"`` / ``"prey"`` /
            ``"cooperative"``) for UI labelling.
    """

    name: str
    agent_prefixes: Tuple[str, ...]
    role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_prefixes": list(self.agent_prefixes),
            "role": self.role,
        }


@dataclass(frozen=True)
class AlgorithmSpec:
    """Catalog description of one trainable algorithm.

    Attributes:
        name: Canonical algorithm identifier used everywhere in the UI and
            in :class:`ExperimentSpec` (e.g. ``"ea"``, ``"cma-es"``,
            ``"pso"``, ``"maddpg"``).
        label: Human-readable display name.
        family: Coarse grouping — ``"evolutionary"`` or ``"marl"``.
        aliases: Alternative names that resolve to this algorithm (e.g.
            ``"cma"`` → ``"cma-es"``), reconciling the historical
            vocabulary drift between the trainer registry and co-evolution.
        hyperparameters: Ordered tuple of :class:`ParamSpec` describing the
            configurable knobs.
        capabilities: The trainer's ``get_capabilities()`` dict
            (``discrete_action``, ``continuous_action``,
            ``needs_global_state``, ``on_policy``, ``supports_turn_based``).
        coevolution_optimizer: The ``Side.optimizer`` token to use when this
            algorithm participates in co-evolution, or ``None`` if it cannot.
        required_modules: Importable module names whose presence gates
            availability (e.g. ``("agilerl", "torch")``). Empty ⇒ always
            available (core deps).
        required_extras: pip extras that provide :attr:`required_modules`
            (e.g. ``("rl-baselines",)``) — surfaced in install hints.
        supported_budget_modes: Budget modes this trainer accepts.
        supported_aggregations: Fitness aggregation methods this trainer can
            apply (population-based trainers support the full set; MARL
            trainers use environment reward directly).
        description: One-line summary for tooltips.
    """

    name: str
    label: str
    family: str
    aliases: Tuple[str, ...] = ()
    hyperparameters: Tuple[ParamSpec, ...] = ()
    capabilities: Dict[str, bool] = field(default_factory=dict)
    coevolution_optimizer: Optional[str] = None
    required_modules: Tuple[str, ...] = ()
    required_extras: Tuple[str, ...] = ()
    supported_budget_modes: Tuple[str, ...] = ("env_steps", "eval_episodes")
    supported_aggregations: Tuple[str, ...] = ("sum", "mean", "min", "max", "team")
    description: str = ""

    @property
    def supports_coevolution(self) -> bool:
        return self.coevolution_optimizer is not None

    def hyperparameter(self, name: str) -> Optional[ParamSpec]:
        """Return the :class:`ParamSpec` named ``name`` (or ``None``)."""
        for spec in self.hyperparameters:
            if spec.name == name:
                return spec
        return None

    def to_dict(self, *, available: Optional[bool] = None) -> Dict[str, Any]:
        """Return a JSON description, optionally stamping ``available``."""
        data: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "aliases": list(self.aliases),
            "hyperparameters": [p.to_dict() for p in self.hyperparameters],
            "capabilities": dict(self.capabilities),
            "coevolution_optimizer": self.coevolution_optimizer,
            "supports_coevolution": self.supports_coevolution,
            "required_modules": list(self.required_modules),
            "required_extras": list(self.required_extras),
            "supported_budget_modes": list(self.supported_budget_modes),
            "supported_aggregations": list(self.supported_aggregations),
            "description": self.description,
        }
        if available is not None:
            data["available"] = available
        return data


@dataclass(frozen=True)
class EnvironmentSpec:
    """Catalog description of one registered environment.

    Wraps the lower-level
    :class:`~deap_pettingzoo.environments.registry.EnvSpec` with the
    declarative metadata the UI needs: a typed ``env_config`` schema and
    explicit team structure.

    Attributes:
        name: Canonical environment identifier (``"simple_spread_v3"``).
        label: Human-readable display name.
        env_config: Ordered :class:`ParamSpec` tuple for the environment
            constructor kwargs.
        teams: Declared teams; ``len(teams) >= 2`` ⇒ multi-team ⇒
            co-evolution-capable.
        api: The registry ``EnvApi`` string (``"parallel"`` /
            ``"aec_masked"`` / ...).
        continuous_default: Whether the env defaults to Box actions.
        tags: Registry tags (``{"mpe2"}`` / ``{"aquarium"}`` ...).
        default_episode_len: Budget-translation episode length.
        notes: Free-form compatibility notes.
        required_modules: Modules gating availability (e.g.
            ``("marl_aquarium",)``); empty ⇒ core.
        required_extras: pip extras providing them.
    """

    name: str
    label: str
    env_config: Tuple[ParamSpec, ...] = ()
    teams: Tuple[TeamSpec, ...] = ()
    api: str = "parallel"
    continuous_default: bool = False
    tags: Tuple[str, ...] = ()
    default_episode_len: int = 25
    notes: str = ""
    required_modules: Tuple[str, ...] = ()
    required_extras: Tuple[str, ...] = ()

    @property
    def multi_team(self) -> bool:
        """``True`` when the env declares two or more teams."""
        return len(self.teams) >= 2

    @property
    def is_turn_based(self) -> bool:
        return self.api in ("aec_masked", "aec_sequential")

    def env_param(self, name: str) -> Optional[ParamSpec]:
        for spec in self.env_config:
            if spec.name == name:
                return spec
        return None

    def to_dict(self, *, available: Optional[bool] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "env_config": [p.to_dict() for p in self.env_config],
            "teams": [t.to_dict() for t in self.teams],
            "multi_team": self.multi_team,
            "api": self.api,
            "is_turn_based": self.is_turn_based,
            "continuous_default": self.continuous_default,
            "tags": list(self.tags),
            "default_episode_len": self.default_episode_len,
            "notes": self.notes,
            "required_modules": list(self.required_modules),
            "required_extras": list(self.required_extras),
        }
        if available is not None:
            data["available"] = available
        return data
