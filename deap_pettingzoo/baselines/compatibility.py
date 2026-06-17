"""
Environment-algorithm compatibility checking and introspection.

This module provides utilities for:
- Inspecting PettingZoo environment properties (action spaces, state availability)
- Validating that a trainer's capabilities match an environment's requirements
- Building compatibility matrices across algorithms and environments
- Querying which registered algorithms are compatible with a given environment

Typical usage::

    from deap_pettingzoo.baselines.compatibility import (
        inspect_environment,
        validate_compatibility,
        build_compatibility_matrix,
        get_compatible_algorithms,
    )

    env_info = inspect_environment('simple_spread_v3')
    result = validate_compatibility(trainer, env_info.env_name, algorithm_name='ea')
    result.raise_if_incompatible()

    matrix = build_compatibility_matrix(
        algorithms=['ea', 'maddpg'],
        env_names=['simple_spread_v3', 'simple_tag_v3'],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from deap_pettingzoo.baselines.base import BaselineTrainer


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentInfo:
    """Introspected properties of a PettingZoo environment.

    Attributes:
        env_name: PettingZoo environment identifier.
        num_agents: Number of agents (at environment reset).
        agent_ids: Ordered list of agent identifiers.
        has_continuous_actions: True if any agent has a Box action space.
        has_discrete_actions: True if any agent has a Discrete action space.
        has_global_state: True if the environment exposes a ``state()`` method.
        obs_dim: Per-agent observation dimension (None if heterogeneous).
        act_dim: Per-agent action dimension (None if heterogeneous).
        action_low: Lower bound for continuous actions (first agent).
        action_high: Upper bound for continuous actions (first agent).
        extra: Any additional metadata captured during introspection.
    """

    env_name: str
    num_agents: int
    agent_ids: List[str]
    has_continuous_actions: bool
    has_discrete_actions: bool
    has_global_state: bool
    obs_dim: Optional[int]
    act_dim: Optional[int]
    action_low: Optional[np.ndarray]
    action_high: Optional[np.ndarray]
    #: Whether the env is a strictly turn-based AEC wrapped via:
    #: :class:`MaskedSimultaneousAdapter` or
    #: :class:`SequentialTurnAdapter`. Used to gate MADDPG/MATD3.
    is_turn_based: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            'env_name': self.env_name,
            'num_agents': self.num_agents,
            'agent_ids': self.agent_ids,
            'has_continuous_actions': self.has_continuous_actions,
            'has_discrete_actions': self.has_discrete_actions,
            'has_global_state': self.has_global_state,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'action_low': self.action_low.tolist() if self.action_low is not None else None,
            'action_high': self.action_high.tolist() if self.action_high is not None else None,
            'is_turn_based': self.is_turn_based,
            'extra': self.extra,
        }


@dataclass
class CompatibilityResult:
    """Result of a trainer-environment compatibility check.

    Attributes:
        compatible: Whether the algorithm can run on this environment.
        algorithm: Name of the algorithm checked.
        env_name: Name of the environment checked.
        issues: Fatal incompatibilities (prevent training).
        warnings: Non-fatal concerns (training may still proceed).
    """

    compatible: bool
    algorithm: str
    env_name: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_incompatible(self) -> None:
        """Raise ``ValueError`` if any fatal issues were detected.

        Raises:
            ValueError: Detailed message listing all incompatibilities.
        """
        if not self.compatible:
            issue_lines = "\n".join(f"  - {i}" for i in self.issues)
            raise ValueError(
                f"Algorithm '{self.algorithm}' is incompatible with "
                f"environment '{self.env_name}':\n{issue_lines}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            'compatible': self.compatible,
            'algorithm': self.algorithm,
            'env_name': self.env_name,
            'issues': self.issues,
            'warnings': self.warnings,
        }


# ---------------------------------------------------------------------------
# Environment introspection
# ---------------------------------------------------------------------------


def inspect_environment(env_name: str) -> EnvironmentInfo:
    """Introspect a PettingZoo environment and return its properties.

    Creates a temporary environment instance, queries agent/space metadata,
    and then closes it.  The environment is *not* stepped.

    Args:
        env_name: PettingZoo environment name (e.g., ``'simple_spread_v3'``).

    Returns:
        :class:`EnvironmentInfo` populated with the environment's properties.

    Raises:
        ValueError: If the environment name is not supported by
            :class:`~deap_pettingzoo.environments.PettingZooWrapper`.
    """
    from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper

    wrapper = PettingZooWrapper(env_name=env_name)
    env = wrapper.env

    # Use possible_agents (always available without reset) for introspection.
    # env.agents is only populated during an active episode.
    agents = list(env.possible_agents)
    num_agents = len(agents)

    has_continuous = False
    has_discrete = False
    obs_dims: set = set()
    act_dims: set = set()
    first_action_low: Optional[np.ndarray] = None
    first_action_high: Optional[np.ndarray] = None

    for agent in agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)

        # Observation dimension
        if hasattr(obs_space, 'shape') and obs_space.shape:
            obs_dims.add(int(np.prod(obs_space.shape)))

        # Action space type
        if hasattr(act_space, 'n'):
            # gymnasium.spaces.Discrete
            has_discrete = True
            act_dims.add(int(act_space.n))
        elif hasattr(act_space, 'shape') and act_space.shape:
            # gymnasium.spaces.Box
            has_continuous = True
            act_dims.add(int(np.prod(act_space.shape)))
            if first_action_low is None:
                first_action_low = act_space.low.copy()
                first_action_high = act_space.high.copy()

    # Global state: PettingZoo parallel envs expose .state() when centralised
    has_global_state = callable(getattr(env, 'state', None))

    # Turn-based detection — consult the registry spec.
    is_turn_based = False
    try:
        from deap_pettingzoo.environments.registry import get_env_spec
        is_turn_based = get_env_spec(env_name).api in (
            "aec_masked",
            "aec_sequential",
        )
    except ValueError:  # pragma: no cover - env_name always registered here
        pass

    # Use the wrapper's close() which guards against SystemExit raised by some third-party envs (e.g. marl-aquarium's aquarium_v0).
    wrapper.close()

    return EnvironmentInfo(
        env_name=env_name,
        num_agents=num_agents,
        agent_ids=agents,
        has_continuous_actions=has_continuous,
        has_discrete_actions=has_discrete,
        has_global_state=has_global_state,
        obs_dim=obs_dims.pop() if len(obs_dims) == 1 else None,
        act_dim=act_dims.pop() if len(act_dims) == 1 else None,
        action_low=first_action_low,
        action_high=first_action_high,
        is_turn_based=is_turn_based,
    )


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------


def check_compatibility(
    trainer: 'BaselineTrainer',
    env_info: EnvironmentInfo,
    algorithm_name: str = 'unknown',
) -> CompatibilityResult:
    """Check compatibility between a trainer and an :class:`EnvironmentInfo`.

    Compares the trainer's :meth:`~BaselineTrainer.get_capabilities` against
    the environment's introspected properties.  Does **not** create any
    environment instance.

    Args:
        trainer: Instantiated trainer to check.
        env_info: Pre-computed environment information.
        algorithm_name: Human-readable name used in error messages.

    Returns:
        :class:`CompatibilityResult` (always returned; never raises).
    """
    caps = trainer.get_capabilities()
    issues: List[str] = []
    warnings: List[str] = []

    # Critical checks

    # Action space compatibility
    if env_info.has_continuous_actions and not caps.get('continuous_action', False):
        issues.append(
            "Environment has continuous (Box) action spaces, but the algorithm "
            "only supports discrete actions."
        )

    if env_info.has_discrete_actions and not caps.get('discrete_action', False):
        issues.append(
            "Environment has discrete action spaces, but the algorithm "
            "only supports continuous actions."
        )

    # Global state requirement
    if caps.get('needs_global_state', False) and not env_info.has_global_state:
        issues.append(
            "Algorithm requires a centralised global state, but the environment "
            "does not expose a state() method."
        )

    # Turn-based carve-out: algorithms that assume simultaneous execution
    # (replay buffer / centralized critic keyed by all-agents-per-step) cannot train on turn-based AEC adapters.
    if env_info.is_turn_based and not caps.get('supports_turn_based', True):
        issues.append(
            "Environment is a turn-based AEC game (api='aec_masked' or "
            "'aec_sequential'), but the algorithm's replay buffer / critic "
            "structure assumes simultaneous-step execution. Use EA, CMA-ES, "
            "or IPPO for turn-based games."
        )

    # --- Warnings -----------------------------------------------------------

    if env_info.num_agents > 8:
        warnings.append(
            f"Environment has {env_info.num_agents} agents; scalability may be a concern."
        )

    if env_info.obs_dim is None:
        warnings.append(
            "Agents have heterogeneous observation spaces; the algorithm must "
            "handle variable-size inputs."
        )

    if env_info.act_dim is None:
        warnings.append(
            "Agents have heterogeneous action spaces; the algorithm must "
            "handle variable-size outputs."
        )

    return CompatibilityResult(
        compatible=len(issues) == 0,
        algorithm=algorithm_name,
        env_name=env_info.env_name,
        issues=issues,
        warnings=warnings,
    )


def validate_compatibility(
    trainer: 'BaselineTrainer',
    env_name: str,
    algorithm_name: str = 'unknown',
) -> CompatibilityResult:
    """Inspect ``env_name`` then check compatibility with ``trainer``.

    Convenience wrapper that calls :func:`inspect_environment` followed by
    :func:`check_compatibility`.

    Args:
        trainer: Instantiated trainer to validate.
        env_name: PettingZoo environment name.
        algorithm_name: Human-readable algorithm name for error messages.

    Returns:
        :class:`CompatibilityResult`.
    """
    env_info = inspect_environment(env_name)
    return check_compatibility(trainer, env_info, algorithm_name)


# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------


def build_compatibility_matrix(
    algorithms: List[str],
    env_names: List[str],
) -> Dict[str, Dict[str, CompatibilityResult]]:
    """Build a compatibility matrix for all algorithm-environment pairs.

    For each ``(algorithm, env_name)`` pair the function:

    1. Instantiates the trainer via :func:`~comparison.get_trainer`.
    2. Introspects the environment.
    3. Runs :func:`check_compatibility`.

    Errors during trainer instantiation or environment introspection are
    captured as incompatible results with a descriptive issue message rather
    than propagating as exceptions.

    Args:
        algorithms: List of registered algorithm names (e.g., ``['ea', 'maddpg']``).
        env_names: List of PettingZoo environment names.

    Returns:
        Nested dict ``{algorithm: {env_name: CompatibilityResult}}``.
    """
    from deap_pettingzoo.baselines.comparison import get_trainer

    # Pre-introspect all environments (cache to avoid redundant work)
    env_infos: Dict[str, Any] = {}
    for env_name in env_names:
        try:
            env_infos[env_name] = inspect_environment(env_name)
        except Exception as exc:
            env_infos[env_name] = exc

    matrix: Dict[str, Dict[str, CompatibilityResult]] = {}

    for algo in algorithms:
        matrix[algo] = {}
        for env_name in env_names:
            env_info = env_infos[env_name]

            # Environment introspection failed
            if isinstance(env_info, Exception):
                matrix[algo][env_name] = CompatibilityResult(
                    compatible=False,
                    algorithm=algo,
                    env_name=env_name,
                    issues=[f"Could not introspect environment: {env_info}"],
                )
                continue

            # Trainer instantiation may fail (e.g. missing dependencies)
            try:
                trainer = get_trainer(algo)
            except Exception as exc:
                matrix[algo][env_name] = CompatibilityResult(
                    compatible=False,
                    algorithm=algo,
                    env_name=env_name,
                    issues=[f"Could not instantiate trainer: {exc}"],
                )
                continue

            matrix[algo][env_name] = check_compatibility(trainer, env_info, algo)

    return matrix


def get_compatible_algorithms(
    env_name: str,
    registered_only: bool = True,
) -> List[str]:
    """Return algorithm names that are compatible with ``env_name``.

    Args:
        env_name: PettingZoo environment name.
        registered_only: If ``True`` (default), only consider algorithms in the
            trainer registry.  If ``False``, raises ``NotImplementedError``
            (custom discovery is not yet supported).

    Returns:
        Sorted list of compatible algorithm names.

    Raises:
        ValueError: If ``env_name`` cannot be introspected.
        NotImplementedError: If ``registered_only=False``.
    """
    if not registered_only:
        raise NotImplementedError(
            "Discovery outside the trainer registry is not yet supported."
        )

    from deap_pettingzoo.baselines.comparison import _TRAINER_REGISTRY, get_trainer

    env_info = inspect_environment(env_name)
    compatible: List[str] = []

    for algo in list(_TRAINER_REGISTRY.keys()):
        try:
            trainer = get_trainer(algo)
            result = check_compatibility(trainer, env_info, algo)
            if result.compatible:
                compatible.append(algo)
        except Exception:
            # Trainer cannot be instantiated — skip silently
            pass

    return sorted(compatible)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_compatibility_matrix(
    matrix: Dict[str, Dict[str, CompatibilityResult]],
    show_warnings: bool = False,
) -> str:
    """Render a compatibility matrix as a human-readable table.

    Args:
        matrix: Output of :func:`build_compatibility_matrix`.
        show_warnings: If ``True``, append warning counts to cells.

    Returns:
        Multi-line string table.
    """
    if not matrix:
        return "(empty matrix)"

    algos = list(matrix.keys())
    envs = list(next(iter(matrix.values())).keys())

    # Column widths
    algo_col = max(len(a) for a in algos + ['Algorithm'])
    env_col = max(max(len(e) for e in envs), 10)

    header = f"{'Algorithm':<{algo_col}}  " + "  ".join(f"{e:<{env_col}}" for e in envs)
    separator = "-" * len(header)
    lines = [header, separator]

    for algo in algos:
        cells = []
        for env in envs:
            result = matrix[algo][env]
            if result.compatible:
                cell = "YES"
                if show_warnings and result.warnings:
                    cell += f"({len(result.warnings)}w)"
            else:
                cell = "NO "
            cells.append(f"{cell:<{env_col}}")
        lines.append(f"{algo:<{algo_col}}  " + "  ".join(cells))

    return "\n".join(lines)
