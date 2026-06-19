"""
Base abstractions and data models for the unified baseline API.

This module defines the core contracts for training algorithms (EA, MARL, etc.)
and evaluating their resulting policies under a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
)
import warnings

import numpy as np


class BudgetMode(Enum):
    """Budget constraint mode for training."""
    ENV_STEPS = "env_steps"
    EVAL_EPISODES = "eval_episodes"
    WALL_CLOCK = "wall_clock"


class RecordingMode(str, Enum):
    """Champion-rollout recording cadence policy (algorithm-agnostic).

    The cadence is expressed once, in terms every trainer shares — the
    normalized training progress (:attr:`ProgressEvent.fraction`) and the
    cumulative env-step count (:attr:`ProgressEvent.env_steps`) — so EA
    "generations" and RL "iterations" behave identically.

    * ``COUNT`` — record :attr:`RecordingConfig.max_recordings` videos spread
      evenly across the run (the default).
    * ``EVERY`` — record on **every** generation/iteration, with no cap.
    * ``INTERVAL`` — record every :attr:`RecordingConfig.every_n_steps` env steps.
    * ``FINAL`` — record only the single end-of-run champion rollout.

    A ``str`` enum so it serializes transparently to/from the catalog JSON,
    experiment spec, and REST payloads.
    """

    COUNT = "count"
    EVERY = "every"
    INTERVAL = "interval"
    FINAL = "final"


@dataclass
class RecordingConfig:
    """Opt-in frame-recording configuration nested on :class:`TrainingConfig`.

    When ``enabled=True``, supported trainers run one extra champion-rollout
    episode per recording cadence and pipe ``env.render('rgb_array')`` frames
    through :class:`~deap_pettingzoo.utils.logging.replay.FrameRecorder`,
    producing ``<run_dir>/frames/{episode_prefix}episode_NNNN.{mp4,gif}``.

    Attributes:
        enabled: Master on/off switch. When False all trainers skip recording.
        mode: Cadence policy (:class:`RecordingMode`), uniform across all
            algorithm families. ``COUNT`` (default) records
            :attr:`max_recordings` videos spread evenly over the run; ``EVERY``
            records every generation/iteration with no cap; ``INTERVAL`` records
            every :attr:`every_n_steps` env steps; ``FINAL`` records only the
            end-of-run champion. See :class:`~deap_pettingzoo.baselines.recording.RecordingController`.
        every_n_steps: Env-step interval used by ``INTERVAL`` mode (applies to
            every algorithm, since all trainers report cumulative env steps).
        fps: Frames-per-second for the encoded artefact.
        fmt: Container format. ``"mp4"`` (requires ``imageio[ffmpeg]``) or
            ``"gif"`` (base ``imageio``). ``mp4`` falls back to ``gif`` if
            ffmpeg is unavailable.
        max_frames_per_episode: Hard cap on captured frames per episode
            (``None`` = no cap).
        max_dim: When set, frames are downscaled (preserving aspect ratio)
            so that their longer dimension is at most ``max_dim`` pixels.
            Useful for very large render surfaces (e.g. 2000×2000 aquarium)
            that exceed H.264 level limits.
        episode_prefix: Filename prefix written before ``episode_NNNN``;
            trainers default to ``"train_"`` so they don't collide with
            future ``"live_"`` rollout captures sharing the same
            ``frames/`` directory.
    """
    enabled: bool = False
    mode: RecordingMode = RecordingMode.COUNT
    every_n_steps: Optional[int] = None
    fps: int = 20
    fmt: str = "mp4"
    max_frames_per_episode: Optional[int] = None
    max_dim: Optional[int] = None
    episode_prefix: str = ""
    #: Number of champion-rollout recordings to produce over a whole run when
    #: :attr:`mode` is ``COUNT``: the first is allocated to the start of the
    #: run, the last to the end, and any remaining are spread evenly in
    #: between. ``1`` records only the final rollout.
    max_recordings: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RecordingMode):
            self.mode = RecordingMode(str(self.mode))
        if self.every_n_steps is not None and self.every_n_steps <= 0:
            raise ValueError(
                "RecordingConfig.every_n_steps must be positive or None, "
                f"got {self.every_n_steps}"
            )
        if self.fps <= 0:
            raise ValueError(f"RecordingConfig.fps must be positive, got {self.fps}")
        fmt = str(self.fmt).lower().lstrip(".")
        if fmt not in {"mp4", "gif"}:
            raise ValueError(
                f"RecordingConfig.fmt must be 'mp4' or 'gif', got {self.fmt!r}"
            )
        self.fmt = fmt
        if (
            self.max_frames_per_episode is not None
            and self.max_frames_per_episode <= 0
        ):
            raise ValueError(
                "RecordingConfig.max_frames_per_episode must be positive or None, "
                f"got {self.max_frames_per_episode}"
            )
        if self.max_dim is not None and self.max_dim <= 0:
            raise ValueError(
                "RecordingConfig.max_dim must be positive or None, "
                f"got {self.max_dim}"
            )
        if self.max_recordings is not None and self.max_recordings < 1:
            raise ValueError(
                "RecordingConfig.max_recordings must be >= 1 or None, "
                f"got {self.max_recordings}"
            )
        if self.mode is RecordingMode.INTERVAL and self.every_n_steps is None:
            raise ValueError(
                "RecordingConfig.mode='interval' requires every_n_steps to be set."
            )


@dataclass
class FitnessConfig:
    """Fitness-aggregation configuration nested on :class:`TrainingConfig`.

    Population-based trainers (EA / CMA-ES / PSO) collapse the per-agent
    rewards of a rollout into a single scalar fitness via
    :class:`~deap_pettingzoo.evaluators.SingleEpisodeEvaluator`. Historically
    every trainer hard-coded ``'sum'``; this config surfaces the other
    aggregation methods (``mean`` / ``min`` / ``max`` / ``team``) through the
    unified API so the experiment-setup UI can configure them.

    Attributes:
        aggregation_method: One of ``SingleEpisodeEvaluator.VALID_AGGREGATIONS``
            (``'sum'`` / ``'mean'`` / ``'min'`` / ``'max'`` / ``'team'``).
        team_mapping: Required for ``'team'`` aggregation — maps each agent id
            to a team name. ``None`` for the scalar methods.
    """

    aggregation_method: str = "sum"
    team_mapping: Optional[Dict[str, str]] = None

    _VALID: ClassVar[frozenset] = frozenset(
        {"sum", "mean", "min", "max", "team"}
    )

    def __post_init__(self) -> None:
        if self.aggregation_method not in self._VALID:
            raise ValueError(
                f"FitnessConfig.aggregation_method must be one of "
                f"{sorted(self._VALID)}, got {self.aggregation_method!r}"
            )
        if self.aggregation_method == "team" and not self.team_mapping:
            raise ValueError(
                "FitnessConfig.aggregation_method='team' requires a non-empty "
                "team_mapping (agent_id -> team_name)."
            )


@dataclass
class ProgressEvent:
    """A single per-iteration training-progress snapshot.

    Emitted by trainers through the optional ``progress_callback`` so the
    subprocess runner / FastAPI service can stream live progress to the UI
    queue page. Deliberately logger-independent: it is built from in-loop
    variables and fires even when ``output_dir is None`` (no run directory).

    Attributes:
        unit: ``"generation"`` for population-based trainers, ``"env_step"``
            for the step-based AgileRL trainers.
        current: Current 1-based generation (or completed env-step count).
        total: Total generations (or budgeted env steps).
        env_steps: Cumulative env steps consumed so far — the
            algorithm-agnostic progress axis. Generation trainers fill
            ``generation * env_steps_per_iteration``; step-based trainers fill
            their running step count. Consumed by the recording controller and
            written to the replay sidecar.
        best_fitness: Best fitness / return so far (``None`` if unknown).
        mean_fitness: Population-mean fitness this iteration (``None`` if N/A).
        wall_time_s: Seconds elapsed since training started.
        extra: Free-form trainer-specific extras (e.g. ``{"side": "..."}``).
    """

    unit: str
    current: int
    total: int
    env_steps: int = 0
    best_fitness: Optional[float] = None
    mean_fitness: Optional[float] = None
    wall_time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        """Completion in ``[0, 1]`` (0 when ``total <= 0``)."""
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.total))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "current": int(self.current),
            "total": int(self.total),
            "env_steps": int(self.env_steps),
            "fraction": self.fraction,
            "best_fitness": (
                None if self.best_fitness is None else float(self.best_fitness)
            ),
            "mean_fitness": (
                None if self.mean_fitness is None else float(self.mean_fitness)
            ),
            "wall_time_s": float(self.wall_time_s),
            "extra": dict(self.extra),
        }


#: Type alias for the optional per-iteration progress hook accepted by
#: :meth:`BaselineTrainer.train`.
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass
class TrainingConfig:
    """Configuration for training a baseline algorithm.
    
    Attributes:
        env_name: PettingZoo environment name (e.g., 'simple_spread_v3').
        budget_mode: Primary budget constraint mode.
        budget_value: Budget amount in units specified by budget_mode.
        num_seeds: Number of independent training runs with different seeds.
        algorithm_params: Algorithm-specific hyperparameters.
        env_config: Optional kwargs forwarded to the underlying
            ``parallel_env(...)`` constructor (e.g. ``max_cycles`` for
            MPE2, ``max_time_steps``/``prey_count`` for aquarium_v0).
        continuous: Ergonomic shortcut. When ``True``, merges
            ``{"continuous_actions": True}`` into ``env_config`` at
            construction time (MPE2 reads this kwarg to switch between
            the ``Discrete`` and ``Box`` action variants). If the caller
            has also set ``env_config["continuous_actions"]`` explicitly
            and it disagrees with ``continuous``, a ``ValueError`` is
            raised rather than silently preferring one.
        output_dir: Directory for saving checkpoints and logs.
        render: Whether to render episodes during training.
    """
    env_name: str
    budget_mode: BudgetMode
    budget_value: int
    num_seeds: int = 1
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    env_config: Dict[str, Any] = field(default_factory=dict)
    continuous: bool = False
    output_dir: Optional[Path] = None
    render: bool = False
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    def __post_init__(self):
        """Validate configuration."""
        if self.budget_value <= 0:
            raise ValueError(f"budget_value must be positive, got {self.budget_value}")
        if self.num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {self.num_seeds}")
        if isinstance(self.budget_mode, str):
            self.budget_mode = BudgetMode(self.budget_mode)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        # Resolve the ``continuous`` shortcut against any explicit ``env_config["continuous_actions"]`` the caller supplied.
        existing = self.env_config.get("continuous_actions")
        if self.continuous:
            if existing is False:
                raise ValueError(
                    "TrainingConfig: continuous=True conflicts with "
                    "env_config['continuous_actions']=False. Pick one."
                )
            self.env_config["continuous_actions"] = True
        elif existing is True:
            self.continuous = True


class PolicyProtocol(Protocol):
    """Protocol for trained policies that can be evaluated.
    
    Both EA genomes (with FeedforwardNeuralPolicy wrapper) and RL networks
    should satisfy this interface.
    """

    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        """Get action for the given observation.

        Args:
            observation: Environment observation vector.
            action_mask: Optional legal-action mask surfaced by
                turn-based AEC adapters (populated from
                ``infos[agent]["action_mask"]``). Implementations that
                do not support masking should accept and ignore this
                keyword.
        """
        ...
    
    def reset(self) -> None:
        """Reset internal state (for recurrent policies)."""
        ...


@dataclass
class TrainingResult:
    """Result of training a baseline algorithm.
    
    Attributes:
        algorithm: Name of the algorithm used.
        policy: Trained policy satisfying PolicyProtocol.
        metrics: Training metrics (e.g., fitness curves, episode rewards).
        metadata: Additional algorithm-specific metadata.
        seed: Random seed used for this training run.
        training_time: Wall-clock time in seconds.
        pareto_front: For multi-objective runs (NSGA-II / SPEA2), the
            nondominated individuals from the final population.
            ``None`` for single-objective runs.
    """
    algorithm: str
    policy: PolicyProtocol
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    training_time: Optional[float] = None
    pareto_front: Optional[List[Any]] = None

    @property
    def best_policy(self) -> PolicyProtocol:
        """Alias for :attr:`policy` (backwards-compatible name)."""
        return self.policy


@dataclass
class EvalResult:
    """Result of evaluating a policy.
    
    Attributes:
        mean_reward: Mean episode reward across evaluation episodes.
        std_reward: Standard deviation of episode rewards.
        min_reward: Minimum episode reward.
        max_reward: Maximum episode reward.
        num_episodes: Number of episodes evaluated.
        episode_rewards: List of individual episode rewards.
        episode_lengths: List of episode lengths.
    """
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    num_episodes: int
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean_reward': float(self.mean_reward),
            'std_reward': float(self.std_reward),
            'min_reward': float(self.min_reward),
            'max_reward': float(self.max_reward),
            'num_episodes': self.num_episodes,
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'episode_lengths': [int(l) for l in self.episode_lengths],
        }


@dataclass
class ComparisonConfig:
    """Configuration for comparing multiple baseline algorithms.
    
    Attributes:
        algorithms: List of algorithm names to compare.
        env_name: PettingZoo environment name.
        budget_mode: Primary budget constraint mode.
        budget_value: Budget amount in units specified by budget_mode.
        num_seeds: Number of independent runs per algorithm.
        eval_episodes: Number of episodes for final evaluation.
        algorithm_configs: Algorithm-specific configurations (nested dicts).
        output_dir: Directory for saving results.
    """
    algorithms: List[str]
    env_name: str
    budget_mode: Union[BudgetMode, str]
    budget_value: int
    num_seeds: int = 3
    eval_episodes: int = 100
    algorithm_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.algorithms:
            raise ValueError("algorithms list cannot be empty")
        if self.budget_value <= 0:
            raise ValueError(f"budget_value must be positive, got {self.budget_value}")
        if self.num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {self.num_seeds}")
        if self.eval_episodes <= 0:
            raise ValueError(f"eval_episodes must be positive, got {self.eval_episodes}")
        if isinstance(self.budget_mode, str):
            self.budget_mode = BudgetMode(self.budget_mode)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)


class BaselineTrainer(ABC):
    """Abstract base class for training baseline algorithms.
    
    Subclasses implement specific training approaches (EA, MARL, etc.)
    while conforming to a unified interface.
    """

    #: Set by :meth:`train` to a
    #: :class:`~deap_pettingzoo.baselines.recording.RecordingController`.
    #: ``None`` (default) means no recording wiring; :meth:`_on_progress`
    #: then only emits progress.
    _recorder: ClassVar[Any] = None

    def _on_progress(
        self,
        callback: Optional[ProgressCallback],
        event: ProgressEvent,
        champion: Optional[Callable[[], Any]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Single per-iteration hook: emit progress **and** maybe record.

        Trainers call this wherever they previously emitted a
        :class:`ProgressEvent`. When a :class:`RecordingController` has been
        attached to ``self._recorder`` and a ``champion`` provider is given,
        the controller decides — from the run's :class:`RecordingMode` — whether
        this ``event`` warrants a champion-rollout recording. ``champion`` is
        called lazily (only when a recording is actually due) and returns the
        policy (or per-agent policy map) to roll out, or ``None`` to skip.
        """
        self._emit_progress(callback, event)
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and champion is not None:
            recorder.observe(event, champion, extra=extra)

    @staticmethod
    def _emit_progress(
        callback: Optional[ProgressCallback],
        event: ProgressEvent,
    ) -> None:
        """Invoke ``callback`` with ``event``, swallowing any exception.

        Progress reporting must never be able to crash a long training run, so
        a misbehaving callback only raises a :class:`RuntimeWarning`. Mirrors
        the error-isolation idiom used by the co-evolution generation hook.
        """
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001 - intentional isolation
            warnings.warn(
                f"progress_callback raised {exc!r}; ignoring this update",
                RuntimeWarning,
                stacklevel=2,
            )

    @abstractmethod
    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Train the algorithm with the given configuration.

        Args:
            config: Training configuration including budget and hyperparameters.
            progress_callback: Optional hook invoked once per generation /
                progress interval with a :class:`ProgressEvent`. Additive and
                error-isolated (see :meth:`_emit_progress`); ``None`` disables
                progress reporting. Not every trainer emits progress — those
                that do are the standard UI-exposed algorithms.

        Returns:
            TrainingResult containing the trained policy and metrics.
        """
        pass
    
    @abstractmethod
    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """Check if this trainer supports the given budget mode.
        
        Args:
            mode: Budget mode to check.
            
        Returns:
            True if supported, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get algorithm capabilities metadata.
        
        Returns:
            Dictionary with keys:
            - 'discrete_action': bool (supports discrete action spaces)
            - 'continuous_action': bool (supports continuous action spaces)
            - 'multidiscrete_action': bool (supports MultiDiscrete action
              spaces; defaults to False when omitted, so MultiDiscrete envs
              are rejected unless the algorithm opts in)
            - 'needs_global_state': bool (requires centralized state)
            - 'on_policy': bool (on-policy vs off-policy)
            - 'supports_turn_based': bool (default True; set False for
              algorithms like MADDPG/MATD3 whose buffer / critic
              structure assumes simultaneous-step execution and thus
              cannot train on turn-based AEC adapters).
        """
        pass


class BaselineEvaluator(ABC):
    """Abstract base class for evaluating trained policies.
    
    Provides a unified interface for policy evaluation regardless of
    the training algorithm used.
    """
    
    @abstractmethod
    def evaluate(
        self,
        policy: PolicyProtocol,
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False
    ) -> EvalResult:
        """Evaluate a policy on the given environment.
        
        Args:
            policy: Trained policy satisfying PolicyProtocol.
            env_name: PettingZoo environment name.
            num_episodes: Number of episodes to evaluate.
            seed: Random seed for reproducibility.
            render: Whether to render episodes.
            
        Returns:
            EvalResult with reward statistics.
        """
        pass
