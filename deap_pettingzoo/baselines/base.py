"""
Base abstractions and data models for the unified baseline API.

This module defines the core contracts for training algorithms (EA, MARL, etc.)
and evaluating their resulting policies under a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Union
import numpy as np


class BudgetMode(Enum):
    """Budget constraint mode for training."""
    ENV_STEPS = "env_steps"
    EVAL_EPISODES = "eval_episodes"
    WALL_CLOCK = "wall_clock"


@dataclass
class RecordingConfig:
    """Opt-in frame-recording configuration nested on :class:`TrainingConfig`.

    When ``enabled=True``, supported trainers run one extra champion-rollout
    episode per recording cadence and pipe ``env.render('rgb_array')`` frames
    through :class:`~deap_pettingzoo.utils.logging.replay.FrameRecorder`,
    producing ``<run_dir>/frames/{episode_prefix}episode_NNNN.{mp4,gif}``.

    Attributes:
        enabled: Master on/off switch. When False all trainers skip recording.
        every_n_generations: Recording cadence in generations / iterations.
            ``None`` defers to the trainer's class-level
            ``_default_recording_every_n`` (1 for EA / CMA-ES / Coevolution,
            10 for AgileRL family).
        fps: Frames-per-second for the encoded artefact.
        fmt: Container format. ``"mp4"`` (requires ``imageio[ffmpeg]``) or
            ``"gif"`` (base ``imageio``). ``mp4`` falls back to ``gif`` if
            ffmpeg is unavailable.
        max_frames_per_episode: Hard cap on captured frames per episode
            (``None`` = no cap).
        episode_prefix: Filename prefix written before ``episode_NNNN``;
            trainers default to ``"train_"`` so they don't collide with
            future ``"live_"`` rollout captures sharing the same
            ``frames/`` directory.
    """
    enabled: bool = False
    every_n_generations: Optional[int] = None
    fps: int = 20
    fmt: str = "mp4"
    max_frames_per_episode: Optional[int] = None
    episode_prefix: str = ""

    def __post_init__(self) -> None:
        if self.every_n_generations is not None and self.every_n_generations <= 0:
            raise ValueError(
                "RecordingConfig.every_n_generations must be positive or None, "
                f"got {self.every_n_generations}"
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
        # Resolve the ``continuous`` shortcut against any explicit
        # ``env_config["continuous_actions"]`` the caller supplied.
        existing = self.env_config.get("continuous_actions")
        if self.continuous:
            if existing is False:
                raise ValueError(
                    "TrainingConfig: continuous=True conflicts with "
                    "env_config['continuous_actions']=False. Pick one."
                )
            self.env_config["continuous_actions"] = True
        elif existing is True:
            # Keep env_config authoritative but reflect it on the flag
            # so downstream introspection stays consistent.
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
    """
    algorithm: str
    policy: PolicyProtocol
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    training_time: Optional[float] = None


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

    #: Default recording cadence (in generations / iterations) used when
    #: ``TrainingConfig.recording.every_n_generations`` is ``None``.
    #: EA / CMA-ES / Coevolution leave this at 1 (record every generation);
    #: AgileRL-family trainers override to 10 since they iterate hundreds of
    #: times per run.
    _default_recording_every_n: ClassVar[int] = 1

    def _resolve_recording_every_n(self, config: TrainingConfig) -> int:
        """Resolve the recording cadence honouring the trainer-aware default.

        User-supplied ``config.recording.every_n_generations`` always wins;
        ``None`` falls back to the class-level
        :attr:`_default_recording_every_n`.
        """
        explicit = config.recording.every_n_generations
        if explicit is not None:
            return int(explicit)
        return int(type(self)._default_recording_every_n)

    @abstractmethod
    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train the algorithm with the given configuration.
        
        Args:
            config: Training configuration including budget and hyperparameters.
            
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
