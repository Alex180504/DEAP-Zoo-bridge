"""Baseline comparisons with reinforcement learning algorithms.

This module provides a unified API for training and comparing different
baseline algorithms (EA, MARL, etc.) under fair budget constraints.

Supported algorithms:
- EA     (Evolutionary Algorithm via DEAP)
- MADDPG (Multi-Agent DDPG via AgileRL)
- MATD3  (Multi-Agent Twin-Delayed DDPG via AgileRL)
- IPPO   (Independent PPO via AgileRL)
- IPPO-Grouped (IPPO with AgileRL prefix-based agent grouping)
"""

from deap_pettingzoo.baselines.agilerl_hpo_trainer import (
    EvolveHPOTrainer,
    MADDPGHPOTrainer,
    MATD3HPOTrainer,
)
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.baselines.agilerl_trainer import (
    AgileRLTrainer,
    GroupedIPPOTrainer,
    IPPOTrainer,
    MADDPGTrainer,
    MATD3Trainer,
)
from deap_pettingzoo.baselines.base import (
    BaselineEvaluator,
    BaselineTrainer,
    BudgetMode,
    ComparisonConfig,
    EvalResult,
    FitnessConfig,
    PolicyProtocol,
    ProgressCallback,
    ProgressEvent,
    RecordingConfig,
    RecordingMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.coevolution import (
    ArchivePool,
    CoEvolutionConfig,
    CoEvolutionTrainer,
    CompositePool,
    ElitistHallOfFame,
    FrozenPolicyPool,
    HallOfFame,
    OpponentPool,
    PopulationPool,
    RingHallOfFame,
    Side,
    fitness_proportional_sampler,
    latest_sampler,
    tournament_sampler,
    uniform_sampler,
)
from deap_pettingzoo.baselines.comparison import (
    compare_baselines,
    get_trainer,
    register_trainer,
)
from deap_pettingzoo.baselines.compatibility import (
    CompatibilityResult,
    EnvironmentInfo,
    build_compatibility_matrix,
    check_compatibility,
    format_compatibility_matrix,
    get_compatible_algorithms,
    inspect_environment,
    validate_compatibility,
)
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.baselines.experiment_spec import (
    BudgetSpec,
    BuiltPlan,
    CoevolutionSettings,
    CoevolutionSideSpec,
    ExperimentSpec,
    FitnessSettings,
    ParallelismSettings,
    RecordingSettings,
    SubRun,
    build_configs,
    deep_merge,
    validate_spec,
)
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator

__all__ = [
    "BaselineTrainer",
    "BaselineEvaluator",
    "BudgetMode",
    "RecordingConfig",
    "RecordingMode",
    "FitnessConfig",
    "ProgressEvent",
    "ProgressCallback",
    "TrainingConfig",
    "TrainingResult",
    "EvalResult",
    "ComparisonConfig",
    "PolicyProtocol",
    "EATrainer",
    "CMAESTrainer",
    "PSOTrainer",
    "AgileRLTrainer",
    "MADDPGTrainer",
    "MATD3Trainer",
    "IPPOTrainer",
    "GroupedIPPOTrainer",
    "EvolveHPOTrainer",
    "MADDPGHPOTrainer",
    "MATD3HPOTrainer",
    "AgileRLPolicyWrapper",
    "UnifiedEvaluator",
    "compare_baselines",
    "register_trainer",
    "get_trainer",
    "EnvironmentInfo",
    "CompatibilityResult",
    "inspect_environment",
    "check_compatibility",
    "validate_compatibility",
    "build_compatibility_matrix",
    "get_compatible_algorithms",
    "format_compatibility_matrix",
    # Co-evolution
    "CoEvolutionConfig",
    "CoEvolutionTrainer",
    "Side",
    "OpponentPool",
    "PopulationPool",
    "ArchivePool",
    "FrozenPolicyPool",
    "CompositePool",
    "HallOfFame",
    "RingHallOfFame",
    "ElitistHallOfFame",
    "uniform_sampler",
    "latest_sampler",
    "fitness_proportional_sampler",
    "tournament_sampler",
    # Experiment spec (UI overhaul)
    "ExperimentSpec",
    "BudgetSpec",
    "FitnessSettings",
    "RecordingSettings",
    "ParallelismSettings",
    "CoevolutionSettings",
    "CoevolutionSideSpec",
    "SubRun",
    "BuiltPlan",
    "build_configs",
    "validate_spec",
    "deep_merge",
]
