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

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BaselineEvaluator,
    BudgetMode,
    RecordingConfig,
    TrainingConfig,
    TrainingResult,
    EvalResult,
    ComparisonConfig,
    PolicyProtocol,
)
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.baselines.agilerl_trainer import (
    AgileRLTrainer,
    MADDPGTrainer,
    MATD3Trainer,
    IPPOTrainer,
    GroupedIPPOTrainer,
)
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.baselines.comparison import (
    compare_baselines,
    register_trainer,
    get_trainer,
)
from deap_pettingzoo.baselines.compatibility import (
    EnvironmentInfo,
    CompatibilityResult,
    inspect_environment,
    check_compatibility,
    validate_compatibility,
    build_compatibility_matrix,
    get_compatible_algorithms,
    format_compatibility_matrix,
)
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
    OpponentPool,
    PopulationPool,
    ArchivePool,
    FrozenPolicyPool,
    CompositePool,
    HallOfFame,
    RingHallOfFame,
    ElitistHallOfFame,
    uniform_sampler,
    latest_sampler,
    fitness_proportional_sampler,
    tournament_sampler,
)

__all__ = [
    "BaselineTrainer",
    "BaselineEvaluator",
    "BudgetMode",
    "RecordingConfig",
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
]
