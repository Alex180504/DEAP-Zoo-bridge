"""Baseline comparisons with reinforcement learning algorithms.

This module provides a unified API for training and comparing different
baseline algorithms (EA, MARL, etc.) under fair budget constraints.

Supported algorithms:
- EA     (Evolutionary Algorithm via DEAP)
- MADDPG (Multi-Agent DDPG via AgileRL)
- MATD3  (Multi-Agent Twin-Delayed DDPG via AgileRL)
- IPPO   (Independent PPO via AgileRL)
- MAPPO  (Multi-Agent PPO via AgileRL)
"""

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BaselineEvaluator,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
    EvalResult,
    ComparisonConfig,
    PolicyProtocol,
)
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.baselines.agilerl_trainer import (
    AgileRLTrainer,
    MADDPGTrainer,
    MATD3Trainer,
    IPPOTrainer,
    MAPPOTrainer,
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

__all__ = [
    "BaselineTrainer",
    "BaselineEvaluator",
    "BudgetMode",
    "TrainingConfig",
    "TrainingResult",
    "EvalResult",
    "ComparisonConfig",
    "PolicyProtocol",
    "EATrainer",
    "CMAESTrainer",
    "AgileRLTrainer",
    "MADDPGTrainer",
    "MATD3Trainer",
    "IPPOTrainer",
    "MAPPOTrainer",
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
]
