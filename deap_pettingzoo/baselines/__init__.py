"""Baseline comparisons with reinforcement learning algorithms.

This module provides a unified API for training and comparing different
baseline algorithms (EA, MARL, etc.) under fair budget constraints.

Core abstractions:
- BaselineTrainer: Algorithm-specific training interface
- BaselineEvaluator: Unified policy evaluation interface
- Data models: TrainingConfig, TrainingResult, EvalResult, ComparisonConfig

Supported algorithms:
- EA (Evolutionary Algorithm via DEAP)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient via cleanMARL)
- Future: IPPO, MAPPO, QMIX, etc.
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
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.baselines.comparison import (
    compare_baselines,
    register_trainer,
    get_trainer,
)

__all__ = [
    'BaselineTrainer',
    'BaselineEvaluator',
    'BudgetMode',
    'TrainingConfig',
    'TrainingResult',
    'EvalResult',
    'ComparisonConfig',
    'PolicyProtocol',
    'EATrainer',
    'UnifiedEvaluator',
    'compare_baselines',
    'register_trainer',
    'get_trainer',
]

__all__ = []
