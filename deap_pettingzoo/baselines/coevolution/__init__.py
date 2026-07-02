"""Co-evolution baselines — multi-population evolution in one trainer.

Public API:

* :class:`CoEvolutionTrainer` / :class:`CoEvolutionConfig` — top-level
  entry point for competitive / self-play / cooperative co-evolution.
* :class:`Side` — declarative per-population config.
* :class:`SideState` (+ :class:`EASideState`, :class:`CMAESSideState`,
  :class:`FrozenSideState`) — runtime state and evolution loop for one
  side.
* :class:`OpponentPool` Protocol + :class:`PopulationPool`,
  :class:`ArchivePool`, :class:`FrozenPolicyPool`,
  :class:`CompositePool` — source-agnostic opponent sampling surface.
  Custom implementations (e.g. backed by a QDax repertoire) only need
  to satisfy the Protocol.
* :class:`HallOfFame` (+ :class:`RingHallOfFame`,
  :class:`ElitistHallOfFame`) — opponent archives for self-play.
* Samplers: :func:`uniform_sampler`, :func:`latest_sampler`,
  :func:`fitness_proportional_sampler`, :func:`tournament_sampler`.
"""

from .archive import ElitistHallOfFame, HallOfFame, RingHallOfFame
from .evaluator import CoEvolutionEvalResult, CoEvolutionEvaluator
from .pools import (
    ArchivePool,
    CompositePool,
    FrozenPolicyPool,
    OpponentPool,
    PolicyFactory,
    PopulationPool,
    RLPolicyPool,
)
from .rl_side import SUPPORTED_RL_ALGORITHMS, RLSideState
from .samplers import (
    Sampler,
    fitness_proportional_sampler,
    latest_sampler,
    tournament_sampler,
    uniform_sampler,
)
from .sides import (
    CMAESSideState,
    EASideState,
    FrozenSideState,
    GenerationStats,
    Side,
    SideState,
    make_side_state,
    register_side_state,
)
from .trainer import (
    VALID_MODES,
    VALID_SCHEDULES,
    CoEvolutionConfig,
    CoEvolutionTrainer,
)

__all__ = [
    # Trainer
    "CoEvolutionTrainer",
    "CoEvolutionConfig",
    "VALID_MODES",
    "VALID_SCHEDULES",
    # Sides
    "Side",
    "SideState",
    "EASideState",
    "CMAESSideState",
    "FrozenSideState",
    "RLSideState",
    "SUPPORTED_RL_ALGORITHMS",
    "GenerationStats",
    "make_side_state",
    "register_side_state",
    # Pools
    "OpponentPool",
    "PopulationPool",
    "ArchivePool",
    "FrozenPolicyPool",
    "CompositePool",
    "RLPolicyPool",
    "PolicyFactory",
    # Archives
    "HallOfFame",
    "RingHallOfFame",
    "ElitistHallOfFame",
    # Samplers
    "Sampler",
    "uniform_sampler",
    "latest_sampler",
    "fitness_proportional_sampler",
    "tournament_sampler",
    # Evaluator
    "CoEvolutionEvaluator",
    "CoEvolutionEvalResult",
]
