"""Multi-objective fitness specification for DEAP evaluators.

Defines the data model for describing fitness objectives —
parsimony, reward-component decomposition, and lexicase
test-case vectors — and wire them through the existing
evaluator stack in a backward-compatible way.

Usage (single-objective, backward-compatible)::

    # No FitnessSpec — evaluators return 1-tuples exactly as before.
    evaluator = SingleEpisodeEvaluator(env_wrapper, max_steps=200)

Usage (multi-objective with parsimony)::

    from deap_pettingzoo.integration.objectives import (
        FitnessSpec, ObjectiveSpec, compute_parsimony,
    )
    spec = FitnessSpec(
        objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ],
    )
    evaluator = SingleEpisodeEvaluator(
        env_wrapper, max_steps=200, fitness_spec=spec,
    )
    # Returns (reward, -num_weights) — compatible with
    # creator.FitnessMax(weights=(1.0, -1.0))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveSpec:
    """One axis of a multi-objective fitness.

    Attributes:
        name: Human-readable label (e.g. ``"reward"``).
        weight: Sign and magnitude contributed to :class:`creator.FitnessMax`
            weights.  Positive = maximise; negative = minimise.
        source: Where the objective value comes from:

            * ``"reward"`` — aggregated reward from the evaluator
              (the only source for backward-compatible single-objective).
            * ``"parsimony"`` — negative weight count from the genome
              (favours smaller networks).
            * ``"reward_component"`` — a named slice of per-agent rewards
              extracted from the env info dict via
              :attr:`FitnessSpec.reward_components_fn`.

        component_key: When ``source == "reward_component"``, the key
            passed to :attr:`FitnessSpec.reward_components_fn` to
            extract this objective's scalar.  Ignored otherwise.
    """

    name: str
    weight: float = 1.0
    source: str = "reward"
    component_key: Optional[str] = None

    _VALID_SOURCES = frozenset({"reward", "parsimony", "reward_component"})

    def __post_init__(self) -> None:
        if self.source not in self._VALID_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(self._VALID_SOURCES)}, "
                f"got {self.source!r}"
            )
        if self.source == "reward_component" and not self.component_key:
            raise ValueError(
                "source='reward_component' requires a non-empty component_key"
            )


@dataclass
class FitnessSpec:
    """Declares the objectives an evaluator should return.

    When ``lexicase_mode`` is ``None`` (default), evaluators return a
    tuple with one scalar per objective, suitable for classic GA / CMA-ES
    / NSGA-II selection.

    When ``lexicase_mode`` is set, evaluators return a **flat vector** of
    per-test-case fitness values (one per episode or one per agent),
    suitable for :func:`deap.tools.selLexicase` /
    :func:`deap.tools.selAutomaticEpsilonLexicase`.

    Attributes:
        objectives: Ordered list of :class:`ObjectiveSpec` entries.  If
            empty or ``None`` the default single-objective reward-only
            contract applies.
        lexicase_mode: When ``None`` (default), objectives are aggregated
            into one scalar each.  ``"per_episode"`` concatenates
            per-episode values across all episodes and objectives.
            ``"per_agent"`` concatenates per-agent values.
        reward_components_fn: Optional callable
            ``(infos: dict, rewards: dict) -> Dict[str, float]``
            called once per env step to extract named reward components
            (e.g. ``"catches"``, ``"energy"``).  See
            :class:`ObjectiveSpec.component_key`.
    """

    objectives: Optional[List[ObjectiveSpec]] = None
    lexicase_mode: Optional[str] = None
    reward_components_fn: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]]
    ] = None

    _VALID_LEXICASE_MODES = frozenset({None, "per_episode", "per_agent"})

    def __post_init__(self) -> None:
        if self.lexicase_mode not in self._VALID_LEXICASE_MODES:
            raise ValueError(
                f"lexicase_mode must be one of "
                f"{sorted(str(m) for m in self._VALID_LEXICASE_MODES)}, "
                f"got {self.lexicase_mode!r}"
            )
        if self.objectives is None:
            self.objectives = []
        self.objectives = list(self.objectives)

    # ---- derived properties ----

    @property
    def is_multiobjective(self) -> bool:
        """``True`` when more than one objective is declared."""
        return len(self.objectives) > 1

    @property
    def is_lexicase(self) -> bool:
        """``True`` when a lexicase mode is active."""
        return self.lexicase_mode is not None

    @property
    def num_objectives(self) -> int:
        """Number of declared objective axes."""
        return len(self.objectives)

    @property
    def weights(self) -> Tuple[float, ...]:
        """Signed weight tuple for :func:`deap.creator.create`.

        Returns ``(1.0,)`` for the implicit single-objective case so
        callers always get a valid tuple.
        """
        if not self.objectives:
            return (1.0,)
        return tuple(o.weight for o in self.objectives)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the spec is inconsistent."""
        if self.objectives and self.lexicase_mode == "per_episode":
            if any(o.source != "reward" for o in self.objectives):
                raise ValueError(
                    "lexicase_mode='per_episode' supports only "
                    "source='reward' objectives (per-episode fitness "
                    "is computed before post-hoc parsimony / components)."
                )
        if (
            self.lexicase_mode == "per_agent"
            and self.objectives
            and any(
                o.source not in ("reward", "reward_component")
                for o in self.objectives
            )
        ):
            raise ValueError(
                "lexicase_mode='per_agent' supports only source='reward' "
                "and source='reward_component' objectives."
            )


# ---------------------------------------------------------------------------
# Parsimony helper
# ---------------------------------------------------------------------------


def compute_parsimony(genome: Any) -> float:
    """Return negative genome size for a parsimony pressure objective.

    Default implementation: :math:`-\\text{num_weights}`.
    Override by passing a custom :attr:`ObjectiveSpec` with a
    different :attr:`~ObjectiveSpec.component_key` and a matching
    ``reward_components_fn`` that extracts the desired metric.

    Args:
        genome: An object with a ``get_weight_count()`` method (e.g.
            :class:`~deap_pettingzoo.genomes.FixedTopologyGenome`).

    Returns:
        Negative integer (higher = smaller network = better when
        weight is positive).
    """
    if hasattr(genome, "get_weight_count"):
        return -float(genome.get_weight_count())
    # Fallback: count flat weight-vector length.
    if hasattr(genome, "weights") and hasattr(genome.weights, "__len__"):
        return -float(len(genome.weights))
    return 0.0


# ---------------------------------------------------------------------------
# Fitness-vector builder
# ---------------------------------------------------------------------------


def build_fitness_vector(
    aggregated_reward: float,
    per_episode_rewards: Optional[List[float]],
    per_agent_rewards: Optional[Dict[str, float]],
    reward_components: Optional[Dict[str, float]],
    genome: Any,
    spec: FitnessSpec,
) -> Tuple[float, ...]:
    """Assemble the fitness tuple/vector the evaluator should return.

    Args:
        aggregated_reward: The single reward scalar produced by the
            evaluator's normal aggregation logic.
        per_episode_rewards: Raw per-episode reward values from a
            multi-episode run (``None`` for single-episode evaluators).
        per_agent_rewards: Raw per-agent reward dict (``None`` when
            the evaluator aggregates internally).
        reward_components: Per-step component dict accumulated across
            the episode, or ``None``.
        genome: The individual's genome (used for parsimony).
        spec: The :class:`FitnessSpec` driving this evaluation.

    Returns:
        Fitness tuple.  When ``spec.lexicase_mode`` is ``None``, the
        length equals ``spec.num_objectives`` (or 1 for backward
        compatibility).  When ``per_episode`` or ``per_agent`` is
        active, the returned tuple is a flat vector of length
        ``num_episodes * num_objectives`` or ``num_agents * num_objectives``,
        with values concatenated in objective-then-episode/agent order.
    """
    objectives = spec.objectives
    if not objectives:
        return (aggregated_reward,)

    # ---- Lexicase "per_episode" path ----
    if spec.lexicase_mode == "per_episode":
        if not per_episode_rewards:
            return tuple(aggregated_reward for _ in objectives)
        # Concatenate: [obj0_ep0, obj0_ep1, ..., obj1_ep0, obj1_ep1, ...]
        vec: List[float] = []
        for obj in objectives:
            if obj.source == "reward":
                vec.extend(per_episode_rewards)
            elif obj.source == "parsimony":
                vec.extend([compute_parsimony(genome)] * len(per_episode_rewards))
            else:
                vec.extend([0.0] * len(per_episode_rewards))
        return tuple(vec)

    # ---- Lexicase "per_agent" path ----
    if spec.lexicase_mode == "per_agent":
        agents = sorted(per_agent_rewards.keys()) if per_agent_rewards else []
        if not agents:
            return tuple(aggregated_reward for _ in objectives)
        vec = []
        for obj in objectives:
            if obj.source == "reward":
                vec.extend(float(per_agent_rewards.get(a, 0.0)) for a in agents)
            elif obj.source == "reward_component" and reward_components:
                key = obj.component_key
                val = float(reward_components.get(key, 0.0)) if key else 0.0
                vec.extend([val] * len(agents))
            elif obj.source == "parsimony":
                vec.extend([compute_parsimony(genome)] * len(agents))
            else:
                vec.extend([0.0] * len(agents))
        return tuple(vec)

    # ---- Standard multi-objective path (no lexicase) ----
    result: List[float] = []
    for obj in objectives:
        if obj.source == "reward":
            result.append(float(aggregated_reward))
        elif obj.source == "parsimony":
            result.append(compute_parsimony(genome))
        elif obj.source == "reward_component":
            key = obj.component_key
            val = 0.0
            if reward_components and key:
                val = float(reward_components.get(key, 0.0))
            result.append(val)
        else:
            result.append(0.0)
    return tuple(result)
