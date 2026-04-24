"""Co-evolution evaluator.

Drives one episode where agents are partitioned across multiple
"sides". The learner side builds its policies from the evaluated
individual's genome; every other side contributes one sampled opponent
policy per agent, drawn from that side's
:class:`~deap_pettingzoo.baselines.coevolution.pools.OpponentPool`.

Rewards are partitioned per-side; the learner's fitness defaults to the
sum of its agents' rewards (symmetric competitive / cooperative with
additive fitness). Aggregation over ``k_opponents`` samples is
configurable (``'mean' | 'min' | 'max'``).

This evaluator reuses the same tolerant ``get_action`` invocation
semantics as :class:`MixedPolicyEvaluator` and consumes the env via
:class:`PettingZooWrapper`'s parallel API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.evaluators.base_evaluator import BaseEvaluator
from deap_pettingzoo.evaluators.mixed_policy import _invoke_get_action

from .pools import OpponentPool


#: Map ``(genome, agent_id, action_space) -> PolicyProtocol``. Matches the
#: learner factory signature used by :class:`MixedPolicyEvaluator` so the
#: same factory works in both evaluators.
LearnerPolicyFactory = Callable[[Any, str, Any], PolicyProtocol]


VALID_AGGREGATIONS = ("mean", "min", "max", "sum")


@dataclass
class CoEvolutionEvalResult:
    """Per-call outcome returned by :meth:`CoEvolutionEvaluator.run`.

    Attributes:
        fitness: Aggregated learner fitness across the ``k_opponents``
            episodes (passed to DEAP).
        per_sample_learner: Per-sample learner rewards (length ``k``).
        per_sample_opponents: Per-sample opponent-side reward totals,
            keyed by side name (each value is length ``k``).
        steps: Total environment steps across all ``k`` episodes.
        num_samples: ``k_opponents`` actually run.
    """

    fitness: float
    per_sample_learner: List[float] = field(default_factory=list)
    per_sample_opponents: Dict[str, List[float]] = field(default_factory=dict)
    steps: int = 0
    num_samples: int = 0


class CoEvolutionEvaluator(BaseEvaluator):
    """Evaluate one individual against sampled opponents from opposing sides.

    Args:
        env_wrapper: :class:`PettingZooWrapper` around the target env.
        learner_role_prefixes: Agent-name prefixes controlled by the
            learner (e.g. ``('adversary',)`` to train pursuers against
            evaders). All other agents must be covered by
            ``opponent_sides``.
        learner_factory: Callable ``(genome, agent_id, action_space) ->
            PolicyProtocol`` building one fresh learner policy per
            learner agent per episode.
        opponent_sides: Ordered list of
            ``(side_name, role_prefixes, OpponentPool)`` tuples. Every
            non-learner agent must match exactly one side's prefixes.
        k_opponents: Number of opponent *tuples* to sample (one
            episode per tuple). Default ``3``.
        aggregation: How to collapse the ``k`` per-sample learner
            rewards into a single fitness. One of
            ``'mean' | 'min' | 'max' | 'sum'``. Default ``'mean'``.
        max_steps: Hard cap on env steps per episode.

    Raises:
        ValueError: On agent-coverage mismatch (unassigned or
            multi-assigned agents) or invalid ``aggregation``.
    """

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        learner_role_prefixes: Sequence[str],
        learner_factory: LearnerPolicyFactory,
        opponent_sides: Sequence[Tuple[str, Sequence[str], OpponentPool]],
        k_opponents: int = 3,
        aggregation: str = "mean",
        max_steps: int = 500,
    ) -> None:
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation={aggregation!r}. "
                f"Must be one of {VALID_AGGREGATIONS}."
            )
        if k_opponents < 1:
            raise ValueError(f"k_opponents must be >= 1, got {k_opponents}.")

        self.env_wrapper = env_wrapper
        self.learner_role_prefixes: Tuple[str, ...] = tuple(learner_role_prefixes)
        self.learner_factory = learner_factory
        self.opponent_sides: List[Tuple[str, Tuple[str, ...], OpponentPool]] = [
            (name, tuple(prefixes), pool)
            for name, prefixes, pool in opponent_sides
        ]
        self.k_opponents = int(k_opponents)
        self.aggregation = aggregation
        self.max_steps = int(max_steps)

        self._partition_agents()
        #: Populated after each :meth:`run` call for trainer logging.
        self.last_details: Optional[CoEvolutionEvalResult] = None

    # ------------------------------------------------------------------
    # Agent partitioning
    # ------------------------------------------------------------------

    def _partition_agents(self) -> None:
        """Validate that every env agent belongs to exactly one side."""
        agents = list(self.env_wrapper.possible_agents)
        learner_agents: List[str] = []
        side_agents: Dict[str, List[str]] = {
            name: [] for name, _, _ in self.opponent_sides
        }

        for agent in agents:
            matched: List[str] = []
            if any(agent.startswith(p) for p in self.learner_role_prefixes):
                matched.append("__learner__")
                learner_agents.append(agent)
            for side_name, prefixes, _ in self.opponent_sides:
                if any(agent.startswith(p) for p in prefixes):
                    matched.append(side_name)
                    side_agents[side_name].append(agent)

            if not matched:
                raise ValueError(
                    f"Agent {agent!r} is not claimed by the learner "
                    f"(prefixes={self.learner_role_prefixes!r}) or by any "
                    f"opponent side. Assign a role prefix that matches."
                )
            if len(matched) > 1:
                raise ValueError(
                    f"Agent {agent!r} is claimed by multiple sides "
                    f"({matched!r}); role prefixes must be disjoint."
                )

        if not learner_agents:
            raise ValueError(
                "CoEvolutionEvaluator: no agents matched "
                f"learner_role_prefixes={self.learner_role_prefixes!r}."
            )

        self.learner_agents: List[str] = learner_agents
        self.side_agents: Dict[str, List[str]] = side_agents

    # ------------------------------------------------------------------
    # BaseEvaluator
    # ------------------------------------------------------------------

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float]:
        """DEAP contract: return ``(fitness,)``.

        Args:
            individual: Must carry a ``.genome`` attribute.
            seed: Base seed for the underlying env (each of the ``k``
                sample episodes uses ``seed + i`` if ``seed`` is given).
            rng: Optional RNG used for opponent sampling. Defaults to a
                fresh ``np.random.default_rng(seed)`` — this keeps
                evaluations deterministic under a fixed ``seed``.
        """
        result = self.run(individual, seed=seed, rng=rng)
        return (result.fitness,)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(
        self,
        individual: Any,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> CoEvolutionEvalResult:
        """Run ``k_opponents`` episodes and aggregate learner fitness."""
        if individual is None or not hasattr(individual, "genome") or individual.genome is None:
            raise ValueError("CoEvolutionEvaluator requires an individual with a 'genome'.")

        if rng is None:
            rng = np.random.default_rng(seed)

        learner_rewards: List[float] = []
        side_reward_samples: Dict[str, List[float]] = {
            name: [] for name, _, _ in self.opponent_sides
        }
        total_steps = 0

        for i in range(self.k_opponents):
            episode_seed = None if seed is None else int(seed) + i
            # Sample ONE opponent policy per side (shared across that
            # side's agents). This matches the canonical self-play
            # contract ("one champion drives the whole team") and
            # massively reduces opponent-sampling variance compared
            # to sampling per-agent.
            opp_policies: Dict[str, PolicyProtocol] = {}
            for side_name, _prefixes, pool in self.opponent_sides:
                side_policy = pool.sample_opponent(rng)
                for agent in self.side_agents[side_name]:
                    opp_policies[agent] = side_policy

            learner_r, side_r, steps = self._run_one_episode(
                individual.genome, opp_policies, episode_seed
            )
            learner_rewards.append(learner_r)
            for side_name, total in side_r.items():
                side_reward_samples[side_name].append(total)
            total_steps += steps

        fitness = self._aggregate(learner_rewards)
        details = CoEvolutionEvalResult(
            fitness=fitness,
            per_sample_learner=learner_rewards,
            per_sample_opponents=side_reward_samples,
            steps=total_steps,
            num_samples=self.k_opponents,
        )
        self.last_details = details
        return details

    def _aggregate(self, rewards: List[float]) -> float:
        arr = np.asarray(rewards, dtype=np.float64)
        if self.aggregation == "mean":
            return float(arr.mean())
        if self.aggregation == "min":
            return float(arr.min())
        if self.aggregation == "max":
            return float(arr.max())
        # "sum"
        return float(arr.sum())

    def _run_one_episode(
        self,
        genome: Any,
        opponent_policies: Dict[str, PolicyProtocol],
        seed: Optional[int],
    ) -> Tuple[float, Dict[str, float], int]:
        """Run one rollout with the pre-sampled opponent policies."""
        observations = self.env_wrapper.reset(seed=seed)

        # Build learner policies fresh per episode (stateful heads etc.).
        learner_policies: Dict[str, PolicyProtocol] = {
            agent: self.learner_factory(
                genome, agent, self.env_wrapper.action_space(agent)
            )
            for agent in self.learner_agents
        }
        for p in learner_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        for p in opponent_policies.values():
            if hasattr(p, "reset"):
                p.reset()

        learner_total = 0.0
        side_totals: Dict[str, float] = {
            name: 0.0 for name, _, _ in self.opponent_sides
        }
        agent_to_side: Dict[str, str] = {}
        for side_name, agents in self.side_agents.items():
            for a in agents:
                agent_to_side[a] = side_name

        steps = 0
        while steps < self.max_steps and observations:
            infos = getattr(self.env_wrapper, "last_infos", {}) or {}
            actions: Dict[str, Any] = {}
            for agent, obs in observations.items():
                mask = infos.get(agent, {}).get("action_mask")
                if agent in learner_policies:
                    actions[agent] = _invoke_get_action(
                        learner_policies[agent], obs, mask, deterministic=True
                    )
                elif agent in opponent_policies:
                    actions[agent] = _invoke_get_action(
                        opponent_policies[agent], obs, mask
                    )

            observations, rewards, terms, truncs, _ = self.env_wrapper.step(actions)

            for agent, reward in rewards.items():
                if agent in learner_policies:
                    learner_total += float(reward)
                else:
                    side = agent_to_side.get(agent)
                    if side is not None:
                        side_totals[side] += float(reward)

            steps += 1
            if terms and all(terms.values()):
                break
            if truncs and all(truncs.values()):
                break

        return learner_total, side_totals, steps


__all__ = ["CoEvolutionEvaluator", "CoEvolutionEvalResult", "LearnerPolicyFactory"]
