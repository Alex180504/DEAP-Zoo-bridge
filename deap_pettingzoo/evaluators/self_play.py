"""Self-play evaluator for turn-based AEC environments.

Drives **every** agent in the environment with a *single shared-weights*
policy produced from one genome. This is the canonical EA-evaluator for
symmetric turn-based games (e.g. tic-tac-toe, connect-four) paired with
:class:`~deap_pettingzoo.environments.MaskedSimultaneousAdapter` or
:class:`~deap_pettingzoo.environments.SequentialTurnAdapter`.

Why a dedicated evaluator rather than :class:`SingleEpisodeEvaluator`?

* Ensures a :class:`~deap_pettingzoo.policies.action_heads.MaskedDiscreteHead`
  is installed so the policy can consume
  ``infos[agent]["action_mask"]`` from the adapter.
* Offers ``first_agent`` and ``team_mapping`` aggregations that make
  sense for competitive self-play fitness (e.g. "reward my genome only
  by its player_1 win-rate" or "sum rewards across a designated team").
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..environments import PettingZooWrapper
from ..genomes import FixedTopologyGenome
from ..policies import FeedforwardNeuralPolicy
from .base_evaluator import BaseEvaluator


class SelfPlayEvaluator(BaseEvaluator):
    """Evaluate one genome by having it play every role in a turn-based env.

    Args:
        env_wrapper: :class:`PettingZooWrapper` around a turn-based
            adapter (``api='aec_masked'`` or ``api='aec_sequential'``).
        max_steps: Hard ceiling on environment steps per episode.
        aggregation: How to collapse per-agent cumulative rewards into a
            single fitness scalar:

            * ``'sum'`` — sum across all agents (symmetric zero-sum games
              would return 0 each episode; use only for cooperative
              boards).
            * ``'mean'`` — arithmetic mean across all agents.
            * ``'first_agent'`` — reward of the first agent in
              ``env.possible_agents`` only. The natural choice for
              symmetric zero-sum games: measures how well the genome
              plays *as the first player*.
            * ``'team_mapping'`` — sum rewards for agents whose
              ``team_mapping[agent]`` matches ``team``. Requires
              ``team_mapping`` and ``team`` to be supplied.
        team_mapping: Maps ``agent_id`` → team name. Required when
            ``aggregation='team_mapping'``.
        team: Team name to sum rewards for. Required when
            ``aggregation='team_mapping'``.
        masked: If ``True`` (default), build the shared policy with a
            :class:`MaskedDiscreteHead` so ``action_mask`` kwargs are
            honoured. Turn off only for non-masked turn-based envs.
        flatten_obs: If ``True`` (default), observations are flattened to
            1-D via ``np.ravel`` before being passed to the policy.
            Classic games (tic-tac-toe, connect-four) present N-D board
            observations that :class:`FeedforwardNeuralPolicy` cannot
            consume directly; flattening sidesteps that without forcing
            a separate observation-wrapper env.
    """

    VALID_AGGREGATIONS = ("sum", "mean", "first_agent", "team_mapping")

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 200,
        aggregation: str = "first_agent",
        team_mapping: Optional[Dict[str, str]] = None,
        team: Optional[str] = None,
        masked: bool = True,
        flatten_obs: bool = True,
        fitness_spec: Optional[Any] = None,
    ) -> None:
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation={aggregation!r}. "
                f"Must be one of {self.VALID_AGGREGATIONS}."
            )
        if aggregation == "team_mapping":
            if not team_mapping:
                raise ValueError(
                    "aggregation='team_mapping' requires a non-empty "
                    "team_mapping dict."
                )
            if team is None:
                raise ValueError(
                    "aggregation='team_mapping' requires a team= kwarg."
                )
        self.env_wrapper = env_wrapper
        self.max_steps = int(max_steps)
        self.aggregation = aggregation
        self.team_mapping = dict(team_mapping or {})
        self.team = team
        self.masked = bool(masked)
        self.flatten_obs = bool(flatten_obs)
        self.fitness_spec = fitness_spec
        if fitness_spec is not None:
            fitness_spec.validate()

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> Tuple[float, ...]:
        """Play one self-play episode and return fitness tuple.

        When ``fitness_spec is None`` returns the historical 1-tuple
        ``(aggregated_scalar,)``.  When a spec is set, the tuple may be
        multi-dimensional (standard) or a flat vector (lexicase).

        Raises:
            ValueError: If ``individual`` is ``None``, is missing a
                ``.genome`` attribute, or the genome is ``None``.
        """
        if individual is None:
            raise ValueError("Individual cannot be None")
        if not hasattr(individual, "genome"):
            raise ValueError(
                f"Individual must have 'genome' attribute, got {type(individual)}"
            )
        genome = individual.genome
        if genome is None:
            raise ValueError("Individual's genome cannot be None")

        observations = self.env_wrapper.reset(seed=seed)
        policies = self._build_shared_policies(genome)

        cumulative: Dict[str, float] = {
            a: 0.0 for a in self.env_wrapper.possible_agents
        }
        reward_components: Optional[Dict[str, float]] = (
            {} if self._has_reward_component_objective() else None
        )
        steps = 0
        done = False
        while not done and steps < self.max_steps and observations:
            infos = getattr(self.env_wrapper, "last_infos", {}) or {}
            actions: Dict[str, Any] = {}
            for agent, obs in observations.items():
                if agent not in policies:
                    # Agent added mid-episode (rare); build on the fly.
                    policies[agent] = self._make_policy(genome, agent)
                mask = infos.get(agent, {}).get("action_mask")
                flat = np.asarray(obs).ravel() if self.flatten_obs else obs
                actions[agent] = policies[agent].get_action(
                    flat, deterministic=True, action_mask=mask
                )
            observations, rewards, terms, truncs, _ = self.env_wrapper.step(actions)
            for agent, reward in rewards.items():
                cumulative[agent] = cumulative.get(agent, 0.0) + float(reward)
            # Accumulate reward components.
            if reward_components is not None and self.fitness_spec is not None:
                rc_fn = self.fitness_spec.reward_components_fn
                if rc_fn is not None:
                    step_rc = rc_fn(infos or {}, rewards)
                    for k, v in step_rc.items():
                        reward_components[k] = reward_components.get(k, 0.0) + float(v)
            done = (
                (terms and all(terms.get(a, False) for a in terms))
                or (truncs and all(truncs.get(a, False) for a in truncs))
            )
            steps += 1

        aggregated = self._aggregate(cumulative)

        if self.fitness_spec is not None and self.fitness_spec.objectives:
            from deap_pettingzoo.integration.objectives import build_fitness_vector
            return build_fitness_vector(
                aggregated_reward=aggregated,
                per_episode_rewards=[aggregated],
                per_agent_rewards=cumulative,
                reward_components=reward_components,
                genome=genome,
                spec=self.fitness_spec,
            )

        return (aggregated,)

    def _build_shared_policies(
        self, genome: FixedTopologyGenome
    ) -> Dict[str, FeedforwardNeuralPolicy]:
        """Create one policy *per agent* that all share the **same** genome.

        Shared weights - one learner plays every side of the board. Per-
        agent policy instances are still created so each can carry its
        own :class:`ActionHead` (action spaces may differ by role, e.g.
        different ``n`` in asymmetric games).
        """
        return {
            agent: self._make_policy(genome, agent)
            for agent in self.env_wrapper.possible_agents
        }

    def _make_policy(
        self, genome: FixedTopologyGenome, agent: str
    ) -> FeedforwardNeuralPolicy:
        return FeedforwardNeuralPolicy(
            genome,
            action_space=self.env_wrapper.action_space(agent),
            masked=self.masked,
        )

    def _aggregate(self, cumulative: Dict[str, float]) -> float:
        if self.aggregation == "sum":
            return float(sum(cumulative.values()))
        if self.aggregation == "mean":
            values = list(cumulative.values())
            return float(np.mean(values)) if values else 0.0
        if self.aggregation == "first_agent":
            agents = list(self.env_wrapper.possible_agents)
            if not agents:
                return 0.0
            return float(cumulative.get(agents[0], 0.0))
        # team_mapping
        return float(
            sum(
                r
                for a, r in cumulative.items()
                if self.team_mapping.get(a) == self.team
            )
        )

    def _has_reward_component_objective(self) -> bool:
        """Return ``True`` if the fitness spec requires per-step reward components."""
        spec = self.fitness_spec
        if spec is None or not spec.objectives:
            return False
        return any(o.source == "reward_component" for o in spec.objectives)


__all__ = ["SelfPlayEvaluator"]
