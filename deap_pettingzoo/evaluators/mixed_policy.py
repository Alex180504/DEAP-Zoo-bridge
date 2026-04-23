"""Mixed-policy evaluator.

Allows different agents in a PettingZoo environment to be controlled by
different policies during a single evaluation episode — useful whenever
*some* agents are the learning subject (e.g. EA-evolved pursuers) and the
remainder are scripted benchmark targets (e.g. APF evaders).

Design
------

A :class:`MixedPolicyEvaluator` is instantiated with:

* an environment wrapper,
* a *learner factory* that maps a genome to a policy and declares which
  agent prefixes (e.g. ``'adversary'``) that policy controls — this is the
  "side being trained",
* a dictionary of pre-built :class:`PolicyProtocol` instances, keyed by
  agent name, that control all remaining agents (e.g. the scripted
  APF evaders).

On ``evaluate(individual)`` the evaluator constructs one learner-policy
instance from ``individual.genome`` per agent belonging to the learner
side, then runs a full episode, summing rewards *only* for agents on the
learner side (this becomes the fitness).  Scripted-side rewards do not
contribute to the DEAP fitness — consistent with the methodology's use of
scripted evaders as a benchmark target rather than a co-optimisation
partner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.evaluators.base_evaluator import BaseEvaluator


LearnerPolicyFactory = Callable[[Any, str, Any], "PolicyLike"]


def _invoke_get_action(
    policy: Any,
    observation: np.ndarray,
    action_mask: Optional[np.ndarray],
    *,
    deterministic: Optional[bool] = None,
) -> Any:
    """Call ``policy.get_action`` tolerantly across protocol variants.

    Heterogeneous policy set (learner EA/RL policies + scripted / rule-
    based opponents) all have slightly different ``get_action``
    signatures. This helper forwards ``action_mask`` and ``deterministic``
    opportunistically and falls back when the policy doesn't accept
    those kwargs.
    """
    kwargs: Dict[str, Any] = {}
    if action_mask is not None:
        kwargs["action_mask"] = action_mask
    if deterministic is not None:
        kwargs["deterministic"] = deterministic
    try:
        return policy.get_action(observation, **kwargs)
    except TypeError:
        # Strip kwargs the policy doesn't accept and retry.
        if "action_mask" in kwargs:
            kwargs.pop("action_mask")
            try:
                return policy.get_action(observation, **kwargs)
            except TypeError:
                pass
        if "deterministic" in kwargs:
            kwargs.pop("deterministic")
            return policy.get_action(observation, **kwargs)
        return policy.get_action(observation)


class PolicyLike:  # structural type; actual classes only need get_action/reset
    """Structural hint matching :class:`PolicyProtocol`."""

    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> Any: ...
    def reset(self) -> None: ...


@dataclass
class MixedPolicyEvalResult:
    """Detailed per-episode outcome returned by :meth:`MixedPolicyEvaluator.run_episode`.

    Attributes:
        fitness: Sum of learner-side rewards over the episode (passed to
            DEAP by :meth:`evaluate`).
        learner_reward: Identical to ``fitness``; kept for clarity when
            both sides' totals are useful.
        scripted_reward: Sum of scripted-side rewards (for diagnostics).
        steps: Number of environment steps taken.
    """

    fitness: float
    learner_reward: float
    scripted_reward: float
    steps: int


class MixedPolicyEvaluator(BaseEvaluator):
    """Evaluate heterogeneous multi-agent policies in one environment.

    Unlike :class:`SingleEpisodeEvaluator`, which wraps one shared genome
    across *all* agents, this evaluator assigns agents to one of two
    disjoint groups:

    * **Learner group** — agents whose name starts with any prefix in
      ``learner_agent_prefixes``.  Each is driven by a fresh policy built
      the individual's genome via
      ``learner_factory(genome, agent_id, action_space)``.
    * **Scripted group** — everyone else.  Each is driven by the
      pre-built :class:`PolicyProtocol` instance found in
      ``scripted_policies[agent_id]``.  If the dict is missing any
      non-learner agent, construction raises.
    """

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        learner_agent_prefixes: List[str],
        learner_factory: LearnerPolicyFactory,
        scripted_policies: Dict[str, PolicyLike],
        max_steps: int = 500,
    ) -> None:
        """Build the evaluator.

        Args:
            env_wrapper: Environment wrapper providing ``reset`` / ``step``.
            learner_agent_prefixes: Agent-name prefixes whose agents are
                driven by the learning side (e.g. ``['adversary']`` to
                train pursuers).
            learner_factory: Callable
                ``(genome, agent_id, action_space) -> policy`` returning a
                fresh policy instance for each learner agent.  The
                ``action_space`` argument was added in the continuous-action
                support drop and is *required* — factories that previously
                took only ``(genome, agent_id)`` must be updated.
            scripted_policies: Mapping from agent_id to a pre-built
                policy for every non-learner agent.
            max_steps: Hard cap on environment steps per episode.

        Raises:
            ValueError: If a non-learner agent has no scripted policy.
        """
        self.env_wrapper = env_wrapper
        self.learner_agent_prefixes = tuple(learner_agent_prefixes)
        self.learner_factory = learner_factory
        self.scripted_policies = dict(scripted_policies)
        self.max_steps = max_steps

        # Partition agents once; validate coverage.
        self.learner_agents: List[str] = []
        self.scripted_agents: List[str] = []
        for agent in env_wrapper.possible_agents:
            if agent.startswith(self.learner_agent_prefixes):
                self.learner_agents.append(agent)
            else:
                self.scripted_agents.append(agent)

        missing = [a for a in self.scripted_agents if a not in self.scripted_policies]
        if missing:
            raise ValueError(
                f"scripted_policies missing entries for non-learner agents: {missing}"
            )
        if not self.learner_agents:
            raise ValueError(
                "No agents matched learner_agent_prefixes "
                f"{self.learner_agent_prefixes!r}; cannot train."
            )

    # ------------------------------------------------------------------
    # BaseEvaluator
    # ------------------------------------------------------------------

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> Tuple[float]:
        """Run one episode and return ``(fitness,)`` (DEAP contract).

        ``fitness`` is the summed reward across all learner-side agents.
        Scripted-side rewards are recorded separately in
        :meth:`run_episode` but do not enter the DEAP fitness.
        """
        result = self.run_episode(individual, seed=seed)
        return (result.fitness,)

    # ------------------------------------------------------------------
    # Core loop (exposed for logging / visualisation)
    # ------------------------------------------------------------------

    def run_episode(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> MixedPolicyEvalResult:
        """Execute one episode with mixed policies and return full stats."""
        if individual is None or not hasattr(individual, "genome") or individual.genome is None:
            raise ValueError("MixedPolicyEvaluator requires an individual with a 'genome'.")

        genome = individual.genome
        observations = self.env_wrapper.reset(seed=seed)

        # Build fresh learner policies for this episode.
        learner_policies: Dict[str, PolicyLike] = {
            agent: self.learner_factory(
                genome, agent, self.env_wrapper.action_space(agent)
            )
            for agent in self.learner_agents
        }
        # Reset all policies (no-op for stateless ones; skip if not implemented).
        for p in learner_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        for p in self.scripted_policies.values():
            if hasattr(p, "reset"):
                p.reset()

        learner_total = 0.0
        scripted_total = 0.0
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
                elif agent in self.scripted_policies:
                    actions[agent] = _invoke_get_action(
                        self.scripted_policies[agent], obs, mask
                    )
                # else: agent terminated — no action required

            observations, rewards, terms, truncs, _ = self.env_wrapper.step(actions)

            for agent, reward in rewards.items():
                if agent.startswith(self.learner_agent_prefixes):
                    learner_total += reward
                else:
                    scripted_total += reward

            steps += 1
            if all(terms.values()) or all(truncs.values()):
                break

        return MixedPolicyEvalResult(
            fitness=learner_total,
            learner_reward=learner_total,
            scripted_reward=scripted_total,
            steps=steps,
        )
