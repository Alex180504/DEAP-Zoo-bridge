import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base_evaluator import BaseEvaluator
from ..environments import PettingZooWrapper
from ..policies import FeedforwardNeuralPolicy

logger = logging.getLogger(__name__)


class SingleEpisodeEvaluator(BaseEvaluator):
    """Evaluates individual genome in a single episode.

    All agents share the same genome and therefore the same network
    architecture. Environments with heterogeneous observation sizes
    across agents (e.g. ``mpe2.simple_adversary_v3``) are not
    supported directly; register the env with :mod:`supersuit`
    padding wrappers (``pad_observations_v0`` +
    ``pad_action_space_v0``) via :attr:`EnvSpec.wrappers`, or evolve
    per-role with :class:`~deap_pettingzoo.evaluators.MixedPolicyEvaluator`.

    Multi-objective / lexicase support
    ----------------------------------
    Pass ``fitness_spec`` (a :class:`~deap_pettingzoo.integration.objectives.FitnessSpec`)
    to return N-tuples for multi-objective fitness or flat vectors for
    lexicase selection.  When ``fitness_spec`` is ``None`` (default)
    the evaluator returns a 1-tuple ``(aggregated_reward,)`` —
    identical to the historical behaviour.
    """

    VALID_AGGREGATIONS = ['sum', 'mean', 'min', 'max', 'team']

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 500,
        aggregation_method: str = 'sum',
        team_mapping: Optional[Dict[str, str]] = None,
        fitness_spec: Optional[Any] = None,
    ) -> None:
        """
        Initialize single episode evaluator.

        Args:
            env_wrapper: PettingZoo environment wrapper
            max_steps: Maximum steps per episode
            aggregation_method: How to combine agent rewards
                - 'sum': Sum of all agent rewards
                - 'mean': Average of agent rewards
                - 'min': Minimum reward (worst agent)
                - 'max': Maximum reward (best agent)
                - 'team': Team-based aggregation (requires team_mapping)
            team_mapping: Maps agent_id to team name (for competitive
                scenarios).
            fitness_spec: Optional
                :class:`~deap_pettingzoo.integration.objectives.FitnessSpec`.
                When ``None`` the evaluator returns the historical
                1-tuple.  When set, the returned tuple may be multi-
                dimensional (standard multi-objective) or a flat vector
                (lexicase mode).

        Raises:
            ValueError: If invalid aggregation method.
        """
        if aggregation_method not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation_method '{aggregation_method}'. "
                f"Must be one of {self.VALID_AGGREGATIONS}"
            )

        self.env_wrapper = env_wrapper
        self.max_steps = max_steps
        self.aggregation_method = aggregation_method
        self.team_mapping = team_mapping or {}
        self.fitness_spec = fitness_spec
        #: Cumulative count of *real* environment steps consumed across every
        #: :meth:`evaluate` call on this instance.
        self.env_steps_consumed = 0
        if fitness_spec is not None:
            fitness_spec.validate()
    
    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None
    ) -> Tuple[float, ...]:
        """
        Evaluate individual in one episode.
        
        Args:
            individual: DEAP individual with genome attribute
            seed: Optional seed for episode reproducibility
            
        Returns:
            Tuple of fitness values (length 1 for backward-compatible
            single-objective; N for multi-objective; flat vector for
            lexicase modes).
            
        Raises:
            ValueError: If individual or genome is None/invalid.
            RuntimeError: If an exception occurs during the episode rollout
                (wraps the original exception with context).
        """
        # Validate inputs
        if individual is None:
            raise ValueError("Individual cannot be None")
        
        if not hasattr(individual, 'genome'):
            raise ValueError(
                f"Individual must have 'genome' attribute, got type {type(individual)}"
            )
        
        if individual.genome is None:
            raise ValueError("Individual's genome cannot be None")
        
        # Reset environment
        observations = self.env_wrapper.reset(seed=seed)
        
        # Create policies for each agent using the same genome
        policies = self._create_policies(individual.genome, observations)
        
        # Run episode
        cumulative_rewards = {agent: 0.0 for agent in observations.keys()}
        reward_components: Optional[Dict[str, float]] = (
            {} if self._has_reward_component_objective() else None
        )
        done = False
        steps = 0
        
        try:
            while not done and steps < self.max_steps:
                # Get actions from policies
                infos = getattr(self.env_wrapper, "last_infos", {}) or {}
                actions = {}
                for agent, obs in observations.items():
                    if agent in policies:
                        mask = infos.get(agent, {}).get("action_mask")
                        actions[agent] = policies[agent].get_action(
                            obs, deterministic=True, action_mask=mask
                        )
                
                # Step environment
                observations, rewards, dones, truncated, infos = self.env_wrapper.step(actions)

                # Accumulate rewards
                for agent, reward in rewards.items():
                    cumulative_rewards[agent] += reward

                # Accumulate reward components (per the fitness spec hook).
                if reward_components is not None and self.fitness_spec is not None:
                    rc_fn = self.fitness_spec.reward_components_fn
                    if rc_fn is not None:
                        step_rc = rc_fn(infos or {}, rewards)
                        for k, v in step_rc.items():
                            reward_components[k] = reward_components.get(k, 0.0) + float(v)
                
                # Check if episode is done
                done = all(dones.values()) or all(truncated.values())
                steps += 1
                self.env_steps_consumed += 1
            
            # Aggregate fitness across agents
            aggregated = self._aggregate_fitness(cumulative_rewards)

            # Build multi-objective / lexicase vector when a spec is set.
            if self.fitness_spec is not None and self.fitness_spec.objectives:
                from deap_pettingzoo.integration.objectives import build_fitness_vector
                return build_fitness_vector(
                    aggregated_reward=aggregated,
                    per_episode_rewards=[aggregated],  # single episode
                    per_agent_rewards=cumulative_rewards,
                    reward_components=reward_components,
                    genome=individual.genome,
                    spec=self.fitness_spec,
                )
            
            return (aggregated,)
        except Exception as e:
            # Log error and re-raise with context
            logger.exception("Error during episode evaluation")
            raise RuntimeError(f"Error during episode evaluation: {e}") from e
    
    def _create_policies(
        self,
        genome,
        observations: Dict[str, np.ndarray]
    ) -> Dict[str, FeedforwardNeuralPolicy]:
        """Create one shared-weights policy per agent.

        Args:
            genome: Base genome shared across all agents.
            observations: Initial observations dict.

        Returns:
            Dictionary mapping agent_id to policy.

        Raises:
            ValueError: If agents have heterogeneous observation sizes.
        """
        obs_sizes = {agent: len(obs) for agent, obs in observations.items()}
        unique_sizes = set(obs_sizes.values())

        if len(unique_sizes) > 1:
            raise ValueError(
                "SingleEpisodeEvaluator requires homogeneous observation "
                f"sizes across agents; got {obs_sizes}. Register the env "
                "with supersuit padding wrappers "
                "(`pad_observations_v0`, `pad_action_space_v0`, optionally "
                "`agent_indicator_v0`) via the registry's `wrappers=(...)` "
                "field, or evolve per-role with MixedPolicyEvaluator."
            )

        policies: Dict[str, FeedforwardNeuralPolicy] = {}
        for agent in observations.keys():
            action_space = self.env_wrapper.action_space(agent)
            policies[agent] = FeedforwardNeuralPolicy(
                genome, action_space=action_space
            )
        return policies

    def _aggregate_fitness(self, cumulative_rewards: Dict[str, float]) -> float:
        """
        Aggregate rewards across agents into single fitness value.
        
        Args:
            cumulative_rewards: Dict mapping agent_id to total reward
            
        Returns:
            Aggregated fitness value
        """
        if not cumulative_rewards:
            return 0.0
        
        rewards = list(cumulative_rewards.values())
        
        if self.aggregation_method == 'sum':
            return float(np.sum(rewards))
        elif self.aggregation_method == 'mean':
            return float(np.mean(rewards))
        elif self.aggregation_method == 'min':
            return float(np.min(rewards))
        elif self.aggregation_method == 'max':
            return float(np.max(rewards))
        elif self.aggregation_method == 'team':
            return self._aggregate_team_fitness(cumulative_rewards)
        
        return 0.0
    
    def _aggregate_team_fitness(self, cumulative_rewards: Dict[str, float]) -> float:
        """
        Aggregate fitness for two-team competitive scenarios.

        Teams are resolved from :attr:`team_mapping` and ordered by team
        name (sorted) so the result is deterministic and independent of
        agent/dict insertion order.

        Args:
            cumulative_rewards: Dict mapping agent_id to reward

        Returns:
            For exactly two teams: the first team's total reward minus the
            second team's, with teams ordered by sorted team name. For a
            single team (or when no ``team_mapping`` was provided): the plain
            sum of all rewards.

        Raises:
            ValueError: If ``team_mapping`` resolves to three or more teams.
                Team aggregation is defined only for the two-team
                (zero-sum-style) case; supporting more teams would require a
                well-defined multi-team rule that this evaluator does not
                assume. Use a different ``aggregation_method`` or a
                ``team_mapping`` that yields at most two teams.
        """
        if not self.team_mapping:
            # Fallback to sum if no team mapping provided
            return float(np.sum(list(cumulative_rewards.values())))

        team_rewards: Dict[str, float] = {}
        for agent, reward in cumulative_rewards.items():
            team = self.team_mapping.get(agent, 'default')
            team_rewards[team] = team_rewards.get(team, 0.0) + reward

        if len(team_rewards) > 2:
            raise ValueError(
                "Team aggregation is defined only for two teams, but "
                f"team_mapping produced {len(team_rewards)} teams: "
                f"{sorted(team_rewards)}. Use a team_mapping that yields at "
                "most two teams, or a different aggregation_method."
            )

        # Order teams by name so the difference is deterministic regardless
        # of agent/dict insertion order.
        ordered_totals = [team_rewards[name] for name in sorted(team_rewards)]
        if len(ordered_totals) == 2:
            return float(ordered_totals[0] - ordered_totals[1])

        return float(sum(ordered_totals))

    def _has_reward_component_objective(self) -> bool:
        """Return ``True`` if the fitness spec requires per-step reward components."""
        spec = self.fitness_spec
        if spec is None or not spec.objectives:
            return False
        return any(o.source == "reward_component" for o in spec.objectives)
