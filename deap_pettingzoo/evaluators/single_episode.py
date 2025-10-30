from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base_evaluator import BaseEvaluator
from ..environments import PettingZooWrapper
from ..policies import FeedforwardNeuralPolicy


class SingleEpisodeEvaluator(BaseEvaluator):
    """Evaluates individual genome in a single episode."""
    
    VALID_AGGREGATIONS = ['sum', 'mean', 'min', 'max', 'team']
    
    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 500,
        fitness_aggregation: str = 'sum',
        team_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize single episode evaluator.
        
        Args:
            env_wrapper: PettingZoo environment wrapper
            max_steps: Maximum steps per episode
            fitness_aggregation: How to combine agent rewards
                - 'sum': Sum of all agent rewards
                - 'mean': Average of agent rewards
                - 'min': Minimum reward (worst agent)
                - 'max': Maximum reward (best agent)
                - 'team': Team-based aggregation (requires team_mapping)
            team_mapping: Maps agent_id to team name (for competitive scenarios)
            
        Raises:
            ValueError: If invalid aggregation method
        """
        if fitness_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid fitness_aggregation '{fitness_aggregation}'. "
                f"Must be one of {self.VALID_AGGREGATIONS}"
            )
        
        self.env_wrapper = env_wrapper
        self.max_steps = max_steps
        self.fitness_aggregation = fitness_aggregation
        self.team_mapping = team_mapping or {}
    
    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None
    ) -> Tuple[float]:
        """
        Evaluate individual in one episode.
        
        Args:
            individual: DEAP individual with genome attribute
            seed: Optional seed for episode reproducibility
            
        Returns:
            Tuple containing single fitness value (DEAP format)
        """
        # Reset environment
        observations = self.env_wrapper.reset(seed=seed)
        
        # Create policies for each agent using the same genome
        policies = self._create_policies(individual.genome, observations)
        
        # Run episode
        cumulative_rewards = {agent: 0.0 for agent in observations.keys()}
        done = False
        steps = 0
        
        while not done and steps < self.max_steps:
            # Get actions from policies
            actions = {}
            for agent, obs in observations.items():
                if agent in policies:
                    actions[agent] = policies[agent].get_action(obs, deterministic=True)
            
            # Step environment
            observations, rewards, dones, truncated, infos = self.env_wrapper.step(actions)
            
            # Accumulate rewards
            for agent, reward in rewards.items():
                cumulative_rewards[agent] += reward
            
            # Check if episode is done
            done = all(dones.values()) or truncated
            steps += 1
        
        # Aggregate fitness across agents
        fitness = self._aggregate_fitness(cumulative_rewards)
        
        return (fitness,)
    
    def _create_policies(
        self,
        genome,
        observations: Dict[str, np.ndarray]
    ) -> Dict[str, FeedforwardNeuralPolicy]:
        """
        Create policies for all agents.
        
        For agents with different observation sizes, creates agent-specific
        genomes with the same weight values but different architectures.
        
        Args:
            genome: Base genome to use
            observations: Initial observations dict
            
        Returns:
            Dictionary mapping agent_id to policy
        """
        from ..genomes import FixedTopologyGenome
        
        policies = {}
        
        # Check if all agents have same observation size
        obs_sizes = {agent: len(obs) for agent, obs in observations.items()}
        unique_sizes = set(obs_sizes.values())
        
        if len(unique_sizes) == 1:
            # All agents have same obs size - share genome
            for agent in observations.keys():
                policies[agent] = FeedforwardNeuralPolicy(genome)
        else:
            # Different obs sizes - create per-agent genomes
            # Use genome's hidden layers and output size as template
            for agent, obs in observations.items():
                obs_dim = len(obs)
                action_space = self.env_wrapper.action_space(agent)
                action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
                
                # Create agent-specific genome with same architecture pattern
                agent_genome = FixedTopologyGenome(
                    layer_sizes=[obs_dim] + genome.layer_sizes[1:-1] + [action_dim],
                    random_seed=42
                )
                
                # Try to transfer weights if architectures are compatible
                # For now, just use random initialization
                policies[agent] = FeedforwardNeuralPolicy(agent_genome)
        
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
        
        if self.fitness_aggregation == 'sum':
            return float(np.sum(rewards))
        elif self.fitness_aggregation == 'mean':
            return float(np.mean(rewards))
        elif self.fitness_aggregation == 'min':
            return float(np.min(rewards))
        elif self.fitness_aggregation == 'max':
            return float(np.max(rewards))
        elif self.fitness_aggregation == 'team':
            return self._aggregate_team_fitness(cumulative_rewards)
        
        return 0.0
    
    def _aggregate_team_fitness(self, cumulative_rewards: Dict[str, float]) -> float:
        """
        Aggregate fitness for team-based competitive scenarios.
        
        Args:
            cumulative_rewards: Dict mapping agent_id to reward
            
        Returns:
            Team fitness (difference between team totals)
        """
        if not self.team_mapping:
            # Fallback to sum if no team mapping provided
            return float(np.sum(list(cumulative_rewards.values())))
        
        team_rewards = {}
        for agent, reward in cumulative_rewards.items():
            team = self.team_mapping.get(agent, 'default')
            team_rewards[team] = team_rewards.get(team, 0.0) + reward
        
        # For competitive scenarios, return difference between teams
        # Assumes two teams; customize as needed
        team_values = list(team_rewards.values())
        if len(team_values) >= 2:
            return float(team_values[0] - team_values[1])
        
        return float(sum(team_values))
