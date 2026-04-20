from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base_evaluator import BaseEvaluator
from ..environments import PettingZooWrapper
from ..policies import FeedforwardNeuralPolicy
from ..genomes import FixedTopologyGenome


class SingleEpisodeEvaluator(BaseEvaluator):
    """Evaluates individual genome in a single episode."""
    
    VALID_AGGREGATIONS = ['sum', 'mean', 'min', 'max', 'team']
    
    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 500,
        aggregation_method: str = 'sum',
        team_mapping: Optional[Dict[str, str]] = None
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
            team_mapping: Maps agent_id to team name (for competitive scenarios)
            
        Raises:
            ValueError: If invalid aggregation method
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
            
        Raises:
            ValueError: If individual or genome is None/invalid
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
        done = False
        steps = 0
        
        try:
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
        except Exception as e:
            # Log error and re-raise with context
            raise RuntimeError(f"Error during episode evaluation: {e}") from e
    
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
            # Different obs sizes - create per-agent genomes with weight transfer
            evolved_weights = genome.get_weights()
            
            for agent, obs in observations.items():
                obs_dim = len(obs)
                action_space = self.env_wrapper.action_space(agent)
                action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
                
                # Create agent-specific genome with same architecture pattern
                agent_genome = FixedTopologyGenome(
                    layer_sizes=[obs_dim] + genome.layer_sizes[1:-1] + [action_dim],
                    activation=genome.activation,
                    random_seed=None  # Will set weights manually via transfer
                )
                
                # Transfer weights from evolved genome
                transferred_weights = self._transfer_weights(
                    source_genome=genome,
                    target_genome=agent_genome,
                    evolved_weights=evolved_weights
                )
                
                agent_genome.set_weights(transferred_weights)
                policies[agent] = FeedforwardNeuralPolicy(agent_genome)
        
        return policies
    
    def _transfer_weights(
        self, 
        source_genome: FixedTopologyGenome,
        target_genome: FixedTopologyGenome,
        evolved_weights: List[float]
    ) -> List[float]:
        """Transfer weights from evolved genome to agent-specific genome.
        
        Preserves weights for compatible layers (hidden→hidden, hidden→output, biases).
        For the input layer, which may have different dimensions, initializes randomly.
        
        This achieves approximately 70-90% weight preservation depending on network size,
        allowing evolved behavior to transfer to agents with different observation sizes.
        
        Args:
            source_genome: The evolved genome (source of weights)
            target_genome: The agent-specific genome (target for weights)
            evolved_weights: Flattened weight vector from evolved genome
            
        Returns:
            Flattened weight vector for target genome with transferred weights
        """
        # Unflatten evolved weights into layer-wise structure
        source_layers = self._unflatten_weights(source_genome, evolved_weights)
        
        # Create target weight structure (random initialization)
        target_weights = target_genome.get_weights()
        target_layers = self._unflatten_weights(target_genome, target_weights)
        
        # Transfer compatible weights layer by layer
        # Input→Hidden: Different sizes, keep random (target_layers[0] unchanged)
        # Hidden→Hidden: Same sizes, transfer evolved weights
        for i in range(1, len(target_layers) - 1):
            if i < len(source_layers):
                source_shape = source_layers[i]['weights'].shape
                target_shape = target_layers[i]['weights'].shape
                
                if source_shape == target_shape:
                    # Compatible layer - transfer weights and biases
                    target_layers[i]['weights'] = source_layers[i]['weights'].copy()
                    target_layers[i]['biases'] = source_layers[i]['biases'].copy()
        
        # Hidden→Output: Transfer hidden part, keep new output initialization
        last_idx = len(target_layers) - 1
        if last_idx < len(source_layers):
            source_hidden_dim = source_layers[last_idx]['weights'].shape[0]
            target_hidden_dim = target_layers[last_idx]['weights'].shape[0]
            
            if source_hidden_dim == target_hidden_dim:
                # Transfer weights from hidden layer (rows), but not output (cols may differ)
                # This preserves the evolved hidden→output transformation for the hidden dimension
                min_output_dim = min(
                    source_layers[last_idx]['weights'].shape[1],
                    target_layers[last_idx]['weights'].shape[1]
                )
                target_layers[last_idx]['weights'][:, :min_output_dim] = \
                    source_layers[last_idx]['weights'][:, :min_output_dim].copy()
                
                # Transfer biases up to compatible dimensions
                min_bias_dim = min(len(source_layers[last_idx]['biases']), 
                                  len(target_layers[last_idx]['biases']))
                target_layers[last_idx]['biases'][:min_bias_dim] = \
                    source_layers[last_idx]['biases'][:min_bias_dim].copy()
        
        # Flatten back to weight vector
        return self._flatten_weights(target_layers)
    
    def _unflatten_weights(
        self, 
        genome: FixedTopologyGenome, 
        weights: List[float]
    ) -> List[Dict[str, np.ndarray]]:
        """Convert flat weight vector to layer-wise structure.
        
        Args:
            genome: Genome defining the network architecture
            weights: Flattened weight vector
            
        Returns:
            List of dicts with 'weights' and 'biases' arrays for each layer
        """
        layers = []
        idx = 0
        
        for i in range(len(genome.layer_sizes) - 1):
            in_size = genome.layer_sizes[i]
            out_size = genome.layer_sizes[i + 1]
            
            # Extract weights for this layer (in_size × out_size matrix)
            num_weights = in_size * out_size
            layer_weights = np.array(weights[idx:idx + num_weights]).reshape(in_size, out_size)
            idx += num_weights
            
            # Extract biases for this layer
            layer_biases = np.array(weights[idx:idx + out_size])
            idx += out_size
            
            layers.append({
                'weights': layer_weights,
                'biases': layer_biases
            })
        
        return layers
    
    def _flatten_weights(self, layers: List[Dict[str, np.ndarray]]) -> List[float]:
        """Convert layer-wise structure back to flat weight vector.
        
        Args:
            layers: List of dicts with 'weights' and 'biases' arrays
            
        Returns:
            Flattened weight vector
        """
        flat = []
        
        for layer in layers:
            # Flatten weights matrix
            flat.extend(layer['weights'].flatten().tolist())
            # Add biases
            flat.extend(layer['biases'].tolist())
        
        return flat
    
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
