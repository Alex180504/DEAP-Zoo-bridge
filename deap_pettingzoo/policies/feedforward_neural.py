from typing import Optional
import numpy as np
from ..genomes import FixedTopologyGenome


class FeedforwardNeuralPolicy:
    """
    Feedforward neural network policy.
    
    Maps observations to actions using a feedforward neural network
    whose weights are stored in a FixedTopologyGenome. The policy
    maintains a reference to the genome, so weight updates are
    immediately reflected in the policy's behavior.
    
    Attributes:
        genome: The genome encoding network weights
        input_size: Number of input features
        output_size: Number of output actions
        layer_sizes: Network architecture
    """
    
    def __init__(self, genome: FixedTopologyGenome) -> None:
        """
        Initialize policy from genome.
        
        Args:
            genome: FixedTopologyGenome encoding network weights
            
        Example:
            >>> genome = FixedTopologyGenome([10, 64, 4], activation='tanh')
            >>> policy = FeedforwardNeuralPolicy(genome)
            >>> observation = np.random.randn(10)
            >>> action = policy.get_action(observation)
        """
        self.genome = genome
    
    @property
    def input_size(self) -> int:
        """Get number of input features."""
        return self.genome.input_size
    
    @property
    def output_size(self) -> int:
        """Get number of output actions."""
        return self.genome.output_size
    
    @property
    def layer_sizes(self):
        """Get network layer sizes."""
        return self.genome.layer_sizes
    
    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            x: Input array
            activation: Activation function name
            
        Returns:
            Activated output
        """
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif activation == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.
        
        Args:
            observation: Input observation vector
            
        Returns:
            np.ndarray: Output action vector
            
        Raises:
            ValueError: If observation size doesn't match input size
            
        Example:
            >>> observation = np.array([1.0, 2.0, 3.0])
            >>> action = policy.forward(observation)
        """
        if len(observation) != self.input_size:
            raise ValueError(
                f"Observation input size mismatch: got {len(observation)}, "
                f"expected {self.input_size}"
            )
        
        # Get weights and layer structure
        weights = self.genome.get_weights()
        layer_ranges = self.genome.get_weight_ranges()
        activation = self.genome.activation
        
        # Start with input
        current_activation = observation
        
        # Forward through each layer
        for i, (start_idx, end_idx) in enumerate(layer_ranges):
            # Get layer dimensions
            input_dim = self.genome.layer_sizes[i]
            output_dim = self.genome.layer_sizes[i + 1]
            
            # Extract weights and biases for the current layer
            layer_weights = weights[start_idx:end_idx]
            
            # First input_dim * output_dim are weights, rest are biases
            W = layer_weights[:input_dim * output_dim].reshape(output_dim, input_dim)
            b = layer_weights[input_dim * output_dim:]
            
            # Linear transformation
            current_activation = W @ current_activation + b
            
            # Apply activation function
            current_activation = self._apply_activation(current_activation, activation)
        
        return current_activation
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action for observation.
        
        Args:
            observation: Input observation vector
            deterministic: If False, add Gaussian noise for exploration of action space            
        Returns:
            np.ndarray: Action vector
            
        Example:
            >>> action = policy.get_action(observation, deterministic=True)
        """
        action = self.forward(observation)
        
        if not deterministic:
            # Add small Gaussian noise for exploration
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = action + noise
        
        return action
    
    def __repr__(self) -> str:
        """String representation of policy."""
        return (
            f"FeedforwardNeuralPolicy("
            f"input_size={self.input_size}, "
            f"output_size={self.output_size}, "
            f"layers={self.layer_sizes})"
        )
