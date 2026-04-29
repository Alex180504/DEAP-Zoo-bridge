from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class FixedTopologyGenome:
    """
    Fixed-topology neural network genome.
    
    Encodes a feedforward neural network with fixed architecture as a flat
    vector of weights and biases. Compatible with DEAP's genetic operators and CMA-ES via cma_integration.py.
    Used as a template for Qdax weight-compatible Flax policy.
    
    Attributes:
        layer_sizes: List of integers defining network architecture [input, hidden..., output]
        activation: Activation function name ('relu', 'tanh', 'sigmoid', 'linear')
        weights: Flat numpy array of all network weights and biases
        input_size: Number of input units
        output_size: Number of output units
        hidden_layers: List of hidden layer sizes
    """
    
    SUPPORTED_ACTIVATIONS = {'relu', 'tanh', 'sigmoid', 'linear'}
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'tanh',
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize fixed-topology genome.
        
        Args:
            layer_sizes: Network architecture [input, hidden1, hidden2, ..., output]
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'linear')
            weight_range: (min, max) for weight initialization
            random_seed: Seed for reproducible initialization
            
        Raises:
            ValueError: If layer_sizes has fewer than 2 layers
            ValueError: If activation not in supported set
            
        Example:
            >>> genome = FixedTopologyGenome(
            ...     layer_sizes=[10, 64, 32, 4],
            ...     activation='tanh',
            ...     weight_range=(-0.5, 0.5),
            ...     random_seed=42
            ... )
        """
        # Validate layer sizes
        if len(layer_sizes) < 2:
            raise ValueError(
                f"Network must have at least 2 layers (input and output), "
                f"got {len(layer_sizes)}"
            )
        
        # Validate activation
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation function '{activation}'. "
                f"Supported: {self.SUPPORTED_ACTIVATIONS}"
            )
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self._weight_range = weight_range
        self._random_seed = random_seed
        
        # Use an encapsulated random number generator
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize weights
        weight_count = self.get_weight_count()
        self.weights = self.rng.uniform(
            low=weight_range[0],
            high=weight_range[1],
            size=weight_count
        )
    
    @property
    def input_size(self) -> int:
        """Get number of input units."""
        return self.layer_sizes[0]
    
    @property
    def output_size(self) -> int:
        """Get number of output units."""
        return self.layer_sizes[-1]
    
    @property
    def hidden_layers(self) -> List[int]:
        """Get list of hidden layer sizes."""
        return self.layer_sizes[1:-1]
    
    def get_weight_count(self) -> int:
        """
        Calculate total number of weights in network.
        
        Returns:
            int: Total weight count (weights + biases for all layers)
            
        Example:
            >>> genome = FixedTopologyGenome([10, 64, 4])
            >>> genome.get_weight_count()
            836  # (10*64 + 64) + (64*4 + 4)
        """
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            # Weights + biases for this layer
            total += input_size * output_size + output_size
        return total
    
    def get_weight_ranges(self) -> List[Tuple[int, int]]:
        """
        Get index ranges for each layer's weights.
        
        Returns:
            List of (start, end) tuples for each layer
            
        Example:
            >>> genome = FixedTopologyGenome([10, 64, 4])
            >>> genome.get_weight_ranges()
            [(0, 704), (704, 836)]  # Layer 1, Layer 2
        """
        ranges = []
        current_idx = 0
        
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            layer_weight_count = input_size * output_size + output_size
            
            ranges.append((current_idx, current_idx + layer_weight_count))
            current_idx += layer_weight_count
        
        return ranges
    
    def get_weights(self) -> np.ndarray:
        """
        Get genome as flat weight vector.
        
        Returns:
            np.ndarray: Flat array of weights
        """
        return self.weights
    
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set genome from flat weight vector.
        
        Args:
            weights: Flat array of weights
            
        Raises:
            ValueError: If weights shape doesn't match architecture
        """
        expected_count = self.get_weight_count()
        if len(weights) != expected_count:
            raise ValueError(
                f"Weight vector shape mismatch: got {len(weights)} elements, "
                f"but network requires {expected_count}"
            )
        
        self.weights = np.array(weights, dtype=np.float64)
    
    def validate(self) -> bool:
        """
        Validate genome structure and values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Check for NaN or Inf
        if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
            return False
        
        # Check correct weight count
        if len(self.weights) != self.get_weight_count():
            return False
        
        return True
    
    def clone(self) -> 'FixedTopologyGenome':
        """
        Create a deep copy of the genome.
        
        Returns:
            FixedTopologyGenome: Cloned genome instance
        """
        cloned_genome = FixedTopologyGenome(
            layer_sizes=self.layer_sizes,
            activation=self.activation,
            weight_range=self._weight_range,
            random_seed=None  # Allows the clone have its own independent RNG stream
        )
        cloned_genome.set_weights(self.get_weights().copy())
        # Keep the clone's own freshly seeded RNG so it produces an independent stream
        return cloned_genome
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize genome to dictionary.
        
        Returns:
            dict: Serialized genome data
        """
        return {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'weights': self.weights.tolist(),  # Convert to list for JSON compatibility
            '_weight_range': self._weight_range,
            '_random_seed': self._random_seed,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'FixedTopologyGenome':
        """
        Deserialize genome from dictionary.
        
        Args:
            data: Serialized genome data
            
        Returns:
            FixedTopologyGenome: Reconstructed genome
        """
        genome = FixedTopologyGenome(
            layer_sizes=data['layer_sizes'],
            activation=data['activation'],
            weight_range=data.get('_weight_range', (-1.0, 1.0)),
            random_seed=data.get('_random_seed', None)
        )
        genome.set_weights(np.array(data['weights'], dtype=np.float64))
        return genome
    
    def mutate(
        self,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.2,
        inplace: bool = True
    ) -> Optional['FixedTopologyGenome']:
        """
        Apply Gaussian mutation to weights.
        
        Args:
            mutation_rate: Probability of mutating each weight (0.0-1.0)
            mutation_std: Standard deviation of mutation noise
            inplace: Modify this genome (True) or return new one (False)
            
        Returns:
            None if inplace, otherwise new mutated genome
            
        Example:
            >>> genome = FixedTopologyGenome([3, 2], random_seed=42)
            >>> original_weights = genome.get_weights().copy()
            >>> genome.mutate(mutation_rate=1.0, mutation_std=0.1)
            >>> np.all(original_weights != genome.get_weights())
            True
        """
        if inplace:
            target = self
        else:
            target = self.clone()
        
        # Create mutation mask (which weights to mutate)
        mutation_mask = target.rng.random(len(target.weights)) < mutation_rate
        
        # Apply Gaussian noise to selected weights
        if np.any(mutation_mask):
            noise = target.rng.normal(
                loc=0.0,
                scale=mutation_std,
                size=np.count_nonzero(mutation_mask)
            )
            target.weights[mutation_mask] += noise
        
        if not inplace:
            return target
        return None
    
    def __repr__(self) -> str:
        """String representation of genome."""
        return (
            f"FixedTopologyGenome("
            f"layers={self.layer_sizes}, "
            f"activation='{self.activation}', "
            f"weights={len(self.weights)})"
        )
