from typing import Any, Optional
import numpy as np
from ..genomes import FixedTopologyGenome
from .action_heads import ActionHead, make_head


class FeedforwardNeuralPolicy:
    """
    Feedforward neural network policy.

    Maps observations to actions using a feedforward neural network
    whose weights are stored in a FixedTopologyGenome. The policy
    maintains a reference to the genome, so weight updates are
    immediately reflected in the policy's behavior.

    When an ``action_space`` is supplied (or an :class:`ActionHead` is
    injected via ``head``), :meth:`get_action` returns a value valid for
    that space (``int`` for ``Discrete``, ``np.ndarray`` for ``Box``).
    When neither is supplied the policy returns the raw network output
    — this preserves the historical behaviour used by most existing
    tests/examples.

    Attributes:
        genome: The genome encoding network weights
        input_size: Number of input features
        output_size: Number of output actions
        layer_sizes: Network architecture
        head: Optional :class:`ActionHead` applied in :meth:`get_action`.
    """

    def __init__(
        self,
        genome: FixedTopologyGenome,
        action_space: Optional[Any] = None,
        head: Optional[ActionHead] = None,
        masked: bool = False,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize policy from genome.

        Args:
            genome: FixedTopologyGenome encoding network weights.
            action_space: Optional gymnasium action space; if provided, a
                suitable :class:`ActionHead` is constructed and applied in
                :meth:`get_action`.
            head: Optional explicit :class:`ActionHead` override (takes
                precedence over ``action_space``).
            masked: If ``True`` and ``action_space`` is ``Discrete``,
                construct a :class:`MaskedDiscreteHead` so that
                :meth:`get_action` can accept an ``action_mask`` kwarg.
                Ignored when ``head`` is supplied explicitly.
            rng: Optional :class:`numpy.random.Generator` used for
                stochastic action noise when ``deterministic=False``.
                Defaults to a fresh ``np.random.default_rng()`` so the
                policy never touches the global legacy RNG.  Only
                relevant for the ``deterministic=False`` exploration
                branch, which production evaluators do not exercise.

        Raises:
            ValueError: If ``genome`` is ``None``.
            TypeError: If ``genome`` is not a :class:`FixedTopologyGenome`.
            NotImplementedError: If ``action_space`` is :class:`MultiDiscrete`,
                :class:`Tuple`, or :class:`Dict` (unsupported by this
                policy; use AgileRL trainers instead).

        Example:
            >>> genome = FixedTopologyGenome([10, 64, 4], activation='tanh')
            >>> policy = FeedforwardNeuralPolicy(genome)
            >>> observation = np.random.randn(10)
            >>> action = policy.get_action(observation)
        """
        if genome is None:
            raise ValueError("Genome cannot be None")
        if not isinstance(genome, FixedTopologyGenome):
            raise TypeError(f"Expected FixedTopologyGenome, got {type(genome)}")
        self.genome = genome
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        if head is not None:
            self.head: Optional[ActionHead] = head
        elif action_space is not None:
            self.head = make_head(action_space, masked=masked)
        else:
            self.head = None
    
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
        hidden_activation = self.genome.activation
        output_activation = getattr(self.genome, 'output_activation', hidden_activation)
        
        # Start with input
        current_activation = observation
        
        # Forward through each layer
        final_layer_idx = len(layer_ranges) - 1
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
            
            # Hidden layers use the genome activation; the final layer can opt out
            # or use a different activation for action logits.
            layer_activation = output_activation if i == final_layer_idx else hidden_activation
            current_activation = self._apply_activation(current_activation, layer_activation)
        
        return current_activation
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Get action for observation.

        When an action head has been configured (via ``action_space`` or
        ``head`` in :meth:`__init__`), the raw network output is decoded
        into an env-valid action (``int`` for Discrete, ``np.ndarray``
        for Box).  Without a head, the raw output vector is returned, as
        in the historical behaviour.

        Args:
            observation: Input observation vector.
            deterministic: If False, add Gaussian noise (std=0.1) to the
                raw network output before decoding.  Not meaningful for
                EA workflows — exploration in evolutionary search happens
                via DEAP mutation operators. Present for interface parity with RL-based policies
                that share the same call sites.
            action_mask: Optional legal-action mask. Forwarded to the
                head's ``decode`` when the head declares
                ``supports_mask=True`` (i.e.
                :class:`MaskedDiscreteHead`); silently ignored
                otherwise for backward compatibility.

        Returns:
            Decoded action.  Type depends on the configured head:
            ``int`` (Discrete), ``np.ndarray`` (Box), or raw ``np.ndarray``
            (no head).

        Example:
            >>> action = policy.get_action(observation, deterministic=True)
        """
        action = self.forward(observation)

        if not deterministic:
            # Add small Gaussian noise for exploration
            # Note: this should not be used with DEAP, which has its own mutation operators
            noise = self._rng.normal(0, 0.1, size=action.shape)
            action = action + noise

        if self.head is not None:
            if action_mask is not None and getattr(self.head, "supports_mask", False):
                return self.head.decode(action, action_mask=action_mask)
            return self.head.decode(action)
        return action

    def reset(self) -> None:
        """No-op to satisfy :class:`PolicyProtocol`."""
        return None
    
    def __repr__(self) -> str:
        """String representation of policy."""
        return (
            f"FeedforwardNeuralPolicy("
            f"input_size={self.input_size}, "
            f"output_size={self.output_size}, "
            f"layers={self.layer_sizes})"
        )
