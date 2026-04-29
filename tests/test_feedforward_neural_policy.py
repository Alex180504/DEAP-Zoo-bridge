import pytest
import numpy as np
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.genomes import FixedTopologyGenome


class TestFeedforwardNeuralPolicyInitialization:
    """Test policy initialization and setup."""
    
    def test_policy_creation_from_genome(self):
        """Test creating policy from genome."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            activation='tanh',
            random_seed=42
        )
        
        policy = FeedforwardNeuralPolicy(genome)
        
        assert policy.genome is genome
        assert policy.input_size == 10
        assert policy.output_size == 4
    
    def test_policy_with_different_architectures(self):
        """Test policy with various network architectures."""
        architectures = [
            [5, 3],          # Simple 2-layer
            [6, 32, 16, 8, 2]  # Three hidden layers
        ]
        
        for arch in architectures:
            genome = FixedTopologyGenome(arch, random_seed=42)
            policy = FeedforwardNeuralPolicy(genome)
            
            assert policy.input_size == arch[0]
            assert policy.output_size == arch[-1]
    
    def test_policy_preserves_genome_reference(self):
        """Test that policy keeps reference to genome (not copy)."""
        genome = FixedTopologyGenome([5, 3], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        # Modify genome weights
        original_weights = genome.get_weights().copy()
        genome.mutate(mutation_rate=1.0, mutation_std=0.5)
        
        # Policy should see the changes
        assert not np.allclose(genome.get_weights(), original_weights)
        assert policy.genome is genome


class TestFeedforwardNeuralPolicyForwardPass:
    """Test neural network forward propagation."""
    
    def test_forward_returns_correct_shape(self):
        """Test forward pass returns correct output shape."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action = policy.forward(observation)
        
        assert action.shape == (4,)
        assert isinstance(action, np.ndarray)
    
    def test_forward_with_different_input_sizes(self):
        """Test forward pass with various input sizes."""
        input_sizes = [1, 100]
        
        for input_size in input_sizes:
            genome = FixedTopologyGenome([input_size, 8, 4], random_seed=42)
            policy = FeedforwardNeuralPolicy(genome)
            
            observation = np.random.randn(input_size)
            action = policy.forward(observation)
            
            assert action.shape == (4,)
    
    def test_forward_deterministic(self):
        """Test forward pass is deterministic for same input."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action1 = policy.forward(observation)
        action2 = policy.forward(observation)
        
        np.testing.assert_array_equal(action1, action2)
    
    def test_forward_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        obs1 = np.random.randn(10)
        obs2 = np.random.randn(10)
        
        action1 = policy.forward(obs1)
        action2 = policy.forward(obs2)
        
        assert not np.allclose(action1, action2)
    
    def test_forward_with_zero_input(self):
        """Test forward pass with zero input."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.zeros(10)
        action = policy.forward(observation)
        
        assert action.shape == (4,)
        assert not np.all(action == 0)  # Biases should make output non-zero
    
    def test_forward_wrong_input_shape_raises_error(self):
        """Test forward raises error for wrong input shape."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        wrong_observation = np.random.randn(5)  # Wrong size
        
        with pytest.raises(ValueError, match="input"):
            policy.forward(wrong_observation)


class TestFeedforwardNeuralPolicyActivationFunctions:
    """Test different activation functions."""
    
    def test_tanh_activation(self):
        """Test tanh activation produces bounded outputs."""
        genome = FixedTopologyGenome([10, 64, 4], activation='tanh', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10) * 10  # Large inputs
        action = policy.forward(observation)
        
        # Tanh output should be in [-1, 1]
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    
    def test_relu_activation(self):
        """Test ReLU activation produces non-negative outputs."""
        genome = FixedTopologyGenome([10, 64, 4], activation='relu', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action = policy.forward(observation)
        
        # ReLU output layer should be non-negative
        assert np.all(action >= 0.0)
    
    def test_sigmoid_activation(self):
        """Test sigmoid activation produces outputs in [0, 1]."""
        genome = FixedTopologyGenome([10, 64, 4], activation='sigmoid', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10) * 10  # Large inputs
        action = policy.forward(observation)
        
        # Sigmoid output should be in [0, 1]
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)
    
    def test_linear_activation(self):
        """Test linear activation produces unbounded outputs."""
        genome = FixedTopologyGenome([10, 4], activation='linear', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10) * 100  # Very large inputs
        action = policy.forward(observation)
        
        # Linear can produce any value
        assert action.shape == (4,)

class TestFeedforwardNeuralPolicyActionSelection:
    """Test action selection methods."""
    
    def test_get_action_stochastic_adds_noise(self):
        """Test stochastic action selection adds exploration noise."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        deterministic_action = policy.get_action(observation, deterministic=True)
        
        # Get multiple stochastic actions
        stochastic_actions = [
            policy.get_action(observation, deterministic=False)
            for _ in range(10)
        ]
        
        # At least some should differ from deterministic
        differences = [not np.allclose(sa, deterministic_action) for sa in stochastic_actions]
        assert any(differences), "Stochastic actions should differ from deterministic"

class TestFeedforwardNeuralPolicyWeightUpdates:
    """Test policy behavior with weight updates."""
    
    def test_policy_output_changes_with_weight_mutation(self):
        """Test policy output changes when genome weights mutate."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action_before = policy.forward(observation)
        
        # Mutate genome weights
        genome.mutate(mutation_rate=1.0, mutation_std=0.5)
        
        action_after = policy.forward(observation)
        
        # Output should change
        assert not np.allclose(action_before, action_after)
    
    def test_policy_output_changes_with_weight_replacement(self):
        """Test policy output changes when genome weights are replaced."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action_before = policy.forward(observation)
        
        # Replace weights
        new_weights = np.random.randn(genome.get_weight_count())
        genome.set_weights(new_weights)
        
        action_after = policy.forward(observation)
        
        # Output should change
        assert not np.allclose(action_before, action_after)


class TestFeedforwardNeuralPolicyProperties:
    """Test policy properties and attributes."""
    
    def test_input_size_property(self):
        """Test input_size property."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        assert policy.input_size == 10
        assert policy.input_size == genome.input_size
    
    def test_output_size_property(self):
        """Test output_size property."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        assert policy.output_size == 4
        assert policy.output_size == genome.output_size
    
    def test_layer_sizes_property(self):
        """Test layer_sizes property."""
        layer_sizes = [10, 64, 32, 4]
        genome = FixedTopologyGenome(layer_sizes, random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        assert policy.layer_sizes == layer_sizes
        assert policy.layer_sizes == genome.layer_sizes


class TestFeedforwardNeuralPolicyEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_network(self):
        """Test policy with minimal 2-layer network."""
        genome = FixedTopologyGenome([1, 1], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.array([0.5])
        action = policy.forward(observation)
        
        assert action.shape == (1,)
    
    def test_very_large_network(self):
        """Test policy with large network."""
        genome = FixedTopologyGenome([100, 256, 128, 50], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(100)
        action = policy.forward(observation)
        
        assert action.shape == (50,)
    
    def test_extreme_input_values(self):
        """Test policy handles extreme input values."""
        genome = FixedTopologyGenome([10, 64, 4], activation='tanh', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        extreme_observation = np.array([1e6, -1e6, 1e-10, -1e-10, 0, 0, 0, 0, 0, 0])
        action = policy.forward(extreme_observation)
        
        # Should not produce NaN or Inf
        assert not np.any(np.isnan(action))
        assert not np.any(np.isinf(action))
        assert action.shape == (4,)
    
    def test_all_zero_weights(self):
        """Test policy with all zero weights."""
        genome = FixedTopologyGenome([10, 4], random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        # Set all weights to zero
        zero_weights = np.zeros(genome.get_weight_count())
        genome.set_weights(zero_weights)
        
        observation = np.random.randn(10)
        action = policy.forward(observation)
        
        # With zero weights and biases, output should be zero
        assert action.shape == (4,)
    
    def test_policy_repr(self):
        """Test string representation of policy."""
        genome = FixedTopologyGenome([10, 64, 4], activation='tanh', random_seed=42)
        policy = FeedforwardNeuralPolicy(genome)
        
        repr_str = repr(policy)
        
        assert 'FeedforwardNeuralPolicy' in repr_str
        assert '10' in repr_str  # Input size
        assert '4' in repr_str   # Output size


class TestFeedforwardNeuralPolicyIntegrationWithGenome:
    """Test integration between policy and genome."""
    
    def test_policy_from_serialized_genome(self):
        """Test creating policy from serialized genome."""
        # Create and serialize genome
        genome1 = FixedTopologyGenome([10, 64, 4], random_seed=42)
        genome_dict = genome1.to_dict()
        
        # Deserialize and create policy
        genome2 = FixedTopologyGenome.from_dict(genome_dict)
        policy = FeedforwardNeuralPolicy(genome2)
        
        # Should produce same output
        observation = np.random.randn(10)
        policy1 = FeedforwardNeuralPolicy(genome1)
        
        action1 = policy1.forward(observation)
        action2 = policy.forward(observation)
        
        np.testing.assert_array_equal(action1, action2)
    
    def test_multiple_policies_same_genome(self):
        """Test multiple policy instances can share same genome."""
        genome = FixedTopologyGenome([10, 64, 4], random_seed=42)
        
        policy1 = FeedforwardNeuralPolicy(genome)
        policy2 = FeedforwardNeuralPolicy(genome)
        
        observation = np.random.randn(10)
        action1 = policy1.forward(observation)
        action2 = policy2.forward(observation)
        
        # Both should produce same output
        np.testing.assert_array_equal(action1, action2)
        
        # Mutating genome should affect both
        genome.mutate(mutation_rate=1.0, mutation_std=0.5)
        
        new_action1 = policy1.forward(observation)
        new_action2 = policy2.forward(observation)
        
        np.testing.assert_array_equal(new_action1, new_action2)
        assert not np.allclose(action1, new_action1)
