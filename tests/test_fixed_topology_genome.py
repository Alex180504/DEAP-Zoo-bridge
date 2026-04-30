import pytest
import numpy as np
from deap_pettingzoo.genomes.fixed_topology import FixedTopologyGenome


class TestFixedTopologyGenomeInitialization:
    """Test genome initialization."""
    
    def test_simple_network_creation(self):
        """Test creating a simple 2-layer network."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            activation='tanh',
            random_seed=42
        )
        
        assert genome.layer_sizes == [10, 5]
        assert genome.activation == 'tanh'
        assert genome.output_activation == 'tanh'
        assert genome.weights is not None
        assert isinstance(genome.weights, np.ndarray)
    
    def test_multilayer_network_creation(self):
        """Test creating a multi-layer network."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 32, 4],
            activation='relu',
            random_seed=42
        )
        
        assert genome.layer_sizes == [10, 64, 32, 4]
        assert len(genome.weights) > 0
    
    def test_weight_initialization_range(self):
        """Test weights are initialized within specified range."""
        weight_range = (-0.5, 0.5)
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            weight_range=weight_range,
            random_seed=42
        )
        
        assert np.all(genome.weights >= weight_range[0])
        assert np.all(genome.weights <= weight_range[1])
    
    def test_reproducible_initialization(self):
        """Test same seed produces same weights."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        genome2 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        np.testing.assert_array_equal(genome1.weights, genome2.weights)
    
    def test_different_seeds_different_weights(self):
        """Test different seeds produce different weights."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        genome2 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=123
        )
        
        assert not np.allclose(genome1.weights, genome2.weights)
    
    def test_invalid_layer_sizes_raises_error(self):
        """Test that invalid layer sizes raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 layers"):
            FixedTopologyGenome(layer_sizes=[10])
        
        with pytest.raises(ValueError, match="at least 2 layers"):
            FixedTopologyGenome(layer_sizes=[])
    
    def test_invalid_activation_raises_error(self):
        """Test that invalid activation function raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            FixedTopologyGenome(
                layer_sizes=[10, 5],
                activation='invalid_activation'
            )
    
    def test_supported_activations(self):
        """Test all supported activation functions."""
        for activation in ['relu', 'tanh', 'sigmoid', 'linear']:
            genome = FixedTopologyGenome(
                layer_sizes=[10, 5],
                activation=activation,
                random_seed=42
            )
            assert genome.activation == activation

    def test_explicit_output_activation(self):
        """Test hidden and output activations can differ."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            activation='relu',
            output_activation='linear',
            random_seed=42
        )

        assert genome.activation == 'relu'
        assert genome.output_activation == 'linear'

    def test_invalid_output_activation_raises_error(self):
        """Test that invalid output activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output activation"):
            FixedTopologyGenome(
                layer_sizes=[10, 5],
                activation='relu',
                output_activation='invalid'
            )


class TestFixedTopologyGenomeWeightManagement:
    """Test weight getter/setter methods."""
    
    def test_get_weights_returns_array(self):
        """Test get_weights returns numpy array."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        weights = genome.get_weights()
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1  # Should be flat array
    
    def test_set_weights_updates_genome(self):
        """Test set_weights correctly updates genome."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        original_weights = genome.get_weights().copy()
        new_weights = np.random.randn(len(original_weights))
        
        genome.set_weights(new_weights)
        
        np.testing.assert_array_equal(genome.get_weights(), new_weights)
        assert not np.allclose(genome.get_weights(), original_weights)
    
    def test_set_weights_wrong_shape_raises_error(self):
        """Test set_weights raises error for wrong shape."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        wrong_size_weights = np.random.randn(100)  # Too many weights
        
        with pytest.raises(ValueError, match="shape"):
            genome.set_weights(wrong_size_weights)
    
    def test_get_weight_count(self):
        """Test get_weight_count returns correct total."""
        # Network: 10 -> 5
        # Weights: (10*5 + 5) = 55
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        expected_count = 10 * 5 + 5  # weights + biases
        assert genome.get_weight_count() == expected_count
        assert len(genome.get_weights()) == expected_count
    
    def test_get_weight_count_multilayer(self):
        """Test get_weight_count for multi-layer network."""
        # Network: 10 -> 64 -> 32 -> 4
        # Layer 1: 10*64 + 64 = 704
        # Layer 2: 64*32 + 32 = 2080
        # Layer 3: 32*4 + 4 = 132
        # Total: 2916
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 32, 4],
            random_seed=42
        )
        
        expected = (10*64 + 64) + (64*32 + 32) + (32*4 + 4)
        assert genome.get_weight_count() == expected
    
    def test_get_weight_ranges(self):
        """Test get_weight_ranges returns correct indices."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        ranges = genome.get_weight_ranges()
        
        # Should have 2 layers (10->64, 64->4)
        assert len(ranges) == 2
        
        # First layer: 10*64 + 64 = 704 weights
        assert ranges[0] == (0, 704)
        
        # Second layer: 64*4 + 4 = 260 weights
        assert ranges[1] == (704, 704 + 260)
        
        # Ranges should be contiguous and cover all weights
        assert ranges[0][0] == 0
        assert ranges[-1][1] == genome.get_weight_count()


class TestFixedTopologyGenomeValidation:
    """Test genome validation."""
    
    def test_validate_returns_true_for_valid_genome(self):
        """Test validate returns True for valid genome."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        assert genome.validate() is True
    
    def test_validate_with_custom_weights(self):
        """Test validate after setting custom weights."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        new_weights = np.random.randn(genome.get_weight_count())
        genome.set_weights(new_weights)
        
        assert genome.validate() is True
    
    def test_validate_detects_invalid_weights(self):
        """Test validate detects NaN or Inf values."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        # Inject NaN
        weights = genome.get_weights().copy()
        weights[0] = np.nan
        genome.set_weights(weights)
        
        assert genome.validate() is False
        
        # Inject Inf
        weights = genome.get_weights().copy()
        weights[0] = np.inf
        genome.set_weights(weights)
        
        assert genome.validate() is False


class TestFixedTopologyGenomeCloning:
    """Test genome cloning."""
    
    def test_clone_creates_deep_copy(self):
        """Test clone creates independent copy."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        genome2 = genome1.clone()
        
        # Should have same weights initially
        np.testing.assert_array_equal(genome1.get_weights(), genome2.get_weights())
        
        # But be independent objects
        assert genome1 is not genome2
        assert genome1.weights is not genome2.weights
    
    def test_clone_independence(self):
        """Test cloned genome is independent."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        genome2 = genome1.clone()
        
        # Modify genome2
        new_weights = np.random.randn(genome2.get_weight_count())
        genome2.set_weights(new_weights)
        
        # genome1 should be unchanged
        assert not np.allclose(genome1.get_weights(), genome2.get_weights())
    
    def test_clone_preserves_attributes(self):
        """Test clone preserves all genome attributes."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 32, 4],
            activation='relu',
            output_activation='linear',
            weight_range=(-0.5, 0.5),
            random_seed=42
        )
        
        genome2 = genome1.clone()
        
        assert genome2.layer_sizes == genome1.layer_sizes
        assert genome2.activation == genome1.activation
        assert genome2.output_activation == genome1.output_activation
        assert genome2.input_size == genome1.input_size
        assert genome2.output_size == genome1.output_size


class TestFixedTopologyGenomeSerialization:
    """Test genome serialization/deserialization."""
    
    def test_to_dict_returns_dict(self):
        """Test to_dict returns dictionary."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            activation='tanh',
            random_seed=42
        )
        
        data = genome.to_dict()
        
        assert isinstance(data, dict)
        assert 'layer_sizes' in data
        assert 'activation' in data
        assert 'output_activation' in data
        assert 'weights' in data
    
    def test_to_dict_contains_all_info(self):
        """Test to_dict contains complete genome information."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            activation='relu',
            output_activation='linear',
            weight_range=(-0.5, 0.5),
            random_seed=42
        )
        
        data = genome.to_dict()
        
        assert data['layer_sizes'] == [10, 64, 4]
        assert data['activation'] == 'relu'
        assert data['output_activation'] == 'linear'
        assert 'weights' in data
        assert isinstance(data['weights'], list)  # Serialized as list
    
    def test_from_dict_reconstructs_genome(self):
        """Test from_dict correctly reconstructs genome."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            activation='tanh',
            random_seed=42
        )
        
        data = genome1.to_dict()
        genome2 = FixedTopologyGenome.from_dict(data)
        
        assert genome2.layer_sizes == genome1.layer_sizes
        assert genome2.activation == genome1.activation
        assert genome2.output_activation == genome1.output_activation
        np.testing.assert_array_equal(genome2.get_weights(), genome1.get_weights())
    
    def test_serialization_roundtrip(self):
        """Test full serialization/deserialization cycle."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 32, 4],
            activation='sigmoid',
            random_seed=42
        )
        
        # Serialize
        data = genome1.to_dict()
        
        # Deserialize
        genome2 = FixedTopologyGenome.from_dict(data)
        
        # Should be equivalent (but not same object)
        assert genome1 is not genome2
        assert genome2.layer_sizes == genome1.layer_sizes
        assert genome2.activation == genome1.activation
        assert genome2.output_activation == genome1.output_activation
        np.testing.assert_array_equal(genome2.get_weights(), genome1.get_weights())

    def test_from_dict_backfills_legacy_output_activation(self):
        """Test legacy serialized genomes default output activation to activation."""
        data = {
            'layer_sizes': [10, 64, 4],
            'activation': 'relu',
            'weights': [0.0] * ((10 * 64 + 64) + (64 * 4 + 4)),
            '_weight_range': (-1.0, 1.0),
            '_random_seed': 42,
        }

        genome = FixedTopologyGenome.from_dict(data)

        assert genome.activation == 'relu'
        assert genome.output_activation == 'relu'


class TestFixedTopologyGenomeMutation:
    """Test genome mutation."""
    
    def test_mutate_inplace_modifies_genome(self):
        """Test inplace mutation modifies genome."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        original_weights = genome.get_weights().copy()
        
        genome.mutate(mutation_rate=1.0, mutation_std=0.1, inplace=True)
        
        # Weights should be different
        assert not np.allclose(genome.get_weights(), original_weights)
    
    def test_mutate_returns_new_genome(self):
        """Test mutation can return new genome."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        original_weights = genome1.get_weights().copy()
        
        genome2 = genome1.mutate(mutation_rate=1.0, mutation_std=0.1, inplace=False)
        
        # Original unchanged
        np.testing.assert_array_equal(genome1.get_weights(), original_weights)
        
        # New genome is different
        assert not np.allclose(genome2.get_weights(), original_weights)
    
    def test_mutation_rate_zero_no_change(self):
        """Test mutation_rate=0 causes no change."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        original_weights = genome.get_weights().copy()
        
        genome.mutate(mutation_rate=0.0, mutation_std=1.0, inplace=True)
        
        np.testing.assert_array_equal(genome.get_weights(), original_weights)
    
    def test_mutation_rate_one_all_change(self):
        """Test mutation_rate=1.0 mutates all weights."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            random_seed=42
        )
        
        original_weights = genome.get_weights().copy()
        
        genome.mutate(mutation_rate=1.0, mutation_std=0.5, inplace=True)
        
        # All weights should have changed
        assert not np.allclose(genome.get_weights(), original_weights)
    
    def test_mutation_std_affects_magnitude(self):
        """Test mutation_std affects change magnitude."""
        genome1 = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        genome2 = genome1.clone()
        
        original_weights = genome1.get_weights().copy()
        
        # Small mutation
        genome1.mutate(mutation_rate=1.0, mutation_std=0.01, inplace=True)
        small_diff = np.abs(genome1.get_weights() - original_weights).mean()
        
        # Large mutation
        genome2.mutate(mutation_rate=1.0, mutation_std=1.0, inplace=True)
        large_diff = np.abs(genome2.get_weights() - original_weights).mean()
        
        # Larger std should cause larger changes (statistically)
        assert large_diff > small_diff


class TestFixedTopologyGenomeProperties:
    """Test genome properties."""
    
    def test_input_size_property(self):
        """Test input_size property."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        assert genome.input_size == 10
    
    def test_output_size_property(self):
        """Test output_size property."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 4],
            random_seed=42
        )
        
        assert genome.output_size == 4
    
    def test_hidden_layers_property(self):
        """Test hidden_layers property."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 64, 32, 4],
            random_seed=42
        )
        
        assert genome.hidden_layers == [64, 32]
    
    def test_no_hidden_layers(self):
        """Test network with no hidden layers."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 4],
            random_seed=42
        )
        
        assert genome.hidden_layers == []
        assert genome.input_size == 10
        assert genome.output_size == 4


class TestFixedTopologyGenomeEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_very_small_network(self):
        """Test smallest valid network (1 input, 1 output)."""
        genome = FixedTopologyGenome(
            layer_sizes=[1, 1],
            random_seed=42
        )
        
        assert genome.get_weight_count() == 1 + 1  # 1 weight + 1 bias
        assert genome.validate() is True
    
    def test_very_large_network(self):
        """Test large network can be created."""
        genome = FixedTopologyGenome(
            layer_sizes=[100, 256, 256, 128, 10],
            random_seed=42
        )
        
        assert genome.validate() is True
        assert genome.get_weight_count() > 0
    
    def test_weight_range_boundary(self):
        """Test weights at range boundaries."""
        genome = FixedTopologyGenome(
            layer_sizes=[10, 5],
            weight_range=(-10.0, 10.0),
            random_seed=42
        )
        
        weights = genome.get_weights()
        assert np.all(weights >= -10.0)
        assert np.all(weights <= 10.0)
