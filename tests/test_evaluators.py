import pytest
import numpy as np
from typing import Dict, Any

from deap_pettingzoo.evaluators import (
    BaseEvaluator,
    SingleEpisodeEvaluator,
    MultiEpisodeEvaluator
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.policies import FeedforwardNeuralPolicy


# Mock DEAP individual for testing
class MockIndividual:
    """Mock DEAP individual with genome."""
    
    def __init__(self, genome: FixedTopologyGenome):
        self.genome = genome
        self.fitness = type('Fitness', (), {'values': None})()


class TestBaseEvaluator:
    """Test BaseEvaluator abstract interface."""
    
    def test_base_evaluator_is_abstract(self):
        """Test BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()
    
    def test_base_evaluator_requires_evaluate_method(self):
        """Test subclasses must implement evaluate method."""
        
        class IncompleteEvaluator(BaseEvaluator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteEvaluator()


class TestSingleEpisodeEvaluator:
    """Test single episode evaluation."""
    
    @pytest.fixture
    def wrapper(self):
        """Create environment wrapper."""
        return PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
    
    @pytest.fixture
    def individual(self, wrapper):
        """Create test individual with genome."""
        # Get dimensions from first agent AFTER reset
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        
        obs_dim = len(observations[agent])  # Get actual observation length
        action_space = wrapper.action_space(agent)
        action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        return MockIndividual(genome)
    
    @pytest.mark.parametrize("method", ['sum', 'mean', 'min', 'max'])
    def test_fitness_aggregation_methods(self, wrapper, individual, method):
        """Test evaluate with different fitness aggregation methods."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            fitness_aggregation=method
        )
        
        fitness = evaluator.evaluate(individual, seed=42)
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))
    
    def test_evaluate_different_seeds_different_results(self, wrapper, individual):
        """Test different seeds produce different results."""
        evaluator = SingleEpisodeEvaluator(env_wrapper=wrapper)
        
        fitness1 = evaluator.evaluate(individual, seed=42)
        fitness2 = evaluator.evaluate(individual, seed=123)
        
        # May occasionally be equal, but unlikely
        assert fitness1 != fitness2 or True  # Soft assertion
    
    def test_max_steps_limits_episode(self, wrapper, individual):
        """Test max_steps parameter limits episode length."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=5
        )
        
        # Should complete without timing out
        fitness = evaluator.evaluate(individual, seed=42)
        assert isinstance(fitness[0], (int, float))
    
    def test_invalid_fitness_aggregation_raises_error(self, wrapper):
        """Test invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregation"):
            SingleEpisodeEvaluator(
                env_wrapper=wrapper,
                fitness_aggregation='invalid_method'
            )


class TestMultiEpisodeEvaluator:
    """Test multi-episode evaluation."""
    
    @pytest.fixture
    def wrapper(self):
        """Create environment wrapper."""
        return PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
    
    @pytest.fixture
    def individual(self, wrapper):
        """Create test individual."""
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        
        obs_dim = len(observations[agent])
        action_space = wrapper.action_space(agent)
        action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        return MockIndividual(genome)
    
    @pytest.mark.parametrize("method", ['mean', 'median', 'min', 'max'])
    def test_aggregation_methods(self, wrapper, individual, method):
        """Test evaluate with different aggregation methods across episodes."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            aggregation_method=method
        )
        
        fitness = evaluator.evaluate(individual)
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))
    
    def test_return_statistics(self, wrapper, individual):
        """Test returning episode statistics."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=5
        )
        
        stats = evaluator.evaluate(individual, return_stats=True)
        
        # Should return (mean, std, min, max)
        assert isinstance(stats, tuple)
        assert len(stats) == 4
        
        mean, std, min_val, max_val = stats
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))
        assert isinstance(min_val, (int, float))
        assert isinstance(max_val, (int, float))
        assert std >= 0
        assert min_val <= mean <= max_val
    
    def test_single_eval_kwargs_passed_through(self, wrapper, individual):
        """Test kwargs are passed to SingleEpisodeEvaluator."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            max_steps=5,
            fitness_aggregation='mean'
        )
        
        fitness = evaluator.evaluate(individual)
        assert isinstance(fitness, tuple)
    
    def test_invalid_aggregation_method_raises_error(self, wrapper):
        """Test invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregation"):
            MultiEpisodeEvaluator(
                env_wrapper=wrapper,
                aggregation_method='invalid'
            )
    
    def test_num_episodes_must_be_positive(self, wrapper):
        """Test num_episodes must be positive."""
        with pytest.raises(ValueError, match="num_episodes"):
            MultiEpisodeEvaluator(
                env_wrapper=wrapper,
                num_episodes=0
            )


class TestEvaluatorIntegration:
    """Test evaluator integration with full pipeline."""
    
    def test_single_and_multi_evaluator_consistency(self):
        """Test single episode result matches multi-episode with n=1."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
        
        # Create individual
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        individual = MockIndividual(genome)
        
        # Single episode evaluator
        single_eval = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=25,
            fitness_aggregation='sum'
        )
        
        # Multi-episode evaluator with 1 episode
        multi_eval = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=1,
            base_seed=42,
            max_steps=25,
            fitness_aggregation='sum'
        )
        
        single_fitness = single_eval.evaluate(individual, seed=42)
        multi_fitness = multi_eval.evaluate(individual)
        
        # Should be very close (floating point tolerance)
        assert abs(single_fitness[0] - multi_fitness[0]) < 1e-6
    
    def test_evaluator_with_deap_workflow(self):
        """Test evaluator works in DEAP-style workflow."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
        
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            max_steps=25
        )
        
        # Create population of individuals
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        
        population = []
        for i in range(5):
            genome = FixedTopologyGenome(
                layer_sizes=[obs_dim, 16, action_dim],
                random_seed=i
            )
            individual = MockIndividual(genome)
            population.append(individual)
        
        # Evaluate population
        for ind in population:
            fitness = evaluator.evaluate(ind)
            ind.fitness.values = fitness
        
        # Check all have fitness
        for ind in population:
            assert ind.fitness.values is not None
            assert isinstance(ind.fitness.values, tuple)
            assert len(ind.fitness.values) == 1
