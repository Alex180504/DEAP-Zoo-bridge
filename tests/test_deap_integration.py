import pytest
import numpy as np
from deap import base, creator, tools

from deap_pettingzoo.integration import (
    create_deap_individual,
    setup_deap_creators,
    create_toolbox,
    evaluate_individual
)
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator


class TestDEAPCreators:
    """Test DEAP creator setup."""
    
    def test_setup_deap_creators_complete(self):
        """Test setup_deap_creators creates all required classes with correct properties."""
        # Clean up any existing creators
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        # Verify classes are registered
        assert hasattr(creator, "FitnessMax")
        assert hasattr(creator, "Individual")
        
        # Verify FitnessMax has correct properties
        fitness = creator.FitnessMax()
        assert hasattr(fitness, 'weights')
        assert fitness.weights == (1.0,)
        
        # Verify Individual has required attributes
        genome = FixedTopologyGenome([4, 8, 2], random_seed=42)
        individual = creator.Individual()
        individual.genome = genome
        
        assert hasattr(individual, 'fitness')
        assert hasattr(individual, 'genome')
        assert individual.genome is genome


class TestIndividualCreation:
    """Test individual creation utilities."""
    
    @pytest.mark.parametrize("activation", ['tanh', 'relu', 'sigmoid', 'linear'])
    def test_create_deap_individual_with_configs(self, activation):
        """Test creating individual with different configurations."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        individual = create_deap_individual(
            layer_sizes=[4, 8, 2],
            activation=activation,
            random_seed=42
        )
        
        assert hasattr(individual, 'genome')
        assert hasattr(individual, 'fitness')
        assert isinstance(individual.genome, FixedTopologyGenome)
        assert individual.genome.activation == activation
    
    def test_individuals_have_different_weights(self):
        """Test different seeds create different individuals."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        ind1 = create_deap_individual([4, 8, 2], random_seed=1)
        ind2 = create_deap_individual([4, 8, 2], random_seed=2)
        
        weights1 = ind1.genome.get_weights()
        weights2 = ind2.genome.get_weights()
        
        assert not np.allclose(weights1, weights2)


class TestToolboxCreation:
    """Test DEAP toolbox configuration."""
    
    @pytest.fixture
    def wrapper(self):
        """Create test environment wrapper."""
        return PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
    
    @pytest.fixture
    def evaluator(self, wrapper):
        """Create test evaluator."""
        return SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=25,
            aggregation_method='sum'
        )
    
    def test_create_toolbox_registers_operators(self, evaluator):
        """Test toolbox has all required operators registered."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        toolbox = create_toolbox(
            genome_config={'layer_sizes': [14, 16, 5]},
            evaluator=evaluator
        )
        
        assert hasattr(toolbox, 'individual')
        assert hasattr(toolbox, 'population')
        assert hasattr(toolbox, 'evaluate')
        assert hasattr(toolbox, 'mate')
        assert hasattr(toolbox, 'mutate')
        assert hasattr(toolbox, 'select')
    
    def test_toolbox_population_creates_multiple_individuals(self, evaluator):
        """Test toolbox.population() creates population."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        toolbox = create_toolbox(
            genome_config={'layer_sizes': [14, 16, 5]},
            evaluator=evaluator
        )
        
        pop = toolbox.population(n=10)
        
        assert len(pop) == 10
        assert all(hasattr(ind, 'genome') for ind in pop)
    
    def test_create_toolbox_with_all_operators(self, evaluator):
        """Test that toolbox has all required operators and they function."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        toolbox = create_toolbox(
            genome_config={'layer_sizes': [14, 16, 5]},
            evaluator=evaluator,
            selection_config={'tournament_size': 3},
            mutation_config={'sigma': 0.5}
        )
        
        # Verify all operators are registered
        assert hasattr(toolbox, 'individual')
        assert hasattr(toolbox, 'population')
        assert hasattr(toolbox, 'evaluate')
        assert hasattr(toolbox, 'mate')
        assert hasattr(toolbox, 'mutate')
        assert hasattr(toolbox, 'select')
        
        # Create individuals for operator testing
        ind1 = toolbox.individual()
        ind2 = toolbox.individual()
        
        # Test crossover changes weights
        weights1_before = ind1.genome.get_weights().copy()
        weights2_before = ind2.genome.get_weights().copy()
        toolbox.mate(ind1, ind2)
        weights1_after = ind1.genome.get_weights()
        weights2_after = ind2.genome.get_weights()
        assert (not np.allclose(weights1_before, weights1_after) or 
                not np.allclose(weights2_before, weights2_after))
        
        # Test mutation changes weights
        ind3 = toolbox.individual()
        weights_before = ind3.genome.get_weights().copy()
        toolbox.mutate(ind3)
        weights_after = ind3.genome.get_weights()
        assert not np.allclose(weights_before, weights_after)
        
        # Test selection works
        pop = toolbox.population(n=5)
        for i, ind in enumerate(pop):
            ind.fitness.values = (float(i),)
        selected = toolbox.select(pop, k=3)
        assert len(selected) == 3


class TestEvaluationIntegration:
    """Test evaluation integration with DEAP."""
    
    def test_evaluate_individual_function(self):
        """Test standalone evaluate_individual function returns valid fitness."""
        wrapper = PettingZooWrapper('simple_tag_v3', env_config={'max_cycles': 25})
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=25,
            fitness_aggregation='sum'
        )
        
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        genome = FixedTopologyGenome([14, 16, 5], random_seed=42)
        individual = creator.Individual()
        individual.genome = genome
        
        fitness = evaluate_individual(individual, evaluator)
        
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))


class TestSimpleEvolutionWorkflow:
    """Test complete evolution workflow."""
    
    def test_simple_evolution_loop(self):
        """Test running simple evolutionary loop."""
        # Setup
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        setup_deap_creators()
        
        wrapper = PettingZooWrapper('simple_tag_v3', env_config={'max_cycles': 25})
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=25,
            fitness_aggregation='sum'
        )
        
        toolbox = create_toolbox(
            genome_config={'layer_sizes': [14, 16, 5], 'random_seed': 42},
            evaluator=evaluator,
            selection_config={'tournament_size': 3}
        )
        
        # Create population
        pop = toolbox.population(n=10)
        
        # Evaluate initial population
        fitnesses = [toolbox.evaluate(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Run one generation
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Re-evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
        
        # Verify all have valid fitness
        assert all(ind.fitness.valid for ind in pop)
        assert len(pop) == 10
        
        # Verify best fitness is meaningful
        best_fitness = max(ind.fitness.values[0] for ind in pop)
        assert isinstance(best_fitness, (int, float))
