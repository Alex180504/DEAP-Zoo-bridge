"""
DEAP integration utilities.

This module provides the bridge between DEAP's evolutionary algorithm framework
and the PettingZoo evaluation system. It handles:
- Individual creation and management
- Toolbox configuration
- Genetic operator registration
- Fitness evaluation integration
"""

from typing import Dict, Any, Optional, List
from deap import base, creator, tools
import numpy as np

from ..genomes import FixedTopologyGenome
from ..evaluators import BaseEvaluator


def setup_deap_creators() -> None:
    """
    Setup DEAP creator classes for fitness and individuals.
    
    Creates:
        - FitnessMax: Fitness class for maximization (weights=(1.0,))
        - Individual: Individual class containing genome attribute
    
    Note:
        This function is idempotent - it's safe to call multiple times.
        Existing classes will be deleted and recreated.
    
    Example:
        >>> setup_deap_creators()
        >>> ind = creator.Individual(genome)
        >>> ind.fitness.values = (10.5,)
    """
    # Remove existing creators if they exist
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    # Create fitness class (maximization)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    # Create individual class - simple object to hold genome and fitness
    creator.create("Individual", object,
                   fitness=creator.FitnessMax,
                   genome=None)


def create_deap_individual(
    layer_sizes: List[int],
    activation: str = 'tanh',
    weight_range: tuple = (-1.0, 1.0),
    random_seed: Optional[int] = None
) -> Any:
    """
    Create a DEAP individual from genome configuration.
    
    Args:
        layer_sizes: Network architecture [input, hidden..., output]
        activation: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
        weight_range: (min, max) for weight initialization
        random_seed: Seed for reproducibility
        
    Returns:
        DEAP Individual with genome attribute
        
    Example:
        >>> setup_deap_creators()
        >>> ind = create_deap_individual([10, 64, 4], random_seed=42)
        >>> ind.genome.layer_sizes
        [10, 64, 4]
    """
    genome = FixedTopologyGenome(
        layer_sizes=layer_sizes,
        activation=activation,
        weight_range=weight_range,
        random_seed=random_seed
    )
    
    individual = creator.Individual()
    individual.genome = genome
    
    return individual


def evaluate_individual(
    individual: Any,
    evaluator: BaseEvaluator,
    **kwargs
) -> tuple:
    """
    Evaluate a DEAP individual using the evaluator.
    
    Args:
        individual: DEAP Individual with genome attribute
        evaluator: Evaluator instance (SingleEpisode or MultiEpisode)
        **kwargs: Additional arguments for evaluator
        
    Returns:
        Tuple containing fitness value (DEAP format)
        
    Example:
        >>> fitness = evaluate_individual(ind, evaluator, seed=42)
        >>> fitness
        (15.3,)
    """
    return evaluator.evaluate(individual, **kwargs)


def create_toolbox(
    genome_config: Dict[str, Any],
    evaluator: BaseEvaluator,
    selection_config: Optional[Dict[str, Any]] = None,
    crossover_config: Optional[Dict[str, Any]] = None,
    mutation_config: Optional[Dict[str, Any]] = None
) -> base.Toolbox:
    """
    Create and configure DEAP toolbox with all genetic operators.
    
    Args:
        genome_config: Configuration for genome creation
            - layer_sizes: List[int] - Network architecture
            - activation: str (optional) - Activation function
            - weight_range: tuple (optional) - Weight initialization range
            - random_seed: int (optional) - Seed for reproducibility
        evaluator: Evaluator for fitness calculation
        selection_config: Selection operator configuration
            - tournament_size: int (default: 3)
        crossover_config: Crossover operator configuration
            - alpha: float (default: 0.5) - Blend crossover alpha
        mutation_config: Mutation operator configuration
            - mu: float (default: 0.0) - Gaussian mean
            - sigma: float (default: 0.2) - Gaussian std
            - indpb: float (default: 0.1) - Independent gene probability
            
    Returns:
        Configured DEAP toolbox
        
    Example:
        >>> toolbox = create_toolbox(
        ...     genome_config={'layer_sizes': [10, 64, 4]},
        ...     evaluator=evaluator
        ... )
        >>> pop = toolbox.population(n=100)
        >>> fitnesses = [toolbox.evaluate(ind) for ind in pop]
    """
    # Setup default configs
    selection_config = selection_config or {}
    crossover_config = crossover_config or {}
    mutation_config = mutation_config or {}
    
    # Extract config values
    layer_sizes = genome_config['layer_sizes']
    activation = genome_config.get('activation', 'tanh')
    weight_range = genome_config.get('weight_range', (-1.0, 1.0))
    base_seed = genome_config.get('random_seed', None)
    
    tournament_size = selection_config.get('tournament_size', 3)
    alpha = crossover_config.get('alpha', 0.5)
    mu = mutation_config.get('mu', 0.0)
    sigma = mutation_config.get('sigma', 0.2)
    indpb = mutation_config.get('indpb', 0.1)
    
    # Create toolbox
    toolbox = base.Toolbox()
    
    # Counter for unique seeds
    seed_counter = {'count': 0}
    
    def make_individual():
        """Create individual with unique seed."""
        if base_seed is not None:
            seed = base_seed + seed_counter['count']
            seed_counter['count'] += 1
        else:
            seed = None
        
        return create_deap_individual(
            layer_sizes=layer_sizes,
            activation=activation,
            weight_range=weight_range,
            random_seed=seed
        )
    
    # Register individual and population
    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation
    toolbox.register("evaluate", evaluate_individual, evaluator=evaluator)
    
    # Register genetic operators
    toolbox.register("mate", crossover_blend, alpha=alpha)
    toolbox.register("mutate", mutate_gaussian, mu=mu, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    # Register clone for deep copying
    toolbox.register("clone", clone_individual)
    
    return toolbox


def crossover_blend(ind1: Any, ind2: Any, alpha: float = 0.5) -> tuple:
    """
    Blend crossover for neural network weights.
    
    Performs blend crossover (BLX-α) on genome weights:
    - child1 = beta * parent1 + (1-beta) * parent2
    - child2 = (1-beta) * parent1 + beta * parent2
    where beta ~ U(0-alpha, 1+alpha)
    
    Args:
        ind1: First parent individual
        ind2: Second parent individual
        alpha: Blend crossover parameter (default: 0.5)
        
    Returns:
        Tuple of (ind1, ind2) after crossover
        
    Example:
        >>> ind1, ind2 = crossover_blend(parent1, parent2, alpha=0.5)
    """
    weights1 = ind1.genome.get_weights()
    weights2 = ind2.genome.get_weights()
    
    # Generate random blending factors
    beta = np.random.uniform(-alpha, 1 + alpha, size=weights1.shape)
    
    # Perform blend crossover
    new_weights1 = beta * weights1 + (1 - beta) * weights2
    new_weights2 = (1 - beta) * weights1 + beta * weights2
    
    # Update genomes
    ind1.genome.set_weights(new_weights1)
    ind2.genome.set_weights(new_weights2)
    
    return ind1, ind2


def mutate_gaussian(
    individual: Any,
    mu: float = 0.0,
    sigma: float = 0.2,
    indpb: float = 0.1
) -> tuple:
    """
    Gaussian mutation for neural network weights.
    
    Each weight has probability `indpb` of being mutated by adding
    Gaussian noise N(mu, sigma).
    
    Args:
        individual: Individual to mutate
        mu: Mean of Gaussian noise (default: 0.0)
        sigma: Standard deviation of Gaussian noise (default: 0.2)
        indpb: Probability of mutating each weight (default: 0.1)
        
    Returns:
        Tuple containing the mutated individual
        
    Example:
        >>> mutated, = mutate_gaussian(ind, sigma=0.3, indpb=0.2)
    """
    weights = individual.genome.get_weights()
    
    # Generate mutation mask (which weights to mutate)
    mutation_mask = np.random.random(weights.shape) < indpb
    
    # Generate Gaussian noise
    noise = np.random.normal(mu, sigma, size=weights.shape)
    
    # Apply mutation only to selected weights
    weights[mutation_mask] += noise[mutation_mask]
    
    # Update genome
    individual.genome.set_weights(weights)
    
    return (individual,)


def clone_individual(individual: Any) -> Any:
    """
    Deep clone a DEAP individual including genome.
    
    Args:
        individual: Individual to clone
        
    Returns:
        Deep copy of the individual with cloned genome
        
    Example:
        >>> clone = clone_individual(original)
        >>> clone.genome is not original.genome  # True
    """
    cloned = creator.Individual()
    cloned.genome = individual.genome.clone()
    
    # Clone fitness if it has been evaluated
    if individual.fitness.valid:
        cloned.fitness.values = individual.fitness.values
    
    return cloned
