"""
Simple evolutionary experiment using DEAP-PettingZoo Bridge.

This script demonstrates basic usage of the library to evolve
neural network policies for PettingZoo environments using DEAP.
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms

from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator


def setup_deap():
    """Setup DEAP toolbox for evolution."""
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", object, fitness=creator.FitnessMax, genome=None)
    
    toolbox = base.Toolbox()
    return toolbox


def create_individual(input_size, hidden_size, output_size):
    """Create a random individual (genome)."""
    genome = FixedTopologyGenome(
        layer_sizes=[input_size, hidden_size, output_size],
        activation='tanh',
        weight_range=(-1.0, 1.0),
        random_seed=None  # Random initialization
    )
    # Create DEAP Individual with genome attribute
    ind = creator.Individual()
    ind.genome = genome
    return ind


def mutate_genome(individual, mutation_rate=0.15, mutation_std=0.3):
    """Mutate an individual's weights."""
    individual.genome.mutate(
        mutation_rate=mutation_rate,
        mutation_std=mutation_std,
        inplace=True
    )
    return individual,


def crossover_genomes(ind1, ind2):
    """Perform uniform crossover on two individuals."""
    weights1 = ind1.genome.get_weights()
    weights2 = ind2.genome.get_weights()
    
    # Uniform crossover
    for i in range(len(weights1)):
        if random.random() < 0.5:
            weights1[i], weights2[i] = weights2[i], weights1[i]
    
    ind1.genome.set_weights(weights1)
    ind2.genome.set_weights(weights2)
    
    return ind1, ind2


def run_experiment():
    """Run a simple evolutionary experiment."""
    
    # Configuration
    POPULATION_SIZE = 20
    GENERATIONS = 10
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.5
    TOURNAMENT_SIZE = 3
    
    print("=" * 70)
    print("DEAP-PettingZoo Evolution Experiment")
    print("=" * 70)
    print(f"Environment: simple_spread_v3")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print(f"Crossover Rate: {CROSSOVER_RATE}")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    print("Initializing environment...")
    env = PettingZooWrapper('simple_spread_v3', render_mode=None)
    print(f"[OK] Environment initialized: {env.env_name}")
    print(f"  Agents: {env.possible_agents}")
    
    # Get a sample observation to determine sizes
    sample_obs = env.reset(seed=42)
    first_agent = env.possible_agents[0]
    input_size = len(sample_obs[first_agent])
    
    # Get action space size
    action_space = env.action_space(first_agent)
    if hasattr(action_space, 'n'):  # Discrete
        output_size = action_space.n
    else:  # Box
        output_size = action_space.shape[0]
    
    hidden_size = 32
    
    print(f"  Observation size: {input_size}")
    print(f"  Action size: {output_size}")
    print()
    
    # Setup evaluator
    evaluator = SingleEpisodeEvaluator(
        env_wrapper=env,
        max_steps=200,
        aggregation_method='sum'
    )
    
    print(f"Network architecture: {input_size} -> {hidden_size} -> {output_size}")
    print()
    
    # Setup DEAP
    toolbox = setup_deap()
    
    # Register genetic operators
    toolbox.register("individual", create_individual, input_size, hidden_size, output_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", crossover_genomes)
    toolbox.register("mutate", mutate_genome, mutation_rate=MUTATION_RATE)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Create initial population
    print("Creating initial population...")
    population = toolbox.population(n=POPULATION_SIZE)
    print(f"[OK] Created {len(population)} individuals")
    print()
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame (best individuals)
    hof = tools.HallOfFame(3)
    
    # Run evolution
    print("Starting evolution...")
    print("-" * 70)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Record initial stats
    record = stats.compile(population)
    initial_best = record['max']
    print(f"Gen 0: Max={record['max']:.3f}, Avg={record['avg']:.3f}, "
          f"Std={record['std']:.3f}, Min={record['min']:.3f}")
    
    # Evolution loop
    for gen in range(1, GENERATIONS + 1):
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Update hall of fame
        hof.update(population)
        
        # Record statistics
        record = stats.compile(population)
        print(f"Gen {gen}: Max={record['max']:.3f}, Avg={record['avg']:.3f}, "
              f"Std={record['std']:.3f}, Min={record['min']:.3f}")
    
    print("-" * 70)
    print()
    
    # Results
    print("Evolution complete!")
    print()
    print("=" * 70)
    print("Best Individuals (Hall of Fame)")
    print("=" * 70)
    for i, ind in enumerate(hof, 1):
        print(f"{i}. Fitness: {ind.fitness.values[0]:.3f}")
        print(f"   Weights: {len(ind.genome.get_weights())} parameters")
        print(f"   Architecture: {ind.genome.layer_sizes}")
    
    print()
    print("=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    print(f"Best fitness achieved: {hof[0].fitness.values[0]:.3f}")
    print(f"Average fitness (final gen): {record['avg']:.3f}")
    print(f"Improvement: {hof[0].fitness.values[0] - initial_best:.3f}")
    print("=" * 70)
    
    return population, hof


if __name__ == "__main__":
    population, hof = run_experiment()
    print("\n[OK] Experiment completed successfully!")
