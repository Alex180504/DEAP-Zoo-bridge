"""
EA trainer implementation using DEAP.

This module wraps the existing DEAP integration to conform to the
unified BaselineTrainer interface.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from deap import tools

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
    PolicyProtocol,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration.deap_integration import (
    setup_deap_creators,
    create_toolbox,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy


class GenomePolicyWrapper:
    """Wraps a genome to satisfy PolicyProtocol.
    
    Converts genome evaluation calls into the policy interface
    expected by UnifiedEvaluator. For homogeneous multi-agent
    environments, all agents use the same policy.
    """
    
    def __init__(self, genome, env_name: str):
        """Initialize policy wrapper.
        
        Args:
            genome: FixedTopologyGenome instance.
            env_name: Environment name (stored for reference).
        """
        self._genome = genome
        self._env_name = env_name
        
        # Create single policy for all agents (homogeneous)
        self._policy = FeedforwardNeuralPolicy(genome)
    
    def get_action(self, observation: np.ndarray) -> Any:
        """Get action for observation.
        
        Args:
            observation: Environment observation.
            
        Returns:
            Action for the given observation.
        """
        return self._policy.get_action(observation)
    
    def reset(self) -> None:
        """Reset policy state (no-op for stateless feedforward policies)."""
    
    @property
    def genome(self):
        """Access underlying genome."""
        return self._genome


class EATrainer(BaselineTrainer):
    """Evolutionary Algorithm trainer using DEAP.
    
    Wraps the existing DEAP integration to provide a unified
    training interface compatible with BaselineTrainer.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_alpha: float = 0.5,
        mutation_sigma: float = 0.2,
        mutation_indpb: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
    ):
        """Initialize EA trainer.
        
        Args:
            hidden_size: Hidden layer size for neural network.
            population_size: Number of individuals in population.
            tournament_size: Tournament selection size.
            crossover_alpha: Blend crossover alpha parameter.
            mutation_sigma: Gaussian mutation standard deviation.
            mutation_indpb: Independent mutation probability per weight.
            crossover_rate: Probability of applying crossover.
            mutation_rate: Probability of applying mutation.
        """
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_alpha = crossover_alpha
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train EA with the given configuration.
        
        Args:
            config: Training configuration.
            
        Returns:
            TrainingResult with best genome wrapped as policy.
        """
        start_time = time.time()
        
        # Translate budget to EA parameters
        ea_params = self._translate_budget(config)
        num_generations = ea_params['num_generations']
        max_steps_per_episode = ea_params['max_steps_per_episode']
        
        # Setup environment and evaluator
        env = PettingZooWrapper(config.env_name, env_config=config.env_config or None)
        sample_obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]
        
        input_size = len(sample_obs[agent_id])
        action_space = env.action_space(agent_id)
        if hasattr(action_space, 'n'):  # Discrete
            output_size = action_space.n
        else:  # Box (continuous)
            output_size = action_space.shape[0]
        
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env,
            max_steps=max_steps_per_episode,
            aggregation_method='sum'
        )
        
        # Setup DEAP
        setup_deap_creators()
        
        # Create genome configuration
        seed = config.algorithm_params.get('seed', None)
        genome_config = {
            'layer_sizes': [input_size, self.hidden_size, output_size],
            'activation': 'tanh',
            'weight_range': (-1.0, 1.0),
            'random_seed': seed if seed is not None else (config.num_seeds if config.num_seeds == 1 else None),
        }
        
        # Create toolbox
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={'tournament_size': self.tournament_size},
            crossover_config={'alpha': self.crossover_alpha},
            mutation_config={
                'sigma': self.mutation_sigma,
                'indpb': self.mutation_indpb,
            }
        )
        
        # Override clone to use our clone_individual
        from deap_pettingzoo.integration.deap_integration import clone_individual
        toolbox.register("clone", clone_individual)
        
        # Create initial population
        population = toolbox.population(n=self.population_size)
        
        # Setup statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        hof = tools.HallOfFame(1)
        
        # Setup random number generator
        rng = np.random.default_rng(config.algorithm_params.get('seed'))
        
        # Track metrics
        fitness_history = []
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        hof.update(population)
        record = stats.compile(population)
        fitness_history.append({
            'generation': 0,
            'mean': record['avg'],
            'std': record['std'],
            'min': record['min'],
            'max': record['max'],
        })
        
        # Evolution loop
        for gen in range(1, num_generations + 1):
            # Select next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if rng.random() < self.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            hof.update(population)
            
            # Record statistics
            record = stats.compile(population)
            fitness_history.append({
                'generation': gen,
                'mean': record['avg'],
                'std': record['std'],
                'min': record['min'],
                'max': record['max'],
            })
        
        # Clean up
        env.close()
        
        training_time = time.time() - start_time
        
        # Get best individual
        best_individual = hof[0]
        best_genome = best_individual.genome
        
        # Wrap genome as policy
        policy = GenomePolicyWrapper(best_genome, config.env_name)
        
        # Create result
        return TrainingResult(
            algorithm='ea',
            policy=policy,
            metrics={
                'fitness_history': fitness_history,
                'final_fitness': best_individual.fitness.values[0],
                'num_generations': num_generations,
                'population_size': self.population_size,
            },
            metadata={
                'genome_config': genome_config,
                'ea_config': {
                    'hidden_size': self.hidden_size,
                    'population_size': self.population_size,
                    'tournament_size': self.tournament_size,
                    'crossover_alpha': self.crossover_alpha,
                    'mutation_sigma': self.mutation_sigma,
                    'mutation_indpb': self.mutation_indpb,
                    'crossover_rate': self.crossover_rate,
                    'mutation_rate': self.mutation_rate,
                },
            },
            training_time=training_time,
        )
    
    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """Check if EA supports the given budget mode.
        
        Args:
            mode: Budget mode to check.
            
        Returns:
            True if supported.
        """
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get EA capabilities.
        
        Returns:
            Capabilities dictionary.
        """
        return {
            'discrete_action': True,
            'continuous_action': True,
            'needs_global_state': False,
            'on_policy': False,  # EA is not on/off-policy (different paradigm)
        }
    
    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        """Translate budget config to EA parameters.
        
        Args:
            config: Training configuration.
            
        Returns:
            Dictionary with EA-specific parameters.
        """
        max_steps_per_episode = config.algorithm_params.get('max_steps_per_episode', 200)
        
        if config.budget_mode == BudgetMode.ENV_STEPS:
            # Calculate generations from environment steps
            # env_steps = population_size * num_generations * max_steps_per_episode
            total_env_steps = config.budget_value
            num_generations = total_env_steps // (self.population_size * max_steps_per_episode)
            num_generations = max(1, num_generations)  # At least 1 generation
            
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            # Each episode evaluates one individual
            # Total episodes = population_size * num_generations
            total_episodes = config.budget_value
            num_generations = total_episodes // self.population_size
            num_generations = max(1, num_generations)
            
        else:
            raise ValueError(f"EA does not support budget mode: {config.budget_mode}")
        
        return {
            'num_generations': num_generations,
            'max_steps_per_episode': max_steps_per_episode,
        }
