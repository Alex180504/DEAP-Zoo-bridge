from typing import Any, Tuple
import numpy as np

from .base_evaluator import BaseEvaluator
from .single_episode import SingleEpisodeEvaluator
from ..environments import PettingZooWrapper


class MultiEpisodeEvaluator(BaseEvaluator):
    """Evaluates individual across multiple episodes for robust fitness estimation."""
    
    VALID_AGGREGATIONS = ['mean', 'median', 'min', 'max']
    
    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        num_episodes: int = 5,
        aggregation_method: str = 'mean',
        base_seed: int = 42,
        **single_eval_kwargs
    ) -> None:
        """
        Initialize multi-episode evaluator.
        
        Args:
            env_wrapper: PettingZoo environment wrapper
            num_episodes: Number of episodes to run
            aggregation_method: How to combine episode fitnesses
                - 'mean': Average fitness across episodes
                - 'median': Median fitness
                - 'min': Worst-case fitness
                - 'max': Best-case fitness
            base_seed: Base seed for generating episode seeds
            **single_eval_kwargs: Additional kwargs for SingleEpisodeEvaluator
            
        Raises:
            ValueError: If invalid parameters
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        
        if aggregation_method not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation_method '{aggregation_method}'. "
                f"Must be one of {self.VALID_AGGREGATIONS}"
            )
        
        self.env_wrapper = env_wrapper
        self.num_episodes = num_episodes
        self.aggregation_method = aggregation_method
        self.base_seed = base_seed
        
        # Create single episode evaluator
        self.single_evaluator = SingleEpisodeEvaluator(
            env_wrapper=env_wrapper,
            **single_eval_kwargs
        )
    
    def evaluate(
        self,
        individual: Any
    ) -> Tuple[float]:
        """
        Evaluate individual across multiple episodes.
        
        Args:
            individual: DEAP individual with genome
            
        Returns:
            Tuple with a single aggregated fitness value (DEAP format)
            
        Raises:
            ValueError: If individual is None or invalid
        """
        fitnesses = self._evaluate_episodes(individual)
        aggregated_fitness = self._aggregate_fitnesses(fitnesses)
        return (aggregated_fitness,)

    def evaluate_with_stats(self, individual: Any) -> Tuple[float, float, float, float]:
        """
        Evaluate individual across multiple episodes and return statistics.

        Args:
            individual: DEAP individual with genome

        Returns:
            Tuple of (mean, std, min, max) fitness values

        Raises:
            ValueError: If individual is None or invalid
        """
        fitnesses = self._evaluate_episodes(individual)

        return (
            float(np.mean(fitnesses)),
            float(np.std(fitnesses)),
            float(np.min(fitnesses)),
            float(np.max(fitnesses))
        )

    def _evaluate_episodes(self, individual: Any) -> np.ndarray:
        """Run all evaluation episodes and return raw fitness values."""
        # Validate inputs
        if individual is None:
            raise ValueError("Individual cannot be None")
        
        if not hasattr(individual, 'genome'):
            raise ValueError(
                f"Individual must have 'genome' attribute, got type {type(individual)}"
            )
        
        if individual.genome is None:
            raise ValueError("Individual's genome cannot be None")
        
        # Run multiple episodes with different seeds
        episode_fitnesses = []
        
        for episode_idx in range(self.num_episodes):
            # Generate deterministic seed for this episode
            episode_seed = self.base_seed + episode_idx
            
            # Evaluate single episode (validation happens in single_evaluator)
            fitness = self.single_evaluator.evaluate(individual, seed=episode_seed)
            episode_fitnesses.append(fitness[0])

        return np.array(episode_fitnesses)
    
    def _aggregate_fitnesses(self, fitnesses: np.ndarray) -> float:
        """
        Aggregate fitness values across episodes.
        
        Args:
            fitnesses: Array of episode fitness values
            
        Returns:
            Aggregated fitness
        """
        if self.aggregation_method == 'mean':
            return float(np.mean(fitnesses))
        elif self.aggregation_method == 'median':
            return float(np.median(fitnesses))
        elif self.aggregation_method == 'min':
            return float(np.min(fitnesses))
        elif self.aggregation_method == 'max':
            return float(np.max(fitnesses))
        
        return 0.0
