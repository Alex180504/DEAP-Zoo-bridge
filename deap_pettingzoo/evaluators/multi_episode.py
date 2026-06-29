from typing import Any, List, Optional, Tuple
import numpy as np

from .base_evaluator import BaseEvaluator
from .single_episode import SingleEpisodeEvaluator
from ..environments import PettingZooWrapper


class MultiEpisodeEvaluator(BaseEvaluator):
    """Evaluates individual across multiple episodes for fitness estimation.

    Multi-objective / lexicase support
    ----------------------------------
    Pass ``fitness_spec`` (a :class:`~deap_pettingzoo.integration.objectives.FitnessSpec`)
    to control the return shape:

    * **Standard (no lexicase)** — each episode returns an N-tuple; the
      final result is the element-wise mean across episodes (N-tuple).
    * **Lexicase ``per_episode``** — the per-episode reward scalars are
      concatenated into a flat vector (length = *num_episodes* ×
      *num_objectives*) for use with :func:`deap.tools.selLexicase`.
    * **Legacy (no spec)** — 1-tuple ``(scalar,)``, identical to the
      historical behaviour.
    """

    VALID_AGGREGATIONS = ['mean', 'median', 'min', 'max']

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        num_episodes: int = 5,
        aggregation_method: str = 'mean',
        base_seed: int = 42,
        fitness_spec: Optional[Any] = None,
        **single_eval_kwargs
    ) -> None:
        """
        Initialize multi-episode evaluator.

        Args:
            env_wrapper: PettingZoo environment wrapper.
            num_episodes: Number of episodes to run.
            aggregation_method: How to combine episode fitnesses
                - 'mean': Average fitness across episodes
                - 'median': Median fitness
                - 'min': Worst-case fitness
                - 'max': Best-case fitness
            base_seed: Base seed for generating episode seeds.
            fitness_spec: Optional
                :class:`~deap_pettingzoo.integration.objectives.FitnessSpec`.
            **single_eval_kwargs: Additional kwargs for
                :class:`SingleEpisodeEvaluator`.

        Raises:
            ValueError: If invalid parameters.
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
        self.fitness_spec = fitness_spec
        if fitness_spec is not None:
            fitness_spec.validate()

        # Create single episode evaluator.
        self.single_evaluator = SingleEpisodeEvaluator(
            env_wrapper=env_wrapper,
            fitness_spec=fitness_spec,
            **single_eval_kwargs,
        )

    @property
    def env_steps_consumed(self) -> int:
        """Cumulative real env steps across every delegated episode.

        Mirrors :attr:`SingleEpisodeEvaluator.env_steps_consumed`; the
        underlying single-episode evaluator accumulates ticks across all
        episodes this evaluator runs, so the count is below the nominal
        ``num_episodes * max_steps`` whenever episodes terminate early
        """
        return self.single_evaluator.env_steps_consumed
    
    def evaluate(
        self,
        individual: Any
    ) -> Tuple[float, ...]:
        """
        Evaluate individual across multiple episodes.

        Args:
            individual: DEAP individual with genome.

        Returns:
            Tuple of fitness values.  When no ``fitness_spec`` is set,
            a 1-tuple ``(aggregated_scalar,)``.  For multi-objective,
            an N-tuple of per-objective means.  For lexicase
            ``per_episode``, a flat concatenated vector.

        Raises:
            ValueError: If individual is None or invalid.
            RuntimeError: If an exception occurs during any episode
                rollout (propagated from
                :class:`SingleEpisodeEvaluator`).
        """
        # ---- lexicase per_episode path ----
        if (
            self.fitness_spec is not None
            and self.fitness_spec.lexicase_mode == "per_episode"
        ):
            fitness_tuples = self._evaluate_episodes(individual)
            # Each tuple is (per_episode_val, ...) per objective already
            # flattened by build_fitness_vector.  Concatenate them.
            flat: List[float] = []
            for ft in fitness_tuples:
                flat.extend(ft)
            return tuple(flat)

        # ---- standard (single- or multi-objective) path ----
        fitness_tuples = self._evaluate_episodes(individual)
        if not fitness_tuples:
            return (0.0,)

        # Element-wise aggregation.
        n_obj = len(fitness_tuples[0])
        aggregated: List[float] = []
        for i in range(n_obj):
            col = [ft[i] for ft in fitness_tuples]
            aggregated.append(self._aggregate_scalars(col))
        return tuple(aggregated)

    def evaluate_with_stats(self, individual: Any) -> Tuple[float, float, float, float]:
        """
        Evaluate individual across multiple episodes and return statistics.

        Only works correctly for the **single-objective** path
        (``fitness_spec is None``).  For multi-objective / lexicase
        configurations use :meth:`evaluate` directly.

        Args:
            individual: DEAP individual with genome

        Returns:
            Tuple of (mean, std, min, max) fitness values.

        Raises:
            ValueError: If individual is None or invalid, or if a
                multi-objective / lexicase spec is active.
            RuntimeError: If an exception occurs during any episode
                rollout (propagated from
                :class:`SingleEpisodeEvaluator`).
        """
        if self.fitness_spec is not None and (
            self.fitness_spec.objectives or self.fitness_spec.is_lexicase
        ):
            raise ValueError(
                "evaluate_with_stats works only for the single-objective "
                "backward-compatible path (fitness_spec=None).  Use "
                "evaluate() for multi-objective / lexicase setups."
            )

        # Delegate directly to avoid double multi-objective aggregation.
        fitnesses = np.asarray(self._evaluate_scalar_episodes(individual))
        return (
            float(np.mean(fitnesses)),
            float(np.std(fitnesses)),
            float(np.min(fitnesses)),
            float(np.max(fitnesses)),
        )

    def _evaluate_episodes(self, individual: Any) -> List[Tuple[float, ...]]:
        """Run all evaluation episodes and return per-episode fitness tuples."""
        # Validate inputs
        if individual is None:
            raise ValueError("Individual cannot be None")

        if not hasattr(individual, 'genome'):
            raise ValueError(
                f"Individual must have 'genome' attribute, got type {type(individual)}"
            )

        if individual.genome is None:
            raise ValueError("Individual's genome cannot be None")

        results: List[Tuple[float, ...]] = []
        for episode_idx in range(self.num_episodes):
            episode_seed = self.base_seed + episode_idx
            fitness = self.single_evaluator.evaluate(individual, seed=episode_seed)
            results.append(fitness)
        return results

    def _evaluate_scalar_episodes(self, individual: Any) -> List[float]:
        """Run all episodes and return scalar fitness values (legacy path)."""
        tuples = self._evaluate_episodes(individual)
        return [t[0] for t in tuples]

    def _aggregate_scalars(self, values: List[float]) -> float:
        """Aggregate a list of scalars using the configured aggregation_method."""
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        return self._aggregate_fitnesses(arr)

    def _aggregate_fitnesses(self, fitnesses: np.ndarray) -> float:
        """
        Aggregate fitness values across episodes (legacy 1-D path).

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
