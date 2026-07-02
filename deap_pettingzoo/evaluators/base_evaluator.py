from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseEvaluator(ABC):
    """Abstract base class for fitness evaluators."""

    @abstractmethod
    def evaluate(self, individual: Any, **kwargs) -> Tuple[float, ...]:
        """
        Evaluate individual and return fitness.

        Args:
            individual: DEAP individual with genome attribute
            **kwargs: Additional evaluation parameters

        Returns:
            Tuple containing fitness value(s) (DEAP format)
        """
        pass
