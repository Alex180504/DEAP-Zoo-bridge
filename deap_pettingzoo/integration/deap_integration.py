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


SUPPORTED_SELECTION_METHODS = (
    "tournament", "roulette", "best", "random",
)
SUPPORTED_CROSSOVER_METHODS = (
    "blend", "one_point", "two_point", "uniform", "sbx", "es_blend",
)
SUPPORTED_MUTATION_METHODS = (
    "gaussian", "polynomial_bounded",
)


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
            - method: str (default: 'tournament') - one of
              ``'tournament'``, ``'roulette'``, ``'best'``, ``'random'``
            - tournament_size: int (default: 3, only for 'tournament')
        crossover_config: Crossover operator configuration
            - method: str (default: 'blend') - one of
              ``'blend'``, ``'one_point'``, ``'two_point'``, ``'uniform'``,
              ``'sbx'`` (simulated-binary-bounded), ``'es_blend'``
            - alpha: float (default: 0.5) - blend/BLX-α parameter
            - indpb: float (default: 0.5) - per-gene swap prob for 'uniform'
            - eta: float (default: 15.0) - SBX distribution index
        mutation_config: Mutation operator configuration
            - method: str (default: 'gaussian') - one of
              ``'gaussian'``, ``'polynomial_bounded'``
            - mu: float (default: 0.0) - Gaussian mean
            - sigma: float (default: 0.2) - Gaussian std
            - indpb: float (default: 0.1) - Independent gene probability
            - eta: float (default: 15.0) - polynomial-bounded distribution index

    Returns:
        Configured DEAP toolbox

    Example:
        >>> toolbox = create_toolbox(
        ...     genome_config={'layer_sizes': [10, 64, 4]},
        ...     evaluator=evaluator,
        ...     selection_config={'method': 'roulette'},
        ...     crossover_config={'method': 'sbx', 'eta': 20.0},
        ...     mutation_config={'method': 'polynomial_bounded', 'eta': 20.0},
        ... )
        >>> pop = toolbox.population(n=100)
        >>> fitnesses = [toolbox.evaluate(ind) for ind in pop]

    Notes:
        * Omitting any ``method`` key preserves the historical defaults
          (``tournament`` / ``blend`` / ``gaussian``), so existing call
          sites remain unchanged.
        * ``'sbx'`` and ``'polynomial_bounded'`` use
          ``genome_config['weight_range']`` as per-gene bounds.
        * Integer/binary/permutation operators are intentionally not
          dispatched here — they require a non-continuous genome and are
          tracked as future work.
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

    # Dispatch genetic operators
    _register_selection(toolbox, selection_config)
    _register_crossover(toolbox, crossover_config, weight_range)
    _register_mutation(toolbox, mutation_config, weight_range)

    # Register clone for deep copying
    toolbox.register("clone", clone_individual)

    return toolbox


def _register_selection(toolbox: base.Toolbox, cfg: Dict[str, Any]) -> None:
    """Register ``toolbox.select`` based on ``cfg['method']``.

    Selection operators only read ``individual.fitness`` so they work on
    the existing ``creator.Individual`` (plain object) without adapters.
    """
    method = cfg.get("method", "tournament")
    if method == "tournament":
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=cfg.get("tournament_size", 3),
        )
    elif method == "roulette":
        # ``tools.selRoulette`` samples proportionally to fitness values and
        # is mathematically only defined for non-negative fitnesses. Several
        # PettingZoo evaluators in this project return summed episode
        # rewards that can be negative, which silently produces invalid
        # sampling (or a ZeroDivisionError) inside DEAP. Wrap the operator
        # so a bad fitness sign fails loudly at selection time with an
        # actionable message instead of corrupting the run.
        def _safe_roulette(individuals, k):
            for ind in individuals:
                fit = ind.fitness.values[0]
                if fit < 0:
                    raise ValueError(
                        "selRoulette requires non-negative fitness values, "
                        f"got {fit!r}. Either set selection_config['method'] "
                        "to 'tournament'/'best'/'random', or shift/rescale "
                        "your evaluator output so all fitnesses are >= 0."
                    )
            return tools.selRoulette(individuals, k)

        toolbox.register("select", _safe_roulette)
    elif method == "best":
        toolbox.register("select", tools.selBest)
    elif method == "random":
        toolbox.register("select", tools.selRandom)
    else:
        raise ValueError(
            f"Unknown selection method {method!r}. "
            f"Supported: {SUPPORTED_SELECTION_METHODS}"
        )


def _register_crossover(
    toolbox: base.Toolbox,
    cfg: Dict[str, Any],
    weight_range: tuple,
) -> None:
    """Register ``toolbox.mate`` based on ``cfg['method']``.

    Non-blend operators delegate to DEAP built-ins via
    :func:`_cx_on_weights`, which extracts each parent's flat weight
    vector into a list, lets DEAP mutate it in place, then writes the
    result back through :meth:`FixedTopologyGenome.set_weights`.
    """
    method = cfg.get("method", "blend")
    if method == "blend":
        toolbox.register("mate", crossover_blend, alpha=cfg.get("alpha", 0.5))
        return
    if method == "one_point":
        toolbox.register("mate", _cx_on_weights, tool=tools.cxOnePoint)
        return
    if method == "two_point":
        toolbox.register("mate", _cx_on_weights, tool=tools.cxTwoPoint)
        return
    if method == "uniform":
        toolbox.register(
            "mate",
            _cx_on_weights,
            tool=tools.cxUniform,
            indpb=cfg.get("indpb", 0.5),
        )
        return
    if method == "sbx":
        low, up = float(weight_range[0]), float(weight_range[1])
        toolbox.register(
            "mate",
            _cx_on_weights,
            tool=tools.cxSimulatedBinaryBounded,
            eta=cfg.get("eta", 15.0),
            low=low,
            up=up,
        )
        return
    if method == "es_blend":
        toolbox.register(
            "mate",
            _cx_on_weights,
            tool=tools.cxBlend,
            alpha=cfg.get("alpha", 0.5),
        )
        return
    raise ValueError(
        f"Unknown crossover method {method!r}. "
        f"Supported: {SUPPORTED_CROSSOVER_METHODS}"
    )


def _register_mutation(
    toolbox: base.Toolbox,
    cfg: Dict[str, Any],
    weight_range: tuple,
) -> None:
    """Register ``toolbox.mutate`` based on ``cfg['method']``.

    Non-gaussian operators delegate to DEAP built-ins via
    :func:`_mut_on_weights`.
    """
    method = cfg.get("method", "gaussian")
    if method == "gaussian":
        toolbox.register(
            "mutate",
            mutate_gaussian,
            mu=cfg.get("mu", 0.0),
            sigma=cfg.get("sigma", 0.2),
            indpb=cfg.get("indpb", 0.1),
        )
        return
    if method == "polynomial_bounded":
        low, up = float(weight_range[0]), float(weight_range[1])
        toolbox.register(
            "mutate",
            _mut_on_weights,
            tool=tools.mutPolynomialBounded,
            eta=cfg.get("eta", 15.0),
            low=low,
            up=up,
            indpb=cfg.get("indpb", 0.1),
        )
        return
    # TODO: ``tools.mutESLogNormal`` requires individuals to carry a
    # per-gene strategy vector (ES self-adaptation). Supporting it would
    # need a new genome variant; deferred.
    raise ValueError(
        f"Unknown mutation method {method!r}. "
        f"Supported: {SUPPORTED_MUTATION_METHODS}"
    )


def _cx_on_weights(ind1: Any, ind2: Any, tool, **kwargs) -> tuple:
    """Adapter that runs a DEAP sequence crossover on genome weights.

    The project's ``creator.Individual`` is a plain object (not a list),
    so DEAP's built-in crossovers (``cxOnePoint``, ``cxUniform`` …) can't
    operate on it directly. This helper extracts the flat weight vector
    into a plain list, defers to ``tool(list1, list2, **kwargs)``, then
    writes the mutated lists back through
    :meth:`FixedTopologyGenome.set_weights`.
    """
    w1 = ind1.genome.get_weights().tolist()
    w2 = ind2.genome.get_weights().tolist()
    tool(w1, w2, **kwargs)
    ind1.genome.set_weights(np.asarray(w1, dtype=np.float64))
    ind2.genome.set_weights(np.asarray(w2, dtype=np.float64))
    return ind1, ind2


def _mut_on_weights(individual: Any, tool, **kwargs) -> tuple:
    """Adapter that runs a DEAP sequence mutation on genome weights.

    Mirrors :func:`_cx_on_weights` for single-parent mutation operators.
    """
    weights = individual.genome.get_weights().tolist()
    tool(weights, **kwargs)
    individual.genome.set_weights(np.asarray(weights, dtype=np.float64))
    return (individual,)


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
