"""
DEAP integration utilities.

This module provides the bridge between DEAP's evolutionary algorithm framework
and the PettingZoo evaluation system. It handles:
- Individual creation and management
- Toolbox configuration
- Genetic operator registration
- Fitness evaluation integration
- Process-pool parallel evaluation (top-level worker helpers)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
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


# --------------------------------------------------------------------------
# Process-pool parallel evaluation plumbing
# --------------------------------------------------------------------------

@dataclass
class _GenomePayload:
    """Picklable payload shipped to worker processes.

    Carries just enough to rebuild a :class:`FixedTopologyGenome` on
    the worker side; the live evaluator is *not* pickled.

    Attributes:
        weights: Flat numpy weight vector (``genome.get_weights()``).
        layer_sizes: Network topology.
        activation: Activation function name.
        weight_range: ``(lo, hi)`` tuple (kept for genome
            round-tripping; evaluation is unaffected).
        seed: Optional per-individual seed forwarded to
            ``evaluator.evaluate(..., seed=seed)``.
    """

    weights: np.ndarray
    layer_sizes: Tuple[int, ...]
    activation: str
    weight_range: Tuple[float, float]
    seed: Optional[int] = None


@dataclass
class _EvalWorkerSpec:
    """Picklable spec for initialising a worker-side evaluator.

    Mirrors the constructor signature of
    :class:`deap_pettingzoo.evaluators.SingleEpisodeEvaluator`; we ship
    the env *name* (plus ``env_config``) rather than a live env so
    nothing JIT-compiled or holding sockets crosses the boundary.
    """

    env_name: str
    env_config: Optional[Dict[str, Any]]
    max_steps: int
    aggregation_method: str
    team_mapping: Optional[Dict[str, str]] = None


# Module-global cache populated lazily on each worker. Using a lazy
# init pattern (instead of ``ProcessPoolExecutor(initializer=...)``)
# keeps a single code path for the process backend, SCOOP, and Dask —
# only the process backend supports ``initializer``.
_WORKER_CACHE: Dict[str, Any] = {}


def _ensure_worker_evaluator(spec: _EvalWorkerSpec) -> Any:
    """Return a worker-local ``SingleEpisodeEvaluator``, building once."""
    cached_spec = _WORKER_CACHE.get("spec")
    evaluator = _WORKER_CACHE.get("evaluator")
    if evaluator is not None and cached_spec == spec:
        return evaluator

    # First call on this worker, or spec changed (should not happen in
    # a well-formed train() call but is safe to rebuild on mismatch).
    from deap_pettingzoo.environments import PettingZooWrapper
    from deap_pettingzoo.evaluators import SingleEpisodeEvaluator

    env = PettingZooWrapper(
        spec.env_name, env_config=spec.env_config or None
    )
    evaluator = SingleEpisodeEvaluator(
        env_wrapper=env,
        max_steps=spec.max_steps,
        aggregation_method=spec.aggregation_method,
        team_mapping=spec.team_mapping or {},
    )
    _WORKER_CACHE["spec"] = spec
    _WORKER_CACHE["evaluator"] = evaluator
    _WORKER_CACHE["env"] = env  # keep ref so env is not GC'd
    return evaluator


class _LightweightIndividual:
    """Minimal ``evaluator.evaluate`` input shaped like a DEAP individual.

    The evaluator contract only requires ``.genome``; constructing a
    real ``creator.Individual`` on the worker would force DEAP's
    ``creator`` side-effects into each subprocess unnecessarily.
    """

    __slots__ = ("genome",)

    def __init__(self, genome: FixedTopologyGenome) -> None:
        self.genome = genome


def _worker_evaluate(
    payload_and_spec: Tuple[_GenomePayload, _EvalWorkerSpec],
) -> tuple:
    """Top-level worker callable.

    Args:
        payload_and_spec: ``(_GenomePayload, _EvalWorkerSpec)`` tuple.
            We pack both so every call carries enough context to
            rebuild the evaluator — necessary because SCOOP / Dask do
            not give us an init hook.

    Returns:
        Fitness tuple from the evaluator, ready for
        ``individual.fitness.values``.
    """
    payload, spec = payload_and_spec
    evaluator = _ensure_worker_evaluator(spec)

    genome = FixedTopologyGenome(
        layer_sizes=list(payload.layer_sizes),
        activation=payload.activation,
        weight_range=tuple(payload.weight_range),
        random_seed=None,
    )
    genome.set_weights(np.asarray(payload.weights, dtype=float))
    ind = _LightweightIndividual(genome)
    return evaluator.evaluate(ind, seed=payload.seed)


def _individual_to_payload(
    individual: Any,
    *,
    seed: Optional[int] = None,
) -> _GenomePayload:
    """Serialise a DEAP individual (or CMA individual) into a payload.

    Handles both ``creator.Individual`` (``.genome`` with full metadata)
    and the CMA-ES individual (list subclass that *also* exposes
    ``.genome``) uniformly.
    """
    genome = individual.genome
    return _GenomePayload(
        weights=np.asarray(genome.get_weights(), dtype=float).copy(),
        layer_sizes=tuple(genome.layer_sizes),
        activation=str(genome.activation),
        weight_range=tuple(getattr(genome, "_weight_range", (-1.0, 1.0))),
        seed=seed,
    )


def make_parallel_evaluate_map(
    executor: Any,
    spec: _EvalWorkerSpec,
) -> Any:
    """Build a ``map``-shaped callable for parallel fitness evaluation.

    The returned callable has the signature
    ``pmap(evaluate_fn, individuals) -> list`` — compatible with
    ``toolbox.register("map", pmap)`` so that DEAP's built-in
    algorithms (:func:`deap.algorithms.eaMuPlusLambda`,
    :func:`deap.algorithms.eaMuCommaLambda`) dispatch evaluations
    through the pool. The ``evaluate_fn`` argument is intentionally
    ignored: the worker rebuilds the evaluator from ``spec`` so the
    driver-side ``toolbox.evaluate`` closure (which holds a live
    :class:`PettingZooWrapper`) is never shipped.

    For the in-house ``_run_simple`` loop the same object is usable
    as a one-arg helper via
    :func:`evaluate_population_parallel`.
    """
    def _pmap(_evaluate_ignored, individuals: Iterable[Any]):
        inds = list(individuals)
        payloads = [
            (_individual_to_payload(ind), spec) for ind in inds
        ]
        return list(executor.map(_worker_evaluate, payloads))

    return _pmap


def evaluate_population_parallel(
    individuals: Iterable[Any],
    *,
    executor: Any,
    spec: _EvalWorkerSpec,
    eval_seeds: Optional[Iterable[Optional[int]]] = None,
) -> List[tuple]:
    """Evaluate a population via ``executor`` using worker rebuilds.

    Thin helper for call sites that do not go through
    ``toolbox.map`` (e.g. the in-house generational loop in
    :class:`EATrainer`).

    Args:
        eval_seeds: Optional iterable of per-individual seeds forwarded
            as the evaluator's ``seed=`` kwarg. Used by tests and the
            ``algorithm_params["deterministic_eval"]`` opt-in to make
            serial and parallel runs produce identical fitness values.
    """
    inds = list(individuals)
    if eval_seeds is None:
        seeds: List[Optional[int]] = [None] * len(inds)
    else:
        seeds = list(eval_seeds)
        if len(seeds) != len(inds):
            raise ValueError(
                f"eval_seeds length {len(seeds)} != population {len(inds)}"
            )
    payloads = [
        (_individual_to_payload(ind, seed=s), spec)
        for ind, s in zip(inds, seeds)
    ]
    return list(executor.map(_worker_evaluate, payloads))


# --------------------------------------------------------------------------
# MixedPolicyEvaluator parallel evaluation plumbing
# --------------------------------------------------------------------------

@dataclass
class _MixedWorkerSpec:
    """Picklable spec for a MixedPolicyEvaluator factory-path worker.

    Carries the **names** of the module-level callables rather than the
    callables themselves, because callables are not always picklable
    (closures, lambdas, nested functions). The worker resolves them via
    :func:`_resolve_callable`.

    Attributes:
        env_factory_qualname: Fully-qualified name of the env factory
            (e.g. ``"examples.pursuit_evasion.ea_vs_apf.build_env"``).
        scripted_factory_qualname: Fully-qualified name of the scripted-
            policy factory.
        learner_factory_qualname: Fully-qualified name of the learner-
            policy factory.
        learner_prefixes: Tuple of agent-name prefixes for learner side.
        max_steps: Episode step cap.
        eval_episodes: Number of episodes to average per call.
        layer_sizes: NN topology ``[in, hidden…, out]``.
        activation: Activation name.
    """

    env_factory_qualname: str
    scripted_factory_qualname: str
    learner_factory_qualname: str
    learner_prefixes: Tuple[str, ...]
    max_steps: int
    eval_episodes: int
    layer_sizes: Tuple[int, ...]
    activation: str


_MIXED_WORKER_CACHE: Dict[str, Any] = {}


def _resolve_callable(qualname: str) -> Any:
    """Import and return a module-level callable by its qualified name.

    Args:
        qualname: Dotted ``module.attribute`` path, e.g.
            ``"mypackage.module.my_function"``.

    Returns:
        The resolved callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute is missing from the module.
    """
    module_name, _, attr = qualname.rpartition(".")
    if not module_name:
        raise ValueError(
            f"_resolve_callable requires a fully-qualified name "
            f"(module.attribute), got {qualname!r}."
        )
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _mixed_worker_evaluate(
    payload_and_spec: Tuple["_GenomePayload", "_MixedWorkerSpec"],
) -> tuple:
    """Top-level worker for MixedPolicyEvaluator parallel evaluation.

    Resolves all factories from their qualified names, builds a fresh
    :class:`~deap_pettingzoo.evaluators.MixedPolicyEvaluator`, and
    evaluates the genome over ``spec.eval_episodes`` episodes.

    This function is module-level and therefore picklable on all
    platforms (including Windows spawn).
    """
    from deap_pettingzoo.evaluators.mixed_policy import MixedPolicyEvaluator

    payload, spec = payload_and_spec

    # Rebuild genome from payload.
    genome = FixedTopologyGenome(
        layer_sizes=list(spec.layer_sizes),
        activation=spec.activation,
        weight_range=tuple(payload.weight_range),
        random_seed=None,
    )
    genome.set_weights(np.asarray(payload.weights, dtype=float))

    # Resolve factories and build the evaluator once per worker process.
    # The cache key includes all fields that influence evaluator construction
    # so that a single worker correctly handles spec changes across runs.
    cache_key = (
        spec.env_factory_qualname,
        spec.scripted_factory_qualname,
        spec.learner_factory_qualname,
        spec.max_steps,
        tuple(spec.learner_prefixes),
    )
    if _MIXED_WORKER_CACHE.get("key") != cache_key:
        _MIXED_WORKER_CACHE["key"] = cache_key
        env_factory = _resolve_callable(spec.env_factory_qualname)
        scripted_factory = _resolve_callable(spec.scripted_factory_qualname)
        learner_factory = _resolve_callable(spec.learner_factory_qualname)
        _MIXED_WORKER_CACHE["env_factory"] = env_factory
        _MIXED_WORKER_CACHE["scripted_factory"] = scripted_factory
        _MIXED_WORKER_CACHE["learner_factory"] = learner_factory
        _MIXED_WORKER_CACHE["evaluator"] = MixedPolicyEvaluator(
            env_factory=env_factory,
            scripted_policy_factory=scripted_factory,
            learner_agent_prefixes=list(spec.learner_prefixes),
            learner_factory=learner_factory,
            max_steps=spec.max_steps,
        )

    evaluator = _MIXED_WORKER_CACHE["evaluator"]

    class _Ind:
        __slots__ = ("genome",)

        def __init__(self, g):
            self.genome = g

    ind = _Ind(genome)
    total = sum(
        evaluator.run_episode(ind, seed=payload.seed).fitness
        for _ in range(spec.eval_episodes)
    )
    return (total / spec.eval_episodes,)


def make_mixed_parallel_evaluate_map(
    executor: Any,
    evaluator: Any,
    *,
    eval_episodes: int = 1,
) -> Any:
    """Build a ``map``-shaped callable for parallel :class:`MixedPolicyEvaluator` eval.

    The returned callable has the signature
    ``pmap(evaluate_fn, individuals) -> list`` — compatible with
    ``toolbox.register("map", pmap)``.

    The evaluator **must** have been constructed via the factory path
    (``env_factory`` + ``scripted_policy_factory`` + ``learner_factory``
    all set as module-level callables).  Live-object construction raises
    ``ValueError``.

    Example::

        evaluator = MixedPolicyEvaluator(
            env_factory=build_env,
            scripted_policy_factory=build_scripted_evaders,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=200,
        )
        with make_executor(n_workers) as executor:
            pmap = make_mixed_parallel_evaluate_map(
                executor, evaluator, eval_episodes=3
            )
            toolbox.register("map", pmap)

    Args:
        executor: A context-managed executor (e.g. from
            :func:`~deap_pettingzoo.utils.parallel.make_executor`).
        evaluator: A :class:`MixedPolicyEvaluator` built with the factory
            path.
        eval_episodes: Episodes averaged per genome per worker call.

    Returns:
        ``pmap`` callable compatible with ``toolbox.register("map", pmap)``.

    Raises:
        ValueError: If the evaluator was not built with the factory path.
        ValueError: If factories are not importable (not module-level).
    """
    from deap_pettingzoo.evaluators.mixed_policy import MixedPolicyEvaluator

    if not isinstance(evaluator, MixedPolicyEvaluator):
        raise TypeError(
            f"evaluator must be a MixedPolicyEvaluator, got {type(evaluator)!r}."
        )
    if evaluator.env_factory is None or evaluator.scripted_policy_factory is None:
        raise ValueError(
            "make_mixed_parallel_evaluate_map requires the evaluator to be "
            "constructed with 'env_factory' and 'scripted_policy_factory'. "
            "Live-object construction (env_wrapper=...) cannot be pickled "
            "across worker processes."
        )

    def _qualname(fn: Any) -> str:
        mod = getattr(fn, "__module__", None)
        name = getattr(fn, "__qualname__", None)
        if mod is None or name is None or "<" in name:
            raise ValueError(
                f"Factory {fn!r} is not a module-level callable (module={mod!r}, "
                f"qualname={name!r}). Lambdas and closures cannot be pickled "
                "across processes. Use a module-level def instead."
            )
        return f"{mod}.{name}"

    # Sample genome from the evaluator's env to discover topology.
    temp_env = evaluator.env_wrapper if evaluator.env_wrapper is not None else evaluator.env_factory()
    try:
        pursuer_id = next(
            a for a in temp_env.possible_agents
            if a.startswith(evaluator.learner_agent_prefixes)
        )
        obs = temp_env.reset(seed=0)
        input_size = int(obs[pursuer_id].shape[0])
        act_space = temp_env.action_space(pursuer_id)
        output_size = act_space.n if hasattr(act_space, "n") else int(act_space.shape[0])
    finally:
        if evaluator.env_wrapper is None:
            temp_env.close()

    spec = _MixedWorkerSpec(
        env_factory_qualname=_qualname(evaluator.env_factory),
        scripted_factory_qualname=_qualname(evaluator.scripted_policy_factory),
        learner_factory_qualname=_qualname(evaluator.learner_factory),
        learner_prefixes=evaluator.learner_agent_prefixes,
        max_steps=evaluator.max_steps,
        eval_episodes=eval_episodes,
        layer_sizes=(input_size, output_size),  # placeholder; overridden per-genome
        activation="tanh",  # placeholder; overridden per-genome
    )

    def _pmap(_evaluate_ignored, individuals):
        inds = list(individuals)
        payloads = []
        for ind in inds:
            genome = ind.genome
            per_ind_spec = _MixedWorkerSpec(
                env_factory_qualname=spec.env_factory_qualname,
                scripted_factory_qualname=spec.scripted_factory_qualname,
                learner_factory_qualname=spec.learner_factory_qualname,
                learner_prefixes=spec.learner_prefixes,
                max_steps=spec.max_steps,
                eval_episodes=spec.eval_episodes,
                layer_sizes=tuple(genome.layer_sizes),
                activation=genome.activation,
            )
            payloads.append((_individual_to_payload(ind), per_ind_spec))
        return list(executor.map(_mixed_worker_evaluate, payloads))

    return _pmap

