"""
CMA-ES integration utilities.

This module mirrors :mod:`deap_pettingzoo.integration.deap_integration` but
wires the `FixedTopologyGenome` flat weight vector into DEAP's built-in
CMA-ES implementation (:class:`deap.cma.Strategy`) instead of the
classical GA toolbox.

Design notes:

* CMA-ES samples **real-valued vectors** of length ``N =
  FixedTopologyGenome(...).get_weight_count()``. Each sample is therefore
  represented as a list-typed DEAP individual (``creator.CMAIndividual``,
  a subclass of ``list``) so that :meth:`deap.cma.Strategy.update` can
  rebuild the sample matrix via ``numpy.array(population)``.
* The project's evaluators contract (``individual.genome`` → fitness
  tuple) is preserved by attaching a transient :class:`FixedTopologyGenome`
  to each sampled individual via ``ind_init`` before evaluation.
* Fitness weights stay ``(1.0,)`` (maximisation) — ``deap.cma.Strategy``
  honours the sign via the underlying :class:`deap.base.Fitness` ordering.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from deap import base, cma, creator, tools

from ..evaluators import BaseEvaluator
from ..genomes import FixedTopologyGenome
from .deap_integration import evaluate_individual, setup_deap_creators


def _setup_cma_creators() -> None:
    """Idempotently create ``creator.CMAIndividual`` (list subclass).

    Reuses ``creator.FitnessMax`` created by
    :func:`setup_deap_creators`. The CMA individual must subclass ``list``
    so that :meth:`deap.cma.Strategy.update` can call
    ``numpy.array(population)`` and recover the ``(lambda_, N)`` matrix.
    """
    setup_deap_creators()
    if hasattr(creator, "CMAIndividual"):
        del creator.CMAIndividual
    creator.create(
        "CMAIndividual",
        list,
        fitness=creator.FitnessMax,
        genome=None,
    )


def setup_cma_strategy(
    genome_config: Dict[str, Any],
    sigma0: float = 0.3,
    population_size: Optional[int] = None,
    centroid: Optional[np.ndarray] = None,
) -> Tuple[cma.Strategy, int]:
    """Build a :class:`deap.cma.Strategy` sized to the genome's weight count.

    Args:
        genome_config: Dict with ``layer_sizes`` (required), plus the
            optional keys ``activation`` and ``weight_range`` used to
            compute the weight dimensionality and to seed the centroid
            when it is not supplied explicitly.
        sigma0: Initial step size.
        population_size: Override ``lambda_``. Defaults to DEAP's
            ``4 + int(3 * log(N))``.
        centroid: Initial mean of the search distribution of length ``N``.
            When ``None``, a zero vector is used.

    Returns:
        ``(strategy, dimension)`` — the strategy instance and the scalar
        weight-vector dimensionality ``N``.

    Raises:
        ValueError: If ``layer_sizes`` is missing from ``genome_config``
            or the supplied ``centroid`` has the wrong length.
    """
    if "layer_sizes" not in genome_config:
        raise ValueError("genome_config must contain 'layer_sizes'")

    # Probe genome to get the scalar weight count without committing weights.
    probe = FixedTopologyGenome(
        layer_sizes=genome_config["layer_sizes"],
        activation=genome_config.get("activation", "tanh"),
        weight_range=genome_config.get("weight_range", (-1.0, 1.0)),
        random_seed=0,
    )
    dimension = probe.get_weight_count()

    if centroid is None:
        centroid = np.zeros(dimension, dtype=np.float64)
    else:
        centroid = np.asarray(centroid, dtype=np.float64)
        if centroid.shape != (dimension,):
            raise ValueError(
                f"centroid has shape {centroid.shape}, expected ({dimension},)"
            )

    kwargs: Dict[str, Any] = {}
    if population_size is not None:
        kwargs["lambda_"] = int(population_size)

    strategy = cma.Strategy(
        centroid=centroid.tolist(),
        sigma=float(sigma0),
        **kwargs,
    )
    return strategy, dimension


def make_cma_ind_init(
    genome_config: Dict[str, Any],
) -> Callable[[Iterable[float]], Any]:
    """Return an ``ind_init`` closure for :meth:`cma.Strategy.generate`.

    The closure consumes a raw weight vector (any iterable of floats of
    length ``N``) and returns a fresh ``creator.CMAIndividual`` whose
    element values are the weights **and** whose ``.genome`` attribute is
    a matching :class:`FixedTopologyGenome`. Downstream evaluators consume
    ``individual.genome`` per the project contract.

    Args:
        genome_config: Same schema as :func:`setup_cma_strategy`.

    Returns:
        Callable suitable as the ``ind_init`` argument to
        :meth:`deap.cma.Strategy.generate`.
    """
    _setup_cma_creators()

    layer_sizes = list(genome_config["layer_sizes"])
    activation = genome_config.get("activation", "tanh")
    weight_range = genome_config.get("weight_range", (-1.0, 1.0))

    def _ind_init(weights: Iterable[float]) -> Any:
        vec = np.asarray(list(weights), dtype=np.float64)
        genome = FixedTopologyGenome(
            layer_sizes=layer_sizes,
            activation=activation,
            weight_range=weight_range,
            random_seed=0,
        )
        genome.set_weights(vec)
        individual = creator.CMAIndividual(vec.tolist())
        individual.genome = genome
        return individual

    return _ind_init


def register_cma_toolbox(
    strategy: cma.Strategy,
    ind_init: Callable[[Iterable[float]], Any],
    evaluator: BaseEvaluator,
) -> base.Toolbox:
    """Build a DEAP toolbox wired for the CMA-ES generate/update loop.

    Registers:

    * ``toolbox.generate`` → ``strategy.generate(ind_init)``
    * ``toolbox.update`` → ``strategy.update``
    * ``toolbox.evaluate`` → :func:`evaluate_individual` bound to ``evaluator``

    Args:
        strategy: The CMA strategy returned by :func:`setup_cma_strategy`.
        ind_init: The closure returned by :func:`make_cma_ind_init`.
        evaluator: Evaluator consuming ``individual.genome``.

    Returns:
        Configured :class:`deap.base.Toolbox`.
    """
    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, ind_init)
    toolbox.register("update", strategy.update)
    toolbox.register("evaluate", evaluate_individual, evaluator=evaluator)
    return toolbox


def make_cma_stats() -> tools.Statistics:
    """Return a :class:`tools.Statistics` instance with the standard keys.

    Mirrors the ``avg``/``std``/``min``/``max`` layout used by
    :class:`deap_pettingzoo.baselines.ea_trainer.EATrainer` so the two
    trainers produce comparably-shaped ``fitness_history`` records.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def setup_cma_one_plus_lambda_strategy(
    genome_config: Dict[str, Any],
    sigma0: float = 0.3,
    population_size: Optional[int] = None,
    parent: Optional[np.ndarray] = None,
) -> Tuple[cma.StrategyOnePlusLambda, int]:
    """Build a :class:`deap.cma.StrategyOnePlusLambda`.

    The (1+λ)-CMA-ES variant keeps a single parent between generations
    and samples ``lambda_`` offspring around it. It is typically more
    sample-efficient than the default (μ/μ, λ) strategy for unimodal
    objectives, at the price of weaker exploration.

    Args:
        genome_config: Dict with ``layer_sizes`` (required); optional
            ``activation`` / ``weight_range`` mirror
            :func:`setup_cma_strategy`.
        sigma0: Initial step size.
        population_size: Override ``lambda_``. Defaults to the DEAP
            builtin ``4 + int(3 * log(N))``.
        parent: Initial parent (shape ``(N,)``). Zero vector by default.

    Returns:
        ``(strategy, dimension)``. The strategy exposes ``.generate`` and
        ``.update`` with the same signatures as :class:`cma.Strategy`,
        so it plugs into :func:`register_cma_toolbox` unchanged.

    Raises:
        ValueError: If ``layer_sizes`` is missing or ``parent`` has the
            wrong shape.
    """
    if "layer_sizes" not in genome_config:
        raise ValueError("genome_config must contain 'layer_sizes'")

    probe = FixedTopologyGenome(
        layer_sizes=genome_config["layer_sizes"],
        activation=genome_config.get("activation", "tanh"),
        weight_range=genome_config.get("weight_range", (-1.0, 1.0)),
        random_seed=0,
    )
    dimension = probe.get_weight_count()

    if parent is None:
        parent = np.zeros(dimension, dtype=np.float64)
    else:
        parent = np.asarray(parent, dtype=np.float64)
        if parent.shape != (dimension,):
            raise ValueError(
                f"parent has shape {parent.shape}, expected ({dimension},)"
            )

    kwargs: Dict[str, Any] = {}
    if population_size is not None:
        kwargs["lambda_"] = int(population_size)

    # DEAP's StrategyOnePlusLambda expects ``parent`` to be an
    # Individual whose ``.fitness`` attribute is already populated — the
    # first call to :meth:`update` does ``self.parent.fitness <= ind.fitness``.
    # We wrap the raw vector as a CMAIndividual and seed its fitness with
    # ``-inf`` so the first generation's offspring are all counted as
    # "improvements" (the canonical DEAP pattern is to evaluate the
    # parent up-front, but callers only have access to the evaluator via
    # the trainer; deferring that to the trainer would leak state).
    _setup_cma_creators()
    parent_ind = creator.CMAIndividual(parent.tolist())
    probe.set_weights(parent)
    parent_ind.genome = probe
    parent_ind.fitness.values = (float("-inf"),)

    strategy = cma.StrategyOnePlusLambda(
        parent=parent_ind,
        sigma=float(sigma0),
        **kwargs,
    )
    return strategy, dimension
