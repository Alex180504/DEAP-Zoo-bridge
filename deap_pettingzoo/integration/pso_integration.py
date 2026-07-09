"""
Particle Swarm Optimisation integration utilities.

This module mirrors :mod:`deap_pettingzoo.integration.cma_integration` but
wires the ``FixedTopologyGenome`` flat weight vector into a classic PSO
search instead of CMA-ES.

Design notes:

* Each particle is a ``creator.PSOParticle`` (``list`` subclass).
* The evaluator contract (``individual.genome`` -> fitness tuple) is
  preserved by attaching a transient ``FixedTopologyGenome`` to each
  particle before evaluation.
* Global best (gbest) is tracked in a per-toolbox mutable dict
  ``_pso_gbest_ctx`` so multiple independent PSO runs do not interfere.
* Velocity is clamped and position bounded to the genome's weight range.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from ..evaluators import BaseEvaluator
from ..genomes import FixedTopologyGenome
from .deap_integration import evaluate_individual, setup_deap_creators


# Creator management


def _setup_pso_creators() -> None:
    """Idempotently create ``creator.PSOParticle`` (list subclass)."""
    setup_deap_creators()
    if hasattr(creator, "PSOParticle"):
        del creator.PSOParticle
    creator.create(
        "PSOParticle",
        list,
        fitness=creator.FitnessMax,
        genome=None,
        speed=None,
        smin=None,
        smax=None,
        best=None,
        best_fitness=None,
    )


# Swarm setup


def setup_pso_swarm(
    genome_config: Dict[str, Any],
    population_size: int = 50,
    phi1: float = 2.0,
    phi2: float = 2.0,
    weight: float = 0.7298,
    smin: Optional[float] = None,
    smax: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """Build PSO swarm state (hyperparameters + dimensionality).

    Args:
        genome_config: Dict with ``layer_sizes`` (required).
        population_size: Number of particles in the swarm.
        phi1: Cognitive acceleration coefficient.
        phi2: Social acceleration coefficient.
        weight: Inertia weight.
        smin/smax: Velocity bounds (default ±half weight range).
        seed: Optional numpy random seed.

    Returns:
        ``(pso_state, dimension)``.
    """
    if "layer_sizes" not in genome_config:
        raise ValueError("genome_config must contain 'layer_sizes'")

    if seed is not None:
        np.random.seed(int(seed))

    probe = FixedTopologyGenome(
        layer_sizes=genome_config["layer_sizes"],
        activation=genome_config.get("activation", "tanh"),
        output_activation=genome_config.get(
            "output_activation", genome_config.get("activation", "tanh")
        ),
        weight_range=genome_config.get("weight_range", (-1.0, 1.0)),
        random_seed=0,
    )
    dimension = probe.get_weight_count()
    wr = genome_config.get("weight_range", (-1.0, 1.0))
    low, high = float(wr[0]), float(wr[1])
    if smin is None:
        smin = -abs(high - low) * 0.5
    if smax is None:
        smax = abs(high - low) * 0.5

    return {
        "dimension": dimension,
        "population_size": int(population_size),
        "low": low,
        "high": high,
        "phi1": float(phi1),
        "phi2": float(phi2),
        "weight": float(weight),
        "smin": float(smin),
        "smax": float(smax),
    }, dimension


# Particle initialiser


def make_pso_ind_init(
    genome_config: Dict[str, Any],
    pso_state: Dict[str, Any],
) -> Callable[[], Any]:
    """Return a zero-arg closure that creates a fresh PSO particle."""
    _setup_pso_creators()
    ls = list(genome_config["layer_sizes"])
    act = genome_config.get("activation", "tanh")
    out_act = genome_config.get("output_activation", act)
    wr = genome_config.get("weight_range", (-1.0, 1.0))
    low, high = float(wr[0]), float(wr[1])
    smin = float(pso_state["smin"])
    smax = float(pso_state["smax"])
    dim = pso_state["dimension"]

    def _ind_init() -> Any:
        vec = np.random.uniform(low, high, size=dim)
        genome = FixedTopologyGenome(
            layer_sizes=ls,
            activation=act,
            output_activation=out_act,
            weight_range=wr,
            random_seed=0,
        )
        genome.set_weights(vec)
        p = creator.PSOParticle(vec.tolist())
        p.genome = genome
        p.speed = np.random.uniform(smin, smax, size=dim)
        p.smin = smin
        p.smax = smax
        p.best = None
        p.best_fitness = None
        return p

    return _ind_init


# Toolbox registration


def register_pso_toolbox(
    pso_state: Dict[str, Any],
    ind_init: Callable[[], Any],
    evaluator: BaseEvaluator,
) -> base.Toolbox:
    """Build a DEAP toolbox for the PSO generate/evaluate/update loop.

    Registers ``toolbox.generate``, ``toolbox.update``, ``toolbox.evaluate``.
    A per-toolbox ``_pso_gbest_ctx`` dict tracks the global best.
    """
    pop_size = int(pso_state["population_size"])
    toolbox = base.Toolbox()
    toolbox.register("particle", ind_init)
    toolbox.register("generate", tools.initRepeat, list, toolbox.particle, n=pop_size)
    gbest_ctx: Dict[str, Any] = {"pos": None, "fitness": None}
    toolbox._pso_gbest_ctx = gbest_ctx  # type: ignore[attr-defined]
    toolbox.register(
        "update", _pso_update_top_level, pso_state=pso_state, gbest_ctx=gbest_ctx
    )
    toolbox.register("evaluate", evaluate_individual, evaluator=evaluator)
    return toolbox


# PSO update (module-level for DEAP picklability)


def _pso_update_top_level(
    population: List[Any],
    pso_state: Dict[str, Any],
    gbest_ctx: Dict[str, Any],
) -> None:
    """Velocity/position update + global-best tracking."""
    phi1 = float(pso_state["phi1"])
    phi2 = float(pso_state["phi2"])
    w = float(pso_state["weight"])
    smin = float(pso_state["smin"])
    smax = float(pso_state["smax"])
    low = float(pso_state["low"])
    high = float(pso_state["high"])

    # Update global best.
    for ind in population:
        if not ind.fitness.valid:
            continue
        f = ind.fitness.values[0]
        if gbest_ctx["fitness"] is None or f > gbest_ctx["fitness"]:
            gbest_ctx["fitness"] = f
            gbest_ctx["pos"] = np.asarray(ind, dtype=np.float64).copy()

    gbest_pos = gbest_ctx["pos"]
    if gbest_pos is None:
        return

    for ind in population:
        x = np.asarray(ind, dtype=np.float64)
        v = np.asarray(ind.speed, dtype=np.float64)
        if ind.fitness.valid:
            f = ind.fitness.values[0]
            if ind.best is None or f > ind.best_fitness:
                ind.best = x.copy()
                ind.best_fitness = f
        pbest = ind.best if ind.best is not None else x
        r1 = np.random.uniform(0, 1, size=x.shape)
        r2 = np.random.uniform(0, 1, size=x.shape)
        v = w * v + phi1 * r1 * (pbest - x) + phi2 * r2 * (gbest_pos - x)
        v = np.clip(v, smin, smax)
        ind.speed = v
        x_new = np.clip(x + v, low, high)
        ind.clear()
        ind.extend(x_new.tolist())
        ind.genome.set_weights(x_new)


def make_pso_stats() -> tools.Statistics:
    """Return a Statistics instance (avg/std/min/max)."""
    s = tools.Statistics(lambda ind: ind.fitness.values)
    s.register("avg", np.mean)
    s.register("std", np.std)
    s.register("min", np.min)
    s.register("max", np.max)
    return s
