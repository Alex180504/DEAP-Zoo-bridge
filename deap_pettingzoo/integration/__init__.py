"""Integration utilities for DEAP framework."""

from .cma_integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_one_plus_lambda_strategy,
    setup_cma_strategy,
)
from .deap_integration import (
    clone_individual,
    create_deap_individual,
    create_toolbox,
    crossover_blend,
    evaluate_individual,
    make_mixed_parallel_evaluate_map,
    make_parallel_evaluate_map,
    mutate_gaussian,
    setup_deap_creators,
)
from .objectives import (
    FitnessSpec,
    ObjectiveSpec,
    build_fitness_vector,
    compute_parsimony,
)
from .pso_integration import (
    make_pso_ind_init,
    make_pso_stats,
    register_pso_toolbox,
    setup_pso_swarm,
)

__all__ = [
    "setup_deap_creators",
    "create_deap_individual",
    "evaluate_individual",
    "create_toolbox",
    "crossover_blend",
    "mutate_gaussian",
    "clone_individual",
    "make_parallel_evaluate_map",
    "make_mixed_parallel_evaluate_map",
    "setup_cma_strategy",
    "setup_cma_one_plus_lambda_strategy",
    "make_cma_ind_init",
    "register_cma_toolbox",
    "make_cma_stats",
    "FitnessSpec",
    "ObjectiveSpec",
    "build_fitness_vector",
    "compute_parsimony",
    "setup_pso_swarm",
    "make_pso_ind_init",
    "register_pso_toolbox",
    "make_pso_stats",
]
