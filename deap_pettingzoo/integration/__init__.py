"""Integration utilities for DEAP framework."""

from .deap_integration import (
    setup_deap_creators,
    create_deap_individual,
    evaluate_individual,
    create_toolbox,
    crossover_blend,
    mutate_gaussian,
    clone_individual,
    make_parallel_evaluate_map,
    make_mixed_parallel_evaluate_map,
)
from .cma_integration import (
    setup_cma_strategy,
    setup_cma_one_plus_lambda_strategy,
    make_cma_ind_init,
    register_cma_toolbox,
    make_cma_stats,
)
from .objectives import (
    FitnessSpec,
    ObjectiveSpec,
    build_fitness_vector,
    compute_parsimony,
)

__all__ = [
    'setup_deap_creators',
    'create_deap_individual',
    'evaluate_individual',
    'create_toolbox',
    'crossover_blend',
    'mutate_gaussian',
    'clone_individual',
    'make_parallel_evaluate_map',
    'make_mixed_parallel_evaluate_map',
    'setup_cma_strategy',
    'setup_cma_one_plus_lambda_strategy',
    'make_cma_ind_init',
    'register_cma_toolbox',
    'make_cma_stats',
    'FitnessSpec',
    'ObjectiveSpec',
    'build_fitness_vector',
    'compute_parsimony',
]
