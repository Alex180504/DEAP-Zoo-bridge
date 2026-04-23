"""Tests for the method-dispatched genetic operators in ``create_toolbox``."""

import pytest

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration import create_toolbox, setup_deap_creators


@pytest.fixture(scope="module")
def evaluator():
    setup_deap_creators()
    wrapper = PettingZooWrapper(
        env_name="simple_tag_v3",
        env_config={"max_cycles": 10},
    )
    ev = SingleEpisodeEvaluator(
        env_wrapper=wrapper,
        max_steps=10,
        aggregation_method="sum",
    )
    yield ev
    wrapper.close()


@pytest.fixture(scope="module")
def genome_config():
    # Small topology keeps runs fast.
    return {
        "layer_sizes": [16, 8, 4],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
        "random_seed": 42,
    }


@pytest.mark.parametrize(
    "method",
    ["tournament", "roulette", "best", "random"],
)
def test_selection_method_dispatch(evaluator, genome_config, method):
    """Every supported selection method must produce a working toolbox."""
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"method": method, "tournament_size": 3},
    )
    pop = toolbox.population(n=6)
    for ind in pop:
        ind.fitness.values = (0.1,)
    chosen = toolbox.select(pop, k=4)
    assert len(chosen) == 4
    for ind in chosen:
        assert hasattr(ind, "genome")


def test_roulette_rejects_negative_fitness(evaluator, genome_config):
    """``roulette`` must fail loudly when any fitness is < 0 (selRoulette
    is mathematically undefined for negative weights, and PettingZoo
    evaluators routinely return negative episode rewards)."""
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"method": "roulette"},
    )
    pop = toolbox.population(n=4)
    for ind in pop:
        ind.fitness.values = (0.5,)
    pop[1].fitness.values = (-3.2,)
    with pytest.raises(ValueError, match="non-negative fitness"):
        toolbox.select(pop, k=2)


@pytest.mark.parametrize(
    "method,params",
    [
        ("blend", {}),
        ("one_point", {}),
        ("two_point", {}),
        ("uniform", {"indpb": 0.3}),
        ("sbx", {"eta": 20.0}),
        ("es_blend", {"alpha": 0.5}),
    ],
)
def test_crossover_method_dispatch(evaluator, genome_config, method, params):
    """Every supported crossover preserves the genome weight count."""
    cfg = {"method": method, **params}
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        crossover_config=cfg,
    )
    p1, p2 = toolbox.population(n=2)
    n_before = p1.genome.get_weight_count()
    c1, c2 = toolbox.mate(p1, p2)
    assert c1.genome.get_weight_count() == n_before
    assert c2.genome.get_weight_count() == n_before


@pytest.mark.parametrize(
    "method,params",
    [
        ("gaussian", {"sigma": 0.1, "indpb": 0.5}),
        ("polynomial_bounded", {"eta": 20.0, "indpb": 0.5}),
    ],
)
def test_mutation_method_dispatch(evaluator, genome_config, method, params):
    """Every supported mutation preserves weight count and modifies weights."""
    cfg = {"method": method, **params}
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        mutation_config=cfg,
    )
    # Grab a fresh individual with its current weight vector snapshot.
    ind = toolbox.population(n=1)[0]
    before = ind.genome.get_weights().copy()
    (mutated,) = toolbox.mutate(ind)
    after = mutated.genome.get_weights()
    assert after.shape == before.shape
    # For polynomial_bounded the bounds are (-1, 1); confirm respected.
    if method == "polynomial_bounded":
        assert after.min() >= -1.0 - 1e-9
        assert after.max() <= 1.0 + 1e-9


def test_unknown_selection_raises(evaluator, genome_config):
    with pytest.raises(ValueError, match="Unknown selection method"):
        create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "nsga2"},
        )


def test_unknown_crossover_raises(evaluator, genome_config):
    with pytest.raises(ValueError, match="Unknown crossover method"):
        create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            crossover_config={"method": "cxOrdered"},
        )


def test_unknown_mutation_raises(evaluator, genome_config):
    with pytest.raises(ValueError, match="Unknown mutation method"):
        create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            mutation_config={"method": "flip_bit"},
        )


def test_default_methods_backwards_compatible(evaluator, genome_config):
    """Omitting ``method`` reproduces the historical tournament/blend/gaussian."""
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
    )
    # Should register all three operators without error.
    assert hasattr(toolbox, "select")
    assert hasattr(toolbox, "mate")
    assert hasattr(toolbox, "mutate")
    pop = toolbox.population(n=4)
    for ind in pop:
        ind.fitness.values = (0.0,)
    # tournament default:
    chosen = toolbox.select(pop, k=2)
    assert len(chosen) == 2
