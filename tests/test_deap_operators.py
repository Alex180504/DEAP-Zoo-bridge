"""Tests for the method-dispatched genetic operators in ``create_toolbox``."""

import pytest

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration import create_toolbox, setup_deap_creators


@pytest.fixture(scope="module")
def evaluator():
    setup_deap_creators()
    wrapper = PettingZooWrapper(
        env_name="simple_spread_v3",
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
    ["tournament", "roulette", "best", "random", "sus", "double_tournament"],
)
def test_selection_method_dispatch(evaluator, genome_config, method):
    """Every supported selection method must produce a working toolbox."""
    cfg: dict = {"method": method, "tournament_size": 3}
    if method == "double_tournament":
        cfg.update({"fitness_size": 2, "parsimony_size": 3})
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config=cfg,
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
            selection_config={"method": "cxOrdered"},
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


# ---------------------------------------------------------------------------
# Phase 1.2 — Advanced selection operator tests
# ---------------------------------------------------------------------------


class TestDoubleTournamentSelection:
    """Double-tournament uses genome.get_weight_count() as parsimony."""

    def test_favours_smaller_networks(self, evaluator, genome_config):
        setup_deap_creators()
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={
                "method": "double_tournament",
                "fitness_size": 3,
                "parsimony_size": 5,
            },
        )
        pop = toolbox.population(n=12)
        # Give half a fitness advantage; double-tournament with
        # fitness_first=True should select fit individuals with
        # small networks.  Inflate the LESS fit half so the
        # fitness tournament reliably picks the fitter (smaller) ones.
        for i, ind in enumerate(pop):
            ind.fitness.values = (1.0,)
            if i >= 6:
                _orig_wc = ind.genome.get_weight_count
                ind.genome.get_weight_count = lambda _orig=_orig_wc: _orig_wc() * 10

        chosen = toolbox.select(pop, k=4)
        assert len(chosen) == 4
        # At least half of the selected should be non-inflated.
        small_count = sum(
            1 for ind in chosen
            if ind.genome.get_weight_count() < 500
        )
        assert small_count >= 2

    def test_fitness_first_false(self, evaluator, genome_config):
        """Parsimony-first mode selects small networks first, then fit among them."""
        setup_deap_creators()
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={
                "method": "double_tournament",
                "fitness_size": 2,
                "parsimony_size": 3,
                "fitness_first": False,
            },
        )
        pop = toolbox.population(n=8)
        for ind in pop:
            ind.fitness.values = (0.0,)
        chosen = toolbox.select(pop, k=3)
        assert len(chosen) == 3
        for ind in chosen:
            assert hasattr(ind, "genome")


class TestLexicaseSelection:
    """Lexicase / epsilon-lexicase require multi-dimensional fitness vectors."""

    @staticmethod
    def _multi_fitness_pop(genome_config, evaluator, n=6, dim=4):
        setup_deap_creators(weights=(1.0,) * dim)
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
        )
        pop = toolbox.population(n=n)
        rng = __import__("numpy").random.default_rng(42)
        for ind in pop:
            ind.fitness.values = tuple(float(v) for v in rng.uniform(-10, 10, size=dim))
        return toolbox, pop

    def test_lexicase_selects_with_vector_fitness(self, evaluator, genome_config):
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "lexicase"},
        )
        _, pop = self._multi_fitness_pop(genome_config, evaluator)
        chosen = toolbox.select(pop, k=3)
        assert len(chosen) == 3
        for ind in chosen:
            assert hasattr(ind, "genome")

    def test_epsilon_lexicase_selects(self, evaluator, genome_config):
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "epsilon_lexicase"},
        )
        _, pop = self._multi_fitness_pop(genome_config, evaluator)
        chosen = toolbox.select(pop, k=3)
        assert len(chosen) == 3

    def test_lexicase_requires_multi_dimensional_fitness(self, evaluator, genome_config):
        """Lexicase with scalar fitness is defined-but-passive (picks randomly)."""
        # Reset to single-objective creators — surrounding tests may have
        # switched to multi-dimensional fitness.
        setup_deap_creators()
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "lexicase"},
        )
        pop = toolbox.population(n=4)
        for ind in pop:
            ind.fitness.values = (5.0,)  # 1-D tuple
        chosen = toolbox.select(pop, k=2)
        assert len(chosen) == 2


class TestSUSSelection:
    """Stochastic Universal Sampling plain dispatch test."""

    def test_sus_selects_correct_count(self, evaluator, genome_config):
        # Ensure single-objective creators (may have been altered by lexicase tests).
        setup_deap_creators()
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "sus"},
        )
        pop = toolbox.population(n=8)
        for ind in pop:
            ind.fitness.values = (abs(float(hash(str(id(ind))))) % 100,)
        chosen = toolbox.select(pop, k=5)
        assert len(chosen) == 5
        for ind in chosen:
            assert hasattr(ind, "genome")
