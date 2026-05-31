"""Tests for the CMA-ES integration helpers."""

import numpy as np
import pytest
from deap import cma, creator

from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_strategy,
    setup_deap_creators,
)


GENOME_CONFIG = {
    "layer_sizes": [4, 8, 2],
    "activation": "tanh",
    "output_activation": "linear",
    "weight_range": (-1.0, 1.0),
}

EXPECTED_DIM = FixedTopologyGenome(**GENOME_CONFIG, random_seed=0).get_weight_count()


class TestSetupCMAStrategy:
    def test_returns_strategy_with_correct_dimension(self):
        strategy, dim = setup_cma_strategy(GENOME_CONFIG, sigma0=0.3, population_size=10)
        assert isinstance(strategy, cma.Strategy)
        assert dim == EXPECTED_DIM
        assert strategy.dim == EXPECTED_DIM
        assert strategy.lambda_ == 10

    def test_default_population_size(self):
        strategy, _ = setup_cma_strategy(GENOME_CONFIG, sigma0=0.3)
        # DEAP default: 4 + int(3 * log(N))
        expected = 4 + int(3 * np.log(EXPECTED_DIM))
        assert strategy.lambda_ == expected

    def test_custom_centroid(self):
        centroid = np.full(EXPECTED_DIM, 0.25, dtype=np.float64)
        strategy, _ = setup_cma_strategy(GENOME_CONFIG, sigma0=0.1, centroid=centroid)
        assert np.allclose(strategy.centroid, centroid)

    def test_centroid_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            setup_cma_strategy(GENOME_CONFIG, centroid=np.zeros(EXPECTED_DIM + 1))

    def test_missing_layer_sizes_raises(self):
        with pytest.raises(ValueError):
            setup_cma_strategy({"activation": "tanh"})


class TestCMAIndInit:
    def test_ind_init_returns_creator_individual(self):
        ind_init = make_cma_ind_init(GENOME_CONFIG)
        vec = np.full(EXPECTED_DIM, 0.1, dtype=np.float64)
        ind = ind_init(vec)
        assert isinstance(ind, list)
        assert isinstance(ind, creator.CMAIndividual)
        assert len(ind) == EXPECTED_DIM

    def test_ind_init_attaches_matching_genome(self):
        ind_init = make_cma_ind_init(GENOME_CONFIG)
        vec = np.linspace(-0.5, 0.5, EXPECTED_DIM)
        ind = ind_init(vec)
        assert isinstance(ind.genome, FixedTopologyGenome)
        assert np.allclose(ind.genome.get_weights(), vec)
        assert ind.genome.layer_sizes == GENOME_CONFIG["layer_sizes"]
        assert ind.genome.activation == GENOME_CONFIG["activation"]
        assert ind.genome.output_activation == GENOME_CONFIG["output_activation"]

    def test_ind_init_propagates_list_values(self):
        ind_init = make_cma_ind_init(GENOME_CONFIG)
        vec = np.arange(EXPECTED_DIM, dtype=np.float64)
        ind = ind_init(vec)
        assert np.allclose(np.asarray(ind), vec)


class TestCMAToolboxLoop:
    def test_generate_update_cycle(self):
        """One full generate → evaluate → update cycle must not raise and must move the centroid."""
        setup_deap_creators()
        strategy, _ = setup_cma_strategy(GENOME_CONFIG, sigma0=0.5, population_size=12)
        ind_init = make_cma_ind_init(GENOME_CONFIG)

        class _StubEvaluator:
            """Reward = -||weights||^2 so mean drifts toward zero."""

            def evaluate(self, individual, **_):
                w = individual.genome.get_weights()
                return (float(-np.dot(w, w)),)

        toolbox = register_cma_toolbox(strategy, ind_init, _StubEvaluator())

        # Non-zero initial centroid to observe drift.
        strategy.centroid = np.full(EXPECTED_DIM, 1.0)

        initial_centroid = np.array(strategy.centroid, copy=True)

        population = toolbox.generate()
        assert len(population) == strategy.lambda_
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        toolbox.update(population)

        new_centroid = np.array(strategy.centroid, copy=True)
        assert not np.allclose(initial_centroid, new_centroid)
        # With reward = -||w||^2, centroid norm should not increase.
        assert np.linalg.norm(new_centroid) < np.linalg.norm(initial_centroid)

    def test_stats_keys(self):
        stats = make_cma_stats()
        assert set(stats.fields) == {"avg", "std", "min", "max"}
