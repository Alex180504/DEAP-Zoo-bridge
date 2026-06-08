"""Tests for PSO integration (Phase 1.4)."""
import pytest
import numpy as np
from deap import creator

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import setup_deap_creators
from deap_pettingzoo.integration.pso_integration import (
    _setup_pso_creators,
    setup_pso_swarm,
    make_pso_ind_init,
    register_pso_toolbox,
    make_pso_stats,
    _pso_update_top_level,
)


@pytest.fixture(scope="module")
def evaluator():
    setup_deap_creators()
    wrapper = PettingZooWrapper(
        env_name="simple_spread_v3",
        env_config={"max_cycles": 10},
    )
    ev = SingleEpisodeEvaluator(env_wrapper=wrapper, max_steps=10,
                                aggregation_method="sum")
    yield ev
    wrapper.close()


@pytest.fixture
def genome_config():
    return {"layer_sizes": [16, 8, 4], "activation": "tanh",
            "weight_range": (-1.0, 1.0)}


class TestPSOSwarmSetup:
    def test_returns_state_and_dimension(self, genome_config):
        state, dim = setup_pso_swarm(genome_config, population_size=20)
        assert dim > 0
        assert state["dimension"] == dim
        assert state["population_size"] == 20

    def test_missing_layer_sizes_raises(self):
        with pytest.raises(ValueError, match="layer_sizes"):
            setup_pso_swarm({}, population_size=10)


class TestPSOCreators:
    def test_particle_creation(self, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=10)
        init_fn = make_pso_ind_init(genome_config, state)
        p = init_fn()
        assert hasattr(p, "genome")
        assert isinstance(p.genome, FixedTopologyGenome)
        assert hasattr(p, "speed")
        assert p.speed is not None
        assert len(p.speed) == state["dimension"]
        assert p.best is None
        assert p.best_fitness is None


class TestPSOToolbox:
    def test_register_toolbox(self, evaluator, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=8)
        init_fn = make_pso_ind_init(genome_config, state)
        toolbox = register_pso_toolbox(state, init_fn, evaluator)
        assert hasattr(toolbox, "generate")
        assert hasattr(toolbox, "update")
        assert hasattr(toolbox, "evaluate")
        pop = toolbox.generate()
        assert len(pop) == 8
        for ind in pop:
            assert hasattr(ind, "genome")

    def test_update_with_evaluated_population(self, evaluator, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=6)
        init_fn = make_pso_ind_init(genome_config, state)
        toolbox = register_pso_toolbox(state, init_fn, evaluator)
        pop = toolbox.generate()
        for i, ind in enumerate(pop):
            ind.fitness.values = (float(i + 1),)
        toolbox.update(pop)
        # After update, gbest should be set.
        assert toolbox._pso_gbest_ctx["fitness"] is not None
        assert toolbox._pso_gbest_ctx["pos"] is not None

    def test_stats(self, evaluator, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=6)
        init_fn = make_pso_ind_init(genome_config, state)
        toolbox = register_pso_toolbox(state, init_fn, evaluator)
        pop = toolbox.generate()
        for ind in pop:
            ind.fitness.values = (1.0,)
        s = make_pso_stats()
        rec = s.compile(pop)
        assert "avg" in rec
        assert "max" in rec


class TestPSOUpdate:
    def test_update_tracks_gbest(self, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=6)
        init_fn = make_pso_ind_init(genome_config, state)
        gbest_ctx = {"pos": None, "fitness": None}
        pop = [init_fn() for _ in range(6)]
        for i, ind in enumerate(pop):
            ind.fitness.values = (float(i),)
        _pso_update_top_level(pop, state, gbest_ctx)
        assert gbest_ctx["fitness"] == 5.0
        assert gbest_ctx["pos"] is not None

    def test_update_moves_particles(self, genome_config):
        _setup_pso_creators()
        state, _ = setup_pso_swarm(genome_config, population_size=6)
        init_fn = make_pso_ind_init(genome_config, state)
        gbest_ctx = {"pos": None, "fitness": None}
        pop = [init_fn() for _ in range(6)]
        before = [np.asarray(ind).copy() for ind in pop]
        for i, ind in enumerate(pop):
            ind.fitness.values = (float(i),)
        _pso_update_top_level(pop, state, gbest_ctx)
        for i, ind in enumerate(pop):
            after = np.asarray(ind)
            assert not np.array_equal(after, before[i])
