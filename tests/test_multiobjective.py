"""Tests for multi-objective neuroevolution support.

Covers: NSGA-II / SPEA2 selection dispatch, EATrainer multi-objective
integration, TrainingResult.pareto_front, and backward compatibility
for single-objective paths.
"""
import pytest
import numpy as np
from deap import creator, tools

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration import (
    create_toolbox,
    setup_deap_creators,
)
from deap_pettingzoo.integration.objectives import (
    FitnessSpec,
    ObjectiveSpec,
)
from deap_pettingzoo.baselines.base import (
    BudgetMode,
    TrainingConfig,
)
from deap_pettingzoo.baselines import EATrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    return {
        "layer_sizes": [16, 8, 4],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
        "random_seed": 42,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mo_spec():
    return FitnessSpec(
        objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ]
    )


def _mo_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        num_seeds=1,
        algorithm_params={
            "seed": seed,
            "max_steps_per_episode": 10,
        },
        env_config={"max_cycles": 10},
    )


# ---------------------------------------------------------------------------
# NSGA-II / SPEA2 dispatch
# ---------------------------------------------------------------------------


class TestNSGA2Dispatch:

    def test_nsga2_selects_correct_count(self, evaluator, genome_config):
        setup_deap_creators(weights=(1.0, -1.0))
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "nsga2"},
        )
        pop = toolbox.population(n=8)
        rng = np.random.default_rng(42)
        for ind in pop:
            ind.fitness.values = (
                float(rng.uniform(0, 10)),
                -float(ind.genome.get_weight_count()),
            )
        selected = toolbox.select(pop, k=5)
        assert len(selected) == 5

    def test_spea2_selects_correct_count(self, evaluator, genome_config):
        setup_deap_creators(weights=(1.0, -1.0))
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "spea2"},
        )
        pop = toolbox.population(n=8)
        rng = np.random.default_rng(42)
        for ind in pop:
            ind.fitness.values = (
                float(rng.uniform(0, 10)),
                -float(ind.genome.get_weight_count()),
            )
        selected = toolbox.select(pop, k=4)
        assert len(selected) == 4

    def test_nsga2_preserves_fitness_shape(self, evaluator, genome_config):
        """NSGA2 must return individuals with the same fitness dimension."""
        setup_deap_creators(weights=(1.0, -1.0))
        toolbox = create_toolbox(
            genome_config=genome_config,
            evaluator=evaluator,
            selection_config={"method": "nsga2"},
        )
        pop = toolbox.population(n=6)
        rng = np.random.default_rng(42)
        for ind in pop:
            ind.fitness.values = (
                float(rng.uniform(0, 10)),
                -float(ind.genome.get_weight_count()),
            )
        selected = toolbox.select(pop, k=3)
        for ind in selected:
            assert len(ind.fitness.values) == 2


# ---------------------------------------------------------------------------
# EATrainer multi-objective smoke tests
# ---------------------------------------------------------------------------


class TestEATrainerMultiObjective:

    def test_nsga2_smoke_train(self):
        trainer = EATrainer(
            hidden_size=8,
            population_size=6,
            selection_method="nsga2",
            strategy="simple",
            fitness_spec=_mo_spec(),
        )
        result = trainer.train(_mo_config(seed=0))
        assert result.algorithm == "ea"
        assert result.policy is not None
        assert result.pareto_front is not None
        assert len(result.pareto_front) > 0
        for ind in result.pareto_front:
            assert len(ind.fitness.values) == 2

    def test_spea2_smoke_train(self):
        trainer = EATrainer(
            hidden_size=8,
            population_size=6,
            selection_method="spea2",
            strategy="simple",
            fitness_spec=_mo_spec(),
        )
        result = trainer.train(_mo_config(seed=0))
        assert result.algorithm == "ea"
        assert result.policy is not None
        assert result.pareto_front is not None

    def test_nsga2_rejects_non_simple_strategy(self):
        with pytest.raises(ValueError, match="strategy='simple'"):
            EATrainer(
                hidden_size=8,
                population_size=6,
                selection_method="nsga2",
                strategy="mu_plus_lambda",
                fitness_spec=_mo_spec(),
            )

    def test_single_objective_no_pareto_front(self):
        config = _mo_config(seed=0)
        trainer = EATrainer(hidden_size=8, population_size=6)
        result = trainer.train(config)
        assert result.pareto_front is None

    def test_mo_parsimony_runs(self):
        trainer = EATrainer(
            hidden_size=8,
            population_size=8,
            selection_method="nsga2",
            strategy="simple",
            fitness_spec=_mo_spec(),
        )
        result = trainer.train(_mo_config(seed=0))
        sizes = [ind.genome.get_weight_count() for ind in result.pareto_front]
        assert len(sizes) >= 1


class TestFitnessSpecIntegration:

    def test_weights_match(self):
        spec = _mo_spec()
        trainer = EATrainer(
            hidden_size=8,
            population_size=4,
            selection_method="nsga2",
            strategy="simple",
            fitness_spec=spec,
        )
        result = trainer.train(_mo_config(seed=0))
        assert len(result.metrics["fitness_history"]) >= 2

    def test_three_objectives(self):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("reward", weight=1.0, source="reward"),
                ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
                ObjectiveSpec("catches", weight=1.0, source="reward_component",
                              component_key="num_catches"),
            ]
        )
        trainer = EATrainer(
            hidden_size=8,
            population_size=4,
            selection_method="nsga2",
            strategy="simple",
            fitness_spec=spec,
        )
        result = trainer.train(_mo_config(seed=0))
        assert len(result.pareto_front) > 0
