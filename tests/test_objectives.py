"""Tests for objectives.py — FitnessSpec, ObjectiveSpec, build_fitness_vector,
compute_parsimony, and the backward-compatible setup_deap_creators signature."""
import pytest
import numpy as np
from deap import creator, base

from deap_pettingzoo.integration.objectives import (
    FitnessSpec,
    ObjectiveSpec,
    build_fitness_vector,
    compute_parsimony,
)
from deap_pettingzoo.integration.deap_integration import setup_deap_creators
from deap_pettingzoo.genomes import FixedTopologyGenome


# ---------------------------------------------------------------------------
# ObjectiveSpec
# ---------------------------------------------------------------------------

class TestObjectiveSpec:
    def test_defaults(self):
        obj = ObjectiveSpec("reward")
        assert obj.name == "reward"
        assert obj.weight == 1.0
        assert obj.source == "reward"
        assert obj.component_key is None

    def test_reward_component_requires_key(self):
        with pytest.raises(ValueError, match="requires a non-empty component_key"):
            ObjectiveSpec("catches", source="reward_component")

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source must be one of"):
            ObjectiveSpec("bad", source="nonsense")


# ---------------------------------------------------------------------------
# FitnessSpec
# ---------------------------------------------------------------------------

class TestFitnessSpec:
    def test_empty_objectives_defaults_to_single_reward(self):
        spec = FitnessSpec()
        assert spec.objectives == []
        assert not spec.is_multiobjective
        assert not spec.is_lexicase
        assert spec.weights == (1.0,)

    def test_single_reward_objective(self):
        spec = FitnessSpec(objectives=[ObjectiveSpec("reward")])
        assert spec.num_objectives == 1
        assert not spec.is_multiobjective
        assert spec.weights == (1.0,)

    def test_multiobjective_reward_parsimony(self):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("reward", weight=1.0, source="reward"),
                ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
            ]
        )
        assert spec.is_multiobjective
        assert spec.num_objectives == 2
        assert spec.weights == (1.0, -1.0)

    def test_lexicase_per_episode_only_reward(self):
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_episode",
        )
        spec.validate()  # OK: single reward

    def test_lexicase_per_episode_rejects_parsimony(self):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("reward", source="reward"),
                ObjectiveSpec("parsimony", source="parsimony"),
            ],
            lexicase_mode="per_episode",
        )
        with pytest.raises(ValueError, match="supports only source='reward'"):
            spec.validate()

    def test_lexicase_per_agent_rejects_parsimony(self):
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("parsimony", source="parsimony")],
            lexicase_mode="per_agent",
        )
        with pytest.raises(ValueError, match="supports only source='reward'"):
            spec.validate()

    def test_invalid_lexicase_mode_raises(self):
        with pytest.raises(ValueError, match="lexicase_mode must be one of"):
            FitnessSpec(lexicase_mode="invalid")


# ---------------------------------------------------------------------------
# compute_parsimony
# ---------------------------------------------------------------------------

class TestComputeParsimony:
    def test_with_genome(self):
        genome = FixedTopologyGenome(layer_sizes=[4, 8, 2], random_seed=0)
        wc = genome.get_weight_count()
        assert compute_parsimony(genome) == -float(wc)

    def test_with_weights_attribute(self):
        class SimpleWeights:
            weights = [0.1, 0.2, 0.3]

        assert compute_parsimony(SimpleWeights()) == -3.0

    def test_fallback_zero(self):
        assert compute_parsimony(object()) == 0.0


# ---------------------------------------------------------------------------
# build_fitness_vector (standard multi-objective)
# ---------------------------------------------------------------------------

class TestBuildFitnessVector:

    @pytest.fixture
    def genome(self):
        return FixedTopologyGenome(layer_sizes=[4, 8, 2], random_seed=0)

    def test_no_objectives_returns_aggregated(self, genome):
        spec = FitnessSpec()
        result = build_fitness_vector(10.0, None, None, None, genome, spec)
        assert result == (10.0,)

    def test_reward_only(self, genome):
        spec = FitnessSpec(objectives=[ObjectiveSpec("reward")])
        result = build_fitness_vector(15.0, None, None, None, genome, spec)
        assert result == (15.0,)

    def test_multiobjective_reward_parsimony(self, genome):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("reward", weight=1.0, source="reward"),
                ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
            ]
        )
        wc = genome.get_weight_count()
        result = build_fitness_vector(20.0, None, None, None, genome, spec)
        assert result == (20.0, -float(wc))

    def test_reward_component(self, genome):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("catches", weight=1.0, source="reward_component",
                              component_key="num_catches"),
            ]
        )
        rc = {"num_catches": 3.0}
        result = build_fitness_vector(0.0, None, None, rc, genome, spec)
        assert result == (3.0,)

    def test_reward_component_missing_key(self, genome):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("catches", weight=1.0, source="reward_component",
                              component_key="num_catches"),
            ]
        )
        result = build_fitness_vector(0.0, None, None, {}, genome, spec)
        assert result == (0.0,)


# ---------------------------------------------------------------------------
# build_fitness_vector (lexicase per_episode)
# ---------------------------------------------------------------------------

class TestBuildFitnessVectorLexicaseEpisode:
    @pytest.fixture
    def genome(self):
        return FixedTopologyGenome(layer_sizes=[4, 8, 2], random_seed=0)

    def test_single_objective(self, genome):
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_episode",
        )
        result = build_fitness_vector(
            10.0,  # aggregated (ignored in lexicase)
            [5.0, 7.0, 3.0],  # per_episode_rewards
            None, None, genome, spec,
        )
        # One objective × 3 episodes = flat vector of 3
        assert result == (5.0, 7.0, 3.0)

    def test_multi_objective_lexicase(self, genome):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("r1", source="reward"),
                ObjectiveSpec("r2", source="reward"),
            ],
            lexicase_mode="per_episode",
        )
        result = build_fitness_vector(
            0.0, [1.0, 2.0], None, None, genome, spec,
        )
        # 2 objectives × 2 episodes = [r1_0, r1_1, r2_0, r2_1]
        assert result == (1.0, 2.0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# build_fitness_vector (lexicase per_agent)
# ---------------------------------------------------------------------------

class TestBuildFitnessVectorLexicaseAgent:
    @pytest.fixture
    def genome(self):
        return FixedTopologyGenome(layer_sizes=[4, 8, 2], random_seed=0)

    def test_single_objective(self, genome):
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_agent",
        )
        result = build_fitness_vector(
            0.0, None,
            {"agent_0": 1.0, "agent_1": 3.0, "agent_2": 2.0},
            None, genome, spec,
        )
        # 1 objective × 3 agents (sorted) = [agent_0, agent_1, agent_2]
        assert result == (1.0, 3.0, 2.0)

    def test_multi_objective(self, genome):
        spec = FitnessSpec(
            objectives=[
                ObjectiveSpec("reward", source="reward"),
                ObjectiveSpec("parsimony", source="parsimony"),
            ],
            lexicase_mode="per_agent",
        )
        result = build_fitness_vector(
            0.0, None,
            {"a": 5.0, "b": 8.0},
            None, genome, spec,
        )
        wc = compute_parsimony(genome)
        # 2 obj × 2 agents sorted: [rew_a, rew_b, par_a, par_b]
        assert result == (5.0, 8.0, wc, wc)


# ---------------------------------------------------------------------------
# setup_deap_creators backward compatibility
# ---------------------------------------------------------------------------

class TestSetupDeapCreators:

    def teardown_method(self):
        # Clean up creators so other tests aren't side-effected.
        for name in ("FitnessMax", "Individual"):
            if hasattr(creator, name):
                delattr(creator, name)

    def test_default_single_objective(self):
        setup_deap_creators()
        assert hasattr(creator, "FitnessMax")
        assert hasattr(creator, "Individual")
        assert creator.FitnessMax.weights == (1.0,)

    def test_multi_objective_weights(self):
        setup_deap_creators(weights=(1.0, -1.0))
        assert creator.FitnessMax.weights == (1.0, -1.0)
        ind = creator.Individual()
        ind.genome = FixedTopologyGenome(layer_sizes=[2, 4, 1], random_seed=0)
        ind.fitness.values = (5.0, -10.0)
        assert ind.fitness.values == (5.0, -10.0)

    def test_empty_weights_defaults(self):
        setup_deap_creators(weights=())
        assert creator.FitnessMax.weights == (1.0,)

    def test_list_weights_converted(self):
        setup_deap_creators(weights=[1.0, -1.0, 1.0])
        assert creator.FitnessMax.weights == (1.0, -1.0, 1.0)

    def test_idempotent(self):
        setup_deap_creators(weights=(1.0, -1.0))
        w1 = creator.FitnessMax.weights
        setup_deap_creators(weights=(1.0,))
        assert creator.FitnessMax.weights == (1.0,)
        assert w1 == (1.0, -1.0)  # original overwritten
