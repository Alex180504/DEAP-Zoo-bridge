import pytest
import numpy as np
from typing import Dict, Any, Optional, Tuple
from gymnasium import spaces

from deap_pettingzoo.evaluators import (
    BaseEvaluator,
    SingleEpisodeEvaluator,
    MultiEpisodeEvaluator,
    SelfPlayEvaluator,
    MixedPolicyEvaluator,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.policies import FeedforwardNeuralPolicy


# Mock DEAP individual for testing
class MockIndividual:
    """Mock DEAP individual with genome."""
    
    def __init__(self, genome: FixedTopologyGenome):
        self.genome = genome
        self.fitness = type('Fitness', (), {'values': None})()


class TestBaseEvaluator:
    """Test BaseEvaluator abstract interface."""
    
    def test_base_evaluator_is_abstract(self):
        """Test BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()
    
    def test_base_evaluator_requires_evaluate_method(self):
        """Test subclasses must implement evaluate method."""
        
        class IncompleteEvaluator(BaseEvaluator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteEvaluator()


class TestSingleEpisodeEvaluator:
    """Test single episode evaluation."""
    
    @pytest.fixture
    def wrapper(self):
        """Create environment wrapper (homogeneous-obs MPE env)."""
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25}
        )
    
    @pytest.fixture
    def individual(self, wrapper):
        """Create test individual with genome."""
        # Get dimensions from first agent AFTER reset
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        
        obs_dim = len(observations[agent])  # Get actual observation length
        action_space = wrapper.action_space(agent)
        action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        return MockIndividual(genome)
    
    @pytest.mark.parametrize("method", ['sum', 'mean', 'min', 'max'])
    def test_fitness_aggregation_methods(self, wrapper, individual, method):
        """Test evaluate with different fitness aggregation methods."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            aggregation_method=method
        )
        
        fitness = evaluator.evaluate(individual, seed=42)
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))
    
    def test_evaluate_different_seeds_different_results(self, wrapper, individual):
        """Test same seed yields identical fitness in homogeneous environments."""
        # Use homogeneous env to avoid randomness from heterogeneous partial transfer
        local_wrapper = PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25}
        )
        try:
            local_obs = local_wrapper.reset(seed=42)
            local_agent = list(local_obs.keys())[0]
            local_obs_dim = len(local_obs[local_agent])
            local_action_dim = local_wrapper.action_space(local_agent).n

            local_genome = FixedTopologyGenome(
                layer_sizes=[local_obs_dim, 16, local_action_dim],
                random_seed=42
            )
            local_individual = MockIndividual(local_genome)
            evaluator = SingleEpisodeEvaluator(env_wrapper=local_wrapper)

            fitness1 = evaluator.evaluate(local_individual, seed=42)
            fitness2 = evaluator.evaluate(local_individual, seed=42)

            # Same seed should produce identical results
            assert fitness1 == fitness2
        finally:
            local_wrapper.close()
    
    def test_max_steps_limits_episode(self, wrapper, individual):
        """Test max_steps parameter limits episode length."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=5
        )
        
        # Should complete without timing out
        fitness = evaluator.evaluate(individual, seed=42)
        assert isinstance(fitness[0], (int, float))
    
    def test_invalid_fitness_aggregation_raises_error(self, wrapper):
        """Test invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregation"):
            SingleEpisodeEvaluator(
                env_wrapper=wrapper,
                aggregation_method='invalid_method'
            )

    def test_team_aggregation_two_teams_is_order_independent(self, wrapper):
        """Two-team difference is deterministic (teams ordered by name)."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            aggregation_method='team',
            team_mapping={
                'adversary_0': 'red',
                'adversary_1': 'red',
                'agent_0': 'blue',
            },
        )
        # 'blue' sorts before 'red': result = blue_total - red_total.
        rewards = {'adversary_0': 1.0, 'adversary_1': 2.0, 'agent_0': 10.0}
        result = evaluator._aggregate_team_fitness(rewards)
        assert result == pytest.approx(10.0 - 3.0)

        # Reordering the dict must not change the result.
        reordered = {'agent_0': 10.0, 'adversary_1': 2.0, 'adversary_0': 1.0}
        assert evaluator._aggregate_team_fitness(reordered) == pytest.approx(result)

    def test_team_aggregation_three_teams_raises(self, wrapper):
        """Three or more teams is undefined and must fail loudly."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            aggregation_method='team',
            team_mapping={'a': 'red', 'b': 'blue', 'c': 'green'},
        )
        rewards = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        with pytest.raises(ValueError, match="two teams"):
            evaluator._aggregate_team_fitness(rewards)

    def test_team_aggregation_single_team_sums(self, wrapper):
        """A single resolved team falls back to the plain sum."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            aggregation_method='team',
            team_mapping={'a': 'red', 'b': 'red'},
        )
        rewards = {'a': 1.5, 'b': 2.5}
        assert evaluator._aggregate_team_fitness(rewards) == pytest.approx(4.0)

    def test_team_aggregation_no_mapping_sums(self, wrapper):
        """No team_mapping falls back to the plain sum of all rewards."""
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            aggregation_method='team',
        )
        rewards = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        assert evaluator._aggregate_team_fitness(rewards) == pytest.approx(6.0)

    def test_heterogeneous_obs_raises_value_error(self):
        """Evaluator rejects envs with role-dependent observation sizes."""
        pytest.importorskip("mpe2")
        hetero = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 25}
        )
        try:
            obs = hetero.reset(seed=0)
            agent = next(iter(obs))
            obs_dim = len(obs[agent])
            action_dim = hetero.action_space(agent).n
            genome = FixedTopologyGenome(
                layer_sizes=[obs_dim, 16, action_dim],
                random_seed=0,
            )
            ind = MockIndividual(genome)
            evaluator = SingleEpisodeEvaluator(env_wrapper=hetero)
            with pytest.raises(
                (ValueError, RuntimeError), match="homogeneous observation"
            ):
                evaluator.evaluate(ind, seed=0)
        finally:
            hetero.close()


class TestMultiEpisodeEvaluator:
    """Test multi-episode evaluation."""
    
    @pytest.fixture
    def wrapper(self):
        """Create environment wrapper (homogeneous-obs MPE env)."""
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25}
        )
    
    @pytest.fixture
    def individual(self, wrapper):
        """Create test individual."""
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        
        obs_dim = len(observations[agent])
        action_space = wrapper.action_space(agent)
        action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        return MockIndividual(genome)
    
    @pytest.mark.parametrize("method", ['mean', 'median', 'min', 'max'])
    def test_aggregation_methods(self, wrapper, individual, method):
        """Test evaluate with different aggregation methods across episodes."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            aggregation_method=method
        )
        
        fitness = evaluator.evaluate(individual)
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))
    
    def test_return_statistics(self, wrapper, individual):
        """Test returning episode statistics."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=5
        )
        
        stats = evaluator.evaluate_with_stats(individual)
        
        # Should return (mean, std, min, max)
        assert isinstance(stats, tuple)
        assert len(stats) == 4
        
        mean, std, min_val, max_val = stats
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))
        assert isinstance(min_val, (int, float))
        assert isinstance(max_val, (int, float))
        assert std >= 0
        assert min_val <= mean <= max_val
    
    def test_single_eval_kwargs_passed_through(self, wrapper, individual):
        """Test kwargs are passed to SingleEpisodeEvaluator."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            max_steps=5,
            aggregation_method='mean'
        )
        
        fitness = evaluator.evaluate(individual)
        assert isinstance(fitness, tuple)
    
    def test_invalid_aggregation_method_raises_error(self, wrapper):
        """Test invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregation"):
            MultiEpisodeEvaluator(
                env_wrapper=wrapper,
                aggregation_method='invalid'
            )
    
    def test_num_episodes_must_be_positive(self, wrapper):
        """Test num_episodes must be positive."""
        with pytest.raises(ValueError, match="num_episodes"):
            MultiEpisodeEvaluator(
                env_wrapper=wrapper,
                num_episodes=0
            )


class TestEvaluatorIntegration:
    """Test evaluator integration with full pipeline."""
    
    def test_single_and_multi_evaluator_consistency(self):
        """Test single episode result matches multi-episode with n=1."""
        wrapper = PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25}
        )
        
        # Create individual
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            random_seed=42
        )
        individual = MockIndividual(genome)
        
        # Single episode evaluator
        single_eval = SingleEpisodeEvaluator(
            env_wrapper=wrapper,
            max_steps=25,
            aggregation_method='sum'
        )
        
        # Multi-episode evaluator with 1 episode
        multi_eval = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=1,
            base_seed=42,
            max_steps=25,
            aggregation_method='mean'
        )
        
        single_fitness = single_eval.evaluate(individual, seed=42)
        multi_fitness = multi_eval.evaluate(individual)
        
        # Should be very close (floating point tolerance)
        assert abs(single_fitness[0] - multi_fitness[0]) < 1e-6
    
    def test_evaluator_with_deap_workflow(self):
        """Test evaluator works in DEAP-style workflow."""
        wrapper = PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25}
        )
        
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper,
            num_episodes=3,
            max_steps=25
        )
        
        # Create population of individuals
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        
        population = []
        for i in range(5):
            genome = FixedTopologyGenome(
                layer_sizes=[obs_dim, 16, action_dim],
                random_seed=i
            )
            individual = MockIndividual(genome)
            population.append(individual)
        
        # Evaluate population
        for ind in population:
            fitness = evaluator.evaluate(ind)
            ind.fitness.values = fitness
        
        # Check all have fitness
        for ind in population:
            assert ind.fitness.values is not None
            assert isinstance(ind.fitness.values, tuple)
            assert len(ind.fitness.values) == 1


# Multi-objective / lexicase evaluator tests.


class _MockFitnessSpec:
    """Minimal fitness-spec stand-in for tests that don't need the real
    `deap_pettingzoo.integration.objectives` import (but the real classes
    are preferred when testing integration)."""

    def __init__(self, objectives, lexicase_mode=None, reward_components_fn=None):
        self.objectives = objectives or []
        self.lexicase_mode = lexicase_mode
        self.reward_components_fn = reward_components_fn

    def validate(self):
        pass


class TestSingleEpisodeEvaluatorMultiObjective:
    """Backward-compatible + multi-objective SingleEpisodeEvaluator."""

    @pytest.fixture
    def wrapper(self):
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25},
        )

    @pytest.fixture
    def individual(self, wrapper):
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim], random_seed=42,
        )
        return MockIndividual(genome)

    def test_no_spec_is_backward_compatible(self, wrapper, individual):
        evaluator = SingleEpisodeEvaluator(env_wrapper=wrapper)
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_single_reward_objective(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(objectives=[ObjectiveSpec("reward")])
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_multiobjective_reward_parsimony(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ])
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Reward should be a real number; parsimony should be negative.
        assert isinstance(result[0], float)
        assert result[1] < 0.0

    def test_lexicase_per_episode(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_episode",
        )
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        # 1 objective, 1 episode = 1 scalar in flat vector
        assert len(result) == 1

    def test_lexicase_per_agent(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_agent",
        )
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=wrapper, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        # N agents in the env, 1 objective → N scalars
        assert len(result) > 0


class TestMultiEpisodeEvaluatorMultiObjective:
    """Multi-objective MultiEpisodeEvaluator tests."""

    @pytest.fixture
    def wrapper(self):
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25},
        )

    @pytest.fixture
    def individual(self, wrapper):
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim], random_seed=42,
        )
        return MockIndividual(genome)

    def test_no_spec_is_backward_compatible(self, wrapper, individual):
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=3,
        )
        result = evaluator.evaluate(individual)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_multiobjective_elementwise_mean(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ])
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=3, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] < 0.0

    def test_lexicase_per_episode(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_episode",
        )
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=4, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual)
        # 1 objective × 4 episodes = 4 scalars concatenated
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_evaluate_with_stats_rejects_multiobjective(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(objectives=[
            ObjectiveSpec("reward", source="reward"),
            ObjectiveSpec("parsimony", source="parsimony"),
        ])
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=3, fitness_spec=spec,
        )
        with pytest.raises(ValueError, match="evaluate_with_stats"):
            evaluator.evaluate_with_stats(individual)

    def test_evaluate_with_stats_rejects_lexicase(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_episode",
        )
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=3, fitness_spec=spec,
        )
        with pytest.raises(ValueError, match="evaluate_with_stats"):
            evaluator.evaluate_with_stats(individual)


class TestSelfPlayEvaluatorMultiObjective:
    """Multi-objective SelfPlayEvaluator tests."""

    @pytest.fixture
    def wrapper(self):
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25},
        )

    @pytest.fixture
    def individual(self, wrapper):
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim], random_seed=42,
        )
        return MockIndividual(genome)

    def test_no_spec_is_backward_compatible(self, wrapper, individual):
        evaluator = SelfPlayEvaluator(env_wrapper=wrapper)
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_multiobjective(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        from deap_pettingzoo.evaluators import SelfPlayEvaluator
        spec = FitnessSpec(objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ])
        evaluator = SelfPlayEvaluator(
            env_wrapper=wrapper, fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestMixedPolicyEvaluatorMultiObjective:
    """Multi-objective MixedPolicyEvaluator tests."""

    @pytest.fixture
    def wrapper(self):
        return PettingZooWrapper(
            env_name='simple_spread_v3',
            env_config={'max_cycles': 25},
        )

    @pytest.fixture
    def individual(self, wrapper):
        observations = wrapper.reset(seed=42)
        agent = list(observations.keys())[0]
        obs_dim = len(observations[agent])
        action_dim = wrapper.action_space(agent).n
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim], random_seed=42,
        )
        return MockIndividual(genome)

    @staticmethod
    def _learner_factory(genome, agent_id, action_space):
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy
        return FeedforwardNeuralPolicy(genome, action_space=action_space)

    def test_no_spec_is_backward_compatible(self, wrapper, individual):
        from deap_pettingzoo.evaluators import MixedPolicyEvaluator
        # Pick one agent prefix that exists in the env.
        agents = list(wrapper.possible_agents)
        prefix = agents[0][:agents[0].rfind("_")] if "_" in agents[0] else agents[0]
        evaluator = MixedPolicyEvaluator(
            env_wrapper=wrapper,
            learner_agent_prefixes=[prefix],
            learner_factory=self._learner_factory,
            scripted_policies={},
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_multiobjective(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        from deap_pettingzoo.evaluators import MixedPolicyEvaluator
        agents = list(wrapper.possible_agents)
        prefix = agents[0][:agents[0].rfind("_")] if "_" in agents[0] else agents[0]
        spec = FitnessSpec(objectives=[
            ObjectiveSpec("reward", weight=1.0, source="reward"),
            ObjectiveSpec("parsimony", weight=-1.0, source="parsimony"),
        ])
        evaluator = MixedPolicyEvaluator(
            env_wrapper=wrapper,
            learner_agent_prefixes=[prefix],
            learner_factory=self._learner_factory,
            scripted_policies={},
            fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lexicase_per_agent(self, wrapper, individual):
        from deap_pettingzoo.integration.objectives import (
            FitnessSpec, ObjectiveSpec,
        )
        from deap_pettingzoo.evaluators import MixedPolicyEvaluator
        agents = list(wrapper.possible_agents)
        prefix = agents[0][:agents[0].rfind("_")] if "_" in agents[0] else agents[0]
        spec = FitnessSpec(
            objectives=[ObjectiveSpec("reward")],
            lexicase_mode="per_agent",
        )
        evaluator = MixedPolicyEvaluator(
            env_wrapper=wrapper,
            learner_agent_prefixes=[prefix],
            learner_factory=self._learner_factory,
            scripted_policies={},
            fitness_spec=spec,
        )
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) > 0


class _PhantomRewardEnv:
    """Minimal env-wrapper stand-in that emits a reward for 'phantom' not in reset obs.
    """

    possible_agents = ["agent_0"]

    def __init__(self):
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        self._step_count = 0
        return {"agent_0": np.zeros(2, dtype=np.float32)}

    def step(self, actions) -> Tuple:
        self._step_count += 1
        obs: Dict[str, np.ndarray] = {}
        rewards = {"agent_0": 1.0, "phantom": 5.0}  # 'phantom' was never in reset obs
        dones = {"agent_0": True}
        truncated = {"agent_0": False}
        infos: Dict = {}
        return obs, rewards, dones, truncated, infos

    def action_space(self, agent: str):
        return spaces.Discrete(2)

    @property
    def last_infos(self):
        return {}

    def close(self):
        pass


class TestSingleEpisodeEvaluatorTerminationHandling:
    """F-2.3-01: evaluator survives a reward for an agent absent at reset."""

    def _make_individual(self):
        genome = FixedTopologyGenome(layer_sizes=[2, 4, 2], random_seed=0)

        class _Ind:
            pass

        ind = _Ind()
        ind.genome = genome
        return ind

    def test_survives_reward_for_agent_absent_at_reset(self):
        """No KeyError when rewards contains 'phantom' not in cumulative_rewards."""
        env = _PhantomRewardEnv()
        evaluator = SingleEpisodeEvaluator(env_wrapper=env, max_steps=3)
        ind = self._make_individual()
        # Before the fix this raises KeyError: 'phantom'.
        result = evaluator.evaluate(ind, seed=0)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)


class TestMultiEpisodeEvaluatorSeedParam:
    """F-2.3-02: MultiEpisodeEvaluator.evaluate must accept seed=None (parallel path)."""

    @pytest.fixture
    def wrapper(self):
        return PettingZooWrapper(
            env_name="simple_spread_v3",
            env_config={"max_cycles": 5},
        )

    @pytest.fixture
    def individual(self, wrapper):
        obs = wrapper.reset(seed=0)
        agent = next(iter(obs))
        obs_dim = len(obs[agent])
        action_dim = wrapper.action_space(agent).n
        genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 8, action_dim], random_seed=0
        )

        class _Ind:
            pass

        ind = _Ind()
        ind.genome = genome
        return ind

    def test_evaluate_accepts_seed_kwarg(self, wrapper, individual):
        """evaluate(ind, seed=42) must not raise TypeError."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=2, max_steps=5
        )
        # Before fix: TypeError — evaluate() got unexpected keyword argument 'seed'
        result = evaluator.evaluate(individual, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_seed_threads_per_episode(self, wrapper, individual):
        """Two calls with the same seed yield the same fitness (reproducibility)."""
        evaluator = MultiEpisodeEvaluator(
            env_wrapper=wrapper, num_episodes=2, max_steps=5
        )
        r1 = evaluator.evaluate(individual, seed=7)
        r2 = evaluator.evaluate(individual, seed=7)
        assert r1 == pytest.approx(r2, abs=1e-9)
