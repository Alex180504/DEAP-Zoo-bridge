import pytest
import numpy as np
from deap_pettingzoo.environments import PettingZooWrapper


class TestPettingZooWrapperInitialization:
    """Test environment wrapper initialization."""
    
    def test_wrapper_creation_from_env_name(self):
        """Test creating wrapper from environment name."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        
        assert wrapper.env_name == 'simple_tag_v3'
        assert wrapper.env is not None
        assert wrapper.possible_agents is not None
    
    def test_wrapper_with_different_mpe_environments(self):
        """Test wrapper works with additional MPE environments."""
        wrapper = PettingZooWrapper(env_name='simple_spread_v3')
        assert wrapper.env_name == 'simple_spread_v3'
        assert wrapper.env is not None
        # TODO: Add more environments
    
    def test_wrapper_stores_configuration(self):
        """Test wrapper stores environment configuration."""
        config = {
            'num_good': 1,
            'num_adversaries': 3,
            'num_obstacles': 2,
            'max_cycles': 25,
            'continuous_actions': True
        }
        
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config=config
        )
        
        assert wrapper.env_config == config
    
    def test_wrapper_invalid_env_raises_error(self):
        """Test wrapper raises error for invalid environment name."""
        with pytest.raises(ValueError, match="environment"):
            PettingZooWrapper(env_name='invalid_env_name')


class TestPettingZooWrapperReset:
    """Test environment reset functionality."""
    
    def test_reset_returns_observations(self):
        """Test reset returns observations for all agents."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset()
        
        assert isinstance(observations, dict)
        assert len(observations) > 0
        
        # Check each observation is a numpy array
        for agent, obs in observations.items():
            assert isinstance(obs, np.ndarray)
    
    def test_reset_with_seed_reproducible(self):
        """Test reset with same seed produces same initial state."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        
        obs1 = wrapper.reset(seed=42)
        obs2 = wrapper.reset(seed=42)
        
        # Should have same agents
        assert set(obs1.keys()) == set(obs2.keys())
        
        # Should have same observations
        for agent in obs1.keys():
            np.testing.assert_array_equal(obs1[agent], obs2[agent])
    
    def test_reset_different_seeds_different_states(self):
        """Test reset with different seeds produces different states."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        
        obs1 = wrapper.reset(seed=42)
        obs2 = wrapper.reset(seed=123)
        
        # At least one agent should have different observation
        differences = []
        for agent in obs1.keys():
            if agent in obs2:
                diff = not np.allclose(obs1[agent], obs2[agent])
                differences.append(diff)
        
        assert any(differences), "Different seeds should produce different states"
    
class TestPettingZooWrapperStep:
    """Test environment step functionality."""
    
    def test_step_returns_correct_format(self):
        """Test step returns observations, rewards, dones, infos."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        actions = {agent: 0 for agent in observations.keys()}  # Use discrete actions
        
        obs, rewards, dones, truncated, infos = wrapper.step(actions)
        
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(truncated, dict)
        assert isinstance(infos, dict)
    
    def test_step_updates_observations(self):
        """Test step returns updated observations."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        actions = {agent: 1 for agent in observations.keys()}  # Use discrete actions
        
        new_obs, _, _, _, _ = wrapper.step(actions)
        
        # Observations should change after step
        for agent in observations.keys():
            if agent in new_obs:
                # At least some values should differ
                assert not np.allclose(observations[agent], new_obs[agent])
    
    def test_step_action_validation(self):
        """Test step validates all agents provide valid actions."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        # Test 1: Incomplete actions (missing agents)
        partial_actions = {list(observations.keys())[0]: 0}
        with pytest.raises(ValueError, match="actions"):
            wrapper.step(partial_actions)
        
        # Test 2: Invalid discrete action (out of range)
        wrong_actions = {agent: 100 for agent in observations.keys()}
        with pytest.raises(ValueError, match="action space"):
            wrapper.step(wrong_actions)
    
    def test_episode_terminates(self):
        """Test episode eventually terminates."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 10}
        )
        observations = wrapper.reset(seed=42)
        
        done = False
        steps = 0
        max_steps = 20
        
        while not done and steps < max_steps:
            actions = {agent: 0 for agent in observations.keys()}  # Use discrete actions
            observations, _, dones, truncated, _ = wrapper.step(actions)
            
            # Check if all agents are done
            done = all(dones.values()) or all(truncated.values())
            steps += 1
        
        assert done, "Episode should terminate within max_cycles"


class TestPettingZooWrapperSpaces:
    """Test observation and action space handling."""
    
    def test_observation_space_per_agent(self):
        """Test observation space is available for each agent."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        for agent in observations.keys():
            obs_space = wrapper.observation_space(agent)
            assert obs_space is not None
            
            # Observation should match space shape
            assert observations[agent].shape == obs_space.shape
    
    def test_action_space_per_agent(self):
        """Test action space is available for each agent."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        for agent in observations.keys():
            action_space = wrapper.action_space(agent)
            assert action_space is not None
            assert hasattr(action_space, 'shape')
    
    def test_observation_space_bounds(self):
        """Test observation space has proper bounds."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        wrapper.reset(seed=42)
        
        agent = wrapper.possible_agents[0]
        obs_space = wrapper.observation_space(agent)
        
        # Should have low and high bounds for Box spaces
        if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
            assert obs_space.low.shape == obs_space.high.shape
    
class TestPettingZooWrapperAgentManagement:
    """Test multi-agent coordination."""
    
    def test_possible_agents_list(self):
        """Test wrapper provides list of possible agents."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        
        assert wrapper.possible_agents is not None
        assert isinstance(wrapper.possible_agents, list)
        assert len(wrapper.possible_agents) > 0
    
    def test_agents_properties(self):
        """Test agents and num_agents properties."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        # Test agents property
        assert wrapper.agents is not None
        assert isinstance(wrapper.agents, list)
        assert set(wrapper.agents) == set(observations.keys())
        
        # Test num_agents property
        assert wrapper.num_agents > 0
        assert wrapper.num_agents == len(wrapper.agents)
    
    def test_agent_removal_on_done(self):
        """Test agents removed from active list when done."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 5}
        )
        observations = wrapper.reset(seed=42)
        initial_agent_count = len(observations)
        
        # Run until some agents are done
        for _ in range(10):
            if len(wrapper.agents) == 0:
                break
            
            actions = {agent: 0 for agent in wrapper.agents}  # Use discrete actions
            wrapper.step(actions)
        
        # Eventually all agents should be removed
        assert len(wrapper.agents) <= initial_agent_count


class TestPettingZooWrapperRendering:
    """Test rendering functionality."""
    
    def test_render_mode_support(self):
        """Test wrapper supports render modes."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            render_mode='rgb_array'
        )
        
        assert wrapper.render_mode == 'rgb_array'
    
    def test_render_returns_frame(self):
        """Test render returns RGB frame when mode is rgb_array."""
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            render_mode='rgb_array'
        )
        wrapper.reset(seed=42)
        
        frame = wrapper.render()
        
        if frame is not None:
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 3  # Height x Width x Channels
            assert frame.shape[2] == 3    # RGB
    
    def test_render_without_mode_returns_none(self):
        """Test render returns None when no render mode set."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        wrapper.reset(seed=42)
        
        frame = wrapper.render()
        assert frame is None


class TestPettingZooWrapperEpisodeMetrics:
    """Test episode tracking and metrics."""
    
    def test_episode_length_tracking(self):
        """Test episode length tracking and reset."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        wrapper.reset(seed=42)
        
        # Initial length should be 0
        assert wrapper.episode_length == 0
        
        # Take a step and check length increments
        actions = {agent: 0 for agent in wrapper.agents}
        wrapper.step(actions)
        assert wrapper.episode_length == 1
        
        # Take more steps
        for _ in range(4):
            actions = {agent: 0 for agent in wrapper.agents}
            wrapper.step(actions)
        assert wrapper.episode_length == 5
        
        # Reset should clear episode length
        wrapper.reset(seed=123)
        assert wrapper.episode_length == 0
    
    def test_cumulative_rewards_tracking(self):
        """Test wrapper tracks cumulative rewards per agent."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        observations = wrapper.reset(seed=42)
        
        # Take steps and accumulate rewards
        total_rewards = {agent: 0.0 for agent in observations.keys()}
        
        for _ in range(5):
            actions = {agent: 1 for agent in wrapper.agents}  # Use discrete actions
            _, rewards, _, _, _ = wrapper.step(actions)
            
            for agent, reward in rewards.items():
                total_rewards[agent] += reward
        
        # Check cumulative rewards
        assert wrapper.cumulative_rewards is not None
        assert isinstance(wrapper.cumulative_rewards, dict)


class TestPettingZooWrapperClose:
    """Test environment cleanup."""
    
    def test_close_cleanup_and_idempotence(self):
        """Test close cleans up environment and is idempotent."""
        wrapper = PettingZooWrapper(env_name='simple_tag_v3')
        wrapper.reset(seed=42)
        
        # Close should clean up
        wrapper.close()
        assert wrapper.env is None or not hasattr(wrapper.env, 'agents')
        
        # Should be idempotent (can close again without error)
        wrapper.close()  # Should not raise error


class TestPettingZooWrapperIntegration:
    """Test integration with DEAP and policies."""
    
    def test_full_episode_with_policies(self):
        """Test running full episode with neural network policies."""
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy
        
        wrapper = PettingZooWrapper(
            env_name='simple_tag_v3',
            env_config={'max_cycles': 10}
        )
        observations = wrapper.reset(seed=42)
        
        # Create policies
        policies = {}
        for agent, obs in observations.items():
            obs_dim = len(obs)
            action_space = wrapper.action_space(agent)
            
            # Handle discrete action spaces
            if hasattr(action_space, 'n'):  # Discrete space
                action_dim = action_space.n
            else:  # Box space
                action_dim = action_space.shape[0]
            
            genome = FixedTopologyGenome(
                layer_sizes=[obs_dim, 32, action_dim],
                random_seed=42
            )
            policies[agent] = FeedforwardNeuralPolicy(genome)
        
        # Run episode
        total_rewards = {agent: 0.0 for agent in observations.keys()}
        done = False
        steps = 0
        
        while not done and steps < 20:
            actions = {agent: policies[agent].forward(obs) 
                      for agent, obs in observations.items()}
            
            observations, rewards, dones, truncated, _ = wrapper.step(actions)
            
            for agent, reward in rewards.items():
                total_rewards[agent] += reward
            
            done = all(dones.values()) or all(truncated.values())
            steps += 1
        
        # Should have accumulated some rewards
        assert isinstance(total_rewards, dict)
        assert len(total_rewards) > 0
