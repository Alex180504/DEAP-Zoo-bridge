from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_spread_v3


class PettingZooWrapper:
    """
    Wrapper for PettingZoo environments compatible with DEAP.
    
    Provides a unified interface for PettingZoo multi-agent environments,
    handling agent coordination, observation/action space management,
    and episode tracking.
    
    Attributes:
        env_name: Name of the PettingZoo environment
        env: The wrapped PettingZoo environment
        env_config: Configuration dict for environment
        render_mode: Rendering mode ('human', 'rgb_array', None)
        possible_agents: List of all possible agent names
        agents: List of currently active agents
        episode_length: Current episode step count
        cumulative_rewards: Dict of cumulative rewards per agent
    """
    
    # Supported MPE environments
    SUPPORTED_ENVS = {
        'simple_tag_v3': simple_tag_v3,
        'simple_adversary_v3': simple_adversary_v3,
        'simple_spread_v3': simple_spread_v3,
    }
    
    def __init__(
        self,
        env_name: str,
        env_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize environment wrapper.
        
        Args:
            env_name: Name of PettingZoo environment ('simple_tag_v3', etc.)
            env_config: Optional configuration dict for environment
            render_mode: Rendering mode ('human', 'rgb_array', None)
            
        Raises:
            ValueError: If environment name is not supported
            
        Example:
            >>> wrapper = PettingZooWrapper(
            ...     env_name='simple_tag_v3',
            ...     env_config={'max_cycles': 25}
            ... )
        """
        if env_name not in self.SUPPORTED_ENVS:
            raise ValueError(
                f"Unsupported environment '{env_name}'. "
                f"Supported: {list(self.SUPPORTED_ENVS.keys())}"
            )
        
        self.env_name = env_name
        self.env_config = env_config or {}
        self.render_mode = render_mode
        
        # Create environment
        env_class = self.SUPPORTED_ENVS[env_name]
        self.env = env_class.parallel_env(
            render_mode=render_mode,
            **self.env_config
        )
        
        # Store agent information
        self.possible_agents = self.env.possible_agents
        
        # Episode tracking
        self.episode_length = 0
        self.cumulative_rewards = {}
    
    @property
    def agents(self) -> List[str]:
        """Get list of currently active agents."""
        return self.env.agents if hasattr(self.env, 'agents') else []
    
    @property
    def num_agents(self) -> int:
        """Get number of currently active agents."""
        return len(self.agents)
    
    def observation_space(self, agent: str):
        """
        Get observation space for agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Observation space for the agent
        """
        return self.env.observation_space(agent)
    
    def action_space(self, agent: str):
        """
        Get action space for agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Action space for the agent
        """
        return self.env.action_space(agent)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional reset options
            
        Returns:
            Dict mapping agent names to initial observations
            
        Example:
            >>> observations = wrapper.reset(seed=42)
            >>> print(observations.keys())
            dict_keys(['adversary_0', 'agent_0', 'agent_1'])
        """
        # Reset environment
        observations, infos = self.env.reset(seed=seed, options=options)
        
        # Reset episode tracking
        self.episode_length = 0
        self.cumulative_rewards = {agent: 0.0 for agent in observations.keys()}
        
        return observations
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            actions: Dict mapping agent names to action arrays
            
        Returns:
            Tuple of (observations, rewards, dones, truncated, infos)
            
        Raises:
            ValueError: If actions missing for active agents or wrong shape
            
        Example:
            >>> actions = {agent: policy.forward(obs) for agent, obs in observations.items()}
            >>> obs, rewards, dones, truncated, infos = wrapper.step(actions)
        """
        # Validate all active agents have actions
        active_agents = set(self.agents)
        action_agents = set(actions.keys())
        
        if not action_agents.issuperset(active_agents):
            missing = active_agents - action_agents
            raise ValueError(
                f"Missing actions for agents: {missing}. "
                f"Expected actions for: {active_agents}"
            )
        
        # Convert and validate actions
        converted_actions = {}
        for agent, action in actions.items():
            if agent in active_agents:
                action_space = self.action_space(agent)
                
                # Convert numpy array to appropriate format
                if hasattr(action, 'shape'):
                    # If action is array, convert based on space type
                    if hasattr(action_space, 'n'):  # Discrete space
                        # Take argmax for discrete actions
                        action_value = int(np.argmax(action)) if action.size > 1 else int(action.item())
                    else:  # Box space
                        action_value = action
                else:
                    # Already a scalar
                    action_value = action
                
                # Validate action
                if not action_space.contains(action_value):
                    raise ValueError(
                        f"Action for agent '{agent}' not in action space: "
                        f"got {action_value}, space is {action_space}"
                    )
                
                converted_actions[agent] = action_value
        
        # Execute step
        observations, rewards, terminations, truncations, infos = self.env.step(converted_actions)
        
        # Update episode tracking
        self.episode_length += 1
        for agent, reward in rewards.items():
            if agent in self.cumulative_rewards:
                self.cumulative_rewards[agent] += reward
            else:
                self.cumulative_rewards[agent] = reward
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            RGB array if render_mode='rgb_array', otherwise None
        """
        if self.render_mode == 'rgb_array':
            return self.env.render()
        elif self.render_mode == 'human':
            self.env.render()
            return None
        else:
            return None
    
    def close(self) -> None:
        """Close environment and cleanup resources."""
        if hasattr(self.env, 'close'):
            self.env.close()
        self.env = None
    
    def __repr__(self) -> str:
        """String representation of wrapper."""
        return (
            f"PettingZooWrapper("
            f"env='{self.env_name}', "
            f"agents={len(self.possible_agents)}, "
            f"episode_length={self.episode_length})"
        )
