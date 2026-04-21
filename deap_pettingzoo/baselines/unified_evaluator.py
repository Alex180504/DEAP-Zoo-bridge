"""
Unified policy evaluator for baseline comparison.

This module provides a common evaluation interface that works with
policies from any training algorithm (EA, MARL, etc.).
"""

from typing import Optional, Dict, Any, List
import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineEvaluator,
    EvalResult,
    PolicyProtocol,
)
from deap_pettingzoo.environments import PettingZooWrapper


class UnifiedEvaluator(BaselineEvaluator):
    """Unified evaluator for any policy type.
    
    Evaluates policies regardless of their training method by using
    the common PolicyProtocol interface.
    """
    
    def __init__(self, max_steps: int = 500):
        """Initialize unified evaluator.
        
        Args:
            max_steps: Maximum steps per episode.
        """
        self.max_steps = max_steps
        self._env_cache: Dict[str, PettingZooWrapper] = {}
    
    def evaluate(
        self,
        policy: PolicyProtocol,
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False
    ) -> EvalResult:
        """Evaluate a policy on the given environment.
        
        Args:
            policy: Trained policy satisfying PolicyProtocol.
            env_name: PettingZoo environment name.
            num_episodes: Number of episodes to evaluate.
            seed: Random seed for reproducibility.
            render: Whether to render episodes.
            
        Returns:
            EvalResult with reward statistics.
        """
        # Get or create environment
        if env_name not in self._env_cache:
            self._env_cache[env_name] = PettingZooWrapper(
                env_name,
                render_mode='human' if render else None
            )
        env = self._env_cache[env_name]
        
        # Setup RNG for episode seeds
        rng = np.random.default_rng(seed)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            # Generate episode seed
            if seed is not None:
                ep_seed = int(rng.integers(0, 2**31 - 1))
            else:
                ep_seed = None
            
            # Run episode
            total_reward, length = self._run_episode(policy, env, ep_seed, render)
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
        
        # Compute statistics
        return EvalResult(
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            num_episodes=num_episodes,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
    
    def _run_episode(
        self,
        policy: PolicyProtocol,
        env: PettingZooWrapper,
        seed: Optional[int],
        render: bool
    ) -> tuple:
        """Run a single episode.
        
        Args:
            policy: Policy to evaluate.
            env: Environment instance.
            seed: Random seed for this episode.
            render: Whether to render.
            
        Returns:
            Tuple of (total_reward, episode_length).
        """
        # Reset policy and environment
        policy.reset()
        observations = env.reset(seed=seed)
        
        # Track rewards per agent
        agent_rewards = dict.fromkeys(env.possible_agents, 0.0)
        steps = 0
        
        # Run episode
        for step in range(self.max_steps):
            steps = step + 1
            
            # Get actions for all agents
            actions = {}
            for agent in env.possible_agents:
                obs = observations[agent]
                actions[agent] = policy.get_action(obs)
            
            # Step environment
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Accumulate rewards
            for agent in env.possible_agents:
                if agent in rewards:
                    agent_rewards[agent] += rewards[agent]
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break
            
            if render:
                env.render()
        
        # Aggregate total reward (sum across all agents)
        total_reward = sum(agent_rewards.values())
        
        return total_reward, steps
    
    def close(self):
        """Close all cached environments."""
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()
