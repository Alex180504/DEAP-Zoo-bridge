"""
Unified policy evaluator for baseline comparison.

This module provides a common evaluation interface that works with
policies from any training algorithm (EA, MARL, etc.).
"""

from typing import Optional, Dict, Any, Mapping, Tuple
import json
import logging
import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineEvaluator,
    EvalResult,
    PolicyProtocol,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.utils.logging.replay import FrameRecorder

logger = logging.getLogger(__name__)


class UnifiedEvaluator(BaselineEvaluator):
    """Unified evaluator for any policy type.
    
    Evaluates policies regardless of their training method by using
    the common PolicyProtocol interface.
    """
    
    def __init__(
        self,
        max_steps: int = 500,
        *,
        frame_recorder: Optional[FrameRecorder] = None,
        record_episode_offset: int = 0,
    ):
        """Initialize unified evaluator.
        
        Args:
            max_steps: Maximum steps per episode.
            frame_recorder: Optional :class:`FrameRecorder`. When supplied
                each evaluation episode is wrapped with
                ``start_episode/end_episode`` and ``env.render('rgb_array')``
                frames are streamed via ``add_frame``. Trainers running a
                champion-rollout pass an enabled recorder here; routine
                evaluations leave this ``None`` (default).
            record_episode_offset: Starting episode index for the recorder
                (so successive generations don't overwrite each other's
                ``episode_NNNN`` artefacts).
        """
        self.max_steps = max_steps
        self._env_cache: Dict[Tuple, PettingZooWrapper] = {}
        self.frame_recorder = frame_recorder
        self.record_episode_offset = int(record_episode_offset)
    
    def evaluate(
        self,
        policy: PolicyProtocol,
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False,
        env_config: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        env_steps: Optional[int] = None,
        record_extra: Optional[Mapping[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate a policy on the given environment.

        Args:
            policy: Trained policy satisfying PolicyProtocol.
            env_name: PettingZoo environment name.
            num_episodes: Number of episodes to evaluate.
            seed: Random seed for reproducibility.
            render: Whether to render episodes.
            env_config: Optional environment configuration dict. Must match
                the config used during training so observation/action shapes
                are consistent.
            iteration: Optional training iteration/generation that produced
                ``policy``. When recording, it is written to the recorder's
                ``frames/recordings.json`` sidecar so the UI can associate
                the artefact with its metrics row.
            env_steps: Optional cumulative env-step count at this iteration
                (the meaningful axis for RL runs). Also written to the sidecar.
            record_extra: Optional extra metadata dict for the sidecar entry.

        Returns:
            EvalResult with reward statistics.
        
        Notes:
            ``human`` mode pops a window but returns None. 
            ``rgb_array`` takes precedence over ``human`` when both are requested.
        """

        if self.frame_recorder is not None and self.frame_recorder.enabled:
            render_mode: Optional[str] = 'rgb_array'
        elif render:
            render_mode = 'human'
        else:
            render_mode = None
        config_key = json.dumps(env_config, sort_keys=True) if env_config else None
        cache_key = (env_name, config_key, render_mode)

        # Get or create environment
        if cache_key not in self._env_cache:
            self._env_cache[cache_key] = PettingZooWrapper(
                env_name,
                env_config=env_config,
                render_mode=render_mode,
            )
        env = self._env_cache[cache_key]
        
        # Setup RNG for episode seeds
        rng = np.random.default_rng(seed)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        recorder = self.frame_recorder
        recording = recorder is not None and recorder.enabled

        for ep_idx in range(num_episodes):
            # Generate episode seed
            if seed is not None:
                ep_seed = int(rng.integers(0, 2**31 - 1))
            else:
                ep_seed = None

            if recording:
                recorder.start_episode(
                    self.record_episode_offset + ep_idx,
                    iteration=iteration,
                    env_steps=env_steps,
                    extra=record_extra,
                )

            # Run episode
            total_reward, length = self._run_episode(
                policy, env, ep_seed, render, recorder=recorder if recording else None
            )
            episode_rewards.append(total_reward)
            episode_lengths.append(length)

            if recording:
                recorder.end_episode()

        if recording:
            # Advance offset so successive evaluate() calls (e.g. one per
            # generation) keep producing distinct episode_NNNN artefacts.
            self.record_episode_offset += num_episodes
        
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
        render: bool,
        *,
        recorder: Optional[FrameRecorder] = None,
    ) -> tuple:
        """Run a single episode.
        
        Args:
            policy: Policy to evaluate.
            env: Environment instance.
            seed: Random seed for this episode.
            render: Whether to render.
            recorder: Optional active :class:`FrameRecorder` collecting
                ``rgb_array`` frames per step.
            
        Returns:
            Tuple of (total_reward, episode_length).
        """
        # Reset policy and environment
        policy.reset()
        observations = env.reset(seed=seed)

        # Capture initial frame so the artefact starts on the reset state.
        if recorder is not None:
            try:
                recorder.add_frame(env.render())
            except Exception:
                logger.debug("frame capture failed", exc_info=True)

        # Track rewards per agent
        agent_rewards = dict.fromkeys(env.possible_agents, 0.0)
        steps = 0
        
        # Run episode
        for step in range(self.max_steps):
            steps = step + 1
            
            # Terminate if no active agents remain
            if not observations:
                break

            # Get actions for active agents only
            infos = getattr(env, "last_infos", {}) or {}
            actions = {}
            for agent in observations:
                mask = infos.get(agent, {}).get("action_mask")
                try:
                    actions[agent] = policy.get_action(
                        observations[agent], action_mask=mask
                    )
                except TypeError:
                    actions[agent] = policy.get_action(observations[agent])
            
            # Step environment
            observations, rewards, terminations, truncations, _ = env.step(actions)

            # Capture post-step frame for recording.
            if recorder is not None:
                try:
                    recorder.add_frame(env.render())
                except Exception:
                    pass

            # Accumulate rewards for active agents
            for agent, reward in rewards.items():
                if agent in agent_rewards:
                    agent_rewards[agent] += reward
            
            # Check if episode is done
            if terminations and all(terminations.values()):
                break
            if truncations and all(truncations.values()):
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
