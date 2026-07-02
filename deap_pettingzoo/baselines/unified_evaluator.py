"""
Unified policy evaluator for baseline comparison.

This module provides a common evaluation interface that works with
policies from any training algorithm (EA, MARL, etc.).
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
        aggregation: str = "sum",
        score_agents: Optional[Sequence[str]] = None,
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
            aggregation: How per-agent rewards collapse into the scalar
                episode reward — ``"sum"`` (default, team return) or
                ``"mean"`` (per-agent average). On a zero-sum competitive
                env the global ``"sum"`` is ~0 for everyone; pair
                ``aggregation`` with ``score_agents`` to score one side.
            score_agents: Optional subset of agent ids whose rewards form
                the scalar episode reward. ``None`` (default) scores every
                agent. Use it to grade a backend on *its* team's return
                (resolving the "trained for X, graded on Y" degeneracy on
                competitive envs). Per-agent rewards in
                :attr:`EvalResult.agent_rewards` always span every agent
                regardless of this filter.
        """
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )
        self.max_steps = max_steps
        self._env_cache: Dict[Tuple, PettingZooWrapper] = {}
        self.frame_recorder = frame_recorder
        self.record_episode_offset = int(record_episode_offset)
        self.aggregation = aggregation
        self.score_agents = set(score_agents) if score_agents is not None else None

    def evaluate(
        self,
        policy: Optional[PolicyProtocol],
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False,
        env_config: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        env_steps: Optional[int] = None,
        record_extra: Optional[Mapping[str, Any]] = None,
        policy_map: Optional[Mapping[str, PolicyProtocol]] = None,
    ) -> EvalResult:
        """Evaluate a policy on the given environment.

        Args:
            policy: Trained policy satisfying PolicyProtocol, broadcast to
                every agent. May be ``None`` when ``policy_map`` is supplied.
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
            policy_map: Optional per-agent ``{agent_id: policy}`` mapping for
                heterogeneous rollouts (e.g. co-evolution, where each side owns
                different agents). When given it takes precedence over
                ``policy``; agents missing from the map act randomly. The
                aggregate reward then spans a mixed-policy episode, so the
                returned statistics are only meaningful for recording, not for
                single-policy fitness comparison.

        Returns:
            EvalResult with reward statistics.

        Notes:
            ``human`` mode pops a window but returns None.
            ``rgb_array`` takes precedence over ``human`` when both are requested.
        """

        if self.frame_recorder is not None and self.frame_recorder.enabled:
            render_mode: Optional[str] = "rgb_array"
        elif render:
            render_mode = "human"
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

        # Guard against a score_agents selection that names no real agent
        # (typo / wrong env): otherwise every episode silently scores 0.0.
        if self.score_agents is not None:
            possible = set(getattr(env, "possible_agents", []) or [])
            if possible and self.score_agents.isdisjoint(possible):
                logger.warning(
                    "score_agents %s match no agent in %s (agents: %s); "
                    "the scalar reward will be 0.0",
                    sorted(self.score_agents),
                    env_name,
                    sorted(possible),
                )

        # Setup RNG for episode seeds
        rng = np.random.default_rng(seed)

        # Run episodes
        episode_rewards = []
        episode_lengths = []
        per_agent_totals: Dict[str, float] = {}
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
            agent_rewards, length = self._run_episode(
                policy,
                env,
                ep_seed,
                render,
                recorder=recorder if recording else None,
                policy_map=policy_map,
            )
            episode_rewards.append(self._aggregate_reward(agent_rewards))
            episode_lengths.append(length)
            for agent, reward in agent_rewards.items():
                per_agent_totals[agent] = per_agent_totals.get(agent, 0.0) + reward

            if recording:
                recorder.end_episode()

        if recording:
            # Advance offset so successive evaluate() calls (e.g. one per
            # generation) keep producing distinct episode_NNNN artefacts.
            self.record_episode_offset += num_episodes

        # Mean-over-episodes reward per agent (decomposition that the
        # team-scalar mean_reward collapses).
        mean_agent_rewards = {
            agent: total / num_episodes for agent, total in per_agent_totals.items()
        }

        # Compute statistics
        return EvalResult(
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            num_episodes=num_episodes,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            agent_rewards=mean_agent_rewards,
        )

    def _aggregate_reward(self, agent_rewards: Mapping[str, float]) -> float:
        """Collapse per-agent rewards into one scalar per the configured policy.

        Restricts to :attr:`score_agents` when set, then applies
        ``"sum"`` or ``"mean"`` aggregation. An empty selection yields
        ``0.0`` (no scoring agent acted).
        """
        if self.score_agents is not None:
            selected = [r for a, r in agent_rewards.items() if a in self.score_agents]
        else:
            selected = list(agent_rewards.values())
        if not selected:
            return 0.0
        if self.aggregation == "mean":
            return float(np.mean(selected))
        return float(np.sum(selected))

    def _run_episode(
        self,
        policy: Optional[PolicyProtocol],
        env: PettingZooWrapper,
        seed: Optional[int],
        render: bool,
        *,
        recorder: Optional[FrameRecorder] = None,
        policy_map: Optional[Mapping[str, PolicyProtocol]] = None,
    ) -> tuple:
        """Run a single episode.

        Args:
            policy: Policy to evaluate (broadcast to all agents). May be
                ``None`` when ``policy_map`` is supplied.
            env: Environment instance.
            seed: Random seed for this episode.
            render: Whether to render.
            recorder: Optional active :class:`FrameRecorder` collecting
                ``rgb_array`` frames per step.
            policy_map: Optional per-agent ``{agent_id: policy}`` mapping;
                agents not present act randomly via ``action_space.sample()``.

        Returns:
            Tuple of (agent_rewards, episode_length), where
            ``agent_rewards`` maps each possible agent to its summed
            reward over the episode. The caller collapses it to a scalar
            via :meth:`_aggregate_reward`.
        """
        # Reset every distinct policy (recurrent state).
        if policy_map is not None:
            for pol in set(policy_map.values()):
                pol.reset()
        if policy is not None:
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
                pol = policy_map.get(agent) if policy_map is not None else policy
                if pol is None:
                    # Heterogeneous rollout with no policy for this agent.
                    actions[agent] = env.action_space(agent).sample()
                    continue
                try:
                    actions[agent] = pol.get_action(
                        observations[agent], action_mask=mask
                    )
                except TypeError:
                    actions[agent] = pol.get_action(observations[agent])

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

        return agent_rewards, steps

    def close(self):
        """Close all cached environments."""
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()
