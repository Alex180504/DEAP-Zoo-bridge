"""Synthetic homogeneous MultiDiscrete PettingZoo parallel env for tests.

The built-in MPE2 environments expose only ``Discrete`` action spaces
(``simple_reference_v3`` flattens movement x communication into a single
``Discrete(50)``), so there is no built-in env to test MultiDiscrete
support end-to-end on. This module provides a tiny, fully-homogeneous
``ParallelEnv`` whose agents share a ``MultiDiscrete`` action space, plus a
top-level ``parallel_env`` factory and a :func:`register_md_env` helper so the
trainers can construct it by name through the registry.

The factory and env class are defined at module top level (no closures/
lambdas) so they remain picklable for the spawn-based parallel paths.
"""

from __future__ import annotations

from typing import Optional

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

#: Default registry name used by the tests.
MD_ENV_NAME = "md_test_v0"

#: Default per-sub-action choice counts (sum = 7 network outputs).
MD_NVEC = (3, 4)

#: Observation dimension (>= 4 so the MPE-style qdax extractor can read the
#: first two position/velocity pairs from the observation vector).
MD_OBS_DIM = 4


class MultiDiscreteParallelEnv(ParallelEnv):
    """Two-agent homogeneous env with a shared ``MultiDiscrete`` action space.

    The reward is ``-sum(action)`` per agent, so lower sub-action indices are
    better — a simple, learnable signal for smoke-testing optimisers.
    """

    metadata = {"name": MD_ENV_NAME, "is_parallelizable": True}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        obs_dim: int = MD_OBS_DIM,
        nvec=MD_NVEC,
        max_cycles: int = 10,
    ) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._obs_dim = int(obs_dim)
        self._nvec = np.asarray(nvec, dtype=np.int64)
        self._max_cycles = int(max_cycles)
        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self._obs_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.MultiDiscrete(self._nvec.copy()) for a in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self._obs_spaces[agent]

    def action_space(self, agent: str):
        return self._act_spaces[agent]

    def _obs(self, agent: str) -> np.ndarray:
        # Keep within the declared Box bounds [-1, 1] (some RL backends
        # validate observations against the space).
        return self._rng.uniform(-1.0, 1.0, size=self._obs_dim).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        observations = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self._step_count += 1
        rewards = {}
        for a in self.agents:
            act = np.asarray(actions[a]).ravel()
            rewards[a] = float(-np.sum(act))
        done = self._step_count >= self._max_cycles
        observations = {a: self._obs(a) for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        return None


def parallel_env(render_mode: Optional[str] = None, **kwargs) -> MultiDiscreteParallelEnv:
    """Top-level factory returning a :class:`MultiDiscreteParallelEnv`."""
    return MultiDiscreteParallelEnv(render_mode=render_mode, **kwargs)


def register_md_env(name: str = MD_ENV_NAME, *, max_cycles: int = 10) -> str:
    """Register the synthetic MultiDiscrete env (idempotent) and return its name."""
    from deap_pettingzoo.environments.registry import is_registered, register_env

    if not is_registered(name):
        register_env(
            name,
            parallel_env,
            default_episode_len=max_cycles,
            notes="Synthetic homogeneous MultiDiscrete env (tests only).",
            tags={"test", "multidiscrete"},
            api="parallel",
        )
    return name
