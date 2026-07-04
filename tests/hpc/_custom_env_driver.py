"""Module-level custom-env factory + registration for the Dask HPC rig.

Baked into the ``deap-hpc-base`` image (see ``docker/hpc-base/Dockerfile``)
so it is importable by dotted path inside every container -- driver and
workers alike. This is the whole point of the test: a runtime-registered
environment (the registry-snapshot mechanism, not an installed entry point)
must survive a genuine multi-container Dask dispatch, not just a local
``spawn``. The factory must stay module-level/importable -- a closure or a
factory defined in a driver script's ``__main__`` would not resolve on a
separate container's worker process.
"""

from __future__ import annotations

from typing import Optional

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

#: Registry name used by the rig.
ENV_NAME = "hpc_rig_custom_env_v0"


class _HpcRigEnv(ParallelEnv):
    """Tiny two-agent Box-observation / Discrete-action smoke-test env."""

    metadata = {"name": ENV_NAME, "is_parallelizable": True}

    def __init__(self, render_mode: Optional[str] = None, **kwargs) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._max_cycles = 10
        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self._obs_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}

    def observation_space(self, agent: str):
        return self._obs_spaces[agent]

    def action_space(self, agent: str):
        return self._act_spaces[agent]

    def _obs(self, agent: str) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        obs = {a: self._obs(a) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        self._step_count += 1
        rewards = {a: float(-abs(int(actions[a]) - 2)) for a in self.agents}
        done = self._step_count >= self._max_cycles
        obs = {a: self._obs(a) for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        return None


def parallel_env(render_mode: Optional[str] = None, **kwargs) -> _HpcRigEnv:
    """Top-level factory returning a :class:`_HpcRigEnv`."""
    return _HpcRigEnv(render_mode=render_mode, **kwargs)


def register() -> str:
    """Register :data:`ENV_NAME` (idempotent) and return its name."""
    from deap_pettingzoo.environments.registry import is_registered, register_env

    if not is_registered(ENV_NAME):
        register_env(
            ENV_NAME,
            parallel_env,
            default_episode_len=10,
            notes="HPC Dask rig smoke-test env (registered at runtime).",
            tags={"test", "hpc"},
            api="parallel",
        )
    return ENV_NAME
