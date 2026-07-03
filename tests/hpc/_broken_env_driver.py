"""Deliberately-broken custom env for the Dask rig's debuggability test.

Raises inside ``step`` with a distinguishable marker so the test can assert
that the real remote traceback reaches the driver -- not an opaque
``KilledWorker`` and not a hang. Same module-level-factory shape as
:mod:`tests.hpc._custom_env_driver` (baked into the image, importable by
dotted path in every container).
"""

from __future__ import annotations

from typing import Optional

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

ENV_NAME = "hpc_rig_broken_env_v0"

#: Grepped for by the test; must appear in the surfaced traceback.
FAILURE_MARKER = "HPC_RIG_INTENTIONAL_FAILURE"


class _BrokenEnv(ParallelEnv):
    metadata = {"name": ENV_NAME, "is_parallelizable": True}

    def __init__(self, render_mode: Optional[str] = None, **kwargs) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
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

    def reset(self, seed: Optional[int] = None, options=None):
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        obs = {a: self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        raise RuntimeError(
            f"{FAILURE_MARKER}: deliberate rollout failure for the HPC rig "
            "debuggability test."
        )

    def render(self):
        return None

    def close(self):
        return None


def parallel_env(render_mode: Optional[str] = None, **kwargs) -> _BrokenEnv:
    return _BrokenEnv(render_mode=render_mode, **kwargs)


def register() -> str:
    """Register :data:`ENV_NAME` (idempotent) and return its name."""
    from deap_pettingzoo.environments.registry import is_registered, register_env

    if not is_registered(ENV_NAME):
        register_env(
            ENV_NAME,
            parallel_env,
            default_episode_len=10,
            notes="HPC Dask rig deliberately-broken env (debuggability test).",
            tags={"test", "hpc"},
            api="parallel",
        )
    return ENV_NAME
