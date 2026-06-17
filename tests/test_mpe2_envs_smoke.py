"""Parametrised smoke tests for every registered MPE2 env.

Builds each env via :class:`PettingZooWrapper`, runs a handful of steps,
and verifies observation / action-space sanity.  Envs with MultiDiscrete
action spaces (``simple_reference_v3``, ``simple_world_comm_v3``) only
execute the build-and-reset portion — they are *not* expected to work
with the Discrete/Box-only :class:`FeedforwardNeuralPolicy`.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import get_env_spec, list_envs


# Smoke suite: only run on main/development CI (and explicit `-m smoke`).
pytestmark = pytest.mark.smoke


# All MPE2 envs auto-registered by deap_pettingzoo.environments.builtin.
_MPE2_ENVS = [n for n in list_envs(tag="mpe2")]


@pytest.mark.parametrize("env_name", _MPE2_ENVS)
class TestMPE2EnvSmoke:
    def test_build(self, env_name):
        wrapper = PettingZooWrapper(env_name)
        assert wrapper.env_name == env_name
        assert len(wrapper.possible_agents) >= 1

    def test_reset(self, env_name):
        wrapper = PettingZooWrapper(env_name)
        observations = wrapper.reset(seed=0)
        assert set(observations.keys()) == set(wrapper.possible_agents)
        for agent, obs in observations.items():
            assert isinstance(obs, np.ndarray)
            assert obs.ndim == 1

    def test_step_once_with_zero_actions(self, env_name):
        wrapper = PettingZooWrapper(env_name)
        wrapper.reset(seed=0)
        # Build a zero-action dict compatible with whatever action-space type the env uses.  
        # For Discrete: int 0.  
        # For Box/MultiDiscrete:np.zeros matching the low/nvec shape.
        actions = {}
        for agent in wrapper.possible_agents:
            space = wrapper.env.action_space(agent)
            if hasattr(space, "n") and not hasattr(space, "nvec"):
                actions[agent] = 0
            elif hasattr(space, "nvec"):
                actions[agent] = np.zeros(space.nvec.shape, dtype=np.int64)
            elif hasattr(space, "low"):
                actions[agent] = np.zeros(space.shape, dtype=np.float32)
            else:  # pragma: no cover - unknown space
                pytest.skip(f"Unsupported action space: {type(space).__name__}")
        observations, rewards, terms, truncs, infos = wrapper.env.step(actions)
        assert set(rewards.keys()).issubset(set(wrapper.possible_agents))

    def test_spec_metadata(self, env_name):
        spec = get_env_spec(env_name)
        assert spec.default_episode_len > 0
        assert "mpe2" in spec.tags
