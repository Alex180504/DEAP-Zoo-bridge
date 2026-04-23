"""Plug a third-party PettingZoo-like env into the registry and drive it
through :class:`PettingZooWrapper`.  This exercises the *open* nature of
the registry: users of deap-bridge can run the EA / AgileRL pipelines on
environments outside the auto-registered MPE2 / aquarium set.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import (
    is_registered,
    register_env,
    unregister_env,
)


# ---------------------------------------------------------------------------
# A minimal parallel-API env with two agents and Box(2) actions.
# ---------------------------------------------------------------------------


class _TinyParallelEnv:
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._t = 0

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        self._t = 0
        self.agents = list(self.possible_agents)
        obs = {a: np.zeros(3, dtype=np.float32) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self._t += 1
        obs = {a: np.full(3, 0.1 * self._t, dtype=np.float32) for a in self.agents}
        rewards = {a: float(np.sum(actions[a])) for a in self.agents}
        terms = {a: self._t >= 5 for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        if self._t >= 5:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self):
        pass


class _TinyEnvModule:
    """Module-style factory."""

    @staticmethod
    def parallel_env(render_mode=None):
        return _TinyParallelEnv(render_mode=render_mode)


@pytest.fixture
def tiny_env_name():
    name = "_tiny_parallel_env_v0"
    register_env(
        name,
        _TinyEnvModule,
        default_episode_len=5,
        continuous_default=True,
        notes="Third-party plug-in smoke test.",
        tags={"test", "third_party"},
    )
    yield name
    if is_registered(name):
        unregister_env(name)


def test_third_party_wrapper_builds(tiny_env_name):
    wrapper = PettingZooWrapper(tiny_env_name)
    assert wrapper.env_name == tiny_env_name
    assert len(wrapper.possible_agents) == 2


def test_third_party_reset_and_step(tiny_env_name):
    wrapper = PettingZooWrapper(tiny_env_name)
    obs = wrapper.reset(seed=0)
    assert set(obs.keys()) == {"agent_0", "agent_1"}

    actions = {a: np.zeros(2, dtype=np.float32) for a in wrapper.possible_agents}
    obs, rewards, terms, truncs, infos = wrapper.env.step(actions)
    assert set(rewards.keys()) == {"agent_0", "agent_1"}


def test_third_party_env_with_feedforward_policy(tiny_env_name):
    """Drive the plug-in env with an EA policy via the Box action head."""
    from deap_pettingzoo.genomes import FixedTopologyGenome
    from deap_pettingzoo.policies import FeedforwardNeuralPolicy

    wrapper = PettingZooWrapper(tiny_env_name)
    genome = FixedTopologyGenome([3, 8, 2], activation="tanh")
    policies = {
        agent: FeedforwardNeuralPolicy(
            genome, action_space=wrapper.env.action_space(agent)
        )
        for agent in wrapper.possible_agents
    }
    obs = wrapper.reset(seed=0)
    actions = {a: policies[a].get_action(obs[a]) for a in wrapper.possible_agents}
    obs, rewards, terms, truncs, _ = wrapper.env.step(actions)
    for a in wrapper.possible_agents:
        assert actions[a].shape == (2,)
        assert (actions[a] >= -1.0).all() and (actions[a] <= 1.0).all()
