"""Tests for :mod:`deap_pettingzoo.environments.jax_parallel`.

These tests use a hand-rolled fake JaxMARL-style env so they do not
depend on the ``jax`` / ``jaxmarl`` extras when those are not installed.
When ``jax`` is installed, the adapter is exercised end-to-end; when it
is not, the tests are skipped cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from gymnasium import spaces  # noqa: E402

from deap_pettingzoo.environments.jax_parallel import (  # noqa: E402
    JaxEnvToPettingZooAdapter,
)
from deap_pettingzoo.environments.registry import (  # noqa: E402
    register_env,
    unregister_env,
    is_registered,
)
from deap_pettingzoo.environments import PettingZooWrapper  # noqa: E402


class _FakeJaxEnv:
    """Minimal JaxMARL-shaped env for adapter tests."""

    agents = ["agent_0", "agent_1"]
    metadata = {"name": "fake_jax_env"}

    def __init__(self, obs_dim: int = 4, n_actions: int = 3, max_steps: int = 5):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.max_steps = max_steps

    def observation_space(self, agent):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def action_space(self, agent):
        return spaces.Discrete(self.n_actions)

    def reset(self, key):
        obs = {a: jnp.zeros((self.obs_dim,), dtype=jnp.float32) for a in self.agents}
        state = {"step": jnp.asarray(0), "key": key}
        return obs, state

    def step(self, key, state, actions):
        step = state["step"] + 1
        obs = {
            a: jnp.full((self.obs_dim,), step, dtype=jnp.float32) for a in self.agents
        }
        rewards = {a: jnp.asarray(1.0) for a in self.agents}
        done_flag = step >= self.max_steps
        dones = {a: done_flag for a in self.agents}
        dones["__all__"] = done_flag
        infos = {a: {} for a in self.agents}
        new_state = {"step": step, "key": key}
        return obs, new_state, rewards, dones, infos


def test_adapter_reset_returns_numpy():
    adapter = JaxEnvToPettingZooAdapter(_FakeJaxEnv(), env_name="fake", seed=0)
    obs, infos = adapter.reset()
    assert set(obs.keys()) == {"agent_0", "agent_1"}
    for v in obs.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (4,)
    assert set(infos.keys()) == {"agent_0", "agent_1"}


def test_adapter_step_round_trips():
    adapter = JaxEnvToPettingZooAdapter(_FakeJaxEnv(), env_name="fake", seed=0)
    adapter.reset()
    actions = {"agent_0": 1, "agent_1": 2}
    obs, rewards, terms, truncs, infos = adapter.step(actions)
    assert all(isinstance(v, np.ndarray) for v in obs.values())
    assert rewards == {"agent_0": 1.0, "agent_1": 1.0}
    assert terms == {"agent_0": False, "agent_1": False}
    assert truncs == {"agent_0": False, "agent_1": False}


def test_adapter_termination_clears_agents():
    env = _FakeJaxEnv(max_steps=2)
    adapter = JaxEnvToPettingZooAdapter(env, env_name="fake", seed=0)
    adapter.reset()
    adapter.step({"agent_0": 0, "agent_1": 0})
    obs, _, terms, _, _ = adapter.step({"agent_0": 0, "agent_1": 0})
    assert all(terms.values())
    assert adapter.agents == []


def test_adapter_seed_is_deterministic():
    env = _FakeJaxEnv()
    a1 = JaxEnvToPettingZooAdapter(env, env_name="fake", seed=42)
    a2 = JaxEnvToPettingZooAdapter(env, env_name="fake", seed=42)
    obs1, _ = a1.reset()
    obs2, _ = a2.reset()
    for k in obs1:
        np.testing.assert_array_equal(obs1[k], obs2[k])


def test_step_before_reset_raises():
    adapter = JaxEnvToPettingZooAdapter(_FakeJaxEnv(), env_name="fake", seed=0)
    with pytest.raises(RuntimeError, match="before reset"):
        adapter.step({"agent_0": 0, "agent_1": 0})


def test_registry_jax_parallel_roundtrip():
    """End-to-end registration via build_env + PettingZooWrapper."""
    name = "__test_fake_jax_env__"
    if is_registered(name):
        unregister_env(name)

    register_env(
        name,
        lambda **cfg: _FakeJaxEnv(**cfg),
        api="jax_parallel",
        default_episode_len=5,
        tags={"jax", "test"},
    )
    try:
        wrapper = PettingZooWrapper(name)
        obs = wrapper.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1"}
        actions = {a: 0 for a in wrapper.agents}
        obs, rewards, terms, truncs, infos = wrapper.step(actions)
        assert all(isinstance(v, np.ndarray) for v in obs.values())
    finally:
        unregister_env(name)
