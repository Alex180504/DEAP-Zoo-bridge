"""AEC environment support: unit + integration tests.

Covers the ``api='parallel' | 'aec' | 'auto'`` branch in
:func:`deap_pettingzoo.environments.registry.build_env`, the
simultaneity guard attached to converted envs, and an end-to-end
:class:`EATrainer` smoke run against a hand-rolled AEC env.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import (
    build_env,
    list_envs,
    register_env,
    unregister_env,
)


# ---------------------------------------------------------------------------
# Minimal parallel-compatible AEC env for testing
# ---------------------------------------------------------------------------


class _SimultaneousAEC(AECEnv):
    """Minimal 2-agent AEC env with simultaneous-step semantics.

    Each call to ``step`` advances the internal agent selector once;
    after both agents have acted, the shared episode counter advances.
    This mirrors the common PettingZoo AEC pattern that
    ``aec_to_parallel`` is designed to handle.
    """

    metadata = {"render_modes": [], "name": "sim_aec_v0", "is_parallelizable": True}

    def __init__(self, render_mode=None, max_cycles: int = 3, obs_dim: int = 3, n_actions: int = 4):
        super().__init__()
        self.possible_agents = ["a", "b"]
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._max_cycles = max_cycles
        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(n_actions)
        self.agents = []
        self._cycles = 0
        self._selector_idx = 0
        self._pending_rewards = {}
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.agent_selection = None

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def observe(self, agent):
        return np.zeros(self._obs_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._cycles = 0
        self._selector_idx = 0
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._pending_rewards = {a: 0.0 for a in self.agents}

    def step(self, action):
        if not self.agents:
            return
        agent = self.agent_selection
        self._pending_rewards[agent] = float(action) * 0.1
        self._cumulative_rewards[agent] += self._pending_rewards[agent]
        self._selector_idx = (self._selector_idx + 1) % len(self.agents)
        if self._selector_idx == 0:
            # Full cycle done — publish rewards and advance step.
            self.rewards = dict(self._pending_rewards)
            self._cycles += 1
            if self._cycles >= self._max_cycles:
                self.truncations = {a: True for a in self.agents}
                self.terminations = {a: False for a in self.agents}
            self._pending_rewards = {a: 0.0 for a in self.agents}
        else:
            self.rewards = {a: 0.0 for a in self.agents}
        self.agent_selection = self.agents[self._selector_idx] if self.agents else None
        if all(self.truncations.values()) or all(self.terminations.values()):
            self.agents = []

    def last(self, observe=True):
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        return (
            obs,
            self._cumulative_rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def close(self):
        pass

    def render(self):
        pass


class _AECFactoryModule:
    """Mimics a PettingZoo env module with an ``env(...)`` entry point."""

    @staticmethod
    def env(render_mode=None, **cfg):
        return _SimultaneousAEC(render_mode=render_mode, **cfg)


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aec_registration():
    name = "sim_aec_v0"
    register_env(
        name,
        _AECFactoryModule,
        default_episode_len=3,
        api="aec",
        tags={"aec", "test"},
    )
    yield name
    unregister_env(name)


@pytest.fixture
def aec_auto_registration():
    name = "sim_aec_auto_v0"
    register_env(
        name,
        _AECFactoryModule,
        default_episode_len=3,
        api="auto",
    )
    yield name
    unregister_env(name)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestRegisterEnvAPIField:
    def test_default_api_is_parallel(self):
        # MPE2 envs are auto-registered with api='parallel'.
        from deap_pettingzoo.environments.registry import get_env_spec

        assert get_env_spec("simple_spread_v3").api == "parallel"

    def test_unsupported_api_rejected(self):
        def _factory(render_mode=None, **cfg):
            return None

        with pytest.raises(ValueError, match="Unsupported api"):
            register_env("bad_api", _factory, api="gibberish")  # type: ignore[arg-type]

    def test_aec_module_accepted_without_parallel_env(self, aec_registration):
        # The fixture registration succeeds despite _AECFactoryModule
        # exposing only .env(...) (no .parallel_env).
        assert aec_registration in list_envs()

    def test_object_without_any_factory_attr_rejected(self):
        class _Empty:
            pass

        with pytest.raises(TypeError, match="parallel_env"):
            register_env("empty_fac", _Empty())

    def test_list_envs_filters_by_api(self, aec_registration):
        aec_names = list_envs(api="aec")
        parallel_names = list_envs(api="parallel")
        assert aec_registration in aec_names
        assert aec_registration not in parallel_names
        assert "simple_spread_v3" in parallel_names


class TestBuildEnvAECConversion:
    def test_aec_env_converts_to_parallel(self, aec_registration):
        env = build_env(aec_registration)
        from pettingzoo.utils.env import ParallelEnv

        assert isinstance(env, ParallelEnv)

    def test_auto_mode_converts_aec_to_parallel(self, aec_auto_registration):
        env = build_env(aec_auto_registration)
        from pettingzoo.utils.env import ParallelEnv

        assert isinstance(env, ParallelEnv)

    def test_aec_env_registered_as_parallel_raises_on_build(self):
        name = "wrong_api_v0"
        register_env(name, _AECFactoryModule, api="parallel")
        try:
            with pytest.raises(TypeError, match="api='aec'"):
                build_env(name)
        finally:
            unregister_env(name)

    def test_parallel_fast_path(self):
        env = build_env("simple_spread_v3", env_config={"max_cycles": 5, "N": 3})
        from pettingzoo.utils.env import ParallelEnv

        assert isinstance(env, ParallelEnv)


class TestPettingZooWrapperOverAEC:
    def test_reset_and_step_roundtrip(self, aec_registration):
        wrapper = PettingZooWrapper(aec_registration)
        obs = wrapper.reset(seed=0)
        assert set(obs.keys()) == {"a", "b"}
        for agent, ob in obs.items():
            assert ob.shape == (3,)

        actions = {agent: 0 for agent in wrapper.agents}
        obs2, rewards, terminations, truncations, infos = wrapper.step(actions)
        assert set(rewards.keys()) >= set(wrapper.possible_agents)
        # After max_cycles (3), truncations must all be True.
        for _ in range(3):
            if not wrapper.agents:
                break
            actions = {agent: 0 for agent in wrapper.agents}
            obs2, rewards, terminations, truncations, infos = wrapper.step(actions)
        assert all(truncations.values()) or not wrapper.agents


class TestSimultaneityGuard:
    def test_warn_on_partial_initial_active_agents(self):
        """Verify the trust-but-verify warning fires on non-simultaneous reset."""

        class _PartialResetAEC(_SimultaneousAEC):
            def reset(self, seed=None, options=None):
                super().reset(seed=seed, options=options)
                # Deliberately drop one agent to trip the guard.
                self.agents = ["a"]
                self.agent_selection = "a"

        class _Mod:
            @staticmethod
            def env(render_mode=None, **cfg):
                return _PartialResetAEC(render_mode=render_mode, **cfg)

        name = "partial_reset_v0"
        register_env(name, _Mod, api="aec")
        try:
            env = build_env(name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                env.reset(seed=0)
                msgs = [str(w.message) for w in caught]
            assert any("simultaneous-step semantics" in m for m in msgs), msgs
        finally:
            unregister_env(name)


# ---------------------------------------------------------------------------
# Integration test — EATrainer over AEC env
# ---------------------------------------------------------------------------


class TestEATrainerOverAEC:
    def test_ea_trains_on_converted_aec_env(self, aec_registration):
        trainer = EATrainer(hidden_size=4, population_size=2)
        config = TrainingConfig(
            env_name=aec_registration,
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=4,  # 2 generations at pop=2
            algorithm_params={"seed": 0},
        )
        result = trainer.train(config)
        assert np.isfinite(result.metrics["final_fitness"])
        assert result.metrics["population_size"] == 2
