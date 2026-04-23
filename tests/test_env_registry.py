"""Tests for the open env registry.

Covers :mod:`deap_pettingzoo.environments.registry`:
register / unregister / lookup / list / build semantics, clear error
messages for unknown names, support for both module-style and callable
factories, and auto-registration of MPE2 / aquarium envs via
:mod:`deap_pettingzoo.environments.builtin`.
"""

from __future__ import annotations

import pytest

from deap_pettingzoo.environments import registry
from deap_pettingzoo.environments.registry import (
    EnvSpec,
    build_env,
    get_env_spec,
    is_registered,
    list_envs,
    register_env,
    unregister_env,
)


# ---------------------------------------------------------------------------
# Fixtures: a throwaway fake env that exposes the PettingZoo parallel API
# ---------------------------------------------------------------------------


class _FakeParallelEnv:
    """Minimal stand-in — tests only need it to be a distinguishable object."""

    def __init__(self, render_mode=None, **cfg):
        self.render_mode = render_mode
        self.cfg = cfg
        self.possible_agents = ["agent_0", "agent_1"]


class _FakeModule:
    """Module-style factory: exposes a ``parallel_env`` callable."""

    @staticmethod
    def parallel_env(render_mode=None, **cfg):
        return _FakeParallelEnv(render_mode=render_mode, **cfg)


@pytest.fixture
def fake_name():
    """A uniquely-named registration that is cleaned up after each test."""
    name = "_fake_test_env_v0"
    yield name
    if is_registered(name):
        unregister_env(name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterAndLookup:
    def test_register_then_get_spec(self, fake_name):
        spec = register_env(fake_name, _FakeModule, default_episode_len=99)
        assert isinstance(spec, EnvSpec)
        assert get_env_spec(fake_name) is spec
        assert spec.default_episode_len == 99

    def test_is_registered(self, fake_name):
        assert not is_registered(fake_name)
        register_env(fake_name, _FakeModule)
        assert is_registered(fake_name)

    def test_duplicate_registration_raises(self, fake_name):
        register_env(fake_name, _FakeModule)
        with pytest.raises(ValueError, match="already registered"):
            register_env(fake_name, _FakeModule)

    def test_duplicate_with_overwrite_succeeds(self, fake_name):
        register_env(fake_name, _FakeModule, default_episode_len=10)
        register_env(
            fake_name, _FakeModule, default_episode_len=20, overwrite=True
        )
        assert get_env_spec(fake_name).default_episode_len == 20


class TestUnknownEnv:
    def test_unknown_env_error_message(self):
        with pytest.raises(ValueError, match="Unsupported / unregistered"):
            get_env_spec("this_env_definitely_does_not_exist_v42")


class TestFactoryForms:
    def test_module_style_factory(self, fake_name):
        register_env(fake_name, _FakeModule)
        env = build_env(fake_name, env_config={"foo": 1}, render_mode="human")
        assert isinstance(env, _FakeParallelEnv)
        assert env.render_mode == "human"
        assert env.cfg == {"foo": 1}

    def test_callable_factory(self, fake_name):
        def _factory(render_mode=None, **cfg):
            return _FakeParallelEnv(render_mode=render_mode, **cfg)

        register_env(fake_name, _factory)
        env = build_env(fake_name, env_config={"bar": 2})
        assert isinstance(env, _FakeParallelEnv)
        assert env.cfg == {"bar": 2}

    def test_invalid_factory_rejected(self, fake_name):
        with pytest.raises(TypeError, match="callable"):
            register_env(fake_name, "not a module or callable")


class TestListAndTags:
    def test_list_envs_returns_sorted_strings(self):
        envs = list_envs()
        assert envs == sorted(envs)
        assert all(isinstance(n, str) for n in envs)

    def test_list_envs_with_tag(self, fake_name):
        register_env(fake_name, _FakeModule, tags={"_unit_test_tag"})
        assert fake_name in list_envs(tag="_unit_test_tag")
        assert fake_name not in list_envs(tag="some_other_tag")


class TestBuiltinRegistrations:
    """Side-effect of ``deap_pettingzoo.environments`` import."""

    def test_mpe2_simple_tag_registered(self):
        # Whenever mpe2 is installed, simple_tag_v3 must be available.
        pytest.importorskip("mpe2")
        assert is_registered("simple_tag_v3")
        spec = get_env_spec("simple_tag_v3")
        assert "mpe2" in spec.tags

    def test_aquarium_registered_when_installed(self):
        try:
            import marl_aquarium  # noqa: F401
        except ImportError:
            pytest.skip("marl-aquarium not installed")
        assert is_registered("aquarium_v0")
