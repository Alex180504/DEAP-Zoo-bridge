"""Tests for the :attr:`EnvSpec.wrappers` hook on :func:`register_env`.

These tests verify three invariants:

1. **Parity**: a registration with ``wrappers=()`` behaves identically
   to not passing the field at all.
2. **Order preservation**: when multiple wrappers are supplied, they
   run left-to-right over the env produced by api coercion.
3. **Composition with turn-based adapters**: wrappers applied on top
   of an ``api='aec_masked'`` registration run *after* the adapter,
   so they always see the parallel-shaped env the rest of the stack
   expects.

The tests avoid any :mod:`supersuit` dependency and use tiny mock wrappers that record which env they saw.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from deap_pettingzoo.environments.registry import (
    build_env,
    get_env_spec,
    register_env,
    unregister_env,
)


class _DummyWrapperEnv:
    """Minimal wrapper around an env that records its construction."""

    def __init__(self, inner: Any, tag: str, trace: List[str]) -> None:
        self.inner = inner
        self.tag = tag
        trace.append(tag)

    # Forward attribute access so downstream code still sees a parallel env.
    def __getattr__(self, item):
        return getattr(self.inner, item)


def _make_wrapper(tag: str, trace: List[str]):
    def _wrap(env):
        return _DummyWrapperEnv(env, tag, trace)

    _wrap.__name__ = f"wrap_{tag}"
    return _wrap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parallel_env_factory():
    """Register a minimal parallel-API env and clean up after the test."""
    pytest.importorskip("mpe2")
    from mpe2 import simple_spread_v3

    name = "test_registry_wrappers_simple_spread"

    def _register(**kwargs):
        register_env(
            name,
            simple_spread_v3,
            default_episode_len=25,
            api="parallel",
            overwrite=True,
            **kwargs,
        )
        return name

    yield _register

    try:
        unregister_env(name)
    except KeyError:
        pass


@pytest.fixture
def masked_aec_factory():
    """Register a minimal turn-based AEC env under ``api='aec_masked'``."""
    classic = pytest.importorskip("pettingzoo.classic")
    tictactoe = classic.tictactoe_v3

    name = "test_registry_wrappers_tictactoe"

    def _register(**kwargs):
        register_env(
            name,
            tictactoe,
            default_episode_len=9,
            api="aec_masked",
            overwrite=True,
            **kwargs,
        )
        return name

    yield _register

    try:
        unregister_env(name)
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_wrappers_default_is_parity(parallel_env_factory):
    """``wrappers=()`` must produce the same env as omitting the field."""
    name = parallel_env_factory()  # wrappers defaulted
    spec = get_env_spec(name)
    assert spec.wrappers == ()

    env = build_env(name)
    # The built env should have no _DummyWrapperEnv in its ancestry.
    assert not isinstance(env, _DummyWrapperEnv)
    env.close() if hasattr(env, "close") else None


def test_wrappers_run_left_to_right(parallel_env_factory):
    """Wrappers run in the order registered; outermost is the last applied."""
    trace: List[str] = []
    wa = _make_wrapper("A", trace)
    wb = _make_wrapper("B", trace)
    wc = _make_wrapper("C", trace)

    name = parallel_env_factory(wrappers=(wa, wb, wc))
    env = build_env(name)

    assert trace == ["A", "B", "C"]
    # Outer wrapper is the last one applied (C wraps B wraps A wraps raw).
    assert isinstance(env, _DummyWrapperEnv) and env.tag == "C"
    assert env.inner.tag == "B"
    assert env.inner.inner.tag == "A"


def test_non_callable_wrapper_rejected(parallel_env_factory):
    """Registering a non-callable wrapper raises ``TypeError`` eagerly."""
    with pytest.raises(TypeError, match="wrappers"):
        parallel_env_factory(wrappers=(lambda e: e, "not-callable"))


def test_wrappers_compose_with_aec_masked(masked_aec_factory):
    """Wrappers run *after* the turn-based adapter on ``aec_masked`` envs."""
    trace: List[str] = []
    outer = _make_wrapper("outer", trace)

    name = masked_aec_factory(wrappers=(outer,))
    env = build_env(name)

    assert trace == ["outer"]
    # The outermost object is our wrapper; its inner is the adapter output.
    assert isinstance(env, _DummyWrapperEnv) and env.tag == "outer"
    inner = env.inner
    # The adapter output must expose the parallel-shaped contract
    # (``possible_agents`` always present, ``metadata`` preserved).
    assert hasattr(inner, "possible_agents")
    assert len(inner.possible_agents) == 2  # tic-tac-toe is 2-player
