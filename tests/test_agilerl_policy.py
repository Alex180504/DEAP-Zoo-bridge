"""Unit tests for :class:`AgileRLPolicyWrapper`.

Specifically targets the action-shape squeeze loop in
:meth:`AgileRLPolicyWrapper.get_action` — the same lineage of bug
that produced the 2-vs-3 tensor-dim mismatch in
``RLSideState`` was a *missing* squeeze; here we test the
analogous *over-applied* squeeze risk: the wrapper must return
actions with the shape an environment expects (no leading vec dim,
scalar `.item()` for 0-d) regardless of what the underlying agent
returns.

Uses a stub agent — no torch, no AgileRL install required.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper


class _StubAgent:
    """Minimal AgileRL-shaped agent.

    Always returns a fixed action ndarray for every requested
    ``agent_id``, with shape controlled by the test.
    """

    def __init__(self, action_value: np.ndarray, agent_ids):
        self._action_value = action_value
        self.agent_ids = list(agent_ids)
        # Mimic the AgileRL surface the wrapper looks up.
        self.observation_spaces = {a: None for a in self.agent_ids}

    def get_action(self, obs_dict, training=False):  # noqa: D401
        # Return-type matches MADDPG: tuple (actions_dict, raw_actions_dict).
        actions = {a: self._action_value for a in obs_dict}
        return actions, actions


@pytest.mark.parametrize(
    "raw_shape, expected_ndim",
    [
        ((1, 5), 1),     # leading vec dim of size 1 over a 5-vector
        ((1, 1, 5), 1),  # double-leading vec dim
        ((5,), 1),       # already 1-D — must stay 1-D
        ((1,), 0),       # (1,) collapses to scalar (.item())
    ],
)
def test_get_action_squeezes_leading_vec_dim(raw_shape, expected_ndim):
    """Continuous action: leading size-1 dims must be stripped, but
    intrinsic feature dims must be preserved.
    """
    raw = np.arange(int(np.prod(raw_shape)), dtype=np.float32).reshape(raw_shape)
    agent = _StubAgent(raw, agent_ids=["a"])
    wrapper = AgileRLPolicyWrapper(agent, agent_ids=["a"])

    obs = np.zeros(3, dtype=np.float32)
    out = wrapper.get_action(obs)

    if expected_ndim == 0:
        # 0-d arrays return a Python scalar via .item().
        assert np.isscalar(out) or (
            isinstance(out, np.generic) and out.ndim == 0
        ), f"expected scalar, got {type(out)} shape={getattr(out, 'shape', None)}"
    else:
        out_arr = np.asarray(out)
        assert out_arr.ndim == expected_ndim, (
            f"expected ndim={expected_ndim}, got ndim={out_arr.ndim} "
            f"shape={out_arr.shape} for raw_shape={raw_shape}"
        )


def test_get_action_discrete_scalar_stays_scalar():
    """A 0-d (true scalar) action must come back as a Python scalar."""
    raw = np.asarray(3, dtype=np.int64)  # ndim==0
    agent = _StubAgent(raw, agent_ids=["a"])
    wrapper = AgileRLPolicyWrapper(agent, agent_ids=["a"])

    out = wrapper.get_action(np.zeros(2, dtype=np.float32))
    # .item() on 0-d returns a Python int.
    assert isinstance(out, int)
    assert out == 3


def test_get_action_rotates_through_agent_ids():
    """The wrapper round-robins through ``agent_ids`` once per call;
    after N calls every agent has been dispatched exactly once for
    a wrapper with N agents.
    """
    raw = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    agent = _StubAgent(raw, agent_ids=["a", "b", "c"])
    wrapper = AgileRLPolicyWrapper(agent, agent_ids=["a", "b", "c"])

    obs = np.zeros(2, dtype=np.float32)
    for _ in range(3):
        wrapper.get_action(obs)
    # Cache must hold one entry per agent post-rotation.
    assert set(wrapper._obs_cache.keys()) == {"a", "b", "c"}


def test_reset_clears_rotation_and_cache():
    raw = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    agent = _StubAgent(raw, agent_ids=["a", "b"])
    wrapper = AgileRLPolicyWrapper(agent, agent_ids=["a", "b"])

    wrapper.get_action(np.zeros(2, dtype=np.float32))
    assert wrapper._call_index == 1
    assert wrapper._obs_cache  # populated

    wrapper.reset()
    assert wrapper._call_index == 0
    assert wrapper._obs_cache == {}
