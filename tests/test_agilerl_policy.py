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


def test_get_action_multidiscrete_vector_preserved():
    """A MultiDiscrete action ``(1, k)`` must come back as a ``(k,)`` vector."""
    raw = np.array([[1, 2]], dtype=np.int64)  # (1, 2) batched MultiDiscrete action
    agent = _StubAgent(raw, agent_ids=["a"])
    wrapper = AgileRLPolicyWrapper(agent, agent_ids=["a"])

    out = np.asarray(wrapper.get_action(np.zeros(3, dtype=np.float32)))
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, [1, 2])


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


# ---------------------------------------------------------------------------
# FrozenRLPolicyWrapper tests (torch-free, using nn.Linear stubs)
# ---------------------------------------------------------------------------


class _TinyLinear:
    """Minimal nn.Module stand-in that does a deterministic linear transform."""

    def __init__(self, in_f: int, out_f: int):
        np.random.seed(0)
        self._W = np.random.randn(out_f, in_f).astype(np.float32)
        self._b = np.zeros(out_f, dtype=np.float32)
        self.training = False

    def __call__(self, x_tensor):
        # x_tensor assumed to be a (1, in_f) numpy array mimic.
        return (x_tensor @ self._W.T) + self._b

    def eval(self):
        self.training = False
        return self

    def cpu(self):
        return self

    def requires_grad_(self, flag: bool):
        return self


def _make_frozen_wrapper(n_agents: int = 2, is_discrete: bool = False):
    """Construct a FrozenRLPolicyWrapper backed by _TinyLinear actors."""
    import importlib
    import sys

    # Provide a minimal torch stand-in so FrozenRLPolicyWrapper can import
    # without the real torch installed in test environments.
    # We mock torch.no_grad as a context manager.
    import contextlib

    class _FakeTorch:
        class Tensor:
            def __init__(self, arr):
                self._arr = np.asarray(arr, dtype=np.float32)

            def unsqueeze(self, dim):
                return _FakeTorch.Tensor(np.expand_dims(self._arr, axis=dim))

            def detach(self):
                return self

            def numpy(self):
                return self._arr

        @staticmethod
        @contextlib.contextmanager
        def no_grad():
            yield

        @staticmethod
        def tensor(arr, dtype=None):
            return _FakeTorch.Tensor(np.asarray(arr, dtype=np.float32))

        float32 = np.float32

    # Only inject the fake if torch is not already importable.
    _torch_was_missing = "torch" not in sys.modules
    if _torch_was_missing:
        sys.modules["torch"] = _FakeTorch()  # type: ignore[assignment]
        sys.modules["torch.nn"] = type(sys)("torch.nn")  # type: ignore

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    if _torch_was_missing:
        # Clean up so other tests are not affected.
        del sys.modules["torch"]
        if "torch.nn" in sys.modules:
            del sys.modules["torch.nn"]

    actors = [_TinyLinear(4, 2) for _ in range(n_agents)]
    agent_ids = [f"agent_{i}" for i in range(n_agents)]
    return FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=agent_ids,
        full_roster=agent_ids,
        algorithm_name="matd3",
        is_discrete=is_discrete,
    )


try:
    import pickle

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    _FROZEN_AVAILABLE = True
except ImportError:
    _FROZEN_AVAILABLE = False

_skip_frozen = pytest.mark.skipif(
    not _FROZEN_AVAILABLE,
    reason="FrozenRLPolicyWrapper not importable (torch missing?)",
)


@_skip_frozen
def test_frozen_wrapper_properties():
    import pickle

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper
    import torch
    import torch.nn as nn

    actors = [nn.Linear(4, 2) for _ in range(2)]
    agent_ids = ["a_0", "a_1"]
    fw = FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=agent_ids,
        full_roster=agent_ids,
        algorithm_name="matd3",
        is_discrete=False,
    )
    assert fw.algorithm_name == "matd3"
    assert fw.agent_ids == agent_ids
    assert not fw.is_discrete


@_skip_frozen
def test_frozen_wrapper_get_action_continuous():
    """Continuous actions: get_action must return an ndarray (1-D)."""
    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    actors = [nn.Linear(4, 2) for _ in range(1)]
    fw = FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=["a_0"],
        full_roster=["a_0"],
        algorithm_name="maddpg",
        is_discrete=False,
    )
    obs = np.zeros(4, dtype=np.float32)
    act = fw.get_action(obs)
    arr = np.asarray(act)
    assert arr.ndim == 1


@_skip_frozen
def test_frozen_wrapper_get_action_discrete():
    """Discrete actions: get_action must return a Python int (argmax)."""
    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    actors = [nn.Linear(4, 3) for _ in range(1)]
    fw = FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=["a_0"],
        full_roster=["a_0"],
        algorithm_name="matd3",
        is_discrete=True,
    )
    obs = np.zeros(4, dtype=np.float32)
    act = fw.get_action(obs)
    assert isinstance(act, int)


@_skip_frozen
def test_frozen_wrapper_reset():
    """reset() must clear call_index and obs_cache."""
    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    actors = [nn.Linear(4, 2) for _ in range(2)]
    agent_ids = ["a_0", "a_1"]
    fw = FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=agent_ids,
        full_roster=agent_ids,
        algorithm_name="matd3",
        is_discrete=False,
    )
    obs = np.zeros(4, dtype=np.float32)
    fw.get_action(obs)
    fw.get_action(obs)
    assert fw._call_index == 2

    fw.reset()
    assert fw._call_index == 0
    assert fw._obs_cache == {}


@_skip_frozen
def test_frozen_wrapper_get_action_multidiscrete():
    """Stochastic (PPO) actors return ``(action, log_prob, entropy)`` with the
    action already sampled; the frozen wrapper must shape a MultiDiscrete
    action as an int vector of length ``len(nvec)``.
    """
    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    class _StochActor(nn.Module):
        def __init__(self, action_vec):
            super().__init__()
            self.register_buffer(
                "_a", torch.tensor([action_vec], dtype=torch.long)  # (1, k)
            )
            self.lin = nn.Linear(4, 1)  # gives the module parameters

        def forward(self, obs, action_mask=None):
            return self._a, torch.zeros(1), torch.zeros(1)

    actor = _StochActor([1, 2])
    fw = FrozenRLPolicyWrapper(
        actors=[actor],
        agent_ids=["agent_0"],
        full_roster=["agent_0"],
        algorithm_name="ippo",
        is_discrete=False,
        multidiscrete_nvecs=[(3, 4)],
    )
    out = np.asarray(fw.get_action(np.zeros(4, dtype=np.float32)))
    assert out.dtype == np.int64
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, [1, 2])


@_skip_frozen
def test_frozen_wrapper_get_action_stochastic_discrete_int():
    """Stochastic actor + Discrete slot returns a Python int."""
    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    class _StochActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, obs, action_mask=None):
            return torch.tensor([2]), torch.zeros(1), torch.zeros(1)  # (1,) sample

    fw = FrozenRLPolicyWrapper(
        actors=[_StochActor()],
        agent_ids=["agent_0"],
        full_roster=["agent_0"],
        algorithm_name="ippo",
        is_discrete=True,
    )
    out = fw.get_action(np.zeros(4, dtype=np.float32))
    assert isinstance(out, int)
    assert out == 2


@_skip_frozen
def test_frozen_wrapper_picklable():
    """FrozenRLPolicyWrapper with CPU actors must survive pickle round-trip."""
    import pickle

    import torch
    import torch.nn as nn

    from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper

    actors = [nn.Linear(4, 2) for _ in range(2)]
    fw = FrozenRLPolicyWrapper(
        actors=actors,
        agent_ids=["a_0", "a_1"],
        full_roster=["a_0", "a_1"],
        algorithm_name="matd3",
        is_discrete=False,
    )
    data = pickle.dumps(fw)
    fw2 = pickle.loads(data)
    assert fw2.algorithm_name == "matd3"
    obs = np.zeros(4, dtype=np.float32)
    act = fw2.get_action(obs)
    assert act is not None
