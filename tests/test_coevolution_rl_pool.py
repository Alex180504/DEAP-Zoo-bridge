"""Tests for RLPolicyPool — the degenerate size-1 opponent pool."""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines.coevolution.pools import RLPolicyPool


class _StubPolicy:
    """Minimal PolicyProtocol stand-in for pool tests."""

    def __init__(self, name: str = "p"):
        self.name = name

    def get_action(self, obs, *, deterministic=False, action_mask=None):
        return 0

    def reset(self):
        pass


class TestRLPolicyPool:
    def test_init_rejects_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            RLPolicyPool(None)

    def test_sample_opponent_returns_cached_policy(self):
        p = _StubPolicy("a")
        pool = RLPolicyPool(p)
        rng = np.random.default_rng(0)
        for _ in range(3):
            assert pool.sample_opponent(rng) is p

    def test_sample_opponents_returns_list_of_same_object(self):
        p = _StubPolicy("a")
        pool = RLPolicyPool(p)
        sampled = pool.sample_opponents(np.random.default_rng(0), 5)
        assert len(sampled) == 5
        assert all(s is p for s in sampled)

    def test_sample_opponents_rejects_zero_k(self):
        pool = RLPolicyPool(_StubPolicy())
        with pytest.raises(ValueError, match="k must be >= 1"):
            pool.sample_opponents(np.random.default_rng(0), 0)

    def test_update_swaps_policy(self):
        p1, p2 = _StubPolicy("a"), _StubPolicy("b")
        pool = RLPolicyPool(p1)
        assert pool.policy is p1
        pool.update(p2)
        assert pool.policy is p2
        assert pool.sample_opponent(np.random.default_rng(0)) is p2

    def test_update_rejects_none(self):
        pool = RLPolicyPool(_StubPolicy())
        with pytest.raises(ValueError, match="must not be None"):
            pool.update(None)

    def test_len_is_one(self):
        pool = RLPolicyPool(_StubPolicy())
        assert len(pool) == 1

    def test_snapshot_opponents_raises(self):
        """Parallel co-evolution with RL is not supported in v1."""
        pool = RLPolicyPool(_StubPolicy())
        with pytest.raises(NotImplementedError, match="not yet supported"):
            pool.snapshot_opponents(np.random.default_rng(0), 2)

    def test_snapshot_sync_returns_independent_pool(self):
        """snapshot_sync() must produce a pool whose contents do not
        change when the source pool is later updated."""
        p1 = _StubPolicy("a")
        pool = RLPolicyPool(p1)
        snapshot = pool.snapshot_sync()

        # Snapshot must be a different RLPolicyPool instance.
        assert snapshot is not pool

        # Now mutate the original.
        p2 = _StubPolicy("b")
        pool.update(p2)

        # The snapshot's policy must still reflect the pre-update state.
        # _StubPolicy is deepcopy-able, so the snapshot got an
        # independent (deep-copied) policy whose .name == "a".
        snap_policy = snapshot.sample_opponent(np.random.default_rng(0))
        assert snap_policy.name == "a"
        # And the live pool advanced.
        assert pool.sample_opponent(np.random.default_rng(0)) is p2
