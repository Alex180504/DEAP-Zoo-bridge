"""Tests for :mod:`deap_pettingzoo.policies.action_heads`.

Covers :class:`DiscreteHead`, :class:`BoxHead` (tanh-squash + rescale),
:class:`MultiDiscreteHead` (per-sub-action argmax), and :func:`make_head`
— including the deliberate ``NotImplementedError`` for :class:`Tuple` /
:class:`Dict` spaces.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from deap_pettingzoo.policies.action_heads import (
    BoxHead,
    DiscreteHead,
    MultiDiscreteHead,
    make_head,
)


class TestDiscreteHead:
    def test_argmax(self):
        head = DiscreteHead(n=4)
        out = head.decode(np.array([0.1, 0.9, 0.2, 0.5]))
        assert out == 1
        assert isinstance(out, int)

    def test_wrong_size_raises(self):
        head = DiscreteHead(n=4)
        with pytest.raises(ValueError, match="size 4"):
            head.decode(np.array([0.1, 0.2]))

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            DiscreteHead(n=0)


class TestBoxHead:
    def test_tanh_rescale(self):
        head = BoxHead(low=np.array([-2.0, 0.0]), high=np.array([2.0, 4.0]))
        out = head.decode(np.array([0.0, 0.0]))
        # tanh(0) = 0 -> midpoint of [low, high]
        np.testing.assert_allclose(out, [0.0, 2.0], atol=1e-6)

    def test_bounds_respected(self):
        head = BoxHead(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        # Very large raw inputs should saturate near the bounds.
        out = head.decode(np.array([1e6, -1e6]))
        assert out[0] == pytest.approx(1.0, abs=1e-4)
        assert out[1] == pytest.approx(-1.0, abs=1e-4)

    def test_dtype_is_float32(self):
        head = BoxHead(low=np.array([-1.0]), high=np.array([1.0]))
        out = head.decode(np.array([0.5]))
        assert out.dtype == np.float32

    def test_infinite_bounds_fallback(self):
        head = BoxHead(
            low=np.array([-np.inf, 0.0]),
            high=np.array([np.inf, 5.0]),
        )
        out = head.decode(np.array([0.0, 0.0]))
        assert np.isfinite(out).all()


class TestMultiDiscreteHead:
    def test_output_size_is_sum_of_nvec(self):
        head = MultiDiscreteHead([3, 4, 2])
        assert head.output_size == 9

    def test_per_subaction_argmax(self):
        head = MultiDiscreteHead([3, 4, 2])
        # chunk0=[0,9,0]->1, chunk1=[0,0,0,7]->3, chunk2=[5,0]->0
        logits = np.array([0, 9, 0, 0, 0, 0, 7, 5, 0], dtype=np.float32)
        out = head.decode(logits)
        np.testing.assert_array_equal(out, [1, 3, 0])

    def test_decode_returns_int_array_in_range(self):
        head = MultiDiscreteHead([5, 3])
        rng = np.random.default_rng(0)
        for _ in range(20):
            out = head.decode(rng.standard_normal(head.output_size))
            assert out.dtype == np.int64
            assert out.shape == (2,)
            assert 0 <= out[0] < 5 and 0 <= out[1] < 3

    def test_wrong_size_raises(self):
        head = MultiDiscreteHead([3, 4])
        with pytest.raises(ValueError, match="size 7"):
            head.decode(np.zeros(5))

    def test_empty_nvec_raises(self):
        with pytest.raises(ValueError):
            MultiDiscreteHead([])

    def test_invalid_nvec_entry_raises(self):
        with pytest.raises(ValueError):
            MultiDiscreteHead([3, 0])


class TestMakeHead:
    def test_discrete_space(self):
        head = make_head(spaces.Discrete(5))
        assert isinstance(head, DiscreteHead)
        assert head.n == 5

    def test_box_space(self):
        head = make_head(spaces.Box(low=-1.0, high=1.0, shape=(3,)))
        assert isinstance(head, BoxHead)
        assert head.output_size == 3

    def test_multidiscrete_space(self):
        head = make_head(spaces.MultiDiscrete([5, 3]))
        assert isinstance(head, MultiDiscreteHead)
        assert head.output_size == 8

    def test_tuple_raises(self):
        with pytest.raises(NotImplementedError):
            make_head(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))))

    def test_none_raises(self):
        with pytest.raises(TypeError):
            make_head(None)


class TestFeedforwardNeuralPolicyWithHead:
    def test_box_space_produces_valid_action(self):
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        genome = FixedTopologyGenome([4, 8, 2], activation="tanh")
        action_space = spaces.Box(low=-3.0, high=3.0, shape=(2,))
        policy = FeedforwardNeuralPolicy(genome, action_space=action_space)
        action = policy.get_action(np.zeros(4, dtype=np.float32))
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert (action >= -3.0).all() and (action <= 3.0).all()

    def test_discrete_space_produces_int(self):
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        genome = FixedTopologyGenome([4, 8, 3], activation="tanh")
        policy = FeedforwardNeuralPolicy(genome, action_space=spaces.Discrete(3))
        action = policy.get_action(np.zeros(4, dtype=np.float32))
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_multidiscrete_space_produces_int_array(self):
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        # nvec [3, 4] -> sum(nvec) = 7 output units.
        genome = FixedTopologyGenome([4, 8, 7], activation="tanh")
        policy = FeedforwardNeuralPolicy(
            genome, action_space=spaces.MultiDiscrete([3, 4])
        )
        action = policy.get_action(np.zeros(4, dtype=np.float32))
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.int64
        assert action.shape == (2,)
        assert 0 <= action[0] < 3 and 0 <= action[1] < 4

    def test_no_head_returns_raw_output(self):
        """Backwards-compat: without ``action_space`` the raw array is returned."""
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        genome = FixedTopologyGenome([4, 8, 2], activation="tanh")
        policy = FeedforwardNeuralPolicy(genome)
        action = policy.get_action(np.zeros(4, dtype=np.float32))
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
