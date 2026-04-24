"""Unit tests for co-evolution opponent pools.

Covers :class:`PopulationPool`, :class:`ArchivePool`,
:class:`FrozenPolicyPool` and :class:`CompositePool`.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines.coevolution import (
    ArchivePool,
    CompositePool,
    ElitistHallOfFame,
    FrozenPolicyPool,
    OpponentPool,
    PopulationPool,
    RingHallOfFame,
    fitness_proportional_sampler,
    latest_sampler,
    tournament_sampler,
    uniform_sampler,
)


class _StubPolicy:
    """Minimal policy double — records which item built it."""

    def __init__(self, tag):
        self.tag = tag

    def get_action(self, observation, action_mask=None):
        return 0

    def reset(self):
        pass


class _Ind:
    """Minimal DEAP-style individual stub."""

    class _F:
        def __init__(self, val, valid=True):
            self.valid = valid
            if valid:
                self.values = (val,)
            # When invalid, omit .values so accessing it raises
            # AttributeError — matching DEAP's del-on-invalidate pattern
            # that PopulationPool._fitnesses() catches.

    def __init__(self, tag, fitness, valid=True):
        self.tag = tag
        self.fitness = _Ind._F(fitness, valid)


def _factory(item):
    return _StubPolicy(getattr(item, "tag", item))


class TestPopulationPool:
    def test_sample_opponent_returns_policy(self):
        pop = [_Ind("a", 1.0), _Ind("b", 2.0)]
        pool = PopulationPool(pop, _factory)
        rng = np.random.default_rng(0)
        p = pool.sample_opponent(rng)
        assert p.tag in {"a", "b"}

    def test_protocol_compliance(self):
        pool = PopulationPool([_Ind("a", 1.0)], _factory)
        assert isinstance(pool, OpponentPool)

    def test_sample_opponents_returns_k(self):
        pool = PopulationPool([_Ind("a", 1.0), _Ind("b", 2.0)], _factory)
        rng = np.random.default_rng(0)
        samples = pool.sample_opponents(rng, 5)
        assert len(samples) == 5
        for p in samples:
            assert p.tag in {"a", "b"}

    def test_empty_pool_raises(self):
        pool = PopulationPool([], _factory)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            pool.sample_opponent(rng)

    def test_invalid_fitness_treated_as_neg_inf(self):
        # Larger pool so tournament(k=4) isn't capped at pool size.
        # One invalid (-inf) + several poor + one strong; tournament
        # must strongly prefer the strong one.
        items = [
            _Ind("bad", 0.0, valid=False),   # -inf
            _Ind("p1", 0.1),
            _Ind("p2", 0.2),
            _Ind("p3", 0.3),
            _Ind("best", 10.0),
        ]
        pool = PopulationPool(items, _factory, sampler=tournament_sampler(4))
        rng = np.random.default_rng(0)
        draws = [pool.sample_opponent(rng).tag for _ in range(400)]
        # Strong individual wins the majority (prob "best" in sample
        # of k=4 from 5 = 1 - (4/5)^4 ≈ 0.59).
        assert draws.count("best") > 200
        # Invalid one never dominates — should appear rarely if at all.
        assert draws.count("bad") < 20

    def test_fitness_proportional_biases_toward_best(self):
        items = [_Ind("a", 0.1), _Ind("b", 100.0)]
        pool = PopulationPool(items, _factory, sampler=fitness_proportional_sampler)
        rng = np.random.default_rng(0)
        draws = [pool.sample_opponent(rng).tag for _ in range(400)]
        # Expect overwhelming bias toward "b".
        assert draws.count("b") > 300

    def test_latest_sampler_returns_last(self):
        items = [_Ind("a", 1.0), _Ind("b", 2.0), _Ind("c", 3.0)]
        pool = PopulationPool(items, _factory, sampler=latest_sampler)
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag == "c"

    def test_update_swaps_population(self):
        pool = PopulationPool([_Ind("a", 1.0)], _factory)
        pool.update([_Ind("x", 5.0)])
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag == "x"


class TestArchivePool:
    def test_reads_from_hof(self):
        hof = RingHallOfFame(capacity=4)
        hof.add(_Ind("a", 1.0), 1.0)
        hof.add(_Ind("b", 2.0), 2.0)
        pool = ArchivePool(hof, _factory, sampler=uniform_sampler)
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag in {"a", "b"}
        assert len(pool) == 2

    def test_empty_archive_raises(self):
        hof = RingHallOfFame(capacity=4)
        pool = ArchivePool(hof, _factory)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            pool.sample_opponent(rng)

    def test_update_is_noop(self):
        hof = RingHallOfFame(capacity=4)
        hof.add(_Ind("a", 1.0), 1.0)
        pool = ArchivePool(hof, _factory)
        pool.update([_Ind("ignored", 99.0)])  # no-op by design
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag == "a"


class TestFrozenPolicyPool:
    def test_samples_from_static_list(self):
        policies = [_StubPolicy("p0"), _StubPolicy("p1")]
        pool = FrozenPolicyPool(policies)
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag in {"p0", "p1"}
        assert len(pool) == 2

    def test_empty_pool_raises_on_sample(self):
        # Construction with an empty list is permitted (the pool may be
        # populated later via update()); only sampling requires at
        # least one policy.
        pool = FrozenPolicyPool([])
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            pool.sample_opponent(rng)

    def test_update_replaces_policies(self):
        pool = FrozenPolicyPool([_StubPolicy("p0")])
        new_policies = [_StubPolicy("new0"), _StubPolicy("new1")]
        pool.update(new_policies)
        assert len(pool) == 2
        rng = np.random.default_rng(0)
        assert pool.sample_opponent(rng).tag.startswith("new")


class TestCompositePool:
    def test_weighted_mixture_zero_weight_never_picked(self):
        pop_a = PopulationPool([_Ind("a", 1.0)], _factory)
        pop_b = PopulationPool([_Ind("b", 1.0)], _factory)
        mix = CompositePool([pop_a, pop_b], weights=(1.0, 0.0))
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert mix.sample_opponent(rng).tag == "a"

    def test_handles_empty_subpool(self):
        pop_a = PopulationPool([_Ind("a", 1.0)], _factory)
        pop_empty = PopulationPool([], _factory)
        mix = CompositePool([pop_a, pop_empty], weights=(0.5, 0.5))
        rng = np.random.default_rng(0)
        # Should route all samples to the non-empty pool.
        for _ in range(20):
            assert mix.sample_opponent(rng).tag == "a"

    def test_all_empty_raises(self):
        pop_empty = PopulationPool([], _factory)
        mix = CompositePool([pop_empty], weights=(1.0,))
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            mix.sample_opponent(rng)

    def test_weights_validation(self):
        pop = PopulationPool([_Ind("a", 1.0)], _factory)
        with pytest.raises(ValueError):
            CompositePool([pop], weights=(0.5, 0.5))  # length mismatch
        with pytest.raises(ValueError):
            CompositePool([pop, pop], weights=(0.0, 0.0))  # all zero
        with pytest.raises(ValueError):
            CompositePool([pop, pop], weights=(-1.0, 1.0))  # negative
