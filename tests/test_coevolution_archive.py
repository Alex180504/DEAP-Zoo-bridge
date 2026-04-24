"""Unit tests for Hall-of-Fame archives."""

from __future__ import annotations

import pytest

from deap_pettingzoo.baselines.coevolution import (
    ElitistHallOfFame,
    HallOfFame,
    RingHallOfFame,
)


class TestRingHallOfFame:
    def test_fifo_eviction(self):
        import numpy as np
        hof = RingHallOfFame(capacity=3)
        for i in range(5):
            hof.add(f"item_{i}", float(i))
        assert len(hof) == 3
        # Oldest two (item_0, item_1) evicted; newest three remain.
        assert hof.items == ["item_2", "item_3", "item_4"]
        np.testing.assert_array_equal(hof.fitnesses, np.array([2.0, 3.0, 4.0]))

    def test_extend(self):
        hof = RingHallOfFame(capacity=4)
        hof.extend(["a", "b", "c"], [1.0, 2.0, 3.0])
        assert len(hof) == 3
        assert hof.items == ["a", "b", "c"]

    def test_is_empty_clear(self):
        hof = RingHallOfFame(capacity=2)
        assert hof.is_empty
        hof.add("x", 1.0)
        assert not hof.is_empty
        hof.clear()
        assert hof.is_empty and len(hof) == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            RingHallOfFame(capacity=0)
        with pytest.raises(ValueError):
            RingHallOfFame(capacity=-1)


class TestElitistHallOfFame:
    def test_keeps_top_k(self):
        hof = ElitistHallOfFame(capacity=3)
        for item, fit in [("a", 1.0), ("b", 5.0), ("c", 3.0), ("d", 10.0)]:
            hof.add(item, fit)
        assert len(hof) == 3
        # Smallest ("a", 1.0) is evicted.
        assert set(hof.items) == {"b", "c", "d"}
        assert sorted(hof.fitnesses) == [3.0, 5.0, 10.0]

    def test_rejects_lesser_when_full(self):
        hof = ElitistHallOfFame(capacity=2)
        hof.add("a", 5.0)
        hof.add("b", 10.0)
        hof.add("c", 1.0)  # lower than both — rejected
        assert set(hof.items) == {"a", "b"}

    def test_replaces_only_strictly_greater(self):
        hof = ElitistHallOfFame(capacity=2)
        hof.add("a", 5.0)
        hof.add("b", 5.0)
        # Adding another at same min value should NOT displace.
        hof.add("c", 5.0)
        assert len(hof) == 2
        assert "c" not in hof.items

    def test_polymorphic_with_base(self):
        hof: HallOfFame = ElitistHallOfFame(capacity=2)
        hof.add("a", 1.0)
        assert isinstance(hof, HallOfFame)
