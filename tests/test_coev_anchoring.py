"""Tests for fitness anchoring: the deterministic co-evolved/anchor split.

Anchoring exists because a co-evolutionary fitness is measured against a
moving population, so it cannot say whether a generation got better or its
opponents got worse.  Reserving a fixed share of every evaluation for
opponents that never change gives selection a reference point.  What these
tests pin down is the part that makes it work: the split has to be exactly
the same for every individual, otherwise it re-introduces the sampling noise
it was meant to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines.coevolution.pools import (
    AnchoredPool,
    FrozenPolicyPool,
    OpponentSnapshot,
)
from deap_pettingzoo.baselines.coevolution.sides import Side


class _StubPolicy:
    """Minimal PolicyProtocol carrying an identity tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def get_action(self, observation, deterministic: bool = True):
        del observation, deterministic
        return 0


class _CountingPool:
    """Co-evolved stand-in that reports how often it was drawn from."""

    def __init__(self, n: int = 5) -> None:
        self._policies = [_StubPolicy(f"coev{i}") for i in range(n)]
        self.draws = 0

    def sample_opponent(self, rng):
        self.draws += 1
        return self._policies[int(rng.integers(0, len(self._policies)))]

    def sample_opponents(self, rng, k):
        return [self.sample_opponent(rng) for _ in range(k)]

    def snapshot_opponents(self, rng, k):
        return [
            OpponentSnapshot(kind="frozen", payload={"policy": self.sample_opponent(rng)})
            for _ in range(k)
        ]

    def update(self, population_like):
        del population_like

    def __len__(self):
        return len(self._policies)


def _anchors(n: int = 3) -> FrozenPolicyPool:
    return FrozenPolicyPool([_StubPolicy(f"anchor{i}") for i in range(n)])


def _tags(policies):
    return [p.tag for p in policies]


class TestDeterministicSplit:
    def test_every_individual_gets_exactly_k_anchor_episodes(self):
        """The whole point: the anchor *count* must not vary between individuals.

        A CompositePool weight would give each individual a random number of
        anchor episodes, so two equally skilled individuals could differ in
        fitness purely by drawing different opponent mixes — exactly the
        selection noise anchoring is meant to remove.
        """
        k, k_anchor, n_individuals = 8, 2, 20
        coev = _CountingPool()
        pool = AnchoredPool(coev, _anchors(), k_opponents=k, k_anchor=k_anchor)
        rng = np.random.default_rng(0)

        for _ in range(n_individuals):
            drawn = _tags(pool.sample_opponents(rng, k))
            n_anchored = sum(1 for t in drawn if t.startswith("anchor"))
            assert n_anchored == k_anchor

        assert coev.draws == n_individuals * (k - k_anchor)

    def test_anchor_identity_is_the_same_for_every_individual(self):
        """Common random numbers on the opponent dimension.

        Slot j must yield the same anchor for everyone in the generation, so
        a fitness difference on the anchored episodes is a skill difference.
        """
        k, k_anchor = 6, 3
        pool = AnchoredPool(
            _CountingPool(), _anchors(3), k_opponents=k, k_anchor=k_anchor
        )
        rng = np.random.default_rng(0)

        blocks = [
            [t for t in _tags(pool.sample_opponents(rng, k)) if t.startswith("anchor")]
            for _ in range(5)
        ]
        assert blocks[0] == ["anchor0", "anchor1", "anchor2"]
        assert all(block == blocks[0] for block in blocks)

    def test_anchors_land_in_the_same_slots_for_every_individual(self):
        k, k_anchor = 4, 2
        pool = AnchoredPool(
            _CountingPool(), _anchors(2), k_opponents=k, k_anchor=k_anchor
        )
        rng = np.random.default_rng(0)
        for _ in range(4):
            drawn = _tags(pool.sample_opponents(rng, k))
            assert [t.startswith("anchor") for t in drawn] == [True, True, False, False]

    def test_batch_draw_splits_per_k_block(self):
        """The population-parallel path asks for ``k * n`` in one call."""
        k, k_anchor, n = 5, 2, 7
        coev = _CountingPool()
        pool = AnchoredPool(coev, _anchors(), k_opponents=k, k_anchor=k_anchor)
        drawn = _tags(pool.sample_opponents(np.random.default_rng(0), k * n))

        for i in range(n):
            block = drawn[i * k:(i + 1) * k]
            assert sum(1 for t in block if t.startswith("anchor")) == k_anchor
        assert coev.draws == n * (k - k_anchor)

    def test_single_draws_hold_the_fraction_across_a_generation(self):
        """RL sides draw one opponent per episode, not k per individual.

        There is no population to make mutually comparable there, so the
        meaningful guarantee is the anchor *fraction* over the generation.
        """
        k, k_anchor, episodes = 4, 1, 40
        pool = AnchoredPool(_CountingPool(), _anchors(), k_opponents=k, k_anchor=k_anchor)
        rng = np.random.default_rng(0)
        drawn = [pool.sample_opponent(rng).tag for _ in range(episodes)]
        assert sum(1 for t in drawn if t.startswith("anchor")) == episodes * k_anchor // k

    def test_snapshots_follow_the_same_split(self):
        """Workers must see the split the serial path produces."""
        k, k_anchor, n = 4, 1, 3
        pool = AnchoredPool(_CountingPool(), _anchors(), k_opponents=k, k_anchor=k_anchor)
        snaps = pool.snapshot_opponents(np.random.default_rng(0), k * n)

        assert len(snaps) == k * n
        for i in range(n):
            block = snaps[i * k:(i + 1) * k]
            tags = [s.payload["policy"].tag for s in block]
            assert sum(1 for t in tags if t.startswith("anchor")) == k_anchor

    def test_registered_anchors_ship_by_name(self):
        """Non-picklable anchors must travel as a registry name, not an object."""
        anchors = FrozenPolicyPool(
            [_StubPolicy("a0"), _StubPolicy("a1")],
            registry_names=["anchor_a0", "anchor_a1"],
        )
        pool = AnchoredPool(_CountingPool(), anchors, k_opponents=2, k_anchor=1)
        snaps = pool.snapshot_opponents(np.random.default_rng(0), 2)

        assert snaps[0].kind == "frozen_registered"
        assert snaps[0].payload == {"name": "anchor_a0"}


class TestDegenerateSettings:
    def test_zero_anchor_is_pure_coevolution(self):
        coev = _CountingPool()
        pool = AnchoredPool(coev, _anchors(), k_opponents=4, k_anchor=0)
        drawn = _tags(pool.sample_opponents(np.random.default_rng(0), 8))
        assert all(t.startswith("coev") for t in drawn)
        assert coev.draws == 8

    def test_full_anchor_never_touches_the_coevolved_pool(self):
        coev = _CountingPool()
        pool = AnchoredPool(coev, _anchors(), k_opponents=4, k_anchor=4)
        drawn = _tags(pool.sample_opponents(np.random.default_rng(0), 8))
        assert all(t.startswith("anchor") for t in drawn)
        assert coev.draws == 0

    @pytest.mark.parametrize("k_anchor", [-1, 5])
    def test_out_of_range_k_anchor_rejected(self, k_anchor):
        with pytest.raises(ValueError, match="k_anchor"):
            AnchoredPool(
                _CountingPool(), _anchors(), k_opponents=4, k_anchor=k_anchor
            )

    def test_empty_anchor_set_rejected(self):
        with pytest.raises(ValueError, match="anchor pool is empty"):
            AnchoredPool(
                _CountingPool(), FrozenPolicyPool([]), k_opponents=4, k_anchor=1
            )

    def test_update_forwards_to_the_coevolved_pool_only(self):
        """Anchors are fixed by definition; a generation update must not move them."""
        anchors = _anchors(2)
        pool = AnchoredPool(_CountingPool(), anchors, k_opponents=2, k_anchor=1)
        pool.update([_StubPolicy("new")])
        assert len(anchors) == 2
        assert _tags([anchors.policy_at(0), anchors.policy_at(1)]) == [
            "anchor0",
            "anchor1",
        ]


class TestSideValidation:
    def test_anchor_mix_requires_policies(self):
        with pytest.raises(ValueError, match="requires a non-empty anchor_policies"):
            Side(name="prey", role_prefixes=("prey",), anchor_mix=0.25)

    @pytest.mark.parametrize("mix", [-0.1, 1.5])
    def test_anchor_mix_range_enforced(self, mix):
        with pytest.raises(ValueError, match="anchor_mix must be in"):
            Side(
                name="prey",
                role_prefixes=("prey",),
                anchor_mix=mix,
                anchor_policies=[_StubPolicy("a")],
            )

    def test_registry_names_must_match_policy_count(self):
        with pytest.raises(ValueError, match="anchor_registry_names"):
            Side(
                name="prey",
                role_prefixes=("prey",),
                anchor_mix=0.5,
                anchor_policies=[_StubPolicy("a"), _StubPolicy("b")],
                anchor_registry_names=["only_one"],
            )

    def test_default_side_is_unanchored(self):
        side = Side(name="prey", role_prefixes=("prey",))
        assert side.anchor_mix == 0.0
        assert side.anchor_policies is None


class TestTrainerWiring:
    """``_anchor_pool`` is where a Side declaration becomes a live pool."""

    @staticmethod
    def _cfg(k: int):
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionConfig

        return CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="predators", role_prefixes=("predator",)),
                Side(name="prey", role_prefixes=("prey",)),
            ],
            k_opponents=k,
        )

    def test_unanchored_side_returns_the_pool_untouched(self):
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        pool = _CountingPool()
        side = Side(name="prey", role_prefixes=("prey",))
        assert CoEvolutionTrainer._anchor_pool(pool, side, self._cfg(4)) is pool

    def test_mix_is_rounded_to_whole_episodes(self):
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        side = Side(
            name="prey",
            role_prefixes=("prey",),
            anchor_mix=0.25,
            anchor_policies=[_StubPolicy("a")],
        )
        wrapped = CoEvolutionTrainer._anchor_pool(
            _CountingPool(), side, self._cfg(16)
        )
        assert isinstance(wrapped, AnchoredPool)
        assert wrapped.k_anchor == 4

    def test_a_declared_anchor_never_rounds_away_to_nothing(self):
        """A silently inert anchor would be indistinguishable from a working one."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        side = Side(
            name="prey",
            role_prefixes=("prey",),
            anchor_mix=0.01,
            anchor_policies=[_StubPolicy("a")],
        )
        wrapped = CoEvolutionTrainer._anchor_pool(_CountingPool(), side, self._cfg(4))
        assert wrapped.k_anchor == 1


class TestEndToEnd:
    """An anchored run has to actually train, not just build a pool."""

    @staticmethod
    def _run(anchor_mix: float, *, common_random_seeds: bool = False):
        from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        pytest.importorskip("pettingzoo")
        tiny = {"population_size": 4, "hidden_size": 8, "seed": 0}
        prey_kwargs = {}
        if anchor_mix > 0.0:
            prey_kwargs = {
                "anchor_mix": anchor_mix,
                "anchor_policies": [_StubPolicy("a0"), _StubPolicy("a1")],
            }
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pred", ("adversary",), "ea", dict(tiny)),
                Side("prey", ("agent",), "ea", dict(tiny), **prey_kwargs),
            ],
            k_opponents=4,
            common_random_seeds=common_random_seeds,
            max_steps_per_episode=5,
        )
        return CoEvolutionTrainer().train(
            TrainingConfig(
                env_name="simple_tag_v3",
                budget_mode=BudgetMode.EVAL_EPISODES,
                budget_value=40,
                coevolution_config=cfg,
            )
        )

    def test_anchored_run_completes_and_reports_its_episode_split(self):
        result = self._run(0.5)
        assert result.policy is not None
        # Realized episodes, not the requested fraction — this is the number
        # that says what the run actually trained against.
        assert result.metadata["anchor_episodes"] == {"prey": 2}

    def test_unanchored_run_reports_no_anchoring(self):
        result = self._run(0.0)
        assert result.policy is not None
        assert result.metadata["anchor_episodes"] is None

    def test_common_random_seeds_is_recorded(self):
        assert self._run(0.0, common_random_seeds=True).metadata[
            "common_random_seeds"
        ] is True
        assert self._run(0.0).metadata["common_random_seeds"] is False
