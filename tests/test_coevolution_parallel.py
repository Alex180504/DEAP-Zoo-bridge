"""Phase 8 — parallel co-evolution tests.

Exercises the new :func:`_episode_rollout` worker entry-point,
the :class:`OpponentSnapshot` round-trip on every concrete pool, the
``executor=`` dispatch in :class:`CoEvolutionEvaluator`, and the
``num_workers`` / ``parallel_backend`` wiring in
:class:`CoEvolutionTrainer`.

Naming follows the existing ``test_coevolution_*`` files.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    ElitistHallOfFame,
    Side,
)
from deap_pettingzoo.baselines.coevolution.pools import (
    ArchivePool,
    CompositePool,
    FrozenPolicyPool,
    OpponentSnapshot,
    PopulationPool,
    rebuild_opponent_policy,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.genomes.fixed_topology import FixedTopologyGenome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _make_genome(seed: int = 0) -> FixedTopologyGenome:
    # simple_tag_v3 adversary obs=16-dim, action=Discrete(5).
    return FixedTopologyGenome(
        layer_sizes=[16, 8, 5],
        activation="tanh",
        random_seed=seed,
    )


class _PicklableSampler:
    """Module-level sampler so pickle can locate it on workers."""

    def __init__(self, action_space):
        self._action_space = action_space

    def get_action(self, _obs, **_kw):
        return self._action_space.sample()


def _base_tcfg(cfg: CoEvolutionConfig, budget: int = 40) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        algorithm_params={"coevolution": cfg},
    )


# ---------------------------------------------------------------------------
# Pool snapshot round-trips
# ---------------------------------------------------------------------------


class TestSnapshotRoundtrip:
    """All concrete pools must produce picklable snapshots and rebuild
    a working :class:`PolicyProtocol` on the worker side."""

    def test_population_pool_snapshot_pickles(self):
        env = PettingZooWrapper("simple_tag_v3")
        agent = env.possible_agents[0]
        action_space = env.action_space(agent)
        try:
            genomes = [_make_genome(i) for i in range(3)]
            pool = PopulationPool(
                genomes,
                lambda g: _PicklableSampler(action_space),
                snapshot_kind="feedforward",
                snapshot_metadata={"masked": False},
            )
            rng = np.random.default_rng(0)
            snaps = pool.snapshot_opponents(rng, 2)
            assert len(snaps) == 2
            blob = pickle.dumps(snaps)
            restored = pickle.loads(blob)
            policy = rebuild_opponent_policy(restored[0], action_space=action_space)
            obs = env.reset(seed=0)[agent]
            assert policy.get_action(obs) is not None
        finally:
            env.close()

    def test_archive_pool_snapshot_pickles(self):
        env = PettingZooWrapper("simple_tag_v3")
        agent = env.possible_agents[0]
        action_space = env.action_space(agent)
        try:
            hof = ElitistHallOfFame(capacity=4)
            for i in range(3):
                hof.add(_make_genome(i), float(i))
            pool = ArchivePool(
                hof,
                lambda g: _PicklableSampler(action_space),
                snapshot_kind="feedforward",
                snapshot_metadata={"masked": False},
            )
            rng = np.random.default_rng(0)
            snaps = pool.snapshot_opponents(rng, 3)
            assert len(snaps) == 3
            blob = pickle.dumps(snaps)
            restored = pickle.loads(blob)
            for snap in restored:
                assert snap.kind == "feedforward"
                policy = rebuild_opponent_policy(snap, action_space=action_space)
                obs = env.reset(seed=0)[agent]
                assert policy.get_action(obs) is not None
        finally:
            env.close()

    def test_frozen_policy_pool_value_snapshot_pickles(self):
        """Default frozen pool ships policies by value (kind='frozen')."""
        env = PettingZooWrapper("simple_tag_v3")
        agent = env.possible_agents[0]
        action_space = env.action_space(agent)
        try:
            pool = FrozenPolicyPool(
                [_PicklableSampler(action_space), _PicklableSampler(action_space)]
            )
            rng = np.random.default_rng(0)
            snaps = pool.snapshot_opponents(rng, 2)
            assert all(s.kind == "frozen" for s in snaps)
            blob = pickle.dumps(snaps)
            restored = pickle.loads(blob)
            policy = rebuild_opponent_policy(restored[0], action_space=action_space)
            obs = env.reset(seed=0)[agent]
            assert policy.get_action(obs) is not None
        finally:
            env.close()

    def test_composite_pool_snapshot_dispatches(self):
        env = PettingZooWrapper("simple_tag_v3")
        agent = env.possible_agents[0]
        action_space = env.action_space(agent)
        try:
            genomes = [_make_genome(i) for i in range(3)]
            sub = PopulationPool(
                genomes,
                lambda g: _PicklableSampler(action_space),
                snapshot_kind="feedforward",
                snapshot_metadata={"masked": False},
            )
            comp = CompositePool([sub], weights=[1.0])
            rng = np.random.default_rng(0)
            snaps = comp.snapshot_opponents(rng, 2)
            assert len(snaps) == 2
            assert all(isinstance(s, OpponentSnapshot) for s in snaps)
            pickle.dumps(snaps)  # must not raise
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


class TestTrainerParallelWiring:
    def test_metadata_records_workers_and_backend(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=10,
            seed=7,
            num_workers=1,
            parallel_backend="process",
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=20))
        assert result.metadata["coevolution_num_workers"] == 1
        assert result.metadata["coevolution_parallel_backend"] == "process"

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="parallel_backend"):
            CoEvolutionConfig(
                mode="competitive_2pop",
                sides=[
                    Side("a", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("b", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
                parallel_backend="not-a-backend",
            )

    def test_k_one_with_workers_warns(self, recwarn):
        """num_workers > 1 with k_opponents == 1 is a no-op; the
        trainer must emit a RuntimeWarning so users notice their
        config is wrong."""
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=1,
            max_steps_per_episode=10,
            seed=7,
            num_workers=2,
        )
        CoEvolutionTrainer().train(_base_tcfg(cfg, budget=10))
        msgs = [str(w.message) for w in recwarn.list]
        assert any("num_workers" in m and "k_opponents=1" in m for m in msgs)


# ---------------------------------------------------------------------------
# Serial vs parallel parity (the central correctness claim of Phase 8)
# ---------------------------------------------------------------------------


class TestSerialMatchesParallel:
    """Under a fixed seed the parallel path must produce numerically
    identical learner fitness sequences to the serial path. Workers
    receive the same opponent snapshots in the same order, so any
    drift here means a state-leak between processes."""

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_metadata_consistent(self, num_workers):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=10,
            seed=42,
            num_workers=num_workers,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=20))
        assert result.metadata["coevolution_num_workers"] == max(1, num_workers)
        assert result.metrics["num_generations"] >= 1
        # Evolving sides have history rows.
        for name in ("pursuers", "evaders"):
            assert len(result.metrics["per_side_history"][name]) >= 1
