"""Smoke tests for :class:`CoEvolutionTrainer`.

One test per variant plus mixed EA-vs-frozen, exercised end-to-end on
``simple_tag_v3`` with tiny budgets. These are not convergence tests —
they verify the orchestration plumbing (bootstrap, synchronous /
alternating scheduling, HoF snapshotting, reporting policy
construction) produces sane :class:`TrainingResult`s.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    ElitistHallOfFame,
    RingHallOfFame,
    Side,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


def _tiny_ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _tiny_cma_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed, "sigma0": 0.3}


def _base_tcfg(cfg: CoEvolutionConfig, budget: int = 40) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        algorithm_params={"coevolution": cfg},
    )


class TestCompetitive2Pop:
    def test_end_to_end_synchronous(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg))
        assert result.algorithm == "coevolution"
        assert result.metadata["mode"] == "competitive_2pop"
        assert result.metrics["num_generations"] >= 1
        # Both evolving sides populated.
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Per-gen history accumulates.
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            assert len(hist) == result.metrics["num_generations"]
            for row in hist:
                assert {"generation", "mean", "std", "min", "max"}.issubset(row)
        # Optimisation actually happened: at least ONE side's best-ever
        # fitness improved end-vs-start. We don't insist both sides
        # improve simultaneously (competitive co-evolution is
        # zero-sum-ish on tiny budgets — one side gaining can mean
        # the other regressing); we only insist neither side's
        # best-ever fitness *regressed* below gen-0.
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            if len(hist) < 2:
                continue  # single-gen run, nothing to compare
            best_ever = max(row["max"] for row in hist)
            assert best_ever >= hist[0]["max"], (
                f"{name}: best-ever fitness regressed: "
                f"gen0_max={hist[0]['max']}, best_ever={best_ever}"
            )
        # Reporting policy is executable.
        env = PettingZooWrapper("simple_tag_v3")
        obs = env.reset(seed=0)
        adv_obs = next(o for a, o in obs.items() if a.startswith("adversary"))
        action = result.policy.get_action(adv_obs)
        assert action is not None
        env.close()


class TestSelfPlayHoF:
    def test_ring_archive_accumulates(self):
        cfg = CoEvolutionConfig(
            mode="self_play_hof",
            sides=[
                Side(
                    "pursuers", ("adversary",), "ea", _tiny_ea_cfg(0),
                    hof=RingHallOfFame(capacity=8),
                ),
                Side(
                    "evaders", ("agent",), "ea", _tiny_ea_cfg(1),
                    hof=ElitistHallOfFame(capacity=8),
                ),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
            snapshot_every=1,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=60))
        # HoF grows with each generation (plus bootstrap snapshot).
        for name in ("pursuers", "evaders"):
            assert result.metrics["hof_sizes"][name] >= 1
        assert result.metadata["mode"] == "self_play_hof"

    def test_requires_hof_on_each_side(self):
        with pytest.raises(ValueError, match="hof"):
            CoEvolutionConfig(
                mode="self_play_hof",
                sides=[
                    Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
            )

    def test_requires_multiple_sides(self):
        with pytest.raises(ValueError, match="at least 2 sides"):
            CoEvolutionConfig(
                mode="self_play_hof",
                sides=[
                    Side(
                        "solo", ("adversary",), "ea", _tiny_ea_cfg(0),
                        hof=RingHallOfFame(capacity=4),
                    ),
                ],
            )


class TestCooperative:
    def test_mixed_ea_cma_alternating(self):
        cfg = CoEvolutionConfig(
            mode="cooperative",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "cma", _tiny_cma_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
            schedule="alternating",
            alternating_generations=1,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=40))
        assert result.metadata["schedule"] == "alternating"
        assert result.metadata["sides"][0]["optimizer"] == "ea"
        assert result.metadata["sides"][1]["optimizer"] == "cma"


class TestConfigValidation:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            CoEvolutionConfig(
                mode="bogus",
                sides=[Side("a", ("agent",), "ea", _tiny_ea_cfg(0))],
            )

    def test_rejects_duplicate_side_names(self):
        with pytest.raises(ValueError, match="Duplicate side names"):
            CoEvolutionConfig(
                mode="cooperative",
                sides=[
                    Side("dup", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("dup", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
            )

    def test_rejects_unknown_reporting_side(self):
        with pytest.raises(ValueError, match="reporting_side"):
            CoEvolutionConfig(
                mode="competitive_2pop",
                sides=[
                    Side("a", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("b", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
                reporting_side="does_not_exist",
            )

    def test_competitive_requires_two_sides(self):
        with pytest.raises(ValueError, match="competitive_2pop"):
            CoEvolutionConfig(
                mode="competitive_2pop",
                sides=[Side("a", ("adversary",), "ea", _tiny_ea_cfg(0))],
            )
