"""End-to-end tests for bidirectional EA-RL co-evolution.

Gated on AgileRL availability (``pytest.importorskip("agilerl")``).
Exercises:

* RL-vs-EA: ``Side(optimizer="rl", ...)`` paired with an EA side,
  one generation budget, asserts both sides advance.
* RL-vs-RL: two RL sides facing each other.
* Side validation: unknown algorithm raises, missing algorithm raises,
  on-policy IPPO raises NotImplementedError from initialise.
* num_workers fallback: a competitive run with num_workers=2 + an RL
  side emits a RuntimeWarning and succeeds.

Tiny budgets keep these under the test-suite time cap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")
pytest.importorskip("agilerl")
pytest.importorskip("torch")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    RLPolicyPool,
    RLSideState,
    Side,
)


def _ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _rl_cfg(algo: str = "maddpg", **overrides) -> dict:
    cfg = {
        "algorithm": algo,
        "rollout_episodes_per_gen": 2,
        "learning_steps_per_gen": 1,
        "batch_size": 8,
        "memory_size": 256,
        "learning_delay": 0,
        "device": "cpu",
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def _tcfg(coev_cfg: CoEvolutionConfig, budget: int = 12) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True},  # MADDPG/MATD3 native
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        algorithm_params={"coevolution": coev_cfg},
    )


# ---------------------------------------------------------------------------
# Side-level validation
# ---------------------------------------------------------------------------


class TestSideValidation:
    def test_optimizer_rl_accepted(self):
        # Should not raise.
        Side("learners", ("agent",), "rl", _rl_cfg())

    def test_unknown_optimizer_still_rejected(self):
        with pytest.raises(ValueError, match="optimizer must be"):
            Side("x", ("agent",), "garbage", {})

    def test_missing_algorithm_raises_at_factory_time(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state

        side = Side("learners", ("agent",), "rl", {})  # no algorithm key
        with pytest.raises(ValueError, match="algorithm.*required"):
            make_side_state(side)

    def test_unknown_algorithm_raises_at_factory_time(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state

        side = Side(
            "learners", ("agent",), "rl",
            {"algorithm": "ddqn_hypothetical"},
        )
        with pytest.raises(ValueError, match="unknown algorithm"):
            make_side_state(side)


class TestOnPolicyGated:
    def test_ippo_raises_notimplemented_at_initialise(self):
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side(
            "learners", ("agent",), "rl",
            _rl_cfg(algo="ippo"),
        )
        state = RLSideState(side)
        env = PettingZooWrapper(
            "simple_tag_v3",
            env_config={"continuous_actions": False},
        )
        try:
            with pytest.raises(NotImplementedError, match="on-policy"):
                state.initialise(env, rng=np.random.default_rng(0))
        finally:
            env.close()


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------


class TestEAvsRL:
    def test_competitive_2pop_runs(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg")),
            ],
            k_opponents=1,  # required: RLPolicyPool cannot snapshot for k>1 parallel
            max_steps_per_episode=15,
            num_workers=1,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_tcfg(cfg))
        assert result.algorithm == "coevolution"
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Both sides accumulated history.
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            assert len(hist) >= 1
            assert all("generation" in row for row in hist)


class TestRLvsRL:
    def test_two_rl_sides_run(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "rl", _rl_cfg("maddpg", seed=0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg", seed=1)),
            ],
            k_opponents=1,
            max_steps_per_episode=15,
            num_workers=1,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_tcfg(cfg, budget=8))
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Both sides ran at least one generation.
        assert result.metrics["num_generations"] >= 1


class TestParallelFallback:
    def test_num_workers_gt_1_with_rl_side_warns_and_falls_back(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg")),
            ],
            k_opponents=1,
            max_steps_per_episode=15,
            num_workers=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CoEvolutionTrainer().train(_tcfg(cfg))

        # Fallback warning must fire.
        msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("RL side is present" in m for m in msgs), (
            f"Expected RL fallback RuntimeWarning, got: {msgs}"
        )
        # Metadata reflects the clamped worker count.
        assert result.metadata["coevolution_num_workers"] == 1


class TestPoolSurface:
    def test_rl_side_exposes_rl_policy_pool(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side("learners", ("agent",), "rl", _rl_cfg("maddpg"))
        state = make_side_state(side)
        env = PettingZooWrapper(
            "simple_tag_v3", env_config={"continuous_actions": True},
        )
        try:
            state.initialise(env, rng=np.random.default_rng(0))
            pool = state.current_pool()
            assert isinstance(pool, RLPolicyPool)
            assert len(pool) == 1
            # snapshot_sync must produce an independent pool
            snap = pool.snapshot_sync()
            assert isinstance(snap, RLPolicyPool)
            assert snap is not pool
        finally:
            env.close()
