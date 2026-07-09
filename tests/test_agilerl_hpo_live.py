"""Live smoke test for AgileRL HPO trainer.

Requires the ``rl-baselines`` extra.  Gated behind ``importorskip`` and
marked ``slow``.
"""
import pytest

agilerl = pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)
pytestmark = pytest.mark.slow

from deap_pettingzoo.baselines.agilerl_hpo_trainer import (
    IPPOHPOTrainer,
    MATD3HPOTrainer,
)
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


def _tiny_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        num_seeds=1,
        algorithm_params={
            "seed": seed,
            "max_steps_per_episode": 10,
            "population_size": 3,
            "evo_every_n_steps": 200,
            "learn_step": 50,
        },
        env_config={"max_cycles": 10},
    )


class TestMATD3HPOLive:
    def test_smoke_train_produces_policy(self):
        trainer = MATD3HPOTrainer(population_size=3, device="cpu")
        result = trainer.train(_tiny_config(seed=0))
        assert result.algorithm == "matd3-hpo"
        assert result.policy is not None
        history = result.metrics["fitness_history"]
        assert len(history) >= 1


class TestIPPOHPOLive:
    def test_smoke_on_policy_train_produces_policy(self):
        trainer = IPPOHPOTrainer(population_size=3, device="cpu")
        events = []
        result = trainer.train(
            _tiny_config(seed=0),
            progress_callback=events.append,
        )
        assert result.algorithm == "ippo-hpo"
        assert result.policy is not None
        assert len(result.metrics["fitness_history"]) >= 1
        # Progress must reach the callback the runner wires in.
        assert len(events) >= 1
