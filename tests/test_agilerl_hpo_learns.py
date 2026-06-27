"""Regression tests asserting that off-policy HPO must store transitions and call learn()."""

import pytest
from unittest.mock import patch

pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)
pytest.importorskip("torch", reason="PyTorch not installed.")
pytestmark = pytest.mark.slow

from deap_pettingzoo.baselines.agilerl_hpo_trainer import MADDPGHPOTrainer
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


def _hpo_config(
    seed: int = 0,
    *,
    budget: int = 200,
    evo_every: int = 50,
    batch_size: int = 2,
    memory_size: int = 500,
    learning_delay: int = 0,
    learn_step: int = 1,
) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=budget,
        num_seeds=1,
        algorithm_params={
            "seed": seed,
            "max_steps_per_episode": 25,
            "population_size": 2,
            "evo_every_n_steps": evo_every,
            "learning_delay": learning_delay,
            "learn_step": learn_step,
            "batch_size": batch_size,
            "memory_size": memory_size,
        },
        env_config={"max_cycles": 25},
    )


class TestHPOStoresAndLearns:
    """TOff-policy HPO must call learn()."""

    def test_hpo_stores_and_learns(self):
        """
        learn() is invoked: proves a real buffer was built and sampled.
        """
        from agilerl.algorithms.maddpg import MADDPG

        learn_calls: list = []
        _orig = MADDPG.learn

        def _spy(self, *args, **kwargs):
            learn_calls.append(True)
            return _orig(self, *args, **kwargs)

        config = _hpo_config(
            seed=0,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=1,
        )
        with patch.object(MADDPG, "learn", _spy):
            trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
            trainer.train(config)

        assert len(learn_calls) > 0, (
            "agent.learn() was never invoked — buffer may be None or never "
            "reached batch_size. This is the B3 regression."
        )

    def test_hpo_champion_improves(self):
        """Peak champion fitness across generations does not fall below gen-1.

        Guards against a catastrophically broken rollout-and-learn loop where
        the population never receives any learning signal.
        """
        config = _hpo_config(
            seed=2,
            budget=800,
            evo_every=100,
            batch_size=4,
            memory_size=2000,
            learning_delay=0,
            learn_step=10,
        )
        trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
        result = trainer.train(config)

        history = result.metrics["fitness_history"]
        assert len(history) >= 2, "Expected at least 2 generations for comparison"
        gen1_max = history[0]["max"]
        # Peak across all generations (not just final) to avoid lucky-dip regression.
        peak_max = max(g["max"] for g in history[1:])
        assert peak_max >= gen1_max, (
            f"Peak post-gen-1 fitness {peak_max:.2f} is below gen-1 max "
            f"{gen1_max:.2f} — the population may be gaining no learning signal."
        )

    def test_hpo_is_reproducible(self):
        """Two same-seed HPO runs yield identical champion fitness histories."""
        config_a = _hpo_config(
            seed=7,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=5,
        )
        config_b = _hpo_config(
            seed=7,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=5,
        )
        trainer_a = MADDPGHPOTrainer(population_size=2, device="cpu")
        trainer_b = MADDPGHPOTrainer(population_size=2, device="cpu")

        result_a = trainer_a.train(config_a)
        result_b = trainer_b.train(config_b)

        hist_a = result_a.metrics["fitness_history"]
        hist_b = result_b.metrics["fitness_history"]

        assert len(hist_a) == len(hist_b), (
            f"Different number of generations: {len(hist_a)} vs {len(hist_b)}"
        )
        for i, (ga, gb) in enumerate(zip(hist_a, hist_b)):
            assert abs(ga["max"] - gb["max"]) < 1e-6, (
                f"Gen {i + 1} max fitness mismatch: "
                f"{ga['max']:.6f} vs {gb['max']:.6f}"
            )
