"""Parallel PSO trainer test (Phase 1.4)."""
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer


class TestPSOTrainerParallel:
    def _config(self, seed=0):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=10,
            num_seeds=1,
            algorithm_params={"seed": seed, "max_steps_per_episode": 10,
                              "num_workers": 2},
            env_config={"max_cycles": 10},
        )

    def test_parallel_smoke(self):
        trainer = PSOTrainer(hidden_size=8, population_size=6, num_workers=2)
        result = trainer.train(self._config(seed=0))
        assert result.algorithm == "pso"
        assert result.policy is not None
