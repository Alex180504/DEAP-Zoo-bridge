"""Tests for parallel NEAT evaluation."""

import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.contrib.neat.trainer import NEATTrainer


class TestNeatParallelEval:
    """Smoke tests for parallel NEAT evaluation."""

    def test_num_workers_stored(self):
        t = NEATTrainer(num_workers=4)
        assert t._num_workers == 4

    def test_num_workers_default(self):
        t = NEATTrainer()
        assert t._num_workers == 1

    def test_num_workers_invalid(self):
        with pytest.raises(ValueError, match="num_workers"):
            NEATTrainer(num_workers=0)

    def test_capabilities_parallel(self):
        t = NEATTrainer(num_workers=4)
        caps = t.get_capabilities()
        assert caps["parallel_training"] is True

    def test_capabilities_serial(self):
        t = NEATTrainer()
        caps = t.get_capabilities()
        assert caps["parallel_training"] is False

    def test_train_smoke_serial(self, tmp_path):
        """Serial path still works."""
        cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=60,
            num_seeds=1,
            algorithm_params={
                "pop_size": 6,
                "eval_episodes_per_genome": 1,
                "seed": 42,
            },
            env_config={"max_cycles": 25, "continuous_actions": False},
            output_dir=tmp_path,
        )
        trainer = NEATTrainer(num_workers=1)
        result = trainer.train(cfg)
        assert result.algorithm == "NEAT"
        assert result.policy is not None

    def test_train_honors_config_num_workers(self, tmp_path, monkeypatch):
        """TrainingConfig.num_workers must actually
        route through parallel evaluation, even when the trainer was
        constructed with the (serial) default of 1.
        """
        import deap_pettingzoo.contrib.neat.parallel as neat_parallel

        calls = []
        _orig = neat_parallel.make_neat_parallel_evaluate

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return _orig(*args, **kwargs)

        monkeypatch.setattr(neat_parallel, "make_neat_parallel_evaluate", _spy)

        n_generations = 2
        cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=1,  # irrelevant: n_generations below bypasses derivation
            num_seeds=1,
            algorithm_params={
                "pop_size": 4,
                "eval_episodes_per_genome": 1,
                "n_generations": n_generations,
            },
            env_config={"max_cycles": 10, "continuous_actions": False},
            seed=0,
            num_workers=2,
            output_dir=tmp_path,
        )
        trainer = NEATTrainer()  # constructor default: serial
        assert trainer._num_workers == 1
        result = trainer.train(cfg)
        assert result.algorithm == "NEAT"
        # The parallel factory must actually have been invoked -- proves the
        # config-supplied num_workers routed to the parallel path, not just
        # that the (serial-safe) generation bookkeeping happens to match.
        assert len(calls) == 1
        assert result.metrics["final_generation"] == n_generations

    def test_train_parallel_updates_fitness_history_and_generation(self, tmp_path):
        n_generations = 3
        cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=1,
            num_seeds=1,
            algorithm_params={
                "pop_size": 4,
                "eval_episodes_per_genome": 1,
                "n_generations": n_generations,
            },
            env_config={"max_cycles": 10, "continuous_actions": False},
            seed=0,
            num_workers=2,
            output_dir=tmp_path,
        )
        trainer = NEATTrainer(num_workers=2)
        result = trainer.train(cfg)
        assert result.metrics["final_generation"] == n_generations
        assert len(result.metrics["fitness_history"]) == n_generations
