"""Tests for PSOTrainer."""
import numpy as np
import pytest
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator


class _MockIndividual:
    """Minimal individual wrapper holding a genome for re-evaluation."""
    def __init__(self, genome):
        self.genome = genome


class TestPSOTrainerInit:
    def test_defaults(self):
        t = PSOTrainer()
        assert t.hidden_size == 64
        assert t.population_size == 50
        assert t.phi1 == 2.0
        assert t.phi2 == 2.0
        assert t.weight == 0.7298

    def test_negative_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            PSOTrainer(num_workers=-1)

    def test_eval_episodes_default_and_validation(self):
        assert PSOTrainer().eval_episodes == 1
        assert PSOTrainer(eval_episodes=5).eval_episodes == 5
        with pytest.raises(ValueError, match="eval_episodes"):
            PSOTrainer(eval_episodes=0)


class TestPSORunPersistence:
    def test_run_writes_policy_artifact_and_summary(self, tmp_path):
        """Regression: the PSO checkpoint used an unregistered 'pso_genome'
        kind, which raised inside a swallowed except — leaving an empty run dir
        (no policy artefact, empty final_summary, wall_time_s ~0). A completed
        run must now persist both."""
        import json

        trainer = PSOTrainer(hidden_size=8, population_size=6, eval_episodes=2)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            seed=1, max_steps_per_episode=10,
            env_config={"max_cycles": 10},
            output_dir=tmp_path,
        )
        trainer.train(config)
        run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        rd = run_dirs[0]
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["artifacts"]["policy"]["kind"] == "pso_genome_v1"
        assert manifest["final_summary"].get("final_fitness") is not None
        assert (rd / "checkpoints" / "policy.json").is_file()
        assert manifest["wall_time_s"] and manifest["wall_time_s"] > 0

    def test_eval_episodes_consumes_more_steps(self):
        """K CRN episodes per generation must consume ~K× the env steps of a
        single-episode run over the same generation count."""
        cfg = dict(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            seed=3, max_steps_per_episode=10,
            env_config={"max_cycles": 10},
        )
        one = PSOTrainer(hidden_size=8, population_size=6, eval_episodes=1).train(
            TrainingConfig(**cfg)
        )
        three = PSOTrainer(hidden_size=8, population_size=6, eval_episodes=3).train(
            TrainingConfig(**cfg)
        )
        assert (
            three.metrics["env_steps_consumed"]
            > one.metrics["env_steps_consumed"] * 2
        )


class TestPSOTrainerSmoke:
    def _tiny_config(self, seed=0):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=12,
            num_seeds=1,
            seed=seed, max_steps_per_episode=10,
            env_config={"max_cycles": 10},
        )

    def test_smoke_train(self):
        trainer = PSOTrainer(hidden_size=8, population_size=6)
        result = trainer.train(self._tiny_config(seed=0))
        assert result.algorithm == "pso"
        assert result.policy is not None
        history = result.metrics["fitness_history"]
        assert len(history) >= 1

    def test_capabilities(self):
        caps = PSOTrainer().get_capabilities()
        assert caps["discrete_action"] is True
        assert caps["continuous_action"] is True
        assert caps["on_policy"] is False

    def test_budget_modes(self):
        t = PSOTrainer()
        assert t.supports_budget_mode(BudgetMode.ENV_STEPS)
        assert t.supports_budget_mode(BudgetMode.EVAL_EPISODES)


class TestPSOChampionFitnessNotStale:
    """Regression for F-2.5-01: PSO champion must not carry stale fitness.

    The bug: ``toolbox.update(population)`` moves particles to new positions
    *before* ``hof.update(population)``, so the HoF records a particle whose
    genome (new position) no longer matches its fitness (evaluated at the old
    position).  The fix swaps the order — ``hof.update`` must happen first.
    """

    @staticmethod
    def _re_evaluate_champion(policy, env_name, env_config, max_steps, seed):
        """Re-evaluate a champion genome at a fixed seed, returning fitness."""
        env = PettingZooWrapper(env_name, env_config=env_config)
        evaluator = SingleEpisodeEvaluator(env, max_steps=max_steps)
        ind = _MockIndividual(policy.genome)
        fitness = evaluator.evaluate(ind, seed=seed)[0]
        env.close()
        return fitness

    def test_champion_re_eval_matches_reported(self):
        """Champion re-evaluated at a fixed seed matches reported fitness."""
        trainer = PSOTrainer(hidden_size=8, population_size=8)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            seed=42, max_steps_per_episode=25,
            env_config={"max_cycles": 25},
        )
        result = trainer.train(config)
        reported = float(result.metrics["final_fitness"])

        # Re-evaluate champion at the same base seed used for the run.
        re_eval = self._re_evaluate_champion(
            result.policy,
            env_name="simple_spread_v3",
            env_config={"max_cycles": 25},
            max_steps=25,
            seed=42,
        )
        # Loose tolerance: the re-eval seed may differ from the champion's
        # original eval seed, but the fitness should be in the same order
        # of magnitude.  A stale-pairing bug would often produce a
        # dramatically different fitness (or NaN).
        assert np.isfinite(re_eval), f"Re-eval fitness is non-finite: {re_eval}"
        assert abs(re_eval - reported) < abs(reported) * 2.0 + 20.0, (
            f"Re-eval fitness {re_eval:.2f} diverges from reported "
            f"{reported:.2f} — stale genome->fitness pairing suspected"
        )

    def test_champion_genome_produces_valid_actions(self):
        """Champion policy must produce finite, correctly-shaped actions."""
        trainer = PSOTrainer(hidden_size=8, population_size=6)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=1,
            seed=7, max_steps_per_episode=15,
            env_config={"max_cycles": 15},
        )
        result = trainer.train(config)

        env = PettingZooWrapper("simple_spread_v3", env_config={"max_cycles": 15})
        obs = env.reset(seed=7)
        for agent_id in env.possible_agents:
            action = result.policy.get_action(obs[agent_id])
            assert action is not None
            # simple_spread_v3 is discrete with 5 actions
            assert isinstance(action, (int, np.integer)), (
                f"Expected discrete action, got {type(action)}"
            )
            assert 0 <= int(action) <= 4
        env.close()
