"""Live smoke test for AgileRL HPO trainer.

Requires the ``rl-baselines`` extra.  Gated behind ``importorskip`` and
marked ``slow``.
"""
from pathlib import Path

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
from deap_pettingzoo.baselines.base import BudgetMode, ProbeConfig, TrainingConfig


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


class TestHPOProbe:
    """The HPO trainer supports the held-out champion probe like other trainers."""

    def test_probe_writes_csv(self, tmp_path):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=200,
            num_seeds=1,
            seed=0,
            output_dir=tmp_path,
            algorithm_params={
                "max_steps_per_episode": 10,
                "population_size": 2,
                "evo_every_n_steps": 50,
                "learn_step": 50,
            },
            env_config={"max_cycles": 10},
            probe=ProbeConfig(
                enabled=True,
                every_n_steps=50,
                n_episodes=2,
                seed=0,
            ),
        )
        trainer = MATD3HPOTrainer(population_size=2, device="cpu")
        result = trainer.train(config)

        assert result.policy is not None
        assert trainer._prober.enabled is True

        probe_files = list(tmp_path.glob("**/probe.csv"))
        assert probe_files, "probe.csv was not written under the run directory"

        import csv

        with open(probe_files[0], newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1, "probe.csv has no data rows"
        expected = {
            "iteration",
            "env_steps",
            "probe_reward_mean",
            "probe_reward_std",
            "probe_episodes",
        }
        assert expected.issubset(rows[0].keys()), (
            f"probe.csv missing expected columns; got {list(rows[0].keys())}"
        )


class TestHPORunDirMetadata:
    """``metadata["run_dir"]`` is how callers find a finished run to patch.

    Experiment drivers write the held-out champion evaluation into the run's
    summary after the trainer has closed its logger, locating it through this
    key.  Every other trainer sets it; when this one did not, the patch became
    a silent no-op and MATD3 runs reported no evaluation at all — a whole
    algorithm missing from the comparison with nothing in the logs to say so.
    """

    def test_run_dir_is_reported_when_output_dir_is_set(self, tmp_path):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=200,
            num_seeds=1,
            seed=0,
            output_dir=tmp_path,
            algorithm_params={
                "max_steps_per_episode": 10,
                "population_size": 2,
                "evo_every_n_steps": 50,
                "learn_step": 50,
            },
            env_config={"max_cycles": 10},
        )
        result = MATD3HPOTrainer(population_size=2, device="cpu").train(config)

        run_dir = result.metadata.get("run_dir")
        assert run_dir, "metadata['run_dir'] missing — summary patching no-ops"
        run_path = Path(run_dir)
        assert run_path.is_dir()
        assert (run_path / "summary.json").is_file()

    def test_run_dir_is_none_without_an_output_dir(self):
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=200,
            num_seeds=1,
            seed=0,
            algorithm_params={
                "max_steps_per_episode": 10,
                "population_size": 2,
                "evo_every_n_steps": 50,
                "learn_step": 50,
            },
            env_config={"max_cycles": 10},
        )
        result = MATD3HPOTrainer(population_size=2, device="cpu").train(config)

        assert result.metadata.get("run_dir") is None


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
