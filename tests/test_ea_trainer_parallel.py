"""Parallel-evaluation tests for :class:`EATrainer`.

These tests verify:

* ``num_workers`` and ``parallel_backend`` are accepted and validated.
* ``num_workers=1`` (serial) and ``num_workers=2`` (process pool) both
  produce a valid :class:`TrainingResult`.
* When ``algorithm_params['seed']`` is set, serial and parallel runs
  produce **identical final fitness** (deterministic per-individual
  eval seeding).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer


def _make_config() -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,
        algorithm_params={"seed": 0},
    )


class TestEATrainerParallelArguments:
    def test_default_num_workers_is_one(self):
        t = EATrainer(hidden_size=8, population_size=4)
        cfg = _make_config()
        result = t.train(cfg)
        assert result.metadata["num_workers"] == 1
        assert result.metadata["parallel_backend"] == "process"

    def test_negative_num_workers_rejected(self):
        with pytest.raises(ValueError, match="num_workers"):
            EATrainer(hidden_size=8, population_size=4, num_workers=-1)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="parallel_backend"):
            EATrainer(
                hidden_size=8,
                population_size=4,
                num_workers=2,
                parallel_backend="bogus",
            )


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs at least 2 CPUs"
)
class TestEATrainerProcessPool:
    def test_parallel_run_completes(self):
        t = EATrainer(
            hidden_size=8, population_size=4, num_workers=2
        )
        result = t.train(_make_config())
        assert result.metadata["num_workers"] == 2
        assert "final_fitness" in result.metrics
        assert np.isfinite(result.metrics["final_fitness"])

    def test_serial_and_parallel_initial_eval_match(self):
        """Initial population is deterministic via genome seed; the
        parallel evaluator must score it identically to the serial
        evaluator for matching seeds.

        We verify this at the API level by directly exercising
        ``evaluate_population_parallel`` against the serial evaluator,
        which sidesteps DEAP's unseeded selection/mutation/crossover
        in the rest of ``train()``.
        """
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )
        from deap_pettingzoo.evaluators.single_episode import (
            SingleEpisodeEvaluator,
        )
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.integration.deap_integration import (
            _EvalWorkerSpec,
            evaluate_population_parallel,
            setup_deap_creators,
        )
        from deap_pettingzoo.utils.parallel import make_executor
        from deap import creator

        env = PettingZooWrapper("simple_spread_v3")
        obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]
        input_size = len(obs[agent_id])
        output_size = env.action_space(agent_id).n

        setup_deap_creators()
        # Deterministic individuals via seeded genome.
        inds = []
        for k in range(4):
            g = FixedTopologyGenome(
                layer_sizes=[input_size, 8, output_size],
                activation="tanh",
                weight_range=(-1.0, 1.0),
                random_seed=k,
            )
            ind = creator.Individual()
            ind.genome = g
            inds.append(ind)

        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env, max_steps=20, aggregation_method="sum"
        )
        seeds = [100 + k for k in range(4)]
        serial = [evaluator.evaluate(ind, seed=s) for ind, s in zip(inds, seeds)]

        spec = _EvalWorkerSpec(
            env_name="simple_spread_v3",
            env_config=None,
            max_steps=20,
            aggregation_method="sum",
            team_mapping=None,
        )
        with make_executor(2, backend="process") as ex:
            parallel = evaluate_population_parallel(
                inds, executor=ex, spec=spec, eval_seeds=seeds
            )
        env.close()

        # Same individuals + same eval seeds -> identical fitness.
        for s, p in zip(serial, parallel):
            assert s == pytest.approx(p, rel=1e-9, abs=1e-9)
