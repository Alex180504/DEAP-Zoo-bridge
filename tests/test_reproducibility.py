"""Reproducibility regression tests

These tests assert that same-seed trainer runs produce identical fitness
histories, and that co-evolution sides properly isolate their RNG streams.
"""

from __future__ import annotations

import random as _random
import pytest
import numpy as np

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)
from deap_pettingzoo.baselines.ea_trainer import EATrainer


# Shared helpers

def _fitness_history(result) -> list:
    """Extract per-generation max-fitness series from a TrainingResult."""
    return [r["max"] for r in result.metrics["fitness_history"]]


# EA reproducibility

class TestEAReproducibility:
    """Two same-seed in-process ``train()`` calls must produce identical
    fitness histories.  This is the keystone test for F-3.1-01 and
    genuinely fails on the buggy HEAD code."""

    def test_ea_train_is_reproducible(self):
        trainer = EATrainer(hidden_size=8, population_size=6)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=24,
            seed=42, max_steps_per_episode=20,
            env_config={"max_cycles": 20},
        )
        result1 = trainer.train(config)
        result2 = trainer.train(config)
        assert _fitness_history(result1) == _fitness_history(result2), (
            "EA train() is not reproducible at a fixed seed — "
            "F-3.1-01 regression."
        )


# CMA reproducibility

class TestCMAReproducibility:
    """CMA-ES already seeds ``np.random`` — confirm it stays reproducible."""

    def test_cma_train_is_reproducible(self):
        trainer = CMAESTrainer(hidden_size=8, population_size=6, sigma0=0.3)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=24,
            seed=42, max_steps_per_episode=20,
            env_config={"max_cycles": 20},
        )
        result1 = trainer.train(config)
        result2 = trainer.train(config)
        assert _fitness_history(result1) == _fitness_history(result2), (
            "CMA train() is not reproducible — regression."
        )

# Co-evolution trainer reproducibility

class TestCoevolutionReproducibility:
    """Two same-seed ``CoEvolutionTrainer.train()`` calls must produce
    identical per-side fitness histories."""

    @staticmethod
    def _config(seed: int) -> TrainingConfig:
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea",
                     {"population_size": 4, "hidden_size": 8, "seed": seed}),
                Side("evaders", ("agent",), "ea",
                     {"population_size": 4, "hidden_size": 8, "seed": seed + 1}),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=seed,
        )
        return TrainingConfig(
            env_name="simple_tag_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=80,
            coevolution_config=cfg,
        )

    def test_coevolution_train_is_reproducible(self):
        result1 = CoEvolutionTrainer().train(self._config(seed=42))
        result2 = CoEvolutionTrainer().train(self._config(seed=42))
        hist1 = result1.metrics["per_side_history"]
        hist2 = result2.metrics["per_side_history"]
        assert set(hist1) == set(hist2)
        for name in hist1:
            series1 = [r["max"] for r in hist1[name]]
            series2 = [r["max"] for r in hist2[name]]
            assert series1 == series2, (
                f"{name}: coevolution train() is not reproducible at a "
                "fixed seed — F-4.2-02 regression."
            )


# num_seeds must not leak into the genome seed

class TestNumSeedsNotSeed:
    """``num_seeds`` must never influence the RNG seed used by trainers.
    The old code path used ``num_seeds`` as a fallback when no explicit
    seed was given (ea_trainer.py:446)."""

    def test_num_seeds_does_not_affect_seeded_run(self):
        """Same explicit seed + different num_seeds → identical results."""
        trainer = EATrainer(hidden_size=8, population_size=6)

        def _run(ns):
            return trainer.train(TrainingConfig(
                env_name="simple_spread_v3",
                budget_mode=BudgetMode.EVAL_EPISODES,
                budget_value=24,
                num_seeds=ns,
                seed=42, max_steps_per_episode=20,
                env_config={"max_cycles": 20},
            ))

        r1 = _run(1)
        r5 = _run(5)
        assert _fitness_history(r1) == _fitness_history(r5), (
            "num_seeds leaked into the RNG seed — F-1.2-04 regression. "
            "Results differ between num_seeds=1 and num_seeds=5 at the "
            "same explicit seed."
        )


# Co-evolution side RNG-isolation contract tests

class _NoopEvaluator:
    """Evaluator that always returns (0.0,) for fitness."""

    def evaluate(self, individual, **_kwargs):
        return (0.0,)


class TestSideRNGIsolation:
    """Each side's ``step_generation`` must restore the global RNG state
    it was handed.  PSO was the sole optimizer missing this block
    (F-2.7-02); EA and CMA had the block but it wasn't exception-safe
    (F-3.1-05).  These are **contract tests** — the PSO test genuinely
    fails on HEAD and passes after the fix."""

    @staticmethod
    def _make_evaluator_factory():
        evaluator = _NoopEvaluator()

        def _factory(_side_state):
            return evaluator

        return _factory

    @staticmethod
    def _seed_globals(seed: int = 12345) -> None:
        _random.seed(seed)
        np.random.seed(seed)

    def _assert_globals_restored(self, side_state, step_fn):
        """Seed globals, set side state, call step_fn, assert globals restored."""
        self._seed_globals(42)
        py_before = _random.getstate()
        np_before = np.random.get_state()

        # Simulate what the trainer does
        _ = np.random.default_rng(99)
        self._seed_globals(42)  # Re-seed so capture is deterministic
        side_state._py_state = _random.getstate()
        side_state._np_state = np.random.get_state()

        # Advance the globals to detect contamination.
        _random.random()
        np.random.random()
        py_mid = _random.getstate()
        np_mid = np.random.get_state()

        # Run step_generation — this must restore globals to mid state.
        step_fn()

        py_after = _random.getstate()
        np_after = np.random.get_state()

        assert py_after == py_mid, (
            f"{type(side_state).__name__}: Python random state was not "
            "restored after step_generation — contamination bug."
        )
        # Compare element-wise since the MT array cannot use == directly.
        assert np_after[0] == np_mid[0], (
            f"{type(side_state).__name__}: NumPy state version changed."
        )
        assert np.array_equal(np_after[1], np_mid[1]), (
            f"{type(side_state).__name__}: NumPy MT state not restored."
        )
        assert np_after[2:] == np_mid[2:], (
            f"{type(side_state).__name__}: NumPy state tail not restored."
        )

    def test_ea_side_restores_global_rng(self):
        from deap_pettingzoo.baselines.coevolution.sides import EASideState
        from deap_pettingzoo.baselines.coevolution import Side
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side("test_ea", ("agent",), "ea",
                     {"population_size": 4, "hidden_size": 8, "seed": 7})
        state = EASideState(side)
        env = PettingZooWrapper("simple_tag_v3")
        state.initialise(env)
        eval_factory = self._make_evaluator_factory()

        def _step():
            state.step_generation(eval_factory)

        self._assert_globals_restored(state, _step)
        env.close()

    def test_pso_side_restores_global_rng(self):
        """Genuine F-2.7-02 pin — fails on HEAD, passes after fix."""
        from deap_pettingzoo.baselines.coevolution.sides import PSOSideState
        from deap_pettingzoo.baselines.coevolution import Side
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side("test_pso", ("agent",), "pso",
                     {"population_size": 4, "hidden_size": 8, "seed": 7})
        state = PSOSideState(side)
        env = PettingZooWrapper("simple_tag_v3")
        state.initialise(env)
        eval_factory = self._make_evaluator_factory()

        def _step():
            state.step_generation(eval_factory)

        self._assert_globals_restored(state, _step)
        env.close()

    def test_cma_side_restores_global_rng(self):
        from deap_pettingzoo.baselines.coevolution.sides import CMAESSideState
        from deap_pettingzoo.baselines.coevolution import Side
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side("test_cma", ("agent",), "cma",
                     {"population_size": 4, "hidden_size": 8, "seed": 7,
                      "sigma0": 0.3})
        state = CMAESSideState(side)
        env = PettingZooWrapper("simple_tag_v3")
        state.initialise(env)
        eval_factory = self._make_evaluator_factory()

        def _step():
            state.step_generation(eval_factory)

        self._assert_globals_restored(state, _step)
        env.close()
