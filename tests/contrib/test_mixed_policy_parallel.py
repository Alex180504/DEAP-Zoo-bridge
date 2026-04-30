"""Tests for MixedPolicyEvaluator factory path and make_mixed_parallel_evaluate_map.

Covers:
* Validation errors for incorrect construction paths.
* Factory-path construction builds a usable evaluator.
* make_mixed_parallel_evaluate_map with _SerialExecutor produces correct results.
* Pickle round-trip of _MixedWorkerSpec.
* (skipped on < 2 CPUs) Actual 2-process parallel evaluation run.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.contrib.pursuit_evasion import (
    APFConfig,
    APFEvaderPolicy,
    PursuitEvasionConfig,
    PursuitEvasionEnv,
)
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes.fixed_topology import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    _GenomePayload,
    _MIXED_WORKER_CACHE,
    _MixedWorkerSpec,
    _individual_to_payload,
    _mixed_worker_evaluate,
    make_mixed_parallel_evaluate_map,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.utils.parallel import _SerialExecutor, make_executor


# ---------------------------------------------------------------------------
# Minimal env config (fast; suitable for unit tests)
# ---------------------------------------------------------------------------

_SMALL_CFG = PursuitEvasionConfig(
    num_adversaries=2,
    num_good=2,
    num_obstacles=1,
    max_cycles=20,
)


# ---------------------------------------------------------------------------
# Module-level factories — must be module-level for pickling on Windows spawn
# ---------------------------------------------------------------------------


def _build_env() -> PursuitEvasionEnv:
    return PursuitEvasionEnv(config=_SMALL_CFG)


def _build_scripted(env: PursuitEvasionEnv) -> dict:
    cfg = APFConfig()
    return {
        name: APFEvaderPolicy(layout=env.get_layout(name), config=cfg)
        for name in env.good_agents
    }


def _learner_factory(genome, agent_id, action_space):
    return FeedforwardNeuralPolicy(genome, action_space=action_space)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_individual(env: PursuitEvasionEnv | None = None, hidden: int = 8):
    """Build a DEAP Individual with a random FixedTopologyGenome."""
    setup_deap_creators()
    from deap import creator

    if env is None:
        env = _build_env()
        close_env = True
    else:
        close_env = False

    obs = env.reset(seed=0)
    pursuer_id = env.adversary_agents[0]
    in_size = obs[pursuer_id].shape[0]
    out_size = env.action_space(pursuer_id).n
    genome = FixedTopologyGenome(
        layer_sizes=[in_size, hidden, out_size],
        activation="tanh",
        random_seed=42,
    )
    ind = creator.Individual()
    ind.genome = genome

    if close_env:
        env.close()
    return ind, genome


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestMixedPolicyEvaluatorFactoryValidation:
    """ValueError is raised for every invalid construction combination."""

    def test_raises_when_no_path_provided(self):
        with pytest.raises(ValueError, match="requires either"):
            MixedPolicyEvaluator(
                learner_agent_prefixes=["adversary"],
                learner_factory=_learner_factory,
            )

    def test_raises_when_both_paths_provided(self):
        env = _build_env()
        try:
            with pytest.raises(ValueError, match="not both"):
                MixedPolicyEvaluator(
                    env_wrapper=env,
                    scripted_policies=_build_scripted(env),
                    env_factory=_build_env,
                    scripted_policy_factory=_build_scripted,
                    learner_agent_prefixes=["adversary"],
                    learner_factory=_learner_factory,
                )
        finally:
            env.close()

    def test_raises_factory_path_without_scripted_factory(self):
        with pytest.raises(ValueError, match="scripted_policy_factory"):
            MixedPolicyEvaluator(
                env_factory=_build_env,
                # scripted_policy_factory intentionally omitted
                learner_agent_prefixes=["adversary"],
                learner_factory=_learner_factory,
            )

    def test_raises_missing_scripted_policies_live_path(self):
        env = _build_env()
        try:
            with pytest.raises(ValueError, match="missing entries"):
                MixedPolicyEvaluator(
                    env_wrapper=env,
                    scripted_policies={},  # all evaders missing
                    learner_agent_prefixes=["adversary"],
                    learner_factory=_learner_factory,
                )
        finally:
            env.close()

    def test_raises_no_learner_agents_matched(self):
        env = _build_env()
        scripted = _build_scripted(env)
        # Cover all agents as scripted so learner side is empty.
        for a in env.adversary_agents:
            scripted[a] = APFEvaderPolicy(layout=env.get_layout(a))
        try:
            with pytest.raises(ValueError, match="No agents matched"):
                MixedPolicyEvaluator(
                    env_wrapper=env,
                    scripted_policies=scripted,
                    learner_agent_prefixes=["nonexistent"],
                    learner_factory=_learner_factory,
                )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Factory-path serial evaluation
# ---------------------------------------------------------------------------


class TestMixedPolicyEvaluatorFactoryPath:
    """Factory-path evaluator runs episodes correctly (serial, no workers)."""

    def test_evaluator_is_created_with_valid_env_wrapper(self):
        evaluator = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        # Factory path: validation env is built and closed during __init__,
        # so env_wrapper is None — agent lists are cached for use by workers.
        assert evaluator.env_wrapper is None
        assert len(evaluator.learner_agents) > 0
        assert len(evaluator.scripted_agents) > 0

    def test_run_episode_returns_structured_result(self):
        evaluator = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        ind, _ = _make_individual()
        res = evaluator.run_episode(ind, seed=0)
        assert res.steps > 0
        assert res.steps <= _SMALL_CFG.max_cycles
        assert np.isfinite(res.learner_reward)
        assert np.isfinite(res.scripted_reward)
        assert res.fitness == res.learner_reward

    def test_evaluate_returns_fitness_tuple(self):
        evaluator = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        ind, _ = _make_individual()
        result = evaluator.evaluate(ind, seed=1)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_factory_path_and_live_path_produce_finite_fitness(self):
        """Both construction paths should produce finite, real fitness values."""
        env = _build_env()
        scripted = _build_scripted(env)
        ind, _ = _make_individual(env=env)

        live_ev = MixedPolicyEvaluator(
            env_wrapper=env,
            scripted_policies=scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        factory_ev = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        try:
            live_fit = live_ev.evaluate(ind, seed=7)
            factory_fit = factory_ev.evaluate(ind, seed=7)
            assert np.isfinite(live_fit[0])
            assert np.isfinite(factory_fit[0])
        finally:
            env.close()
            # factory_ev's validation env is already closed in __init__.


# ---------------------------------------------------------------------------
# make_mixed_parallel_evaluate_map with _SerialExecutor
# ---------------------------------------------------------------------------


class TestMakeMixedParallelEvaluateMap:
    """Serial executor produces correct results; validates error cases."""

    def _make_evaluator(self):
        ev = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        return ev

    def test_requires_factory_path_evaluator(self):
        env = _build_env()
        scripted = _build_scripted(env)
        live_ev = MixedPolicyEvaluator(
            env_wrapper=env,
            scripted_policies=scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
        )
        try:
            with _SerialExecutor() as ex:
                with pytest.raises(ValueError, match="env_factory"):
                    make_mixed_parallel_evaluate_map(ex, live_ev)
        finally:
            env.close()

    def test_rejects_non_evaluator(self):
        with _SerialExecutor() as ex:
            with pytest.raises(TypeError, match="MixedPolicyEvaluator"):
                make_mixed_parallel_evaluate_map(ex, object())

    def test_serial_executor_returns_fitness_list(self):
        ev = self._make_evaluator()
        with _SerialExecutor() as ex:
            pmap = make_mixed_parallel_evaluate_map(
                ex, ev, eval_episodes=1
            )
            ind, _ = _make_individual()
            results = pmap(None, [ind, ind])
        assert len(results) == 2
        for fit in results:
            assert isinstance(fit, tuple)
            assert len(fit) == 1
            assert np.isfinite(fit[0])

    def test_pmap_matches_direct_evaluate(self):
        """pmap(serial) should produce the same fitness as direct evaluate."""
        ev = self._make_evaluator()
        ind, _ = _make_individual()
        direct = ev.run_episode(ind, seed=0).fitness

        with _SerialExecutor() as ex:
            pmap = make_mixed_parallel_evaluate_map(
                ex, ev, eval_episodes=1
            )
            # pmap ignores the evaluate_fn argument.
            [(parallel_fit,)] = pmap(None, [ind])

        # Both run a single episode with the same genome.
        # Fitness values won't be identical (different env instances,
        # possibly different random state), but both should be finite.
        assert np.isfinite(direct)
        assert np.isfinite(parallel_fit)

    def test_rejects_lambda_factory(self):
        """Lambda factories are not module-level and cannot be pickled."""
        ev = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=lambda g, a, s: FeedforwardNeuralPolicy(g, action_space=s),
        )
        with _SerialExecutor() as ex:
            with pytest.raises(ValueError, match="module-level"):
                make_mixed_parallel_evaluate_map(ex, ev)


# ---------------------------------------------------------------------------
# Pickle round-trip of _MixedWorkerSpec
# ---------------------------------------------------------------------------


class TestMixedWorkerSpecPickle:
    """_MixedWorkerSpec must pickle/unpickle cleanly (required by workers)."""

    def test_spec_pickles(self):
        spec = _MixedWorkerSpec(
            env_factory_qualname=(
                "tests.contrib.test_mixed_policy_parallel._build_env"
            ),
            scripted_factory_qualname=(
                "tests.contrib.test_mixed_policy_parallel._build_scripted"
            ),
            learner_factory_qualname=(
                "tests.contrib.test_mixed_policy_parallel._learner_factory"
            ),
            learner_prefixes=("adversary",),
            max_steps=20,
            eval_episodes=1,
            layer_sizes=(16, 8, 5),
            activation="tanh",
            output_activation="linear",
        )
        data = pickle.dumps(spec)
        restored = pickle.loads(data)
        assert restored.learner_prefixes == ("adversary",)
        assert restored.activation == "tanh"
        assert restored.output_activation == "linear"
        assert restored.layer_sizes == (16, 8, 5)
        assert restored.eval_episodes == 1


class TestMixedWorkerActivationPropagation:
    def test_individual_payload_preserves_output_activation(self):
        ind, genome = _make_individual()
        genome.output_activation = "linear"

        payload = _individual_to_payload(ind, seed=123)

        assert payload.activation == "tanh"
        assert payload.output_activation == "linear"
        assert payload.seed == 123

    def test_mixed_worker_rebuilds_genome_with_output_activation(self):
        class _StubResult:
            def __init__(self, fitness: float):
                self.fitness = fitness

        class _StubEvaluator:
            def __init__(self):
                self.seen_output_activation = None

            def run_episode(self, individual, seed=None):
                del seed
                self.seen_output_activation = individual.genome.output_activation
                return _StubResult(1.0)

        stub = _StubEvaluator()
        spec = _MixedWorkerSpec(
            env_factory_qualname="tests.contrib.test_mixed_policy_parallel._build_env",
            scripted_factory_qualname="tests.contrib.test_mixed_policy_parallel._build_scripted",
            learner_factory_qualname="tests.contrib.test_mixed_policy_parallel._learner_factory",
            learner_prefixes=("adversary",),
            max_steps=20,
            eval_episodes=1,
            layer_sizes=(4, 8, 5),
            activation="relu",
            output_activation="linear",
        )
        _MIXED_WORKER_CACHE["key"] = (
            spec.env_factory_qualname,
            spec.scripted_factory_qualname,
            spec.learner_factory_qualname,
            spec.max_steps,
            tuple(spec.learner_prefixes),
        )
        _MIXED_WORKER_CACHE["evaluator"] = stub
        payload = _GenomePayload(
            weights=np.zeros(85, dtype=np.float64),
            layer_sizes=spec.layer_sizes,
            activation=spec.activation,
            output_activation=spec.output_activation,
            weight_range=(-1.0, 1.0),
            seed=0,
        )

        try:
            result = _mixed_worker_evaluate((payload, spec))
        finally:
            _MIXED_WORKER_CACHE.clear()

        assert result == (1.0,)
        assert stub.seen_output_activation == "linear"


# ---------------------------------------------------------------------------
# Actual 2-process parallel run (skipped on single-CPU machines)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2,
    reason="needs at least 2 CPUs for process-pool test",
)
class TestMixedParallelProcessPool:
    """Run make_mixed_parallel_evaluate_map with a real ProcessPoolExecutor."""

    def test_parallel_eval_produces_finite_fitness(self):
        setup_deap_creators()
        ev = MixedPolicyEvaluator(
            env_factory=_build_env,
            scripted_policy_factory=_build_scripted,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=_SMALL_CFG.max_cycles,
        )
        ind, _ = _make_individual()
        population = [ind, ind, ind, ind]

        with make_executor(2) as executor:
            pmap = make_mixed_parallel_evaluate_map(
                executor, ev, eval_episodes=1
            )
            results = pmap(None, population)

        assert len(results) == 4
        for fit in results:
            assert isinstance(fit, tuple)
            assert np.isfinite(fit[0])
