"""Unit tests for :class:`CoEvolutionEvaluator`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionEvalResult,
    CoEvolutionEvaluator,
    FrozenPolicyPool,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.policies import FeedforwardNeuralPolicy


class _DeapLikeIndividual:
    def __init__(self, genome):
        self.genome = genome


class _RandomPolicy:
    def __init__(self, action_space):
        self.sp = action_space

    def get_action(self, observation, action_mask=None):
        return self.sp.sample()

    def reset(self):
        pass


def _build_env_and_genome():
    env = PettingZooWrapper("simple_tag_v3")
    obs = env.reset(seed=0)
    adv = next(a for a in env.possible_agents if a.startswith("adversary"))
    input_size = len(obs[adv])
    output_size = env.action_space(adv).n
    genome = FixedTopologyGenome(
        layer_sizes=[input_size, 8, output_size],
        activation="tanh",
        weight_range=(-1.0, 1.0),
        random_seed=0,
    )
    return env, genome


def _learner_factory(genome, agent_id, action_space):
    return FeedforwardNeuralPolicy(genome, action_space=action_space)


def test_evaluate_returns_single_tuple():
    env, genome = _build_env_and_genome()
    evader_policies = [
        _RandomPolicy(env.action_space(a))
        for a in env.possible_agents
        if a.startswith("agent")
    ]
    evaluator = CoEvolutionEvaluator(
        env_wrapper=env,
        learner_role_prefixes=("adversary",),
        learner_factory=_learner_factory,
        opponent_sides=[("evaders", ("agent",), FrozenPolicyPool(evader_policies))],
        k_opponents=2,
        max_steps=20,
    )
    fitness = evaluator.evaluate(_DeapLikeIndividual(genome), seed=0)
    assert isinstance(fitness, tuple) and len(fitness) == 1
    assert isinstance(fitness[0], float)
    # Sanity: the aggregated learner fitness must be finite and within
    # a sane range. simple_tag rewards are O(few hundred) per episode;
    # values outside this window indicate either NaN/inf leakage or a
    # reward-aggregation bug.
    import math
    assert math.isfinite(fitness[0]), f"non-finite fitness: {fitness[0]}"
    assert -1e6 < fitness[0] < 1e6, f"fitness out of sane range: {fitness[0]}"
    assert evaluator.last_details is not None
    assert isinstance(evaluator.last_details, CoEvolutionEvalResult)
    assert len(evaluator.last_details.per_sample_learner) == 2
    env.close()


def test_k_opponents_consistent_with_samples():
    env, genome = _build_env_and_genome()
    evader_policies = [
        _RandomPolicy(env.action_space(a))
        for a in env.possible_agents
        if a.startswith("agent")
    ]
    evaluator = CoEvolutionEvaluator(
        env_wrapper=env,
        learner_role_prefixes=("adversary",),
        learner_factory=_learner_factory,
        opponent_sides=[("evaders", ("agent",), FrozenPolicyPool(evader_policies))],
        k_opponents=5,
        max_steps=10,
    )
    result = evaluator.run(_DeapLikeIndividual(genome), seed=1)
    assert result.num_samples == 5
    assert len(result.per_sample_learner) == 5
    assert len(result.per_sample_opponents["evaders"]) == 5
    env.close()


def test_aggregation_min_max_sum():
    env, genome = _build_env_and_genome()
    evader_policies = [
        _RandomPolicy(env.action_space(a))
        for a in env.possible_agents
        if a.startswith("agent")
    ]

    def _evaluator(agg):
        return CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=("adversary",),
            learner_factory=_learner_factory,
            opponent_sides=[
                ("evaders", ("agent",), FrozenPolicyPool(evader_policies))
            ],
            k_opponents=3,
            aggregation=agg,
            max_steps=10,
        )

    ind = _DeapLikeIndividual(genome)
    rng = np.random.default_rng(0)
    # Same rng state -> same sampled opponent order -> comparable per-sample
    # learner rewards across aggregations (within numerical noise).
    res_mean = _evaluator("mean").run(ind, seed=0, rng=np.random.default_rng(0))
    res_min = _evaluator("min").run(ind, seed=0, rng=np.random.default_rng(0))
    res_max = _evaluator("max").run(ind, seed=0, rng=np.random.default_rng(0))
    res_sum = _evaluator("sum").run(ind, seed=0, rng=np.random.default_rng(0))
    assert res_min.fitness <= res_mean.fitness <= res_max.fitness
    assert res_sum.fitness == pytest.approx(sum(res_mean.per_sample_learner))
    env.close()


def test_rejects_unassigned_agent():
    env, _ = _build_env_and_genome()
    with pytest.raises(ValueError, match="not claimed"):
        CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=("adversary",),
            learner_factory=_learner_factory,
            opponent_sides=[],  # leaves 'agent_*' unassigned
            k_opponents=1,
        )
    env.close()


def test_rejects_overlapping_prefixes():
    env, _ = _build_env_and_genome()
    evader_policies = [
        _RandomPolicy(env.action_space(a))
        for a in env.possible_agents
        if a.startswith("agent")
    ]
    with pytest.raises(ValueError, match="multiple sides"):
        CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=("a",),  # matches both adversary_* and agent_*
            learner_factory=_learner_factory,
            opponent_sides=[("evaders", ("agent",), FrozenPolicyPool(evader_policies))],
            k_opponents=1,
        )
    env.close()


def test_rejects_bad_aggregation():
    env, _ = _build_env_and_genome()
    evader_policies = [
        _RandomPolicy(env.action_space(a))
        for a in env.possible_agents
        if a.startswith("agent")
    ]
    with pytest.raises(ValueError, match="aggregation"):
        CoEvolutionEvaluator(
            env_wrapper=env,
            learner_role_prefixes=("adversary",),
            learner_factory=_learner_factory,
            opponent_sides=[("evaders", ("agent",), FrozenPolicyPool(evader_policies))],
            k_opponents=1,
            aggregation="geometric",
        )
    env.close()
