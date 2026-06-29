"""Continuous-action end-to-end tests for the EA / CMA-ES stack.

Covers:
- :func:`required_output_size` helper.
- ``FeedforwardNeuralPolicy`` wired with a ``BoxHead`` via
  :class:`SingleEpisodeEvaluator` on a continuous MPE2 env.
- :class:`EATrainer` / :class:`CMAESTrainer` training on continuous envs
  and producing a ``BoxHead``-bearing policy whose actions stay inside
  the env's action-space bounds.
- :func:`_resolve_shared_action_space` rejecting heterogeneous action
  spaces with a clear ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.ea_trainer import (
    EATrainer,
    _resolve_shared_action_space,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import setup_deap_creators
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.policies.action_heads import BoxHead, required_output_size


CONTINUOUS_ENVS = [
    pytest.param("simple_spread_v3", id="simple_spread_v3"),
    pytest.param("simple_tag_v3", id="simple_tag_v3"),
]


# ---------------------------------------------------------------------------
# required_output_size helper
# ---------------------------------------------------------------------------


class TestRequiredOutputSize:
    def test_discrete(self):
        assert required_output_size(spaces.Discrete(5)) == 5

    def test_box_1d(self):
        assert required_output_size(spaces.Box(low=-1.0, high=1.0, shape=(3,))) == 3

    def test_box_multi_dim(self):
        # Flattens: shape (2, 3) -> 6 outputs.
        assert required_output_size(spaces.Box(low=-1.0, high=1.0, shape=(2, 3))) == 6

    def test_multi_discrete_sum_of_nvec(self):
        # MultiDiscrete is now supported: sum(nvec) = 3 + 4 = 7 outputs.
        assert required_output_size(spaces.MultiDiscrete([3, 4])) == 7

    def test_tuple_raises(self):
        with pytest.raises(NotImplementedError, match="Tuple"):
            required_output_size(spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2))))

    def test_dict_raises(self):
        with pytest.raises(NotImplementedError, match="Tuple"):
            required_output_size(
                spaces.Dict({"a": spaces.Discrete(2), "b": spaces.Discrete(2)})
            )

    def test_none_raises(self):
        with pytest.raises(TypeError):
            required_output_size(None)


# ---------------------------------------------------------------------------
# Evaluator wires BoxHead for continuous envs
# ---------------------------------------------------------------------------


def _make_individual(env: PettingZooWrapper, hidden: int = 8):
    setup_deap_creators()
    from deap import creator

    obs = env.reset(seed=0)
    agent = env.possible_agents[0]
    input_size = int(np.asarray(obs[agent]).shape[0])
    output_size = required_output_size(env.action_space(agent))
    genome = FixedTopologyGenome(
        layer_sizes=[input_size, hidden, output_size],
        activation="tanh",
        output_activation="linear",
        random_seed=0,
    )
    ind = creator.Individual()
    ind.genome = genome
    return ind


@pytest.mark.parametrize("env_name", CONTINUOUS_ENVS)
def test_box_head_wired_in_single_episode_evaluator(env_name):
    _maybe_skip_heterogeneous(env_name)
    env = PettingZooWrapper(env_name, env_config={"continuous_actions": True})
    try:
        ind = _make_individual(env)
        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env, max_steps=20, aggregation_method="sum"
        )
        # Evaluate end-to-end: the evaluator builds a FeedforwardNeuralPolicy
        # with action_space=env.action_space(agent), attaching a BoxHead.
        fitness = evaluator.evaluate(ind, seed=0)
        assert isinstance(fitness, tuple) and len(fitness) == 1
        assert np.isfinite(fitness[0])

        # Re-drive the policy directly to inspect the head + action bounds.
        obs = env.reset(seed=0)
        policies = evaluator._create_policies(ind.genome, obs)
        for agent, policy in policies.items():
            assert isinstance(policy.head, BoxHead), (
                f"agent={agent} should have BoxHead, got {type(policy.head).__name__}"
            )
            low = env.action_space(agent).low
            high = env.action_space(agent).high
            action = policy.get_action(obs[agent])
            assert action.dtype == np.float32
            assert np.all(action >= low - 1e-6)
            assert np.all(action <= high + 1e-6)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Trainer smoke tests on continuous envs
# ---------------------------------------------------------------------------


def _maybe_skip_heterogeneous(env_name: str) -> None:
    """Skip the trainer test if the env has role-dependent spaces.

    ``simple_tag_v3`` may publish different Box bounds for adversaries
    and good agents; the EA/CMA-ES trainers require a single shared
    space. Likewise, observation sizes may differ across roles, which
    :class:`SingleEpisodeEvaluator` rejects outright (no more
    weight-transfer branch — see ``docs/known_issues.md``). In either
    case we skip — the action-space error path is covered explicitly
    by ``test_rejects_heterogeneous_spaces`` below.
    """
    env = PettingZooWrapper(env_name, env_config={"continuous_actions": True})
    try:
        agents = list(env.possible_agents)
        ref_action = env.action_space(agents[0])
        for a in agents[1:]:
            if env.action_space(a) != ref_action:
                pytest.skip(
                    f"{env_name}: heterogeneous action spaces "
                    f"(agent {a!r} differs from {agents[0]!r})"
                )
        obs = env.reset(seed=0)
        sizes = {a: len(o) for a, o in obs.items()}
        if len(set(sizes.values())) > 1:
            pytest.skip(
                f"{env_name}: heterogeneous observation sizes {sizes}"
            )
    finally:
        env.close()


@pytest.mark.parametrize("env_name", CONTINUOUS_ENVS)
def test_ea_trainer_continuous_smoke(env_name):
    _maybe_skip_heterogeneous(env_name)

    trainer = EATrainer(
        hidden_size=8,
        population_size=4,
        crossover_rate=0.5,
        mutation_rate=0.5,
    )
    config = TrainingConfig(
        env_name=env_name,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,  # ~2 generations at pop=4
        algorithm_params={"seed": 0},
        env_config={"continuous_actions": True},
    )
    result = trainer.train(config)

    assert np.isfinite(result.metrics["final_fitness"])
    assert isinstance(result.policy._policy.head, BoxHead)

    # Sampled actions must respect the env's Box bounds.
    env = PettingZooWrapper(env_name, env_config={"continuous_actions": True})
    try:
        obs = env.reset(seed=0)
        agent = env.possible_agents[0]
        action = result.policy.get_action(obs[agent])
        low = env.action_space(agent).low
        high = env.action_space(agent).high
        assert action.dtype == np.float32
        assert np.all(action >= low - 1e-6)
        assert np.all(action <= high + 1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("env_name", CONTINUOUS_ENVS)
def test_cmaes_trainer_continuous_smoke(env_name):
    _maybe_skip_heterogeneous(env_name)

    trainer = CMAESTrainer(hidden_size=8, population_size=6, sigma0=0.3)
    config = TrainingConfig(
        env_name=env_name,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,  # ~2 generations at pop=6
        algorithm_params={"seed": 0},
        env_config={"continuous_actions": True},
    )
    result = trainer.train(config)

    assert np.isfinite(result.metrics["final_fitness"])
    assert isinstance(result.policy._policy.head, BoxHead)


# ---------------------------------------------------------------------------
# Heterogeneous-space guard tests
# ---------------------------------------------------------------------------


class _FakeHeteroEnv:
    """Stand-in wrapper exposing two agents with different action spaces."""

    def __init__(self):
        self.possible_agents = ["a", "b"]
        self._spaces = {
            "a": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "b": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
        }

    def action_space(self, agent: str):
        return self._spaces[agent]


def test_resolve_shared_action_space_rejects_heterogeneous():
    env = _FakeHeteroEnv()
    with pytest.raises(ValueError, match="shared action space"):
        _resolve_shared_action_space(env)


def test_resolve_shared_action_space_returns_reference_when_homogeneous():
    class _FakeHomoEnv:
        possible_agents = ["a", "b", "c"]
        _space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        def action_space(self, agent: str):
            return self._space

    env = _FakeHomoEnv()
    result = _resolve_shared_action_space(env)
    assert result == env._space


# ---------------------------------------------------------------------------
# TrainingConfig.continuous shortcut
# ---------------------------------------------------------------------------


class TestIdentityActivationAlias:
    """'identity' is an accepted alias for 'linear' in FixedTopologyGenome."""

    def test_identity_constructs_without_error(self):
        g = FixedTopologyGenome([3, 4, 2], output_activation="identity", random_seed=0)
        assert g is not None

    def test_identity_behaves_like_linear(self):
        g_id = FixedTopologyGenome(
            [3, 4, 2], activation="tanh", output_activation="identity", random_seed=42
        )
        g_lin = FixedTopologyGenome(
            [3, 4, 2], activation="tanh", output_activation="linear", random_seed=42
        )
        obs = np.array([0.5, -0.3, 1.2])
        p_id = FeedforwardNeuralPolicy(g_id)
        p_lin = FeedforwardNeuralPolicy(g_lin)
        np.testing.assert_array_almost_equal(p_id.forward(obs), p_lin.forward(obs))

    def test_identity_as_hidden_activation_behaves_like_linear(self):
        g = FixedTopologyGenome([3, 4, 2], activation="identity", random_seed=0)
        g_ref = FixedTopologyGenome([3, 4, 2], activation="linear", random_seed=0)
        obs = np.array([1.0, -1.0, 0.5])
        p = FeedforwardNeuralPolicy(g)
        p_ref = FeedforwardNeuralPolicy(g_ref)
        np.testing.assert_array_almost_equal(p.forward(obs), p_ref.forward(obs))


def test_ea_trainer_box_genome_uses_linear_output():
    """EATrainer sets output_activation='linear' in genome_config for Box spaces.
    """
    trainer = EATrainer(hidden_size=4, population_size=2)
    config = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=2,
        algorithm_params={"seed": 0},
        env_config={"continuous_actions": True},
    )
    result = trainer.train(config)
    genome_cfg = result.metadata["genome_config"]
    assert genome_cfg.get("output_activation") == "linear", (
        "EATrainer should set output_activation='linear' for Box action spaces; "
        f"got {genome_cfg.get('output_activation')!r}"
    )
    # Behavioural check: extreme weights should saturate toward the declared bounds.
    genome = result.policy._policy.genome
    genome.set_weights(np.ones(genome.get_weight_count()) * 20.0)
    env = PettingZooWrapper("simple_spread_v3", env_config={"continuous_actions": True})
    try:
        obs = env.reset(seed=0)
        agent = env.possible_agents[0]
        action = result.policy.get_action(obs[agent])
        high = env.action_space(agent).high
        low = env.action_space(agent).low
        mid = (high + low) / 2.0
        half_range = (high - low) / 2.0
        # Single tanh in BoxHead.decode with extreme input → normalized deviation near 1.
        # Double-tanh would cap it at tanh(1) ≈ 0.762.
        normalized_dev = np.abs(action - mid) / half_range
        assert np.all(normalized_dev > 0.9), (
            f"Expected action near bounds (norm-dev > 0.9 per dim), got {normalized_dev}"
        )
    finally:
        env.close()


class TestTrainingConfigContinuousShortcut:
    def _cfg(self, **kw):
        base = dict(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=1,
        )
        base.update(kw)
        return TrainingConfig(**base)

    def test_default_off(self):
        cfg = self._cfg()
        assert cfg.continuous is False
        assert "continuous_actions" not in cfg.env_config

    def test_shortcut_sets_env_config(self):
        cfg = self._cfg(continuous=True)
        assert cfg.continuous is True
        assert cfg.env_config["continuous_actions"] is True

    def test_shortcut_preserves_other_env_config_keys(self):
        cfg = self._cfg(continuous=True, env_config={"max_cycles": 25, "N": 3})
        assert cfg.env_config == {
            "max_cycles": 25,
            "N": 3,
            "continuous_actions": True,
        }

    def test_explicit_true_syncs_flag(self):
        cfg = self._cfg(env_config={"continuous_actions": True})
        assert cfg.continuous is True
        assert cfg.env_config["continuous_actions"] is True

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            self._cfg(continuous=True, env_config={"continuous_actions": False})

    def test_explicit_false_without_shortcut_is_kept(self):
        cfg = self._cfg(env_config={"continuous_actions": False})
        assert cfg.continuous is False
        assert cfg.env_config["continuous_actions"] is False
