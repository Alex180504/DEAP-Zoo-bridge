"""End-to-end MultiDiscrete tests for the evolutionary stack + the
environment-compatibility gate.

The built-in MPE2 environments expose only ``Discrete`` action spaces, so
these tests run against the synthetic homogeneous ``MultiDiscrete`` env in
:mod:`tests.multidiscrete_env`.

Covers:
- :class:`EATrainer` / :class:`CMAESTrainer` / :class:`PSOTrainer` training on
  a MultiDiscrete env and producing a ``MultiDiscreteHead``-bearing policy
  whose actions are valid integer vectors.
- :func:`inspect_environment` correctly flagging MultiDiscrete spaces.
- :func:`check_compatibility` allowing the EA family + IPPO and rejecting the
  off-policy MADDPG / MATD3 learners on MultiDiscrete envs.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.baselines.ea_trainer import EATrainer
from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
from deap_pettingzoo.baselines.compatibility import (
    check_compatibility,
    inspect_environment,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import MultiDiscreteHead

from tests.multidiscrete_env import MD_ENV_NAME, register_md_env


@pytest.fixture(scope="module", autouse=True)
def _md_env():
    """Register the synthetic MultiDiscrete env for this module."""
    from deap_pettingzoo.environments.registry import is_registered, unregister_env

    register_md_env()
    yield MD_ENV_NAME
    if is_registered(MD_ENV_NAME):
        unregister_env(MD_ENV_NAME)


def _assert_valid_md_action(action, nvec=(3, 4)):
    arr = np.asarray(action)
    assert arr.dtype == np.int64
    assert arr.shape == (len(nvec),)
    for i, n in enumerate(nvec):
        assert 0 <= arr[i] < n


# ---------------------------------------------------------------------------
# EA / CMA-ES / PSO smoke training on a MultiDiscrete env
# ---------------------------------------------------------------------------


def test_ea_trainer_multidiscrete_smoke():
    trainer = EATrainer(
        hidden_size=8, population_size=4, crossover_rate=0.5, mutation_rate=0.5
    )
    config = TrainingConfig(
        env_name=MD_ENV_NAME,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,
        algorithm_params={"seed": 0},
    )
    result = trainer.train(config)

    assert np.isfinite(result.metrics["final_fitness"])
    assert isinstance(result.policy._policy.head, MultiDiscreteHead)

    env = PettingZooWrapper(MD_ENV_NAME)
    try:
        obs = env.reset(seed=1)
        agent = env.possible_agents[0]
        _assert_valid_md_action(result.policy.get_action(obs[agent]))
    finally:
        env.close()


def test_cmaes_trainer_multidiscrete_smoke():
    trainer = CMAESTrainer(hidden_size=8, population_size=6, sigma0=0.3)
    config = TrainingConfig(
        env_name=MD_ENV_NAME,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        algorithm_params={"seed": 0},
    )
    result = trainer.train(config)

    assert np.isfinite(result.metrics["final_fitness"])
    assert isinstance(result.policy._policy.head, MultiDiscreteHead)


def test_pso_trainer_multidiscrete_smoke():
    trainer = PSOTrainer(hidden_size=8, population_size=6)
    config = TrainingConfig(
        env_name=MD_ENV_NAME,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        algorithm_params={"seed": 0},
    )
    result = trainer.train(config)

    assert np.isfinite(result.metrics["final_fitness"])
    assert isinstance(result.policy._policy.head, MultiDiscreteHead)


# ---------------------------------------------------------------------------
# Compatibility introspection + gate
# ---------------------------------------------------------------------------


def test_inspect_environment_flags_multidiscrete():
    info = inspect_environment(MD_ENV_NAME)
    assert info.has_multidiscrete_actions is True
    assert info.has_continuous_actions is False
    # act_dim is the sum(nvec) output size = 3 + 4 = 7.
    assert info.act_dim == 7
    # to_dict must round-trip the new field.
    assert info.to_dict()["has_multidiscrete_actions"] is True


def test_ea_is_compatible_with_multidiscrete():
    info = inspect_environment(MD_ENV_NAME)
    result = check_compatibility(EATrainer(), info, "ea")
    assert result.compatible


@pytest.mark.parametrize("algo_cls_name", ["MADDPGTrainer", "MATD3Trainer"])
def test_offpolicy_rejected_on_multidiscrete(algo_cls_name):
    pytest.importorskip("agilerl")
    pytest.importorskip("torch")
    import deap_pettingzoo.baselines.agilerl_trainer as at

    trainer = getattr(at, algo_cls_name)()
    info = inspect_environment(MD_ENV_NAME)
    result = check_compatibility(trainer, info, algo_cls_name)
    assert not result.compatible
    assert any("MultiDiscrete" in issue for issue in result.issues)
