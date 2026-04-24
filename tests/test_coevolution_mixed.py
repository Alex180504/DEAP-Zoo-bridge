"""Tests for mixing EA/CMA-ES learners with frozen (RL-substitute) opponents.

Frozen opponents cover the "train EA vs RL" use case: the RL side is
pre-trained and wrapped as a static pool of
:class:`PolicyProtocol` implementations, then dropped into the
co-evolution run as a ``Side(optimizer='frozen', ...)``. These tests
exercise that wiring with random-action stand-ins standing in for a
trained RL policy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


class _RandomPolicy:
    """Stand-in for any pre-trained RL policy. Stateless."""

    def __init__(self, action_space, rng: np.random.Generator):
        self.sp = action_space
        self._rng = rng

    def get_action(self, observation, action_mask=None):
        return self.sp.sample()

    def reset(self):
        pass


def _build_frozen_side(env: PettingZooWrapper, prefix: str) -> list:
    rng = np.random.default_rng(0)
    return [
        _RandomPolicy(env.action_space(a), rng)
        for a in env.possible_agents
        if a.startswith(prefix)
    ]


def test_ea_vs_frozen_runs_and_reports_ea_policy():
    env = PettingZooWrapper("simple_tag_v3")
    env.reset(seed=0)
    evader_policies = _build_frozen_side(env, "agent")
    env.close()

    cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side(
                "pursuers", ("adversary",), "ea",
                {"population_size": 4, "hidden_size": 8, "seed": 0},
            ),
            Side(
                "evaders", ("agent",), "frozen",
                frozen_policies=evader_policies,
            ),
        ],
        k_opponents=2,
        max_steps_per_episode=20,
        seed=42,
        reporting_side="pursuers",
    )
    result = CoEvolutionTrainer().train(
        TrainingConfig(
            env_name="simple_tag_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=32,
            algorithm_params={"coevolution": cfg},
        )
    )
    # Frozen side reports None fitness (no evolution); EA side has a
    # concrete float.
    assert result.metrics["final_best"]["evaders"] is None
    assert isinstance(result.metrics["final_best"]["pursuers"], float)
    # Reporting policy comes from the EA side.
    assert type(result.policy).__name__ == "FeedforwardNeuralPolicy"


def test_cma_vs_frozen():
    env = PettingZooWrapper("simple_tag_v3")
    env.reset(seed=0)
    evader_policies = _build_frozen_side(env, "agent")
    env.close()

    cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side(
                "pursuers", ("adversary",), "cma",
                {
                    "population_size": 4, "hidden_size": 8, "seed": 0,
                    "sigma0": 0.3,
                },
            ),
            Side(
                "evaders", ("agent",), "frozen",
                frozen_policies=evader_policies,
            ),
        ],
        k_opponents=2,
        max_steps_per_episode=20,
        seed=7,
    )
    result = CoEvolutionTrainer().train(
        TrainingConfig(
            env_name="simple_tag_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=32,
            algorithm_params={"coevolution": cfg},
        )
    )
    assert result.metadata["sides"][0]["optimizer"] == "cma"
    assert isinstance(result.metrics["final_best"]["pursuers"], float)


def test_frozen_without_policies_raises():
    with pytest.raises(ValueError, match="frozen_policies"):
        Side("evaders", ("agent",), optimizer="frozen", frozen_policies=[])
