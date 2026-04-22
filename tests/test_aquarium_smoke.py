"""
Smoke tests for marl-aquarium (``aquarium_v0``) integration.

Verifies that ``aquarium_v0`` is accepted by:

- :class:`PettingZooWrapper` (generic env wrapper),
- :class:`EATrainer` (DEAP neuroevolution path),
- :class:`MADDPGTrainer` (AgileRL multi-agent RL path),

and that both trainers return a policy whose ``get_action`` maps a real
aquarium observation to a valid action.

Requires the optional ``aquarium`` extra (``pip install marl-aquarium``).
RL test additionally requires the ``rl-baselines`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest

# Gate the entire module on marl-aquarium being installed.
pytest.importorskip(
    "marl_aquarium",
    reason="marl-aquarium not installed; run `pip install marl-aquarium`.",
)

from deap_pettingzoo.baselines import (
    BudgetMode,
    EATrainer,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.environments import PettingZooWrapper


# Keep agent counts and episode length small so tests stay fast.
SMALL_AQUARIUM_CONFIG = {
    "predator_count": 1,
    "prey_count": 2,
    "predator_observe_count": 1,
    "prey_observe_count": 2,
    "max_time_steps": 30,
    "keep_prey_count_constant": True,
}


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Wrapper smoke
# ---------------------------------------------------------------------------


class TestPettingZooWrapperAquarium:
    def test_wrapper_accepts_aquarium(self):
        wrapper = PettingZooWrapper(
            "aquarium_v0", env_config=SMALL_AQUARIUM_CONFIG
        )
        try:
            obs = wrapper.reset(seed=0)
            assert isinstance(obs, dict)
            assert len(obs) == len(wrapper.possible_agents) == 3
            a0 = wrapper.possible_agents[0]
            # Aquarium obs dim depends on observe-counts; just assert it's
            # a 1-D Box matching the per-agent obs array shape.
            space_shape = wrapper.observation_space(a0).shape
            assert len(space_shape) == 1
            assert obs[a0].shape == space_shape
            assert wrapper.action_space(a0).n == 16
        finally:
            wrapper.close()

    def test_wrapper_step_cycle(self):
        wrapper = PettingZooWrapper(
            "aquarium_v0", env_config=SMALL_AQUARIUM_CONFIG
        )
        try:
            obs = wrapper.reset(seed=0)
            # Build one-hot-like action arrays (wrapper takes argmax).
            actions = {}
            for agent in wrapper.agents:
                logits = np.zeros(wrapper.action_space(agent).n)
                logits[0] = 1.0
                actions[agent] = logits
            next_obs, rewards, terms, truncs, infos = wrapper.step(actions)
            assert set(next_obs.keys()).issubset(set(wrapper.possible_agents))
            assert all(isinstance(v, (int, float)) for v in rewards.values())
        finally:
            wrapper.close()


# ---------------------------------------------------------------------------
# EA smoke
# ---------------------------------------------------------------------------


class TestEATrainerAquarium:
    def test_ea_trains_on_aquarium(self):
        """Tiny EA run on aquarium_v0: 2 generations, pop 4."""
        trainer = EATrainer(
            hidden_size=16,
            population_size=4,
            tournament_size=2,
            mutation_rate=0.5,
        )
        config = TrainingConfig(
            env_name="aquarium_v0",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=2,  # 2 generations worth of evaluations
            algorithm_params={"seed": 0},
            env_config=SMALL_AQUARIUM_CONFIG,
        )
        result = trainer.train(config)
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "ea"
        assert result.policy is not None

        # The resulting policy should produce a valid discrete action
        # from a real aquarium observation.
        wrapper = PettingZooWrapper(
            "aquarium_v0", env_config=SMALL_AQUARIUM_CONFIG
        )
        try:
            obs = wrapper.reset(seed=1)
            agent = wrapper.possible_agents[0]
            action_vec = result.policy.get_action(obs[agent])
            # FeedforwardNeuralPolicy returns raw logits; wrapper.step
            # converts via argmax, but we check shape here.
            assert np.asarray(action_vec).shape[-1] == 16
        finally:
            wrapper.close()


# ---------------------------------------------------------------------------
# RL smoke (MADDPG)
# ---------------------------------------------------------------------------


agilerl = pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)

from deap_pettingzoo.baselines import (  # noqa: E402 - gated import
    AgileRLPolicyWrapper,
    MADDPGTrainer,
)


class TestMADDPGAquarium:
    def test_maddpg_trains_on_aquarium(self):
        """Tiny MADDPG run on aquarium_v0: 200 env steps."""
        config = TrainingConfig(
            env_name="aquarium_v0",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=200,
            algorithm_params={
                "seed": 0,
                "batch_size": 16,
                "memory_size": 500,
                "learning_delay": 50,
            },
            env_config=SMALL_AQUARIUM_CONFIG,
        )
        result = MADDPGTrainer().train(config)
        assert isinstance(result, TrainingResult)
        assert result.algorithm == "maddpg"
        assert isinstance(result.policy, AgileRLPolicyWrapper)
        assert result.metadata["env_name"] == "aquarium_v0"
        assert result.metrics["env_steps"] == 200
