"""Regression tests for the rl_hpo co-evolution side crash.

Drives the full producer -> rebuild -> infer chain that raised
``AttributeError: 'list' object has no attribute 'get_action'`` when
``current_pool()`` wrapped the champion in a list and ``RLPolicyPool``
stored that list as its policy.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)
pytest.importorskip("torch", reason="PyTorch not installed.")
pytestmark = pytest.mark.slow

from deap_pettingzoo.baselines.coevolution.pools import rebuild_opponent_policy
from deap_pettingzoo.baselines.coevolution.rl_hpo_side import RLEvolveHPOSideState
from deap_pettingzoo.baselines.coevolution.sides import Side
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


def _hpo_side_cfg() -> dict:
    return {
        "algorithm": "maddpg_hpo",
        "population_size": 2,
        "evo_every_n_steps": 20,
        "device": "cpu",
        "seed": 0,
    }


class TestRLHPOPoolRoundtrip:
    """Regression tests for B4: RLPolicyPool([list]) -> AttributeError crash."""

    def test_rl_hpo_pool_roundtrip(self):
        """Drives the full producer -> rebuild -> infer chain.
        """
        side = Side("hpo", ("agent",), "rl_hpo", _hpo_side_cfg())
        state = RLEvolveHPOSideState(side)
        env = PettingZooWrapper("simple_spread_v3", env_config={"max_cycles": 10})
        try:
            state.initialise(env, rng=np.random.default_rng(0))

            pool = state.current_pool()
            rng = np.random.default_rng(42)
            snapshots = pool.snapshot_opponents(rng, k=1)
            assert len(snapshots) == 1

            # Pickle round-trip — validates picklability across process boundaries.
            snap = pickle.loads(pickle.dumps(snapshots[0]))
            action_space = env.action_space(env.possible_agents[0])
            policy = rebuild_opponent_policy(snap, action_space=action_space)

            assert hasattr(policy, "get_action"), (
                f"rebuild_opponent_policy returned {type(policy)!r} which has no "
                "'get_action' — the list was not unwrapped before the fix"
            )

            obs = env.reset()
            first_agent = state._learner_ids[0]
            if hasattr(policy, "reset"):
                policy.reset()
            action = policy.get_action(obs[first_agent])
            assert action is not None
        finally:
            env.close()

    def test_rl_hpo_side_detected_as_rl(self):
        """The is_rl_side marker must be True so co-evolution detectors treat it as RL.
        """
        side = Side("hpo", ("agent",), "rl_hpo", _hpo_side_cfg())
        state = RLEvolveHPOSideState(side)
        assert getattr(state, "is_rl_side", False) is True, (
            "RLEvolveHPOSideState.is_rl_side must be True for the co-evolution "
            "GPU-contention and worker-fallback detectors to fire correctly"
        )
