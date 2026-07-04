"""On-policy (IPPO / GroupedIPPO) co-evolution tests.

Covers the on-policy additions to the RL co-evolution sides:

* :class:`OnPolicyRolloutBuffer` assembles the 8-tuple AgileRL's
  ``IPPO.learn`` expects, with the leading ``num_envs == 1`` axis.
* :class:`RLSideState` accepts on-policy algorithms (no longer raises),
  keeps no replay buffer, and learns from a freshly collected rollout.
* ``RLPolicyPool`` snapshots an IPPO learner without the deepcopy fallback.
* :class:`RLEvolveHPOSideState` trains an on-policy population and exposes a
  picklable champion snapshot.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

pytest.importorskip("agilerl")
pytest.importorskip("torch")
pytest.importorskip("pettingzoo.mpe")

import torch  # noqa: E402

from deap_pettingzoo.baselines.agilerl_onpolicy import OnPolicyRolloutBuffer  # noqa: E402
from deap_pettingzoo.baselines.agilerl_policy import FrozenRLPolicyWrapper  # noqa: E402
from deap_pettingzoo.baselines.coevolution import Side, make_side_state  # noqa: E402
from deap_pettingzoo.baselines.coevolution.pools import (  # noqa: E402
    FrozenPolicyPool,
    RLPolicyPool,
)
from deap_pettingzoo.baselines.coevolution.rl_side import _RLDriverContext  # noqa: E402
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubPolicy:
    """Trivial opponent policy returning the no-op discrete action."""

    def get_action(self, obs, **_):
        return 0

    def reset(self):
        pass


def _ippo_rl_side(env, role=("adversary",), episodes=2):
    side = Side(
        "rl", role, "rl",
        {"algorithm": "ippo", "rollout_episodes_per_gen": episodes,
         "device": "cpu", "seed": 0},
    )
    state = make_side_state(side)
    state.initialise(env, rng=np.random.default_rng(0))
    return state


# ---------------------------------------------------------------------------
# OnPolicyRolloutBuffer
# ---------------------------------------------------------------------------


class TestOnPolicyRolloutBuffer:
    def test_assembles_8tuple_with_num_envs_axis(self):
        ids = ["a", "b"]
        buf = OnPolicyRolloutBuffer(ids)
        for _ in range(5):
            buf.add(
                obs={a: np.zeros(4, np.float32) for a in ids},
                action={a: np.array([1], np.int64) for a in ids},
                log_prob={a: np.array([-0.5], np.float32) for a in ids},
                value={a: np.array([0.1], np.float32) for a in ids},
                reward={a: 1.0 for a in ids},
                done={a: np.zeros(1, np.float32) for a in ids},
            )
        assert len(buf) == 5

        exp = buf.experiences(
            {a: np.zeros(4, np.float32) for a in ids},
            {a: np.ones(1, np.float32) for a in ids},
        )
        assert len(exp) == 8
        states, actions, log_probs, rewards, dones, values, next_states, next_dones = exp
        assert set(states) == set(ids)
        assert len(states["a"]) == 5
        # Leading num_envs == 1 axis on every stored entry.
        assert states["a"][0].shape == (1, 4)
        assert rewards["a"][0].shape == (1,)
        assert dones["a"][0].shape == (1,)
        assert next_states["a"].shape == (1, 4)
        assert next_dones["a"].shape == (1,)


# ---------------------------------------------------------------------------
# On-policy RLSideState
# ---------------------------------------------------------------------------


class TestOnPolicyRLSide:
    def test_ippo_initialise_no_longer_raises(self):
        env = PettingZooWrapper("simple_tag_v3", env_config={})
        state = _ippo_rl_side(env)
        assert state.is_on_policy
        # On-policy keeps no replay buffer.
        assert state._memory is None
        assert state._wrapper is not None

    def test_onpolicy_step_generation_learns_and_updates(self):
        env = PettingZooWrapper("simple_tag_v3", env_config={})
        state = _ippo_rl_side(env, episodes=2)
        actor = list(state._agent.actors.values())[0]
        before = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()

        ctx = _RLDriverContext(
            env_wrapper=env,
            learner_role_prefixes=("adversary",),
            opponent_sides=[("evaders", ("agent",), FrozenPolicyPool([_StubPolicy()]))],
            max_steps=12,
            rng=np.random.default_rng(1),
        )
        state.step_generation(lambda s: ctx)

        after = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()
        assert state._total_learn_calls == 1
        assert state._total_learn_failures == 0
        assert state._total_transitions > 0
        assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# RLPolicyPool snapshots for an IPPO learner
# ---------------------------------------------------------------------------


class TestRLPoolSnapshotIPPO:
    def test_snapshot_opponents_no_fallback(self):
        env = PettingZooWrapper("simple_tag_v3", env_config={})
        pool = _ippo_rl_side(env).current_pool()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            snaps = pool.snapshot_opponents(np.random.default_rng(0), 2)
        fallback = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "make_frozen_cpu_snapshot failed" in str(w.message)
        ]
        assert not fallback
        assert len(snaps) == 2
        # The frozen IPPO policy is picklable across worker boundaries.
        pickle.loads(pickle.dumps(snaps[0].payload["policy"]))

    def test_snapshot_sync_returns_frozen_isolated_pool(self):
        env = PettingZooWrapper("simple_tag_v3", env_config={})
        pool = _ippo_rl_side(env).current_pool()
        synced = pool.snapshot_sync()
        assert isinstance(synced, RLPolicyPool)
        assert isinstance(synced.policy, FrozenRLPolicyWrapper)


# ---------------------------------------------------------------------------
# On-policy evolvable-HPO side
# ---------------------------------------------------------------------------


class TestOnPolicyHPO:
    def test_ippo_hpo_registered(self):
        from deap_pettingzoo.baselines.coevolution.rl_hpo_side import (
            SUPPORTED_HPO_RL_ALGORITHMS,
        )

        assert "ippo_hpo" in SUPPORTED_HPO_RL_ALGORITHMS
        assert SUPPORTED_HPO_RL_ALGORITHMS["ippo_hpo"][2] is True  # on-policy

    def test_ippo_hpo_trains_and_snapshots(self):
        mpe2 = pytest.importorskip("mpe2")

        env = mpe2.simple_spread_v3.parallel_env(N=3, max_cycles=10)
        env.reset(seed=0)

        side = Side(
            "learners", ("agent",), "rl_hpo",
            {"algorithm": "ippo_hpo", "population_size": 2,
             "evo_every_n_steps": 25, "device": "cpu", "seed": 0},
        )
        state = make_side_state(side)
        state.initialise(env, rng=np.random.default_rng(0))
        assert state._on_policy

        actor = list(state._population[0].actors.values())[0]
        before = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()

        class _Ev:
            env_wrapper = env

        state.step_generation(lambda s: _Ev())

        after = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()
        assert not torch.allclose(before, after)

        snap = pickle.loads(pickle.dumps(state.best_snapshot()))
        assert isinstance(snap, FrozenRLPolicyWrapper)
