"""End-to-end parallel RL co-evolution tests (Phase 2).

Gated on AgileRL + torch availability.  Verifies:

* EA side evaluates against FrozenRLPolicyWrapper snapshots in
  parallel (num_workers=2) without crashing.
* _RLRolloutPayload is picklable (required for ProcessPoolExecutor).
* _rl_rollout_worker_fn can be called directly (serial unit test).
* FrozenRLPolicyWrapper produced by make_frozen_cpu_snapshot is
  picklable and produces valid actions.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")
pytest.importorskip("agilerl")
pytest.importorskip("torch")

from deap_pettingzoo.baselines.agilerl_policy import (
    AgileRLPolicyWrapper,
    FrozenRLPolicyWrapper,
)
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)
from deap_pettingzoo.baselines.coevolution.rl_side import (
    _RLRolloutPayload,
    _RLTrajectory,
    _rl_rollout_worker_fn,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ea_cfg(seed: int = 0) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _rl_cfg(algo: str = "maddpg", **overrides) -> dict:
    base = {
        "algorithm": algo,
        "rollout_episodes_per_gen": 2,
        "learning_steps_per_gen": 1,
        "batch_size": 8,
        "memory_size": 256,
        "learning_delay": 0,
        "device": "cpu",
        "seed": 0,
    }
    base.update(overrides)
    return base


def _tcfg(
    coev_cfg: CoEvolutionConfig,
    budget: int = 12,
    env_config: dict | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True} if env_config is None else env_config,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        algorithm_params={"coevolution": coev_cfg},
    )


def _build_rl_wrapper(
    env_name: str = "simple_tag_v3",
    *,
    algo: str = "maddpg",
    role: tuple = ("agent",),
    env_config: dict | None = None,
) -> AgileRLPolicyWrapper:
    """Initialise an AgileRLPolicyWrapper by running a minimal RLSideState.

    ``algo="ippo"`` with ``role=("adversary",)`` exercises a grouped
    StochasticActor (shared across the homogeneous adversaries).
    """
    from deap_pettingzoo.baselines.coevolution import make_side_state

    if env_config is None:
        env_config = {} if algo.startswith("ippo") else {"continuous_actions": True}
    side = Side("side", role, "rl", _rl_cfg(algo))
    state = make_side_state(side)
    env = PettingZooWrapper(env_name, env_config=env_config)
    state.initialise(env, rng=np.random.default_rng(0))
    assert isinstance(state._wrapper, AgileRLPolicyWrapper)
    return state._wrapper


# ---------------------------------------------------------------------------
# FrozenRLPolicyWrapper unit tests
# ---------------------------------------------------------------------------


class TestFrozenWrapper:
    def test_make_frozen_cpu_snapshot_returns_frozen(self):
        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()
        assert isinstance(frozen, FrozenRLPolicyWrapper)

    def test_frozen_wrapper_picklable(self):
        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()
        data = pickle.dumps(frozen)
        frozen2 = pickle.loads(data)
        assert isinstance(frozen2, FrozenRLPolicyWrapper)
        assert frozen2.algorithm_name == frozen.algorithm_name

    def test_frozen_wrapper_get_action_continuous(self):
        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()
        env = PettingZooWrapper(
            "simple_tag_v3", env_config={"continuous_actions": True}
        )
        obs_spaces = {a: env.observation_space(a) for a in env.possible_agents}
        rl_agents = [a for a in env.possible_agents if a.startswith("agent")]
        if not rl_agents:
            pytest.skip("No 'agent' prefixed agents in simple_tag_v3")
        obs = obs_spaces[rl_agents[0]].sample()
        act = frozen.get_action(obs)
        assert act is not None

    def test_frozen_wrapper_actor_in_eval_mode(self):
        """All actors in the snapshot must be in eval mode (no gradient)."""
        import torch.nn as nn
        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()
        for actor in frozen._actors:
            if isinstance(actor, nn.Module):
                assert not actor.training, "Actor must be in eval mode"


class TestFrozenWrapperIPPO:
    """IPPO (on-policy StochasticActor) frozen snapshots — previously blocked
    by copy.deepcopy, now built via state_dict reconstruction."""

    def test_ippo_snapshot_builds_without_fallback(self):
        """make_frozen_cpu_snapshot must succeed for an IPPO StochasticActor
        (grouped across the homogeneous adversaries) — no exception/warning."""
        wrapper = _build_rl_wrapper(algo="ippo", role=("adversary",))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            frozen = wrapper.make_frozen_cpu_snapshot()
        assert isinstance(frozen, FrozenRLPolicyWrapper)

    def test_ippo_frozen_picklable_and_acts(self):
        """Frozen IPPO snapshot round-trips through pickle and yields a valid
        discrete (int) action for the env's action space."""
        wrapper = _build_rl_wrapper(algo="ippo", role=("adversary",))
        frozen = pickle.loads(pickle.dumps(wrapper.make_frozen_cpu_snapshot()))
        assert isinstance(frozen, FrozenRLPolicyWrapper)

        env = PettingZooWrapper("simple_tag_v3", env_config={})
        adversaries = [a for a in env.possible_agents if a.startswith("adversary")]
        obs = env.observation_space(adversaries[0]).sample()
        act = frozen.get_action(obs)
        assert isinstance(act, int)
        assert 0 <= act < env.action_space(adversaries[0]).n

    def test_ippo_grouped_actor_mapping(self):
        """A grouped IPPO shares one actor across adversaries; the frozen
        wrapper must carry the per-agent → actor index map."""
        wrapper = _build_rl_wrapper(algo="ippo", role=("adversary",))
        frozen = wrapper.make_frozen_cpu_snapshot()
        # 3 adversaries share a single actor.
        assert len(frozen._actors) < len(frozen.agent_ids)
        assert frozen._actor_idx_map is not None
        assert len(frozen._actor_idx_map) == len(frozen.agent_ids)


# ---------------------------------------------------------------------------
# _RLRolloutPayload picklability
# ---------------------------------------------------------------------------


class TestRLRolloutPayloadPickle:
    def test_payload_picklable(self):
        """An _RLRolloutPayload with a FrozenRLPolicyWrapper must round-trip
        through pickle without error."""
        from deap_pettingzoo.baselines.coevolution.pools import OpponentSnapshot

        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()

        payload = _RLRolloutPayload(
            env_name="simple_tag_v3",
            env_config={"continuous_actions": True},
            max_steps=10,
            episode_seed=42,
            frozen_rl_snapshot=frozen,
            rl_agent_ids=list(frozen.agent_ids),
            rl_role_prefixes=("agent",),
            opponent_snapshots={},
            side_role_prefixes={},
        )

        data = pickle.dumps(payload)
        payload2 = pickle.loads(data)
        assert payload2.env_name == "simple_tag_v3"
        assert payload2.episode_seed == 42


# ---------------------------------------------------------------------------
# _rl_rollout_worker_fn direct invocation (serial)
# ---------------------------------------------------------------------------


class TestRLRolloutWorkerFn:
    def test_worker_fn_returns_trajectory(self):
        """_rl_rollout_worker_fn called directly must return _RLTrajectory."""
        from deap_pettingzoo.baselines.coevolution.pools import OpponentSnapshot

        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()

        payload = _RLRolloutPayload(
            env_name="simple_tag_v3",
            env_config={"continuous_actions": True},
            max_steps=5,
            episode_seed=7,
            frozen_rl_snapshot=frozen,
            rl_agent_ids=list(frozen.agent_ids),
            rl_role_prefixes=("agent",),
            opponent_snapshots={},
            side_role_prefixes={},
        )

        traj = _rl_rollout_worker_fn(payload)
        assert isinstance(traj, _RLTrajectory)
        assert isinstance(traj.episode_return, float)
        assert isinstance(traj.steps, list)
        assert len(traj.steps) <= payload.max_steps

    def test_worker_fn_step_data_keys_match_rl_agents(self):
        """Each step's obs/action dicts must be keyed by rl_agent_ids."""
        from deap_pettingzoo.baselines.coevolution.pools import OpponentSnapshot

        wrapper = _build_rl_wrapper()
        frozen = wrapper.make_frozen_cpu_snapshot()
        rl_ids = list(frozen.agent_ids)

        payload = _RLRolloutPayload(
            env_name="simple_tag_v3",
            env_config={"continuous_actions": True},
            max_steps=5,
            episode_seed=99,
            frozen_rl_snapshot=frozen,
            rl_agent_ids=rl_ids,
            rl_role_prefixes=("agent",),
            opponent_snapshots={},
            side_role_prefixes={},
        )

        traj = _rl_rollout_worker_fn(payload)
        for step in traj.steps:
            for aid in step.obs:
                assert aid in rl_ids, f"Unexpected agent key {aid!r} in step obs"


# ---------------------------------------------------------------------------
# End-to-end parallel EA-vs-RL with num_workers=2
# ---------------------------------------------------------------------------


class TestEAvsRLParallel:
    def test_ea_vs_rl_num_workers_2_succeeds(self):
        """A competitive run with num_workers=2 and an RL side must complete
        without error. EA evaluates against FrozenRLPolicyWrapper in
        parallel; RL still collects serially (Phase 2 serial fallback)."""
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg")),
            ],
            k_opponents=1,
            max_steps_per_episode=10,
            num_workers=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CoEvolutionTrainer().train(_tcfg(cfg, budget=8))

        # Informational warning fires (not a crash).
        msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("RL side is present" in m for m in msgs)

        # Worker count is preserved.
        assert result.metadata["coevolution_num_workers"] == 2

        # A result was produced.
        assert result.best_policy is not None

    def test_matd3_ea_parallel(self):
        """MATD3 RL side with EA opponent in parallel (num_workers=2)."""
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(1)),
                Side("evaders", ("agent",), "rl", _rl_cfg("matd3")),
            ],
            k_opponents=1,
            max_steps_per_episode=10,
            num_workers=2,
            seed=1,
        )
        result = CoEvolutionTrainer().train(_tcfg(cfg, budget=8))
        assert result.best_policy is not None

    def test_ea_vs_ippo_num_workers_2_succeeds(self):
        """On-policy IPPO RL side co-evolving against an EA side, with the EA
        evaluating in parallel against the frozen IPPO opponent snapshot.

        The IPPO side controls the homogeneous adversaries (grouped actor);
        the run must complete and never fall back to the live-policy snapshot
        path (which is what would break parallel pickling)."""
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "rl",
                     _rl_cfg("ippo", rollout_episodes_per_gen=2)),
                Side("evaders", ("agent",), "ea", _ea_cfg(3)),
            ],
            k_opponents=1,
            max_steps_per_episode=10,
            num_workers=2,
            seed=7,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CoEvolutionTrainer().train(
                _tcfg(cfg, budget=8, env_config={})
            )
        fallback = [
            str(w.message) for w in caught
            if w.category is RuntimeWarning
            and "make_frozen_cpu_snapshot failed" in str(w.message)
        ]
        assert not fallback, f"unexpected snapshot fallback: {fallback}"
        assert result.best_policy is not None
