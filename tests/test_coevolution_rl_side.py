"""End-to-end tests for bidirectional EA-RL co-evolution.

Gated on AgileRL availability (``pytest.importorskip("agilerl")``).
Exercises:

* RL-vs-EA: ``Side(optimizer="rl", ...)`` paired with an EA side,
  one generation budget, asserts both sides advance.
* RL-vs-RL: two RL sides facing each other.
* Side validation: unknown algorithm raises, missing algorithm raises,
  on-policy IPPO raises NotImplementedError from initialise.
* num_workers fallback: a competitive run with num_workers=2 + an RL
  side emits a RuntimeWarning and succeeds.

Tiny budgets keep these under the test-suite time cap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")
pytest.importorskip("agilerl")
pytest.importorskip("torch")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    RLPolicyPool,
    RLSideState,
    Side,
)


def _ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _rl_cfg(algo: str = "maddpg", **overrides) -> dict:
    cfg = {
        "algorithm": algo,
        "rollout_episodes_per_gen": 2,
        "learning_steps_per_gen": 1,
        "batch_size": 8,
        "memory_size": 256,
        "learning_delay": 0,
        "device": "cpu",
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def _tcfg(coev_cfg: CoEvolutionConfig, budget: int = 12) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True},  # MADDPG/MATD3 native
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        coevolution_config=coev_cfg,
    )


# ---------------------------------------------------------------------------
# Side-level validation
# ---------------------------------------------------------------------------


class TestSideValidation:
    def test_optimizer_rl_accepted(self):
        # Should not raise.
        Side("learners", ("agent",), "rl", _rl_cfg())

    def test_unknown_optimizer_still_rejected(self):
        with pytest.raises(ValueError, match="optimizer must be"):
            Side("x", ("agent",), "garbage", {})

    def test_missing_algorithm_raises_at_factory_time(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state

        side = Side("learners", ("agent",), "rl", {})  # no algorithm key
        with pytest.raises(ValueError, match="algorithm.*required"):
            make_side_state(side)

    def test_unknown_algorithm_raises_at_factory_time(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state

        side = Side(
            "learners", ("agent",), "rl",
            {"algorithm": "ddqn_hypothetical"},
        )
        with pytest.raises(ValueError, match="unknown algorithm"):
            make_side_state(side)


class TestOnPolicySupported:
    def test_ippo_initialises_without_replay_buffer(self):
        """On-policy IPPO is now supported: initialise() builds the agent +
        wrapper + pool and keeps no replay buffer (PPO collects fresh
        rollouts each generation).  Detailed learning behaviour is covered by
        ``tests/test_coevolution_rl_onpolicy.py``."""
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side(
            "learners", ("agent",), "rl",
            _rl_cfg(algo="ippo"),
        )
        state = RLSideState(side)
        assert state.is_on_policy
        env = PettingZooWrapper(
            "simple_tag_v3",
            env_config={"continuous_actions": False},
        )
        try:
            state.initialise(env, rng=np.random.default_rng(0))
            assert state._memory is None
            assert state._wrapper is not None
            assert state._pool is not None
        finally:
            env.close()


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------


class TestEAvsRL:
    def test_competitive_2pop_runs(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg")),
            ],
            k_opponents=1,  # RLPolicyPool is size-1; num_workers=1 is the hard constraint
            max_steps_per_episode=15,
            num_workers=1,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_tcfg(cfg))
        assert result.algorithm == "coevolution"
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Both sides accumulated history.
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            assert len(hist) >= 1
            assert all("generation" in row for row in hist)


class TestRLvsRL:
    def test_two_rl_sides_run(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "rl", _rl_cfg("maddpg", seed=0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg", seed=1)),
            ],
            k_opponents=1,
            max_steps_per_episode=15,
            num_workers=1,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_tcfg(cfg, budget=8))
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Both sides ran at least one generation.
        assert result.metrics["num_generations"] >= 1


class TestParallelFallback:
    def test_num_workers_gt_1_with_rl_side_warns_but_does_not_fall_back(self):
        """num_workers > 1 with an RL side now emits an informational
        RuntimeWarning (EA runs in parallel against frozen RL snapshots)
        but does NOT force num_workers back to 1."""
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _ea_cfg(0)),
                Side("evaders", ("agent",), "rl", _rl_cfg("maddpg")),
            ],
            k_opponents=1,
            max_steps_per_episode=15,
            num_workers=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CoEvolutionTrainer().train(_tcfg(cfg))

        # Informational warning must fire.
        msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("RL side is present" in m for m in msgs), (
            f"Expected RL-side RuntimeWarning, got: {msgs}"
        )
        # Worker count is NOT clamped back to 1 any more.
        assert result.metadata["coevolution_num_workers"] == 2


class TestPoolSurface:
    def test_rl_side_exposes_rl_policy_pool(self):
        from deap_pettingzoo.baselines.coevolution import make_side_state
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )

        side = Side("learners", ("agent",), "rl", _rl_cfg("maddpg"))
        state = make_side_state(side)
        env = PettingZooWrapper(
            "simple_tag_v3", env_config={"continuous_actions": True},
        )
        try:
            state.initialise(env, rng=np.random.default_rng(0))
            pool = state.current_pool()
            assert isinstance(pool, RLPolicyPool)
            assert len(pool) == 1
            # snapshot_sync must produce an independent pool
            snap = pool.snapshot_sync()
            assert isinstance(snap, RLPolicyPool)
            assert snap is not pool
        finally:
            env.close()


class _RandomBoxPolicy:
    """Tiny opponent policy that samples uniformly from a Box space.

    Module-local helper for the learning-happened tests below; lives
    here so the tests don't depend on any other side-state machinery.
    """

    def __init__(self, action_space) -> None:
        self._space = action_space

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def get_action(self, obs, mask=None, deterministic=False):  # noqa: D401
        return self._space.sample()


def _build_rl_state_and_ctx(seed: int = 0):
    """Bring up an RLSideState with a frozen Box-random opponent.

    Returns ``(state, env, ctx, rng)``. Caller must ``env.close()``.
    The returned ``ctx`` is the same ``_RLDriverContext`` shape the
    trainer hands to ``step_generation``.
    """
    from deap_pettingzoo.baselines.coevolution import (
        FrozenPolicyPool,
        make_side_state,
    )
    from deap_pettingzoo.baselines.coevolution.rl_side import (
        _RLDriverContext,
    )
    from deap_pettingzoo.environments.pettingzoo_wrapper import (
        PettingZooWrapper,
    )

    env = PettingZooWrapper(
        "simple_tag_v3", env_config={"continuous_actions": True}
    )
    side = Side(
        "evaders", ("agent",), "rl",
        _rl_cfg("maddpg", seed=seed, rollout_episodes_per_gen=2,
                learning_steps_per_gen=2, batch_size=8, learning_delay=0),
    )
    state = make_side_state(side)
    rng = np.random.default_rng(seed)
    state.initialise(env, rng=rng)

    # One shared random policy across all opponent (pursuer) agents.
    pursuer_ids = [a for a in env.possible_agents if a.startswith("adversary")]
    opp_policy = _RandomBoxPolicy(env.action_space(pursuer_ids[0]))
    opp_pool = FrozenPolicyPool([opp_policy])

    ctx = _RLDriverContext(
        env_wrapper=env,
        learner_role_prefixes=("agent",),
        opponent_sides=[("pursuers", ("adversary",), opp_pool)],
        max_steps=15,
        rng=rng,
    )
    state._bootstrap_ctx = ctx
    return state, env, ctx, rng


class TestLearningActuallyHappens:
    """Assertions that the optimisation step actually fires."""

    def test_buffer_shape_invariant_obs_and_action(self):
        """``obs`` and ``action`` tensors stored in the replay buffer
        must have matching ``ndim`` per agent. This is the exact
        invariant whose violation produced the original 2-vs-3
        ``agent.learn()`` dim mismatch.
        """
        state, env, ctx, _ = _build_rl_state_and_ctx(seed=0)
        try:
            # Fill the buffer with one episode's transitions.
            state._collect_episode(ctx)
            state._collect_episode(ctx)
            assert len(state._memory) >= 4, (
                f"buffer too small: {len(state._memory)}"
            )

            sample = state._memory.sample(min(8, len(state._memory)))
            # Buffer.sample returns (obs_dict, action_dict, reward_dict,
            # next_obs_dict, done_dict) per the field_names contract.
            obs_dict, action_dict = sample[0], sample[1]
            for agent_id in state._side_agent_ids:
                obs_t = obs_dict[agent_id]
                act_t = action_dict[agent_id]
                # Either both 2-D (B, feat) or both 1-D scalars; what
                # MUST NOT happen is action with an extra leading
                # vec-dim that obs lacks.
                assert obs_t.ndim == act_t.ndim, (
                    f"buffer shape mismatch for {agent_id!r}: "
                    f"obs.ndim={obs_t.ndim}, action.ndim={act_t.ndim}. "
                    f"obs.shape={tuple(obs_t.shape)}, "
                    f"action.shape={tuple(act_t.shape)}"
                )
                assert obs_t.shape[0] == act_t.shape[0], (
                    f"batch-size mismatch for {agent_id!r}: "
                    f"obs.shape={tuple(obs_t.shape)}, "
                    f"action.shape={tuple(act_t.shape)}"
                )
        finally:
            env.close()

    def test_learning_actually_runs(self):
        """``step_generation`` must successfully invoke ``agent.learn``
        at least once on the configured budget, and zero failures
        must be recorded. The conftest escalates every
        RuntimeWarning from ``rl_side`` to an error, so any
        ``agent.learn() failed`` warning would already break this
        test before it reached the asserts; the explicit counter
        check belt-and-braces guards against the warning being
        suppressed by a future refactor.
        """
        state, env, ctx, _ = _build_rl_state_and_ctx(seed=0)
        try:
            state.evaluate_population()  # bootstrap buffer
            stats = state.step_generation(lambda s: ctx)

            assert state._total_learn_failures == 0, (
                f"agent.learn() recorded "
                f"{state._total_learn_failures} failures"
            )
            # learning_steps_per_gen=2 + buffer >= batch_size=8
            # after 2 bootstrap rollouts + 2 step_generation rollouts,
            # so all 2 learn() calls should fire.
            assert state._total_learn_calls >= 1, (
                f"expected >=1 successful agent.learn() calls, "
                f"got {state._total_learn_calls}"
            )
            assert stats.generation == 1
        finally:
            env.close()

    def test_actor_weights_change_after_step_generation(self):
        """At least one actor parameter must shift after one full
        ``step_generation``. Catches the silent-no-op failure mode
        where ``learn()`` runs but does nothing useful (e.g. zero
        gradients, frozen optimiser state).
        """
        import torch

        state, env, ctx, _ = _build_rl_state_and_ctx(seed=0)
        try:
            state.evaluate_population()

            # Snapshot first actor param tensor before learning.
            # AgileRL's MADDPG.actors is a ModuleDict keyed on agent_id.
            first_agent_id = state._side_agent_ids[0]
            actor = state._agent.actors[first_agent_id]
            before = {
                name: p.detach().clone()
                for name, p in actor.named_parameters()
            }

            state.step_generation(lambda s: ctx)

            assert state._total_learn_calls >= 1, (
                "no learn() calls fired - test cannot judge weight delta"
            )

            # Refetch the same actor module post-learn (the agent
            # object is not replaced, only mutated in place).
            after_actor = state._agent.actors[first_agent_id]
            after = dict(after_actor.named_parameters())
            deltas = [
                float(torch.abs(after[name] - before[name]).sum().item())
                for name in before
            ]
            assert any(d > 0.0 for d in deltas), (
                f"no actor parameter changed after step_generation; "
                f"max delta = {max(deltas) if deltas else 0.0}"
            )
        finally:
            env.close()
