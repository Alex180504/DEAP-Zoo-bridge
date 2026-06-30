"""Turn-based AEC env support: unit + integration tests.

Covers the two turn-based adapters in
:mod:`deap_pettingzoo.environments.turn_based`, action-mask plumbing
through :class:`MaskedDiscreteHead` + :class:`FeedforwardNeuralPolicy`,
the :class:`SelfPlayEvaluator`, and the MADDPG/MATD3 compatibility gap.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from deap_pettingzoo.environments import (
    MaskedSimultaneousAdapter,
    PettingZooWrapper,
    SequentialTurnAdapter,
)
from deap_pettingzoo.environments.registry import (
    build_env,
    clear_registry,
    register_env,
)
from deap_pettingzoo.environments import builtin  # noqa: F401 — re-register builtins
from deap_pettingzoo.environments.jax_parallel import JaxEnvToPettingZooAdapter


# ---------------------------------------------------------------------------
# Minimal strictly-turn-based AEC env for testing
# ---------------------------------------------------------------------------


class _TurnCountAEC(AECEnv):
    """Strictly turn-based 2-player AEC env.

    Each turn, the active player picks an action in ``[0, 2]``. The
    reward is ``action``. Mask forbids repeating the previous action
    (so the mask has at most one zero). After ``n_turns`` full cycles
    the episode terminates. Observation is the flat 2-vector
    ``[my_last_action, opponent_last_action]``.
    """

    metadata = {"render_modes": [], "name": "turn_count_aec_v0", "is_parallelizable": False}

    def __init__(self, render_mode=None, n_turns: int = 3):
        super().__init__()
        self.possible_agents = ["a", "b"]
        self.n_turns = int(n_turns)
        self._obs_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0.0, high=2.0, shape=(2,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
            }
        )
        self._act_space = spaces.Discrete(3)
        self.agents = []
        self._turn = 0
        self._last = {}
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.agent_selection = None

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def observe(self, agent):
        other = "b" if agent == "a" else "a"
        obs = np.array(
            [self._last.get(agent, 0), self._last.get(other, 0)],
            dtype=np.float32,
        )
        mask = np.ones(3, dtype=np.int8)
        last = self._last.get(agent)
        if last is not None:
            mask[last] = 0
        return {"observation": obs, "action_mask": mask}

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._turn = 0
        self._last = {}
        self.agent_selection = "a"
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def step(self, action):
        agent = self.agent_selection
        # Enforce the mask — refuse to repeat the last action.
        if self._last.get(agent) == int(action):
            raise AssertionError(
                f"_TurnCountAEC: {agent} repeated action {action} "
                "despite mask zeroing it out."
            )
        self.rewards = {a: 0.0 for a in self.agents}
        self.rewards[agent] = float(action)
        self._cumulative_rewards[agent] += float(action)
        self._last[agent] = int(action)
        # Alternate.
        self.agent_selection = "b" if agent == "a" else "a"
        if agent == "b":
            self._turn += 1
        if self._turn >= self.n_turns:
            self.terminations = {a: True for a in self.agents}
            self.agents = []

    def render(self):
        return None

    def close(self):
        return None


class _StaggeredDeathAEC(AECEnv):
    """2-agent AEC env where 'a' is removed from agents immediately on termination.

    Step sequence:
    1. reset: agents=["a","b"], agent_selection="a"
    2. step(action): a acts, terminations["a"]=True, agents=["b"], agent_selection="b"
    3. step(action): b acts, terminations["b"]=True, agents=[], agent_selection=None
    """

    metadata = {"render_modes": [], "name": "staggered_death_aec_v0", "is_parallelizable": False}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = ["a", "b"]
        self._act_space = spaces.Discrete(2)
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.agents = []
        self.agent_selection = None
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self._cumulative_rewards = {}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def observe(self, agent):
        return np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = ["a", "b"]
        self.agent_selection = "a"
        self.rewards = {"a": 0.0, "b": 0.0}
        self._cumulative_rewards = {"a": 0.0, "b": 0.0}
        self.terminations = {"a": False, "b": False}
        self.truncations = {"a": False, "b": False}
        self.infos = {"a": {}, "b": {}}

    def step(self, action):
        agent = self.agent_selection
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.rewards[agent] = float(int(action)) + 1.0
        if agent == "a":
            self.terminations["a"] = True
            self.agents.remove("a")
            self.agent_selection = "b"
        else:
            self.terminations["b"] = True
            self.agents = []
            self.agent_selection = None

    def render(self):
        return None

    def close(self):
        return None


class _DrainRequiredAEC(AECEnv):
    """2-agent AEC env where terminated 'a' stays in agents until step(None).

    Step sequence:
    1. reset: agents=["a","b"], agent_selection="a"
    2. step(action): a acts, terminations["a"]=True, agent_selection="b"
       — 'a' is STILL in agents (requires step(None) to drain)
    3. step(action): b acts, agent_selection="a" (the terminated a!)
    4. step(None): drain "a", agents=["b"], agent_selection="b"
    5. step(action): b acts again, terminations["b"]=True, agents=[]
    """

    metadata = {"render_modes": [], "name": "drain_required_aec_v0", "is_parallelizable": False}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = ["a", "b"]
        self._act_space = spaces.Discrete(2)
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.agents = []
        self.agent_selection = None
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self._cumulative_rewards = {}
        self._b_turn_count = 0

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def observe(self, agent):
        return np.array([float(self._b_turn_count)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = ["a", "b"]
        self.agent_selection = "a"
        self._b_turn_count = 0
        self.rewards = {"a": 0.0, "b": 0.0}
        self._cumulative_rewards = {"a": 0.0, "b": 0.0}
        self.terminations = {"a": False, "b": False}
        self.truncations = {"a": False, "b": False}
        self.infos = {"a": {}, "b": {}}

    def step(self, action):
        agent = self.agent_selection
        self.rewards = {"a": 0.0, "b": 0.0}

        if action is None:
            # Drain: remove terminated agent, advance selection.
            if agent in self.agents:
                self.agents.remove(agent)
            remaining = [a for a in ["a", "b"] if a in self.agents]
            self.agent_selection = remaining[0] if remaining else None
            return

        if agent == "a":
            self.rewards["a"] = 1.0
            self.terminations["a"] = True
            self.agent_selection = "b"
        else:
            self._b_turn_count += 1
            self.rewards["b"] = 1.0
            if "a" not in self.agents:
                # a already drained; b's second turn → episode over.
                self.terminations["b"] = True
                self.agents = []
                self.agent_selection = None
            else:
                # a still in agents (not yet drained) → cycle back to terminated a.
                self.agent_selection = "a"

    def render(self):
        return None

    def close(self):
        return None


def _register_turn_env(name: str, *, api: str) -> None:
    register_env(
        name,
        lambda render_mode=None, **cfg: _TurnCountAEC(render_mode=render_mode, **cfg),
        default_episode_len=6,
        api=api,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Registry + adapter round-trip
# ---------------------------------------------------------------------------


class TestRegistryAcceptsTurnBasedAPIs:
    def test_api_aec_masked_accepted(self):
        _register_turn_env("tc_masked_v0", api="aec_masked")
        env = build_env("tc_masked_v0")
        assert isinstance(env, MaskedSimultaneousAdapter)

    def test_api_aec_sequential_accepted(self):
        _register_turn_env("tc_seq_v0", api="aec_sequential")
        env = build_env("tc_seq_v0")
        assert isinstance(env, SequentialTurnAdapter)

    def test_invalid_api_rejected(self):
        with pytest.raises(ValueError, match="Unsupported api"):
            register_env(
                "tc_bogus_v0",
                lambda render_mode=None, **_: _TurnCountAEC(),
                api="aec_bogus",
                overwrite=True,
            )

    def test_api_aec_raises_for_non_parallelizable(self):
        _register_turn_env("tc_pure_aec_v0", api="aec")
        with pytest.raises(TypeError, match="aec_masked"):
            build_env("tc_pure_aec_v0")


# ---------------------------------------------------------------------------
# MaskedSimultaneousAdapter
# ---------------------------------------------------------------------------


class TestMaskedSimultaneousAdapter:
    def _make(self):
        _register_turn_env("tc_masked_roundtrip_v0", api="aec_masked")
        return build_env("tc_masked_roundtrip_v0")

    def test_reset_returns_all_agents(self):
        env = self._make()
        obs, infos = env.reset(seed=0)
        assert set(obs.keys()) == {"a", "b"}
        assert set(infos.keys()) == {"a", "b"}
        assert obs["a"].shape == (2,)
        # Only the current player gets a live mask.
        assert np.any(infos["a"]["action_mask"])
        assert not np.any(infos["b"]["action_mask"])

    def test_step_only_applies_current_players_action(self):
        env = self._make()
        env.reset(seed=0)
        # 'a' is current; action 2. 'b' action is ignored but must be provided.
        obs, rewards, terms, truncs, infos = env.step({"a": 2, "b": 0})
        assert rewards["a"] == pytest.approx(2.0)
        assert rewards["b"] == pytest.approx(0.0)
        assert not any(terms.values())
        # Now 'b' is current.
        assert np.any(infos["b"]["action_mask"])
        assert not np.any(infos["a"]["action_mask"])

    def test_episode_terminates_cleanly(self):
        env = self._make()
        env.reset(seed=0)
        # n_turns=3 → 6 turns total.
        for _ in range(6):
            # pick an action that differs from player's last action
            masks = {a: env.aec_env.observe(a).get("action_mask") for a in env.agents}
            actions = {a: int(np.argmax(m)) for a, m in masks.items()}
            obs, rewards, terms, truncs, _ = env.step(actions)
        assert all(terms.values())


# ---------------------------------------------------------------------------
# SequentialTurnAdapter
# ---------------------------------------------------------------------------


class TestSequentialTurnAdapter:
    def _make(self):
        _register_turn_env("tc_seq_roundtrip_v0", api="aec_sequential")
        return build_env("tc_seq_roundtrip_v0")

    def test_reset_returns_single_agent(self):
        env = self._make()
        obs, infos = env.reset(seed=0)
        assert len(obs) == 1
        current = next(iter(obs))
        assert current == "a"
        assert obs[current].shape == (2,)
        assert "action_mask" in infos[current]

    def test_step_advances_single_agent(self):
        env = self._make()
        env.reset(seed=0)
        obs, rewards, terms, truncs, infos = env.step({"a": 1})
        # Next current is 'b'; obs dict keyed by 'b'.
        assert "b" in obs
        assert rewards["a"] == pytest.approx(1.0)

    def test_terminal_step_reports_all_possible_agents(self):
        env = self._make()
        env.reset(seed=0)
        # 3 turns × 2 players = 6 steps — run 5 and check last carries both.
        terms_final = {}
        for i in range(20):
            if not env.agents:
                break
            current = env.agents[0]
            mask = env.aec_env.observe(current).get("action_mask")
            action = int(np.argmax(mask))
            _, _, terms_final, _, _ = env.step({current: action})
        assert set(terms_final.keys()) == {"a", "b"}
        assert all(terms_final.values())


# ---------------------------------------------------------------------------
# Mask-aware policy head
# ---------------------------------------------------------------------------


class TestMaskedDiscreteHead:
    def test_argmax_restricted_to_legal_actions(self):
        from deap_pettingzoo.policies.action_heads import MaskedDiscreteHead
        head = MaskedDiscreteHead(4)
        logits = np.array([5.0, 3.0, 4.0, 2.0])  # unmasked argmax = 0
        mask = np.array([0, 1, 1, 1])  # block index 0
        assert head.decode(logits, action_mask=mask) == 2  # second-best legal

    def test_mask_none_behaves_like_plain(self):
        from deap_pettingzoo.policies.action_heads import MaskedDiscreteHead
        head = MaskedDiscreteHead(3)
        logits = np.array([0.1, 0.9, 0.2])
        assert head.decode(logits) == 1

    def test_all_zero_mask_returns_zero(self):
        from deap_pettingzoo.policies.action_heads import MaskedDiscreteHead
        head = MaskedDiscreteHead(3)
        logits = np.array([0.1, 0.9, 0.2])
        assert head.decode(logits, action_mask=np.zeros(3, dtype=np.int8)) == 0


class TestFeedforwardPolicyMaskThreading:
    def test_masked_flag_installs_masked_head(self):
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy
        from deap_pettingzoo.policies.action_heads import MaskedDiscreteHead

        genome = FixedTopologyGenome([2, 4, 3], activation="tanh", random_seed=0)
        policy = FeedforwardNeuralPolicy(
            genome, action_space=spaces.Discrete(3), masked=True
        )
        assert isinstance(policy.head, MaskedDiscreteHead)

    def test_get_action_honours_mask(self):
        from deap_pettingzoo.genomes import FixedTopologyGenome
        from deap_pettingzoo.policies import FeedforwardNeuralPolicy

        genome = FixedTopologyGenome([2, 4, 3], activation="tanh", random_seed=0)
        policy = FeedforwardNeuralPolicy(
            genome, action_space=spaces.Discrete(3), masked=True
        )
        obs = np.array([0.5, -0.5], dtype=np.float32)
        # Run 10 observations; with mask = [0, 1, 0] only index 1 is legal.
        mask = np.array([0, 1, 0], dtype=np.int8)
        actions = {policy.get_action(obs, action_mask=mask) for _ in range(10)}
        assert actions == {1}


# ---------------------------------------------------------------------------
# SelfPlayEvaluator smoke
# ---------------------------------------------------------------------------


class TestSelfPlayEvaluator:
    def test_runs_episode_with_masked_adapter(self):
        from deap_pettingzoo.evaluators import SelfPlayEvaluator
        from deap_pettingzoo.genomes import FixedTopologyGenome

        _register_turn_env("tc_selfplay_v0", api="aec_masked")
        wrapper = PettingZooWrapper(env_name="tc_selfplay_v0")

        # Genome sized for flattened 2-D obs → 3 discrete actions.
        genome = FixedTopologyGenome([2, 8, 3], activation="tanh", random_seed=42)

        class _Ind:
            pass

        ind = _Ind()
        ind.genome = genome

        evaluator = SelfPlayEvaluator(wrapper, max_steps=20, aggregation="first_agent")
        fitness = evaluator.evaluate(ind, seed=0)
        assert isinstance(fitness, tuple) and len(fitness) == 1
        assert fitness[0] >= 0.0  # env only dispenses non-negative rewards

    def test_aggregation_validation(self):
        from deap_pettingzoo.evaluators import SelfPlayEvaluator
        _register_turn_env("tc_selfplay_agg_v0", api="aec_masked")
        wrapper = PettingZooWrapper(env_name="tc_selfplay_agg_v0")
        with pytest.raises(ValueError, match="Invalid aggregation"):
            SelfPlayEvaluator(wrapper, aggregation="nope")
        with pytest.raises(ValueError, match="team_mapping"):
            SelfPlayEvaluator(wrapper, aggregation="team_mapping")


# ---------------------------------------------------------------------------
# Compatibility carve-out
# ---------------------------------------------------------------------------


class TestTurnBasedCompatibility:
    def test_inspect_detects_turn_based(self):
        from deap_pettingzoo.baselines.compatibility import inspect_environment
        _register_turn_env("tc_compat_v0", api="aec_masked")
        info = inspect_environment("tc_compat_v0")
        assert info.is_turn_based is True

    def test_maddpg_blocked_on_turn_based(self):
        from deap_pettingzoo.baselines.compatibility import (
            EnvironmentInfo,
            check_compatibility,
        )

        _register_turn_env("tc_compat_block_v0", api="aec_masked")

        class _FakeTrainer:
            def get_capabilities(self):
                return {
                    "discrete_action": True,
                    "continuous_action": True,
                    "needs_global_state": False,
                    "on_policy": False,
                    "supports_turn_based": False,
                }

        info = EnvironmentInfo(
            env_name="tc_compat_block_v0",
            num_agents=2,
            agent_ids=["a", "b"],
            has_continuous_actions=False,
            has_discrete_actions=True,
            has_global_state=False,
            obs_dim=2,
            act_dim=3,
            action_low=None,
            action_high=None,
            is_turn_based=True,
        )
        result = check_compatibility(_FakeTrainer(), info, algorithm_name="maddpg")
        assert result.compatible is False
        assert any("turn-based" in i for i in result.issues)

    def test_ea_allowed_on_turn_based(self):
        from deap_pettingzoo.baselines.compatibility import (
            EnvironmentInfo,
            check_compatibility,
        )

        class _FakeEA:
            def get_capabilities(self):
                return {
                    "discrete_action": True,
                    "continuous_action": True,
                    "needs_global_state": False,
                    "on_policy": False,
                    "supports_turn_based": True,
                }

        info = EnvironmentInfo(
            env_name="tc_dummy",
            num_agents=2,
            agent_ids=["a", "b"],
            has_continuous_actions=False,
            has_discrete_actions=True,
            has_global_state=False,
            obs_dim=2,
            act_dim=3,
            action_low=None,
            action_high=None,
            is_turn_based=True,
        )
        result = check_compatibility(_FakeEA(), info, algorithm_name="ea")
        assert result.compatible is True


class TestMaskedAdapterStaggeredDeath:
    """Terminated agents must be dropped from obs_dict, not given empty arrays."""

    def _make_adapter(self):
        aec = _StaggeredDeathAEC()
        return MaskedSimultaneousAdapter(aec, env_name="staggered_masked")

    def test_terminated_agent_absent_from_obs(self):
        """Just-terminated 'a' must not appear in obs_dict after it dies."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        # "a" is the current agent; after this step "a" terminates and is removed.
        obs, rewards, terms, truncs, infos = adapter.step({"a": 0, "b": 0})
        assert terms["a"] is True
        assert "a" not in obs, (
            "Terminated agent 'a' should be absent from obs_dict, "
            f"but got obs['a'] = {obs.get('a')!r}"
        )
        assert "a" not in infos

    def test_terminated_agent_still_in_rewards_and_terms(self):
        """Rewards/terminations still carry the just-terminated agent (bookkeeping)."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        obs, rewards, terms, truncs, infos = adapter.step({"a": 0, "b": 0})
        assert "a" in rewards
        assert "a" in terms
        assert terms["a"] is True

    def test_no_empty_obs_for_any_agent(self):
        """No observation returned by step has size 0 (policy-safe invariant)."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        obs, _, _, _, _ = adapter.step({"a": 0, "b": 0})
        for agent, arr in obs.items():
            assert arr.size > 0, f"Agent {agent!r} received a zero-size observation"

    def test_live_agent_obs_has_correct_shape(self):
        """The surviving agent 'b' still receives a valid observation."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        obs, _, _, _, _ = adapter.step({"a": 0, "b": 0})
        assert "b" in obs
        assert obs["b"].shape == (2,)


class TestSequentialAdapterDrainLoop:
    """SequentialTurnAdapter must drain terminated agents with step(None)."""

    def _make_adapter(self):
        aec = _DrainRequiredAEC()
        return SequentialTurnAdapter(aec, env_name="drain_seq")

    def test_drain_loop_skips_terminated_agent(self):
        """After b acts and agent_selection flips to terminated 'a', adapter drains 'a'
        and returns the next live agent ('b') — not the terminated one."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)

        # Step 1: 'a' acts (terminates, stays in agents until step(None)).
        obs1, _, terms1, _, _ = adapter.step({"a": 0})
        assert "b" in obs1, "After a acts, b should be the next agent"
        assert terms1.get("a", False) is True

        # Step 2: 'b' acts. Internally AEC sets agent_selection='a' (terminated).
        # Adapter must drain 'a' and return 'b' (next live agent).
        obs2, _, _, _, _ = adapter.step({"b": 0})
        assert "b" in obs2, (
            "After draining terminated 'a', the live agent 'b' should be returned. "
            f"Got obs keys: {list(obs2.keys())}"
        )
        assert "a" not in obs2, "Terminated 'a' must NOT appear as the next actor"

    def test_dead_agent_not_re_acted_on(self):
        """Once 'a' is terminated, the adapter never requests an action for it again."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        adapter.step({"a": 0})
        obs2, _, _, _, _ = adapter.step({"b": 0})
        # If the drain loop works, 'a' never shows up in obs.
        actor = next(iter(obs2)) if obs2 else None
        assert actor != "a", f"Terminated agent 'a' reappeared as next actor: {obs2}"

    def test_episode_eventually_terminates(self):
        """Full episode with staggered death reaches a clean terminal state."""
        adapter = self._make_adapter()
        adapter.reset(seed=0)
        done = False
        for _ in range(20):
            if not adapter.agents:
                done = True
                break
            current = adapter.agents[0]
            _, _, terms, truncs, _ = adapter.step({current: 0})
            if all(terms.values()) or all(truncs.values()):
                done = True
                break
        assert done


class TestJaxAdapterTruncationAndPruning:
    """JaxEnvToPettingZooAdapter maps truncation signals and prunes done agents."""

    def _make_mock_jax_env(self, *, global_truncated=False, per_agent_done=None):
        """Build a minimal JaxEnvProtocol-compatible mock without importing JAX."""
        import types

        per_agent_done = per_agent_done or {}

        class _MockJaxEnv:
            agents = ["a", "b"]

            def observation_space(self, agent):
                from gymnasium import spaces as gym_spaces
                return gym_spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

            def action_space(self, agent):
                from gymnasium import spaces as gym_spaces
                return gym_spaces.Discrete(2)

            def reset(self, key):
                obs = {"a": np.zeros(2, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
                state = types.SimpleNamespace(step=0)
                return obs, state

            def step(self, key, state, actions):
                obs = {"a": np.ones(2, dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
                new_state = types.SimpleNamespace(step=state.step + 1)
                rewards = {"a": 1.0, "b": 1.0}
                dones = dict(per_agent_done)
                dones["__all__"] = all(per_agent_done.get(ag, False) for ag in ["a", "b"])
                if global_truncated:
                    dones["__truncated__"] = True
                infos = {}
                return obs, new_state, rewards, dones, infos

        return _MockJaxEnv()

    def test_truncation_flag_mapped(self):
        """When __truncated__ is set, truncations[agent] must be True (not False)."""
        jax = pytest.importorskip("jax")
        mock_env = self._make_mock_jax_env(
            global_truncated=True,
            per_agent_done={"a": True, "b": True},
        )
        adapter = JaxEnvToPettingZooAdapter(mock_env, env_name="jax_trunc_test", seed=0)
        adapter.reset(seed=0)
        _, _, terminations, truncations, _ = adapter.step({"a": 0, "b": 0})
        assert any(truncations.values()), (
            "When __truncated__ is set, at least one agent should have truncations=True"
        )

    def test_per_agent_done_prunes_obs(self):
        """An agent marked done-but-not-global-done is pruned from obs/infos."""
        jax = pytest.importorskip("jax")
        mock_env = self._make_mock_jax_env(
            per_agent_done={"a": True, "b": False},
        )
        adapter = JaxEnvToPettingZooAdapter(mock_env, env_name="jax_prune_test", seed=0)
        adapter.reset(seed=0)
        obs, _, terminations, _, _ = adapter.step({"a": 0, "b": 0})
        assert terminations.get("a", False) is True
        assert "a" not in obs, (
            "Done agent 'a' should be pruned from obs when the episode continues"
        )
