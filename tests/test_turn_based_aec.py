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
