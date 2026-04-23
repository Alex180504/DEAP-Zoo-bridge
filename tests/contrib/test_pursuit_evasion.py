"""Smoke tests for the pursuit-evasion prototype: env wrapper, APF policy,
mixed evaluator.  These tests execute real environment steps but with a
very small scenario to keep runtime under a few seconds."""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.contrib.pursuit_evasion import (
    PursuitEvasionEnv,
    PursuitEvasionConfig,
    ObsLayout,
    APFEvaderPolicy,
    APFConfig,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import setup_deap_creators


SMALL_CFG = PursuitEvasionConfig(
    num_adversaries=2,
    num_good=2,
    num_obstacles=2,
    max_cycles=20,
    equal_max_speed=1.0,
    equal_accel=3.0,
    num_agent_neighbors=2,
    num_landmark_neighbors=2,
)


# ---------------------------------------------------------------------------
# PursuitEvasionEnv
# ---------------------------------------------------------------------------


class TestPursuitEvasionEnv:
    def test_construction_and_agents(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        assert len(env.adversary_agents) == 2
        assert len(env.good_agents) == 2
        assert all(a.startswith("adversary") for a in env.adversary_agents)
        assert all(not a.startswith("adversary") for a in env.good_agents)

    def test_reset_returns_obs_for_all_agents(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        obs = env.reset(seed=1)
        assert set(obs) == set(env.possible_agents)
        for name, vec in obs.items():
            assert vec.ndim == 1
            assert vec.shape[0] > 0

    def test_equal_ability_applied(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        env.reset(seed=0)
        world = env._get_world()
        for a in world.agents:
            assert a.max_speed == SMALL_CFG.equal_max_speed
            assert a.accel == SMALL_CFG.equal_accel

    def test_step_progresses_and_masks(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        obs = env.reset(seed=2)
        actions = {name: 0 for name in obs}  # all no-op
        new_obs, rewards, terms, truncs, _ = env.step(actions)
        assert set(new_obs) == set(obs)
        for v in new_obs.values():
            assert np.isfinite(v).all()

    def test_layout_slices_partition_observation(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        obs = env.reset(seed=3)
        for name, vec in obs.items():
            lay: ObsLayout = env.get_layout(name)
            # slices should cover exactly the vector length
            total = (
                (lay.self_vel.stop - lay.self_vel.start)
                + (lay.self_pos.stop - lay.self_pos.start)
                + (lay.landmark_rel.stop - lay.landmark_rel.start)
                + (lay.adversary_rel.stop - lay.adversary_rel.start)
                + (lay.good_rel.stop - lay.good_rel.start)
                + (lay.good_vel.stop - lay.good_vel.start)
            )
            assert total == vec.shape[0], f"layout mismatch for {name}"

    def test_boundary_penalty_increases_with_distance(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        pen_inside = env._boundary_penalty(np.array([0.0, 0.0]))
        pen_edge = env._boundary_penalty(np.array([1.0, 1.0]))
        pen_far = env._boundary_penalty(np.array([2.0, 2.0]))
        assert pen_inside == 0.0
        assert pen_edge > 0.0
        assert pen_far > pen_edge

    def test_full_obs_mode_layout_splits_adversary_and_good(self):
        """Without K-nearest masking the layout exposes type-separated slices."""
        cfg = PursuitEvasionConfig(
            num_adversaries=2, num_good=2, num_obstacles=2, max_cycles=5,
            num_agent_neighbors=None, num_landmark_neighbors=None,
        )
        env = PursuitEvasionEnv(config=cfg)
        env.reset(seed=0)
        lay = env.get_layout(env.adversary_agents[0])
        assert not lay.partial_obs
        assert lay.num_adv_in_other == 1      # num_adversaries - 1
        assert lay.num_good_in_other == 2     # num_good
        # adversary_rel + good_rel must partition other_rel.
        assert lay.adversary_rel.start == lay.other_rel.start
        assert lay.good_rel.stop == lay.other_rel.stop
        assert lay.adversary_rel.stop == lay.good_rel.start

    def test_curriculum_advance_changes_prey_speed(self):
        cfg = PursuitEvasionConfig(
            num_adversaries=2, num_good=2, num_obstacles=2, max_cycles=5,
            equal_ability=False, curriculum=True,
            num_agent_neighbors=2, num_landmark_neighbors=2,
        )
        env = PursuitEvasionEnv(config=cfg)
        env.reset(seed=0)
        assert env.curriculum_stage == 0
        speeds_stage0 = [a.max_speed for a in env._get_world().agents if not a.adversary]
        env.advance_curriculum()
        env.reset(seed=0)
        assert env.curriculum_stage == 1
        speeds_stage1 = [a.max_speed for a in env._get_world().agents if not a.adversary]
        # Prey get faster as curriculum advances (50% -> 75% of full speed).
        assert speeds_stage1[0] > speeds_stage0[0]

    def test_curriculum_with_equal_ability_emits_warning(self):
        import warnings
        cfg = PursuitEvasionConfig(
            num_adversaries=2, num_good=2, num_obstacles=2, max_cycles=5,
            equal_ability=True, curriculum=True,
            num_agent_neighbors=2, num_landmark_neighbors=2,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PursuitEvasionEnv(config=cfg)
        assert any("equal_ability" in str(w.message) for w in caught)

    def test_set_curriculum_stage_clamped(self):
        cfg = PursuitEvasionConfig(
            num_adversaries=2, num_good=2, num_obstacles=2, max_cycles=5,
            equal_ability=False, curriculum=True,
            num_agent_neighbors=2, num_landmark_neighbors=2,
        )
        env = PursuitEvasionEnv(config=cfg)
        env.set_curriculum_stage(2)
        env.reset(seed=0)
        assert env.curriculum_stage == 2

    def test_terminate_on_success_flag_reaches_env(self):
        """When enabled, the underlying scenario has terminate_on_success=True."""
        cfg = PursuitEvasionConfig(
            num_adversaries=2, num_good=2, num_obstacles=2, max_cycles=5,
            terminate_on_success=True,
            num_agent_neighbors=2, num_landmark_neighbors=2,
        )
        env = PursuitEvasionEnv(config=cfg)
        env.reset(seed=0)
        # The scenario lives on env.env.unwrapped.scenario.
        scenario = env.env.unwrapped.scenario
        assert getattr(scenario, "terminate_on_success", False) is True


# ---------------------------------------------------------------------------
# APFEvaderPolicy
# ---------------------------------------------------------------------------


class TestAPFEvaderPolicy:
    def test_returns_valid_discrete_action(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        obs = env.reset(seed=0)
        evader = env.good_agents[0]
        pol = APFEvaderPolicy(layout=env.get_layout(evader))
        a = pol.get_action(obs[evader])
        assert isinstance(a, int)
        assert 0 <= a <= 4

    def test_pursuer_repulsion_pushes_away(self):
        """A pursuer to the +x side should produce a -x (left) action."""
        env = PursuitEvasionEnv(config=SMALL_CFG)
        env.reset(seed=0)
        evader = env.good_agents[0]
        layout = env.get_layout(evader)
        pol = APFEvaderPolicy(layout=layout)

        obs = np.zeros(layout.good_vel.stop, dtype=np.float32)
        # Put a single pursuer directly to the right (+x) at distance 0.3
        adv_rel = obs[layout.adversary_rel].reshape(layout.num_adv_in_other, 2)
        adv_rel[0] = [0.3, 0.0]
        obs[layout.adversary_rel] = adv_rel.reshape(-1)
        action = pol.get_action(obs)
        assert action == 1  # left

    def test_boundary_push_inward(self):
        """Agent past +x boundary should be pushed in the -x direction."""
        env = PursuitEvasionEnv(config=SMALL_CFG)
        env.reset(seed=0)
        evader = env.good_agents[0]
        layout = env.get_layout(evader)
        pol = APFEvaderPolicy(layout=layout)

        obs = np.zeros(layout.good_vel.stop, dtype=np.float32)
        obs[layout.self_pos] = [1.2, 0.0]  # past +x boundary
        action = pol.get_action(obs)
        assert action == 1  # left

    def test_all_zero_obs_returns_noop(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        env.reset(seed=0)
        evader = env.good_agents[0]
        layout = env.get_layout(evader)
        pol = APFEvaderPolicy(layout=layout)
        obs = np.zeros(layout.good_vel.stop, dtype=np.float32)
        assert pol.get_action(obs) == 0

    def test_reset_is_noop(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        layout = env.get_layout(env.good_agents[0])
        pol = APFEvaderPolicy(layout=layout)
        pol.reset()  # must not raise


# ---------------------------------------------------------------------------
# MixedPolicyEvaluator
# ---------------------------------------------------------------------------


class TestMixedPolicyEvaluator:
    def _make_individual(self, env: PursuitEvasionEnv, hidden: int = 16):
        setup_deap_creators()
        from deap import creator
        obs = env.reset(seed=0)
        pursuer = env.adversary_agents[0]
        in_size = obs[pursuer].shape[0]
        out_size = env.action_space(pursuer).n
        genome = FixedTopologyGenome(
            layer_sizes=[in_size, hidden, out_size],
            activation="tanh",
            random_seed=0,
        )
        ind = creator.Individual()
        ind.genome = genome
        return ind

    def _make_evaluator(self, env: PursuitEvasionEnv):
        apf_cfg = APFConfig()
        scripted = {
            name: APFEvaderPolicy(layout=env.get_layout(name), config=apf_cfg)
            for name in env.good_agents
        }

        def _factory(genome, agent_id):
            return FeedforwardNeuralPolicy(genome)

        return MixedPolicyEvaluator(
            env_wrapper=env,
            learner_agent_prefixes=["adversary"],
            learner_factory=_factory,
            scripted_policies=scripted,
            max_steps=SMALL_CFG.max_cycles,
        )

    def test_evaluate_returns_single_float_tuple(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        ind = self._make_individual(env)
        ev = self._make_evaluator(env)
        result = ev.evaluate(ind, seed=0)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_run_episode_returns_structured_result(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        ind = self._make_individual(env)
        ev = self._make_evaluator(env)
        res = ev.run_episode(ind, seed=1)
        assert res.steps > 0
        assert res.steps <= SMALL_CFG.max_cycles
        assert res.fitness == res.learner_reward
        assert np.isfinite(res.learner_reward)
        assert np.isfinite(res.scripted_reward)

    def test_requires_individual_with_genome(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        ev = self._make_evaluator(env)
        with pytest.raises(ValueError):
            ev.evaluate(None)

    def test_raises_on_missing_scripted_policy(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)

        def _factory(genome, agent_id):
            return FeedforwardNeuralPolicy(genome)

        with pytest.raises(ValueError, match="missing entries"):
            MixedPolicyEvaluator(
                env_wrapper=env,
                learner_agent_prefixes=["adversary"],
                learner_factory=_factory,
                scripted_policies={},  # empty -> missing all evaders
                max_steps=5,
            )

    def test_raises_when_no_learner_agents(self):
        env = PursuitEvasionEnv(config=SMALL_CFG)
        scripted = {
            name: APFEvaderPolicy(layout=env.get_layout(name))
            for name in env.possible_agents
        }

        def _factory(genome, agent_id):
            return FeedforwardNeuralPolicy(genome)

        with pytest.raises(ValueError, match="No agents matched"):
            MixedPolicyEvaluator(
                env_wrapper=env,
                learner_agent_prefixes=["nonexistent_prefix"],
                learner_factory=_factory,
                scripted_policies=scripted,
                max_steps=5,
            )
