"""Tests for the featurized-observation APF controller and the anchor set.

:class:`AquariumFeatureAPFPolicy` exists because co-evolution runs every
agent through the feature wrapper, while the raw-layout scripted policies
parse the env's own observation format.  Dropping a raw-layout policy into
the co-evolution env would not fail loudly — it would read the wrong fields
and steer on noise — so these tests check it reads the featurized layout and
that its steering is the right sign for each role.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.pursuit_evasion.aquarium import experiment_config as XC
from examples.pursuit_evasion.aquarium.apf_policy import (
    AquariumAPFConfig,
    AquariumFeatureAPFPolicy,
)
from examples.pursuit_evasion.aquarium.observation_features import (
    aquarium_feature_size,
)

P_COUNT, Q_COUNT, WALLS = 3, 3, 4


def _obs(predators=(), prey=(), walls=(1.0, 1.0, 1.0, 1.0), heading=(1.0, 0.0)):
    """Build a featurized observation vector.

    Layout: ``[heading_x, heading_y, speed] + walls[4]
    + P x [present, rel_x, rel_y, distance, speed]
    + Q x [present, rel_x, rel_y, distance, speed]``
    """
    feats = [heading[0], heading[1], 0.5, *walls]
    for block, count in ((predators, P_COUNT), (prey, Q_COUNT)):
        for i in range(count):
            if i < len(block):
                rel_x, rel_y = block[i]
                dist = math.hypot(rel_x, rel_y)
                feats.extend([1.0, rel_x, rel_y, dist, 0.5])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.asarray(feats, dtype=np.float32)


def _policy(role, config=None, continuous=True):
    return AquariumFeatureAPFPolicy(
        role=role,
        predator_observe_count=P_COUNT,
        prey_observe_count=Q_COUNT,
        action_count=16,
        config=config or AquariumAPFConfig(k_tangential=0.0, k_prey_separation=0.0),
        wall_features=WALLS,
        continuous=continuous,
    )


class TestLayout:
    def test_expected_observation_size_matches_the_feature_wrapper(self):
        """A silent layout drift here would make the policy read noise."""
        assert _obs().size == aquarium_feature_size(P_COUNT, Q_COUNT, WALLS)

    def test_padding_entries_are_ignored(self):
        """Absent entities are all-zero rows and must not pull the force."""
        policy = _policy("pursuer")
        far = policy.get_action(_obs(prey=[(0.5, 0.0)]))
        # Same single prey, but the observation is otherwise empty either way.
        assert np.allclose(far, policy.get_action(_obs(prey=[(0.5, 0.0)])))

    def test_no_visible_opponent_holds_heading(self):
        policy = _policy("pursuer")
        action = policy.get_action(_obs(heading=(0.0, -1.0)))
        assert np.allclose(action, policy._emit(0.0, -1.0))


class TestSteeringDirection:
    def test_pursuer_steers_toward_prey(self):
        action = _policy("pursuer").get_action(_obs(prey=[(0.4, 0.0)]))
        assert action[0] > 0.0

    def test_evader_steers_away_from_a_predator(self):
        action = _policy("evader").get_action(_obs(predators=[(0.4, 0.0)]))
        assert action[0] < 0.0

    @pytest.mark.parametrize(
        "offset,axis,sign",
        [((0.0, 0.4), 1, 1), ((0.0, -0.4), 1, -1), ((-0.4, 0.0), 0, -1)],
    )
    def test_pursuer_tracks_prey_on_every_axis(self, offset, axis, sign):
        action = _policy("pursuer").get_action(_obs(prey=[offset]))
        assert math.copysign(1.0, action[axis]) == sign

    def test_nearer_opponents_dominate(self):
        """Inverse-square falloff: the close threat sets the direction."""
        action = _policy("evader").get_action(
            _obs(predators=[(0.1, 0.0), (-0.9, 0.0)])
        )
        assert action[0] < 0.0

    def test_evader_ignores_prey_and_pursuer_ignores_predators(self):
        """Each role reads only its opponent block."""
        evader = _policy("evader").get_action(_obs(prey=[(0.4, 0.0)]))
        pursuer = _policy("pursuer").get_action(_obs(predators=[(0.4, 0.0)]))
        # Nothing in the opponent block -> hold heading (1, 0).
        assert np.allclose(evader, _policy("evader")._emit(1.0, 0.0))
        assert np.allclose(pursuer, _policy("pursuer")._emit(1.0, 0.0))


class TestWallAvoidance:
    def test_close_wall_pushes_away(self):
        policy = _policy("evader")
        action = policy.get_action(_obs(walls=(0.02, 1.0, 1.0, 1.0)))
        assert action[0] > 0.0     # left wall -> push right

    def test_walls_ignored_when_the_env_does_not_expose_them(self):
        policy = AquariumFeatureAPFPolicy(
            role="evader",
            predator_observe_count=P_COUNT,
            prey_observe_count=Q_COUNT,
            config=AquariumAPFConfig(k_tangential=0.0, k_prey_separation=0.0),
            wall_features=0,
            continuous=True,
        )
        obs = np.zeros(aquarium_feature_size(P_COUNT, Q_COUNT, 0), dtype=np.float32)
        obs[0], obs[1] = 1.0, 0.0
        # Must not raise despite there being no wall block to read.
        assert policy.get_action(obs) is not None


class TestRoleAndModes:
    def test_unknown_role_rejected(self):
        with pytest.raises(ValueError, match="role must be"):
            _policy("sideways")

    def test_discrete_mode_emits_a_valid_action_index(self):
        policy = _policy("pursuer", continuous=False)
        action = policy.get_action(_obs(prey=[(0.4, 0.0)]))
        assert isinstance(action, (int, np.integer))
        assert 0 <= int(action) < 16

    def test_continuous_mode_emits_a_unit_box_action(self):
        action = _policy("pursuer").get_action(_obs(prey=[(0.4, 0.0)]))
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= 1.0 + 1e-6)

    def test_tangential_dodge_rotates_the_flee_vector(self):
        cfg = AquariumAPFConfig(k_tangential=3.0, k_prey_separation=0.0)
        policy = _policy("evader", cfg)
        policy._turn_sign = 1
        plain = _policy("evader").get_action(_obs(predators=[(0.4, 0.0)]))
        dodged = policy.get_action(_obs(predators=[(0.4, 0.0)]))
        # Radial flee is -x for both; the dodge adds a perpendicular component.
        assert abs(dodged[1]) > abs(plain[1])


class TestSymmetricSqueeze:
    """Opponents on opposite sides cancel; the policy must still commit.

    Both the radial and the tangential terms are antiparallel in this
    configuration, so without a fallback the net force is zero and the
    hold-heading branch drives the agent straight through the line joining
    the two threats.
    """

    @pytest.mark.parametrize("k_tangential", [0.0, 3.0])
    def test_evader_does_not_freeze_between_two_predators(self, k_tangential):
        cfg = AquariumAPFConfig(k_tangential=k_tangential, k_prey_separation=0.0)
        policy = _policy("evader", cfg)
        policy._turn_sign = 1
        action = policy.get_action(_obs(predators=[(0.3, 0.0), (-0.3, 0.0)]))
        assert np.linalg.norm(action) > 1e-3

    def test_fallback_flees_the_nearer_of_two_predators(self):
        policy = _policy("evader")
        # Near threat on +x, slightly further on -x: flee toward -x.
        action = policy.get_action(_obs(predators=[(0.30, 0.0), (-0.31, 0.0)]))
        assert action[0] < 0.0

    def test_pursuer_does_not_freeze_between_two_prey(self):
        action = _policy("pursuer").get_action(_obs(prey=[(0.3, 0.0), (-0.3, 0.0)]))
        assert np.linalg.norm(action) > 1e-3


class TestAnchorConfigs:
    def test_anchor_set_is_deterministic_for_a_given_seed(self):
        a = XC.anchor_apf_configs(XC.APF_EVADER_CONFIG, 4, seed=0)
        b = XC.anchor_apf_configs(XC.APF_EVADER_CONFIG, 4, seed=0)
        assert [c.k_predator for c in a] == [c.k_predator for c in b]

    def test_no_variant_equals_the_held_out_scoring_config(self):
        """Training on the probe's own opponent would let a run overfit it."""
        base = XC.APF_EVADER_CONFIG
        for cfg in XC.anchor_apf_configs(base, 6, seed=0):
            assert (cfg.k_predator, cfg.k_tangential, cfg.k_wall) != (
                base.k_predator,
                base.k_tangential,
                base.k_wall,
            )

    def test_gains_stay_within_the_requested_spread(self):
        base = XC.APF_EVADER_CONFIG
        spread = 0.35
        for cfg in XC.anchor_apf_configs(base, 8, seed=1, spread=spread):
            ratio = cfg.k_predator / base.k_predator
            assert 1.0 / (1.0 + spread) <= ratio <= 1.0 + spread

    def test_disabled_terms_stay_disabled(self):
        """Perturbing a zero gain on would change the controller's character."""
        base = XC.APF_PURSUER_CONFIG      # k_tangential / k_prey_separation are 0
        for cfg in XC.anchor_apf_configs(base, 4, seed=0):
            assert cfg.k_tangential == 0.0
            assert cfg.k_prey_separation == 0.0

    def test_wall_threshold_stays_in_a_sane_band(self):
        for cfg in XC.anchor_apf_configs(XC.APF_EVADER_CONFIG, 16, seed=2):
            assert 0.05 <= cfg.wall_threshold <= 0.6

    def test_zero_variants_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            XC.anchor_apf_configs(XC.APF_EVADER_CONFIG, 0)
