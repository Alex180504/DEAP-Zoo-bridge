import numpy as np
from gymnasium import spaces

from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
    aquarium_feature_size,
    extract_prey_caught_total,
    featurize_aquarium_observation,
)


class _StubAquariumEnv:
    def __init__(self):
        self.predator_observe_count = 1
        self.prey_observe_count = 1
        self.possible_agents = ["predator_0"]
        self.agents = ["predator_0"]
        self.metadata = {"name": "stub_aquarium"}
        self._obs = np.array([
            0.0, 0.25, 0.75, 0.75, 0.50,
            0.0, 0.90, 0.20, 0.40, 0.75, 0.60,
            0.0, 0.00, 0.00, 0.00, 0.00, 0.00,
        ], dtype=np.float32)

    def observation_space(self, agent):
        return spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(16)

    def reset(self, seed=None, options=None):
        del seed, options
        return {"predator_0": self._obs.copy()}, {"predator_0": {"prey_caught_total": 0}}

    def step(self, actions):
        del actions
        return (
            {"predator_0": self._obs.copy()},
            {"predator_0": 1.0},
            {"predator_0": False},
            {"predator_0": False},
            {"predator_0": {"prey_caught_total": 2}},
        )


def test_aquarium_feature_size():
    assert aquarium_feature_size(5, 5) == 53


def test_featurize_aquarium_observation_projects_relative_geometry():
    raw = np.array([
        0.0, 0.25, 0.75, 0.75, 0.50,
        0.0, 0.90, 0.20, 0.40, 0.75, 0.60,
        0.0, 0.00, 0.00, 0.00, 0.00, 0.00,
    ], dtype=np.float32)

    features = featurize_aquarium_observation(raw, 1, 1)

    np.testing.assert_allclose(
        features,
        np.array([
            1.0, 0.0, 0.5,
            1.0, 0.4, 0.0, 0.4, 0.6,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32),
        atol=1e-6,
    )


def test_extract_prey_caught_total_uses_max_reported_value():
    infos = {
        "predator_0": {"prey_caught_total": 1},
        "predator_1": {"prey_caught_total": 3},
        "predator_2": {},
    }

    assert extract_prey_caught_total(infos) == 3


def test_feature_wrapper_updates_obs_space_and_last_infos():
    env = AquariumObservationFeatureWrapper(_StubAquariumEnv())

    obs, infos = env.reset(seed=0)
    assert obs["predator_0"].shape == (13,)
    assert env.observation_space("predator_0").shape == (13,)
    assert infos["predator_0"]["prey_caught_total"] == 0
    assert env.last_infos["predator_0"]["prey_caught_total"] == 0

    obs, _, _, _, infos = env.step({"predator_0": 0})
    assert obs["predator_0"].shape == (13,)
    assert infos["predator_0"]["prey_caught_total"] == 2
    assert env.last_infos["predator_0"]["prey_caught_total"] == 2