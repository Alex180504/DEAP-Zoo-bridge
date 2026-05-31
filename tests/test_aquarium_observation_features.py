import numpy as np
from gymnasium import spaces

from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
    LazyActionPenaltyWrapper,
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

    # sdirection=0.75 -> angle_deg=90 (north).  The env encodes directions
    # via get_angle_from_vector = -atan2(vy, vx), so the inverse decode is
    # (cos(rad), -sin(rad)) = (0, -1) for 90°.  See apf_policy module
    # docstring for the convention reference.
    np.testing.assert_allclose(
        features,
        np.array([
            0.0, -1.0, 0.5,
            1.0, 0.0, -0.4, 0.4, 0.6,
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
    # append_agent_id=False keeps the legacy 13-dim featurized obs so this
    # test stays focused on the geometric feature layout.  See
    # ``test_feature_wrapper_appends_agent_one_hot`` for the identity tail.
    env = AquariumObservationFeatureWrapper(_StubAquariumEnv(), append_agent_id=False)

    obs, infos = env.reset(seed=0)
    assert obs["predator_0"].shape == (13,)
    assert env.observation_space("predator_0").shape == (13,)
    assert infos["predator_0"]["prey_caught_total"] == 0
    assert env.last_infos["predator_0"]["prey_caught_total"] == 0

    obs, _, _, _, infos = env.step({"predator_0": 0})
    assert obs["predator_0"].shape == (13,)
    assert infos["predator_0"]["prey_caught_total"] == 2
    assert env.last_infos["predator_0"]["prey_caught_total"] == 2


def test_feature_wrapper_appends_agent_one_hot_by_default():
    """Default wrapper appends a one-hot agent ID so a shared homogeneous
    genome can express role differentiation across symmetric predators."""

    class _MultiPredStub(_StubAquariumEnv):
        def __init__(self):
            super().__init__()
            self.possible_agents = ["predator_0", "predator_1", "predator_2"]
            self.agents = list(self.possible_agents)

        def reset(self, seed=None, options=None):
            del seed, options
            return (
                {a: self._obs.copy() for a in self.agents},
                {a: {"prey_caught_total": 0} for a in self.agents},
            )

    env = AquariumObservationFeatureWrapper(_MultiPredStub())
    obs, _ = env.reset(seed=0)
    # 13 geometric features + 3 one-hot agent IDs.
    assert obs["predator_0"].shape == (16,)
    assert env.observation_space("predator_1").shape == (16,)
    np.testing.assert_array_equal(obs["predator_0"][-3:], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(obs["predator_1"][-3:], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(obs["predator_2"][-3:], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_lazy_penalty_uses_heading_concentration_for_adjacent_bins():
    env = LazyActionPenaltyWrapper(
        _StubAquariumEnv(),
        concentration_threshold=0.7,
        penalty=-8.0,
        min_actions=4,
        mode="terminal",
    )
    env._history["predator_0"] = [0, 1, 0, 1]

    rewards = {"predator_0": 1.0}
    infos = {"predator_0": {}}
    env._apply_terminal_penalty(rewards, infos)

    assert rewards["predator_0"] == -7.0
    assert env.last_penalised["predator_0"] is True
    assert env.last_heading_concentration["predator_0"] > 0.9
    assert infos["predator_0"]["lazy_heading_concentration"] > 0.9


def test_lazy_penalty_does_not_fire_for_spread_headings():
    env = LazyActionPenaltyWrapper(
        _StubAquariumEnv(),
        concentration_threshold=0.7,
        penalty=-8.0,
        min_actions=4,
        mode="terminal",
    )
    env._history["predator_0"] = [0, 4, 8, 12]

    rewards = {"predator_0": 1.0}
    infos = {"predator_0": {}}
    env._apply_terminal_penalty(rewards, infos)

    assert rewards["predator_0"] == 1.0
    assert env.last_penalised["predator_0"] is False
    assert env.last_heading_concentration["predator_0"] < 0.1