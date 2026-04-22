"""Tests for :class:`GroupedIPPOTrainer` grouping validation and registry."""

import pytest

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

from deap_pettingzoo.baselines import GroupedIPPOTrainer, get_trainer


def test_registered_as_ippo_grouped():
    """Registry exposes the trainer under ``'ippo-grouped'``."""
    trainer = get_trainer("ippo-grouped")
    assert isinstance(trainer, GroupedIPPOTrainer)


def test_group_prefix_strips_trailing_index():
    assert GroupedIPPOTrainer._group_prefix("adversary_0") == "adversary"
    assert GroupedIPPOTrainer._group_prefix("agent_12") == "agent"
    # Agents without an underscore become singleton groups.
    assert GroupedIPPOTrainer._group_prefix("predator") == "predator"


def test_validation_rejects_all_singletons():
    obs = {"alice": gym.spaces.Box(-1, 1, (4,)),
           "bob": gym.spaces.Box(-1, 1, (4,))}
    act = {"alice": gym.spaces.Discrete(3),
           "bob": gym.spaces.Discrete(3)}
    with pytest.raises(ValueError, match="singleton groups"):
        GroupedIPPOTrainer._validate_grouping(
            ["alice", "bob"], obs, act,
        )


def test_validation_rejects_mismatched_obs_in_group():
    obs = {
        "agent_0": gym.spaces.Box(-1, 1, (4,)),
        "agent_1": gym.spaces.Box(-1, 1, (6,)),  # mismatch!
    }
    act = {
        "agent_0": gym.spaces.Discrete(3),
        "agent_1": gym.spaces.Discrete(3),
    }
    with pytest.raises(ValueError, match="mismatched observation spaces"):
        GroupedIPPOTrainer._validate_grouping(
            ["agent_0", "agent_1"], obs, act,
        )


def test_validation_rejects_mismatched_action_in_group():
    obs = {
        "adversary_0": gym.spaces.Box(-1, 1, (4,)),
        "adversary_1": gym.spaces.Box(-1, 1, (4,)),
    }
    act = {
        "adversary_0": gym.spaces.Discrete(3),
        "adversary_1": gym.spaces.Discrete(5),  # mismatch!
    }
    with pytest.raises(ValueError, match="mismatched action spaces"):
        GroupedIPPOTrainer._validate_grouping(
            ["adversary_0", "adversary_1"], obs, act,
        )


def test_validation_accepts_valid_grouping():
    """simple_tag-style agent set is a canonical valid example."""
    obs = {
        "adversary_0": gym.spaces.Box(-1, 1, (4,)),
        "adversary_1": gym.spaces.Box(-1, 1, (4,)),
        "adversary_2": gym.spaces.Box(-1, 1, (4,)),
        "agent_0": gym.spaces.Box(-1, 1, (5,)),
    }
    act = {
        "adversary_0": gym.spaces.Discrete(5),
        "adversary_1": gym.spaces.Discrete(5),
        "adversary_2": gym.spaces.Discrete(5),
        "agent_0": gym.spaces.Discrete(5),
    }
    groups = GroupedIPPOTrainer._validate_grouping(
        list(obs.keys()), obs, act,
    )
    assert groups == {
        "adversary": ["adversary_0", "adversary_1", "adversary_2"],
        "agent": ["agent_0"],
    }


def test_capabilities_flags_on_policy():
    caps = GroupedIPPOTrainer().get_capabilities()
    assert caps["on_policy"] is True
    assert caps["discrete_action"] is True
    assert caps["continuous_action"] is True
    assert caps["needs_global_state"] is False
