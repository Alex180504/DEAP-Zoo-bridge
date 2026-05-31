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
    # Non-numeric suffixes must NOT be stripped — ``team_blue`` and
    # ``team_red`` are different singleton groups, not one ``team`` group.
    assert GroupedIPPOTrainer._group_prefix("team_blue") == "team_blue"
    assert GroupedIPPOTrainer._group_prefix("agent_alpha") == "agent_alpha"


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


# --- AgileRL grouping monkey-patch regression guards (review S2) -----------

agilerl_algo_utils = pytest.importorskip("agilerl.utils.algo_utils")


def test_agilerl_patch_target_exists_with_expected_signature():
    """The function we monkey-patch must exist with the expected leading
    parameters. If a dependency bump renames/re-signatures it, this test
    fails loudly instead of grouped IPPO silently training without the fix.
    """
    import inspect

    from deap_pettingzoo.baselines.agilerl_trainer import (
        _AGILERL_TARGET_NAME,
        _AGILERL_TARGET_PARAMS,
    )

    target = getattr(agilerl_algo_utils, _AGILERL_TARGET_NAME, None)
    assert callable(target), (
        f"agilerl.utils.algo_utils.{_AGILERL_TARGET_NAME} missing or not "
        "callable; the grouping patch target has drifted."
    )
    params = tuple(inspect.signature(target).parameters)
    assert params[: len(_AGILERL_TARGET_PARAMS)] == _AGILERL_TARGET_PARAMS


def test_apply_grouping_patch_is_idempotent_and_installs():
    """Applying the patch installs the corrected function and is a no-op on
    repeat calls (guarded by the ``_grouping_patch_applied`` flag).
    """
    from deap_pettingzoo.baselines import agilerl_trainer as at

    at._apply_agilerl_grouping_patch()
    assert getattr(agilerl_algo_utils, "_grouping_patch_applied", False) is True
    assert (
        agilerl_algo_utils.concatenate_experiences_into_batches
        is at._patched_concatenate_experiences_into_batches
    )
    # Second call must not raise and must keep the patch installed.
    at._apply_agilerl_grouping_patch()
    assert (
        agilerl_algo_utils.concatenate_experiences_into_batches
        is at._patched_concatenate_experiences_into_batches
    )


def test_apply_grouping_patch_raises_on_missing_target(monkeypatch):
    """If the target function disappears, the patch raises a clear error
    rather than silently leaving grouped IPPO unpatched.
    """
    from deap_pettingzoo.baselines import agilerl_trainer as at

    monkeypatch.setattr(agilerl_algo_utils, "_grouping_patch_applied", False, raising=False)
    monkeypatch.delattr(
        agilerl_algo_utils, at._AGILERL_TARGET_NAME, raising=False
    )
    with pytest.raises(RuntimeError, match="missing"):
        at._apply_agilerl_grouping_patch()

