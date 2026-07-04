"""Tests for UI experiment setup.
"""

from __future__ import annotations

import pytest

from deap_pettingzoo.catalog import algorithms as cat_algorithms
from deap_pettingzoo.catalog.schema import ParamSpec, ParamType
from deap_pettingzoo.ui.components.param_widgets import (
    available_enum_choices,
    coerce_touched,
)
from deap_pettingzoo.ui.services.spec_utils import spec_to_form_defaults
from deap_pettingzoo.utils import device_check


class TestBudgetRoundTrip:
    def test_spec_to_form_defaults_uses_mode_value_keys(self):
        """The Setup page's budget param specs are named "mode"/"value" —
        ``spec_to_form_defaults`` must emit those exact keys or the
        pre-populated form silently falls back to defaults on load."""
        spec = {"budget": {"mode": "eval_episodes", "value": 200, "num_seeds": 3}}
        form = spec_to_form_defaults(spec)
        assert form["budget"] == {
            "mode": "eval_episodes",
            "value": 200,
            "num_seeds": 3,
            "budget_on_consumed": False,
        }

    def test_budget_survives_form_widget_round_trip(self):
        """Simulate the Setup page's ``coerce_touched(_BUDGET_PARAMS, ...)``
        call fed the *converted* form defaults, and check the submitted
        spec preserves the original mode/value instead of resetting to the
        hand-authored defaults (env_steps / 50000)."""
        original = {"mode": "eval_episodes", "value": 200}
        form = spec_to_form_defaults({"budget": original})

        budget_params = [
            {"name": "mode", "type": "enum",
             "choices": ["env_steps", "eval_episodes"], "default": "env_steps"},
            {"name": "value", "type": "int", "default": 50000, "minimum": 1},
        ]
        # ``coerce_touched`` models a no-op rerun: values equal to what was
        # pre-populated (nothing re-touched by the user this render).
        submitted, touched = coerce_touched(budget_params, form["budget"])

        assert submitted["mode"] == "eval_episodes"
        assert submitted["value"] == 200
        assert touched == {"mode", "value"}


class TestUnavailableChoiceGating:
    def test_unavailable_choice_rejected_by_coerce(self):
        spec = ParamSpec(
            "device", ParamType.ENUM, "cpu",
            choices=("cpu", "cuda"), unavailable_choices=("cuda",),
        )
        assert spec.coerce("cpu") == "cpu"
        with pytest.raises(ValueError):
            spec.coerce("cuda")

    def test_unavailable_choice_fails_validation(self):
        spec = ParamSpec(
            "device", ParamType.ENUM, "cpu",
            choices=("cpu", "cuda"), unavailable_choices=("cuda",),
        )
        assert spec.validate("cpu") == []
        issues = spec.validate("cuda")
        assert issues and issues[0].severity == "error"

    def test_available_choices_property_excludes_unavailable(self):
        spec = ParamSpec(
            "device", ParamType.ENUM, "cpu",
            choices=("cpu", "cuda"), unavailable_choices=("cuda",),
        )
        assert spec.available_choices == ("cpu",)

    def test_to_dict_carries_unavailable_choices(self):
        spec = ParamSpec(
            "device", ParamType.ENUM, "cpu",
            choices=("cpu", "cuda"), unavailable_choices=("cuda",),
        )
        assert spec.to_dict()["unavailable_choices"] == ["cuda"]

    def test_available_enum_choices_helper_drops_unavailable(self):
        """The pure filtering helper ``render_param`` uses to build the
        selectbox's ``options=`` list never includes an unavailable choice."""
        spec_dict = {
            "name": "device", "type": "enum", "choices": ["cpu", "cuda"],
            "unavailable_choices": ["cuda"],
        }
        assert available_enum_choices(spec_dict) == ["cpu"]

    def test_available_enum_choices_helper_no_op_when_all_available(self):
        spec_dict = {"name": "device", "type": "enum", "choices": ["cpu", "cuda"]}
        assert available_enum_choices(spec_dict) == ["cpu", "cuda"]


class TestDeviceChoiceGatedByCudaAvailability(object):
    @pytest.fixture(autouse=True)
    def _clear_algorithm_cache(self):
        # ``get_algorithm`` is ``@lru_cache``-d; the device choice is only
        # recomputed on a cache miss, so each variant needs a fresh build.
        cat_algorithms.clear_cache()
        yield
        cat_algorithms.clear_cache()

    def test_cuda_pruned_when_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(device_check, "cuda_available", lambda: False)
        spec = cat_algorithms.get_algorithm("maddpg")
        device_param = spec.hyperparameter("device")
        assert device_param is not None
        assert "cuda" in device_param.unavailable_choices

    def test_cuda_available_when_gpu_present(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(device_check, "cuda_available", lambda: True)
        spec = cat_algorithms.get_algorithm("maddpg")
        device_param = spec.hyperparameter("device")
        assert device_param is not None
        assert device_param.unavailable_choices == ()


class TestHPONumEnvsPruned:
    def test_hpo_trainer_does_not_surface_num_envs(self):
        """``num_envs`` must be hidden for the AgileRL-HPO rollout loop (it
        always builds a single, non-vectorised env). The plain AgileRL
        trainers still surface it, but the HPO variants must not."""
        hpo_spec = cat_algorithms.get_algorithm("maddpg-hpo")
        assert hpo_spec.hyperparameter("num_envs") is None

        plain_spec = cat_algorithms.get_algorithm("maddpg")
        assert plain_spec.hyperparameter("num_envs") is not None
