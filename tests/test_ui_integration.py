"""Tests for UI: api_client, param_widget coercion, validation grouping,
and the Setup page's form→spec logic (streamlit-free helpers).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from deap_pettingzoo.ui.components.param_widgets import (
    coerce_param,
    coerce_touched,
    _is_default,
)
from deap_pettingzoo.ui.components.validation_view import group_issues_by_tab
from deap_pettingzoo.ui.services.api_client import (
    ApiClient,
    ApiUnavailable,
    ApiValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps(name, ptype, default=None, **kw) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name, "type": ptype, "default": default}
    result.update(kw)
    return result


_MOCK_ENV_ALGO_JSON = {
    "name": "simple_tag_v3",
    "label": "Simple Tag",
    "env_config": [
        _ps("num_good", "int", 1, minimum=1, step=1),
        _ps("num_adversaries", "int", 3, minimum=1, step=1),
        _ps("max_cycles", "int", 25, minimum=1, step=1),
    ],
    "teams": [
        {"name": "predators", "agent_prefixes": ["adversary"], "role": "predator"},
        {"name": "prey", "agent_prefixes": ["agent"], "role": "prey"},
    ],
    "multi_team": True,
}
_MOCK_ALGO_EA_JSON = {
    "name": "ea",
    "label": "Evolutionary Algorithm",
    "hyperparameters": [
        _ps("population_size", "int", 50, minimum=2, step=1),
        _ps("hidden_size", "int", 64, minimum=1, step=1),
        _ps("strategy", "enum", "simple", choices=["simple", "mu_plus_lambda", "mu_comma_lambda"]),
        _ps("mutation_rate", "float", 0.3, minimum=0.0, maximum=1.0, step=0.05),
        _ps("num_workers", "int", 1, minimum=0, step=1, advanced=True),
    ],
    "available": True,
    "required_extras": [],
}
_MOCK_ALGO_MADDPG_JSON = {
    "name": "maddpg",
    "label": "MADDPG",
    "hyperparameters": [
        _ps("batch_size", "int", 64, minimum=1, step=1),
        _ps("lr_actor", "float", None, minimum=0.0, nullable=True),
    ],
    "available": False,
    "required_extras": ["rl-baselines"],
}

_STUB_CATALOG = {
    "catalog_version": "1",
    "algorithms": [_MOCK_ALGO_EA_JSON, _MOCK_ALGO_MADDPG_JSON],
    "environments": [_MOCK_ENV_ALGO_JSON],
    "coevolution": {
        "modes": [{"name": "competitive_2pop", "label": "Competitive (2 pop)"}],
        "schedules": ["synchronous", "alternating"],
        "aggregations": ["mean", "min", "max", "sum"],
    },
    "budget_modes": [{"name": "env_steps"}, {"name": "eval_episodes"}],
}


# ---------------------------------------------------------------------------
# param_widgets — coercion
# ---------------------------------------------------------------------------


class TestCoerceParam:
    def test_int_from_string_and_float(self):
        assert coerce_param(_ps("n", "int"), "50") == 50
        assert coerce_param(_ps("n", "int"), 3.0) == 3

    def test_int_rejects_bool(self):
        with pytest.raises(ValueError):
            coerce_param(_ps("n", "int"), True)

    def test_float(self):
        assert coerce_param(_ps("x", "float"), "0.5") == 0.5
        assert coerce_param(_ps("x", "float"), 3) == 3.0

    def test_bool(self):
        assert coerce_param(_ps("b", "bool"), True) is True
        assert coerce_param(_ps("b", "bool"), False) is False

    def test_enum_by_value_and_string(self):
        spec = _ps("m", "enum", "a", choices=["a", "b", "c"])
        assert coerce_param(spec, "b") == "b"
        with pytest.raises(ValueError):
            coerce_param(spec, "z")

    def test_nullable_accepts_none(self):
        assert coerce_param(_ps("s", "int", None, nullable=True), None) is None
        assert coerce_param(_ps("s", "int", None, nullable=True), "") is None

    def test_list(self):
        spec = _ps("g", "list", None)
        assert coerce_param(spec, [1, 2, 3]) == [1, 2, 3]
        with pytest.raises(ValueError):
            coerce_param(spec, "not-a-list")


class TestCoerceTouched:
    def test_only_non_default_coerced_and_marked_touched(self):
        specs = [
            _ps("a", "int", 5),
            _ps("b", "str", "hello"),
            _ps("c", "float", 0.0, nullable=True),
        ]
        values = {"a": 5, "b": "world", "c": None}
        coerced, touched = coerce_touched(specs, values)
        # a matches default (5==5), so not in coerced.
        # b differs ("world" != "hello"), so touched.
        # c is nullable: None != 0.0, so it IS touched.
        assert coerced == {"b": "world", "c": None}
        assert touched == {"b", "c"}

    def test_unknown_key_skipped(self):
        specs = [_ps("a", "int", 1)]
        coerced, touched = coerce_touched(specs, {"z": 99})
        assert coerced == {}
        assert touched == set()


# ---------------------------------------------------------------------------
# validation_view — grouping
# ---------------------------------------------------------------------------


class TestGroupIssues:
    def test_groups_by_prefix(self):
        issues = [
            {"path": "budget.mode", "code": "x", "message": "bad"},
            {"path": "budget.value", "code": "y", "message": "bad2"},
            {"path": "fitness.aggregation_method", "code": "z", "message": "w"},
        ]
        groups = group_issues_by_tab(issues)
        assert "Budget" in groups
        assert "Fitness" in groups
        assert len(groups["Budget"]) == 2

    def test_empty(self):
        assert group_issues_by_tab([]) == {}


# ---------------------------------------------------------------------------
# api_client — mock httpx
# ---------------------------------------------------------------------------


class TestApiClient:
    def test_get_catalog_mocked(self):
        """Mock httpx.Client so the api_client hits no real server."""
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _STUB_CATALOG
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/catalog"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        cat = client.get_catalog()
        assert cat["catalog_version"] == "1"
        assert len(cat["algorithms"]) == 2

    def test_422_raises_api_validation_error(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {
            "ok": False,
            "issues": [{"path": "env", "code": "x", "message": "bad"}],
        }
        mock_resp.text = "unprocessable"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments"

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        with pytest.raises(ApiValidationError) as exc:
            client.submit({"environment": ""})
        assert len(exc.value.report["issues"]) == 1

    def test_connect_error_maps_to_api_unavailable(self):
        client = ApiClient(base_url="http://test")
        mock_client = MagicMock()

        import httpx
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client.post.side_effect = httpx.ConnectError("refused")
        client._client = mock_client

        with pytest.raises(ApiUnavailable):
            client.get_catalog()

    @patch.object(ApiClient, "client", new_callable=MagicMock)
    def test_validate_dry_run(self, mock_client_prop):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "issues": []}
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments/validate"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_prop.return_value = mock_client

        client = ApiClient(base_url="http://test")
        client._client = mock_client
        report = client.validate({"environment": "simple_v3", "algorithms": ["ea"]})
        assert report["ok"]


# ---------------------------------------------------------------------------
# UIState setup_config round-trip
# ---------------------------------------------------------------------------


class TestUIStateSetupConfig:
    def test_setup_config_persists_and_restores(self):
        from deap_pettingzoo.ui.services.state import UIState

        store: Dict[str, Any] = {}
        state = UIState.from_mapping(store)
        state.setup_config = '{"env": "simple_v3"}'
        state.persist()

        restored = UIState.from_mapping(store)
        assert restored.setup_config == '{"env": "simple_v3"}'

    def test_default_is_none(self):
        from deap_pettingzoo.ui.services.state import UIState

        store: Dict[str, Any] = {}
        state = UIState.from_mapping(store)
        assert state.setup_config is None
        assert state.queue_expanded == ""


# ---------------------------------------------------------------------------
# Form validation integration (using real validate_spec via catalog)
# ---------------------------------------------------------------------------


class TestFormValidationRoundtrip:
    def test_valid_spec_validates(self):
        """Construct an ExperimentSpec from raw Python dataclasses,
        call validate_spec, and assert it's ok."""
        import deap_pettingzoo.environments  # noqa: F401

        from deap_pettingzoo.baselines.experiment_spec import (
            BudgetSpec,
            ExperimentSpec,
            validate_spec,
        )

        spec = ExperimentSpec(
            environment="simple_spread_v3",
            algorithms=["ea"],
            budget=BudgetSpec(mode="eval_episodes", value=16, num_seeds=1),
        )
        report = validate_spec(spec)
        assert report.ok, [i.message for i in report.errors]

    def test_invalid_spec_reports_errors(self):
        import deap_pettingzoo.environments  # noqa: F401

        from deap_pettingzoo.baselines.experiment_spec import (
            BudgetSpec,
            ExperimentSpec,
            validate_spec,
        )

        spec = ExperimentSpec(
            environment="nope_v99",
            algorithms=["skynet"],
            budget=BudgetSpec(mode="wall_clock", value=-1, num_seeds=0),
        )
        report = validate_spec(spec)
        assert not report.ok
        codes = {i.code for i in report.errors}
        assert "unknown_env" in codes
