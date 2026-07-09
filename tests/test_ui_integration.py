"""Tests for UI: api_client, param_widget coercion, validation grouping,
and the Setup page's form->spec logic (streamlit-free helpers).
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
        _ps("lr_actor", "float", 1e-3, minimum=0.0, step=1e-4),
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


# ---------------------------------------------------------------------------
# Queue page — api_client methods used by 9_Queue.py
# ---------------------------------------------------------------------------


_MOCK_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": "abc1234567890def",
        "label": "my run",
        "status": "queued",
        "spec": {"environment": "simple_tag_v3", "algorithms": ["ea"]},
        "created_at": "2025-01-01T12:00:00Z",
        "started_at": None,
        "finished_at": None,
        "run_dirs": [],
        "error": None,
        "position": 0,
        "progress": None,
    },
    {
        "id": "run2222222222",
        "label": "run 2",
        "status": "running",
        "spec": {"environment": "simple_spread_v3", "algorithms": ["maddpg"]},
        "created_at": "2025-01-01T13:00:00Z",
        "started_at": "2025-01-01T13:00:05Z",
        "finished_at": None,
        "run_dirs": [],
        "error": None,
        "position": 0,
        "progress": {
            "unit": "generation",
            "current": 3,
            "total": 10,
            "fraction": 0.3,
            "best_fitness": 12.5,
            "mean_fitness": 8.2,
            "wall_time_s": 15.0,
            "run_index": 0,
            "total_runs": 1,
            "algorithm": "maddpg",
            "seed": 0,
            "extra": {},
        },
    },
    {
        "id": "done9999999999",
        "label": "done",
        "status": "completed",
        "spec": {"environment": "simple_v3", "algorithms": ["ea", "cma-es"]},
        "created_at": "2025-01-01T11:00:00Z",
        "started_at": "2025-01-01T11:00:01Z",
        "finished_at": "2025-01-01T11:02:30Z",
        "run_dirs": ["/tmp/runs/ea_s0"],
        "error": None,
        "position": -1,
        "progress": None,
    },
]

_MOCK_QUEUE_STATE: Dict[str, Any] = {
    "paused": False,
    "running_id": "run2222222222",
    "queued_ids": ["abc1234567890def"],
    "worker_limit": 1,
}

_MOCK_PROGRESS: Dict[str, Any] = {
    "unit": "generation",
    "current": 5,
    "total": 10,
    "fraction": 0.5,
    "best_fitness": 15.0,
    "mean_fitness": 10.0,
    "wall_time_s": 30.0,
    "run_index": 0,
    "total_runs": 1,
    "algorithm": "ea",
    "seed": 0,
    "extra": {},
}

_MOCK_STDOUT: Dict[str, Any] = {
    "text": "[ea] gen 5/10  best=15.0  mean=10.0\n",
    "next_offset": 60,
    "eof": False,
}


class TestApiClientQueue:
    """Test the api_client methods that the Queue page calls."""

    def test_list_experiments(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _MOCK_EXPERIMENTS
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        client._client = mock_client

        exps = client.list_experiments()
        assert len(exps) == 3
        assert exps[1]["status"] == "running"
        assert exps[2]["status"] == "completed"

    def test_list_experiments_by_status(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [e for e in _MOCK_EXPERIMENTS if e["status"] == "queued"]
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        client._client = mock_client

        exps = client.list_experiments(status="queued")
        assert len(exps) == 1

    def test_get_progress(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _MOCK_PROGRESS
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments/abc/progress"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        client._client = mock_client

        prog = client.get_progress("abc")
        assert prog["fraction"] == 0.5
        assert prog["best_fitness"] == 15.0

    def test_get_stdout(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _MOCK_STDOUT
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments/abc/stdout"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        client._client = mock_client

        chunk = client.get_stdout("abc", lines=50)
        assert chunk["eof"] is False
        assert "gen 5/10" in chunk["text"]

    def test_get_queue_state(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _MOCK_QUEUE_STATE
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/queue"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        client._client = mock_client

        qs = client.get_queue()
        assert qs["running_id"] == "run2222222222"
        assert len(qs["queued_ids"]) == 1

    def test_pause_and_resume(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"paused": True}
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/queue/pause"

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        r = client.pause_queue()
        assert r["paused"] is True

    def test_cancel_experiment(self):
        client = ApiClient(base_url="http://test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc123", "status": "cancelled"}
        mock_resp.text = "ok"
        mock_resp.headers = {}
        mock_resp.url = MagicMock()
        mock_resp.url.path = "/experiments/abc123/cancel"

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        client._client = mock_client

        r = client.cancel_experiment("abc123")
        assert r["status"] == "cancelled"


class TestQueueStatusBadge:
    """Test the _status_badge helper from queue_helpers."""

    def test_known_statuses(self):
        from deap_pettingzoo.ui.services.queue_helpers import status_badge
        assert "\U0001f504" in status_badge("running")
        assert "\u2705" in status_badge("completed")
        assert "\u274c" in status_badge("failed")
        assert "\u23f3" in status_badge("queued")
        assert "\U0001f6ab" in status_badge("cancelled")

    def test_unknown_status(self):
        from deap_pettingzoo.ui.services.queue_helpers import status_badge
        badge = status_badge("bogus")
        assert "Bogus" in badge


class TestQueueFormatTime:
    def test_iso_formatted(self):
        from deap_pettingzoo.ui.services.queue_helpers import format_time
        assert "2025" in format_time("2025-06-15T14:30:00Z")

    def test_none_returns_dash(self):
        from deap_pettingzoo.ui.services.queue_helpers import format_time
        assert format_time(None) == "\u2014"


class TestQueueShortId:
    def test_long_id_truncated(self):
        from deap_pettingzoo.ui.services.queue_helpers import short_id
        result = short_id("abcdef1234567890")
        assert len(result) <= 12  # 8 chars + ellipsis
        assert result.startswith("abcdef12")

    def test_short_id_preserved(self):
        from deap_pettingzoo.ui.services.queue_helpers import short_id
        assert short_id("abc") == "abc"


class TestQueueAlgoLabel:
    def test_extracts_algorithm_names(self):
        from deap_pettingzoo.ui.services.queue_helpers import algo_label
        spec = {"algorithms": ["ea", "cma-es", "pso", "maddpg"]}
        result = algo_label(spec)
        assert "ea" in result
        assert "cma-es" in result
        assert "pso" in result
        # Only first 3 shown
        assert "maddpg" not in result

    def test_falls_back_to_env(self):
        from deap_pettingzoo.ui.services.queue_helpers import algo_label
        spec = {"environment": "simple_tag_v3", "algorithms": []}
        assert algo_label(spec) == "simple_tag_v3"

    def test_none_spec(self):
        from deap_pettingzoo.ui.services.queue_helpers import algo_label
        assert algo_label(None) == "\u2014"


class TestQueueProgressDescription:
    def test_label_format(self):
        from deap_pettingzoo.ui.services.queue_helpers import progress_description
        desc = progress_description({
            "unit": "generation", "current": 3, "total": 10,
            "run_index": 0, "total_runs": 2,
            "algorithm": "ea", "seed": 1,
        })
        assert "Run 1/2" in desc
        assert "ea s1" in desc
        assert "generation 3/10" in desc

    def test_empty_dict(self):
        from deap_pettingzoo.ui.services.queue_helpers import progress_description
        assert progress_description({}) == ""

    def test_none(self):
        from deap_pettingzoo.ui.services.queue_helpers import progress_description
        assert progress_description(None) == ""
