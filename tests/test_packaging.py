"""Tests for packaging import smoke, launcher arg parsing, and integrated-launch mode.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from deap_pettingzoo.ui._launcher import main as launcher_main, run as launcher_run


# ---------------------------------------------------------------------------
# Import smoke — every new subpackage must import without heavy deps
# ---------------------------------------------------------------------------


class TestImportSmoke:
    def test_catalog_imports_without_torch(self):
        """Catalog must never import torch/jax/agilerl at top level."""
        import deap_pettingzoo.catalog

        assert hasattr(deap_pettingzoo.catalog, "to_json")
        assert deap_pettingzoo.catalog.to_json()

    def test_service_imports_without_streamlit(self):
        """Service must never import streamlit at top level."""
        import deap_pettingzoo.service

        assert hasattr(deap_pettingzoo.service, "run")

    def test_baselines_exports_experiment_spec(self):
        from deap_pettingzoo.baselines import (
            ExperimentSpec,
            BudgetSpec,
            build_configs,
            validate_spec,
        )

        assert ExperimentSpec is not None
        assert BudgetSpec is not None
        assert build_configs is not None
        assert validate_spec is not None

    def test_baselines_exports_new_base_types(self):
        from deap_pettingzoo.baselines import (
            FitnessConfig,
            ProgressEvent,
            ProgressCallback,
        )

        assert FitnessConfig is not None
        assert ProgressEvent is not None

    def test_ui_api_client_imports_without_torch(self):
        from deap_pettingzoo.ui.services.api_client import ApiClient

        assert ApiClient is not None


# ---------------------------------------------------------------------------
# Launcher — arg parsing
# ---------------------------------------------------------------------------


class TestLauncherArgParsing:
    def test_no_flags_forwards_all(self):
        """Without --with-service, all args are forwarded to Streamlit."""
        with patch.object(sys, "argv", ["deap_pettingzoo.ui", "--server.port=8765"]):
            with patch(
                "deap_pettingzoo.ui._launcher.run", return_value=0
            ) as mock_run:
                result = launcher_main()
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["extra_args"] == ["--server.port=8765"]
                assert call_kwargs["with_service"] is False

    def test_with_service_flag_consumed(self):
        """--with-service is consumed by the launcher, not forwarded."""
        with patch.object(
            sys, "argv",
            ["deap_pettingzoo.ui", "--with-service", "--server.port=8765"],
        ):
            with patch(
                "deap_pettingzoo.ui._launcher.run", return_value=0
            ) as mock_run:
                launcher_main()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["with_service"] is True
                assert call_kwargs["extra_args"] == ["--server.port=8765"]

    def test_service_port_flag_consumed(self):
        """--service-port is consumed by the launcher."""
        with patch.object(
            sys, "argv",
            [
                "deap_pettingzoo.ui",
                "--with-service",
                "--service-port",
                "9999",
                "--server.port=8765",
            ],
        ):
            with patch(
                "deap_pettingzoo.ui._launcher.run", return_value=0
            ) as mock_run:
                launcher_main()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["service_port"] == 9999
                assert call_kwargs["with_service"] is True

    def test_invalid_service_port_ignored(self):
        """Non-integer --service-port is silently ignored (default used)."""
        with patch.object(
            sys, "argv",
            ["deap_pettingzoo.ui", "--with-service", "--service-port", "abc"],
        ):
            with patch(
                "deap_pettingzoo.ui._launcher.run", return_value=0
            ) as mock_run:
                launcher_main()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["service_port"] == 8100  # default


# ---------------------------------------------------------------------------
# Launcher — run() FASTAPI_URL wiring
# ---------------------------------------------------------------------------


class TestLauncherRun:
    def test_service_not_started_when_fastapi_url_already_set(self, monkeypatch):
        """When FASTAPI_URL is already set, with_service is a no-op."""
        monkeypatch.setenv("FASTAPI_URL", "http://remote:8100")
        with patch(
            "deap_pettingzoo.ui._launcher._ensure_streamlit_installed"
        ), patch("subprocess.call", return_value=0) as mock_call:
            launcher_run(with_service=True, service_port=9999)
            # subprocess.call should be called exactly once (for streamlit).
            assert mock_call.call_count == 1
            # The env passed to streamlit should still have the original FASTAPI_URL.
            env = mock_call.call_args[1]["env"]
            assert env["FASTAPI_URL"] == "http://remote:8100"

    def test_with_service_false_does_not_spawn(self):
        """with_service=False never spawns the service."""
        with patch(
            "deap_pettingzoo.ui._launcher._ensure_streamlit_installed"
        ), patch("subprocess.call", return_value=0) as mock_call, patch(
            "subprocess.Popen"
        ) as mock_popen:
            launcher_run(with_service=False)
            mock_popen.assert_not_called()
            assert mock_call.call_count == 1

    def test_extra_args_forwarded(self):
        """Extra args are forwarded to streamlit run."""
        with patch(
            "deap_pettingzoo.ui._launcher._ensure_streamlit_installed"
        ), patch("subprocess.call", return_value=0) as mock_call:
            launcher_run(extra_args=["--server.headless", "true"])
            cmd = mock_call.call_args[0][0]
            assert "--server.headless" in cmd
            assert "true" in cmd

    def test_run_returns_streamlit_exit_code(self):
        with patch(
            "deap_pettingzoo.ui._launcher._ensure_streamlit_installed"
        ), patch("subprocess.call", return_value=3) as mock_call:
            result = launcher_run()
            assert result == 3
