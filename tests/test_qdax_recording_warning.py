"""Once-per-process warning when frame recording is requested on QDax.

Frame recording is unsupported on the QDax/JAX path because scoring
runs in JIT-compiled JAX functions with no live PettingZoo env. We
nevertheless want to emit a single ``RuntimeWarning`` so users see
why ``RecordingConfig.enabled=True`` produced no artefacts.

The flag is module-level (`_QDAX_RECORDING_WARNED`) so even multiple
trainer instances in the same process only warn once.
"""

from __future__ import annotations

import warnings

import pytest

from deap_pettingzoo.baselines.base import (
    BudgetMode,
    RecordingConfig,
    TrainingConfig,
)


def test_warn_once_helper_emits_runtime_warning():
    qmod = pytest.importorskip("deap_pettingzoo.contrib.qdax.trainer")
    qmod._QDAX_RECORDING_WARNED = False  # reset between test runs

    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=100,
        recording=RecordingConfig(enabled=True),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qmod._warn_recording_unsupported_once(cfg)
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "QDax/JAX path" in str(w[0].message)


def test_warn_once_is_idempotent():
    qmod = pytest.importorskip("deap_pettingzoo.contrib.qdax.trainer")
    qmod._QDAX_RECORDING_WARNED = False  # reset

    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=100,
        recording=RecordingConfig(enabled=True),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qmod._warn_recording_unsupported_once(cfg)
        qmod._warn_recording_unsupported_once(cfg)
        qmod._warn_recording_unsupported_once(cfg)
        # Exactly one warning across three calls.
        assert len(w) == 1


def test_warn_skipped_when_recording_disabled():
    qmod = pytest.importorskip("deap_pettingzoo.contrib.qdax.trainer")
    qmod._QDAX_RECORDING_WARNED = False  # reset

    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=100,
        # No recording= kwarg → default RecordingConfig(enabled=False).
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qmod._warn_recording_unsupported_once(cfg)
        assert len(w) == 0
        assert qmod._QDAX_RECORDING_WARNED is False
