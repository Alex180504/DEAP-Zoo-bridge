"""Tests for device fallback resolution (CPU-only torch safety net)."""

from __future__ import annotations

import warnings

import pytest

from deap_pettingzoo.utils import device_check
from deap_pettingzoo.utils.device_check import resolve_device


@pytest.fixture(autouse=True)
def _reset_warned():
    device_check._warned_devices.clear()
    yield
    device_check._warned_devices.clear()


def _set_cuda(monkeypatch, available: bool):
    monkeypatch.setattr(device_check, "_torch_cuda_available", lambda: available)


def test_cpu_passthrough(monkeypatch):
    _set_cuda(monkeypatch, False)
    assert resolve_device("cpu") == "cpu"


def test_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    _set_cuda(monkeypatch, False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_device("cuda") == "cpu"
        assert resolve_device("cuda:0") == "cpu"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_cuda_kept_when_available(monkeypatch):
    _set_cuda(monkeypatch, True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"


def test_none_autodetects(monkeypatch):
    _set_cuda(monkeypatch, True)
    assert resolve_device(None) == "cuda"
    _set_cuda(monkeypatch, False)
    assert resolve_device(None) == "cpu"


def test_warning_emitted_once_per_device(monkeypatch):
    _set_cuda(monkeypatch, False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_device("cuda")
        resolve_device("cuda")  # same device -> no second warning
    assert sum(issubclass(w.category, RuntimeWarning) for w in caught) == 1
