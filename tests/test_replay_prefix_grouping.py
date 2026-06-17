"""Tests for the Replay page's frame-prefix grouping helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Skip when streamlit isn't available (e.g. CI runners without the
# `[ui]` extra). The module imports streamlit at top level.
pytest.importorskip("streamlit")


def _load_replay_module():
    """Load 5_Replay.py without going through Streamlit's page runner.

    The filename starts with a digit which makes it impossible to
    `import` directly. We bypass that with `importlib.util`.
    """
    here = Path(__file__).resolve()
    page_path = (
        here.parents[1]
        / "deap_pettingzoo"
        / "ui"
        / "pages"
        / "5_Replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_replay_under_test", page_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_replay_under_test"] = module
    # The page calls `st.set_page_config` / `st.stop()` at import time.
    # Patch streamlit so importing the module is side-effect-free.
    import streamlit as st

    # Stub out anything the script calls eagerly during import.
    class _Stop(Exception):  # noqa: N801 — local sentinel only
        pass

    st.set_page_config = lambda *a, **k: None  # type: ignore[assignment]
    st.title = lambda *a, **k: None  # type: ignore[assignment]
    st.info = lambda *a, **k: None  # type: ignore[assignment]
    st.warning = lambda *a, **k: None  # type: ignore[assignment]
    st.selectbox = lambda *a, **k: None  # type: ignore[assignment]
    st.radio = lambda *a, **k: None  # type: ignore[assignment]
    st.video = lambda *a, **k: None  # type: ignore[assignment]
    st.image = lambda *a, **k: None  # type: ignore[assignment]
    st.stop = lambda: (_ for _ in ()).throw(_Stop())  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    except _Stop:
        # The script reaches `st.stop()` because no run is selected;
        # by that point `_prefix_of` is already bound on the module.
        pass
    return module


def test_prefix_of_extracts_train():
    mod = _load_replay_module()
    assert mod._prefix_of(Path("train_episode_0001.mp4")) == "train_"


def test_prefix_of_extracts_live():
    mod = _load_replay_module()
    assert mod._prefix_of(Path("live_episode_0042.gif")) == "live_"


def test_prefix_of_unprefixed():
    mod = _load_replay_module()
    assert mod._prefix_of(Path("episode_0000.mp4")) == ""


def test_prefix_of_unrelated_filename():
    mod = _load_replay_module()
    assert mod._prefix_of(Path("not_a_frame.mp4")) == ""
