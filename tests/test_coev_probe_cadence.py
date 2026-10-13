"""Scripted-APF probe cadence for the aquarium coevolution suite.

Under ``side_pacing="equal_steps"`` a generation is no longer a fixed amount
of experience, so the probe fires on consumed env steps instead.
"""

from __future__ import annotations

from examples.pursuit_evasion.aquarium_coevolution import _ProbeCadence


def test_step_cadence_fires_once_per_bucket() -> None:
    cadence = _ProbeCadence(every_n=5, every_n_steps=1000)

    # First call establishes the baseline checkpoint.
    assert cadence.due(1, 0) is True
    # Still inside the first bucket, whatever the generation index.
    assert cadence.due(2, 400) is False
    assert cadence.due(5, 999) is False
    # Crossing 1000 consumed steps fires again.
    assert cadence.due(6, 1000) is True
    assert cadence.due(7, 1500) is False


def test_step_cadence_skips_buckets_a_big_generation_jumped() -> None:
    """A single expensive generation must not queue up a burst of probes."""
    cadence = _ProbeCadence(every_n=0, every_n_steps=1000)

    assert cadence.due(1, 0) is True
    assert cadence.due(2, 7500) is True    # jumped 7 buckets, fires once
    assert cadence.due(3, 7900) is False


def test_falls_back_to_generation_stride_without_a_consumed_count() -> None:
    cadence = _ProbeCadence(every_n=3, every_n_steps=1000)

    assert cadence.due(1, None) is False
    assert cadence.due(3, None) is True
    assert cadence.due(4, None) is False
    assert cadence.due(6, None) is True


def test_generation_stride_alone_is_honoured() -> None:
    cadence = _ProbeCadence(every_n=2, every_n_steps=0)

    assert cadence.due(1, 10_000) is False
    assert cadence.due(2, 20_000) is True


def test_cadence_can_be_disabled() -> None:
    assert _ProbeCadence(every_n=0, every_n_steps=0).enabled is False
    assert _ProbeCadence(every_n=5, every_n_steps=0).enabled is True
    assert _ProbeCadence(every_n=0, every_n_steps=1000).enabled is True
