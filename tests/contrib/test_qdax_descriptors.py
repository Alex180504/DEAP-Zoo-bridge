"""Tests for QDax pursuit-evasion descriptors.

Shape/range checks on hand-crafted (T, N, 2) position trajectories.
"""

import numpy as np
import pytest

pytest.importorskip("jax")


def _positions(T: int, n_p: int, n_e: int) -> np.ndarray:
    """Hand-crafted trajectory: pursuers on the x-axis, evaders at (0.5, 0.5)."""
    pos = np.zeros((T, n_p + n_e, 2), dtype=np.float32)
    for i in range(n_p):
        pos[:, i, 0] = float(i)
    pos[:, n_p:, :] = 0.5
    return pos


def test_formation_spread_and_mean_distance_shape_and_range() -> None:
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe

    fn = pe.formation_spread_and_mean_distance(num_pursuers=2, num_evaders=1)
    assert fn.output_shape == 2
    positions = _positions(10, 2, 1)
    d = np.asarray(fn({"positions": positions}))
    assert d.shape == (2,)
    assert np.all(d >= np.array(fn.min_bd) - 1e-6)
    assert np.all(d <= np.array(fn.max_bd) + 1e-6)


def test_with_capture_efficiency_is_3d() -> None:
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe

    fn = pe.with_capture_efficiency(num_pursuers=2, num_evaders=1)
    assert fn.output_shape == 3
    positions = _positions(8, 2, 1)
    d = np.asarray(fn({"positions": positions}))
    assert d.shape == (3,)


def test_formation_spread_raises_on_empty_teams() -> None:
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe

    with pytest.raises(ValueError):
        pe.formation_spread_and_mean_distance(num_pursuers=0, num_evaders=1)
    with pytest.raises(ValueError):
        pe.with_capture_efficiency(num_pursuers=1, num_evaders=0)
