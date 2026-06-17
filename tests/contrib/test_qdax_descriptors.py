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
    # num_evaders=0 is allowed (scripted/hidden opponents).
    assert pe.formation_spread_and_mean_distance(num_pursuers=1, num_evaders=0) is not None
    # num_evaders < 0 still raises.
    with pytest.raises(ValueError):
        pe.formation_spread_and_mean_distance(num_pursuers=1, num_evaders=-1)


def test_norm_scale_default_matches_mpe_diagonal() -> None:
    """Default norm (2*sqrt(2)) should normalise MPE-world positions into [0,1]."""
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe
    import math

    fn_default = pe.formation_spread_and_mean_distance(num_pursuers=2, num_evaders=1)
    fn_explicit = pe.formation_spread_and_mean_distance(
        num_pursuers=2, num_evaders=1, norm_scale=2.0 * math.sqrt(2.0)
    )
    positions = _positions(10, 2, 1)
    d_default = np.asarray(fn_default({"positions": positions}))
    d_explicit = np.asarray(fn_explicit({"positions": positions}))
    np.testing.assert_allclose(d_default, d_explicit, atol=1e-5)


def test_norm_scale_large_arena_saturates_less() -> None:
    """A large norm_scale should yield smaller (unsaturated) descriptor values."""
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe

    # Positions with pursuer spread ~1 unit: with default MPE norm ~0.35
    fn_mpe = pe.formation_spread_and_mean_distance(num_pursuers=2, num_evaders=1)
    fn_big = pe.formation_spread_and_mean_distance(
        num_pursuers=2, num_evaders=1, norm_scale=3000.0
    )
    positions = _positions(10, 2, 1)
    d_mpe = np.asarray(fn_mpe({"positions": positions}))
    d_big = np.asarray(fn_big({"positions": positions}))
    # With a 3000-unit norm the same physical spread produces much smaller values.
    assert float(d_big[0]) < float(d_mpe[0]) or float(d_big[1]) < float(d_mpe[1])


def test_with_capture_efficiency_norm_scale_propagates() -> None:
    """norm_scale should propagate through with_capture_efficiency's base descriptor."""
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe
    import math

    fn = pe.with_capture_efficiency(
        num_pursuers=2, num_evaders=1, norm_scale=2.0 * math.sqrt(2.0)
    )
    positions = _positions(8, 2, 1)
    d = np.asarray(fn({"positions": positions}))
    assert d.shape == (3,)
    assert np.all(d >= 0.0 - 1e-6)
    assert np.all(d <= 1.0 + 1e-6)
