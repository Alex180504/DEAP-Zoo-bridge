"""Pursuit-evasion behavioural descriptors.

Two variants ship:

* :func:`formation_spread_and_mean_distance` \u2014 2-D, recommended v1
  descriptor. Cheap, interpretable, low-collinearity axes.
* :func:`with_capture_efficiency` \u2014 3-D extension adding a
  capture-efficiency axis for experiments where QD archives should
  separate "tight formations that capture" from "tight formations
  that do not".

Both descriptors assume the trajectory summary dict contains
``positions`` of shape ``(T, N, 2)`` where ``N = num_pursuers + num_evaders``
and that pursuer indices precede evader indices. The shipped
:mod:`deap_pettingzoo.contrib.qdax.scoring_jax` rollouts respect this
ordering.

The descriptor objects returned here are **picklable** (top-level
classes, not closures) so they can be shipped to worker processes by
the parallel scoring path
(:func:`deap_pettingzoo.contrib.qdax.scoring_numpy.make_numpy_scoring_function`
with ``num_workers > 1``).
"""

from __future__ import annotations

from typing import Callable, Tuple


class _FormationSpreadAndMeanDistance:
    """Picklable 2-D pursuit-evasion descriptor.

    Equivalent to the closure previously returned by
    :func:`formation_spread_and_mean_distance`, but as a top-level
    class so :mod:`pickle` can serialise it for cross-process
    transport.
    """

    output_shape: int = 2
    min_bd: Tuple[float, ...] = (0.0, 0.0)
    max_bd: Tuple[float, ...] = (1.0, 1.0)

    def __init__(self, num_pursuers: int, num_evaders: int) -> None:
        n_p = int(num_pursuers)
        n_e = int(num_evaders)
        if n_p <= 0 or n_e <= 0:
            raise ValueError(
                f"num_pursuers and num_evaders must be positive, got "
                f"{n_p}, {n_e}."
            )
        self.n_p = n_p
        self.n_e = n_e

    def __call__(self, trajectory):
        import jax.numpy as jnp

        n_p = self.n_p
        n_e = self.n_e
        norm = 2.0 * jnp.sqrt(2.0)

        positions = trajectory["positions"]  # (T, N, 2)
        pursuer_pos = positions[:, :n_p, :]  # (T, n_p, 2)
        evader_pos = positions[:, n_p : n_p + n_e, :]  # (T, n_e, 2)

        centroid = jnp.mean(pursuer_pos, axis=1, keepdims=True)
        spread = jnp.mean(
            jnp.linalg.norm(pursuer_pos - centroid, axis=-1)
        )

        diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)
        mean_dist = jnp.mean(dists)

        return jnp.stack(
            [
                jnp.clip(spread / norm, 0.0, 1.0),
                jnp.clip(mean_dist / norm, 0.0, 1.0),
            ]
        )


def formation_spread_and_mean_distance(
    num_pursuers: int,
    num_evaders: int,
) -> Callable:
    """Return a 2-D descriptor ``(formation_spread, mean_dist_to_evader)``.

    ``formation_spread`` is the mean distance of each pursuer to the
    pursuer centroid (averaged over time). ``mean_dist_to_evader`` is
    the mean distance from every pursuer to every evader (averaged
    over time). Both are normalised to ``[0, 1]`` by a divisor of
    ``2 * sqrt(2)`` (the diagonal of a 2x2 MPE world).

    The returned object is a picklable
    :class:`_FormationSpreadAndMeanDistance` instance, so it survives
    cross-process transport via :mod:`pickle` (used by the parallel
    scoring path).
    """
    return _FormationSpreadAndMeanDistance(num_pursuers, num_evaders)


class _WithCaptureEfficiency:
    """Picklable 3-D pursuit-evasion descriptor (formation + capture rate)."""

    output_shape: int = 3
    min_bd: Tuple[float, ...] = (0.0, 0.0, 0.0)
    max_bd: Tuple[float, ...] = (1.0, 1.0, 1.0)

    def __init__(
        self,
        num_pursuers: int,
        num_evaders: int,
        capture_radius: float = 0.1,
    ) -> None:
        self.base = _FormationSpreadAndMeanDistance(num_pursuers, num_evaders)
        self.n_p = int(num_pursuers)
        self.n_e = int(num_evaders)
        self.r = float(capture_radius)

    def __call__(self, trajectory):
        import jax.numpy as jnp

        n_p = self.n_p
        n_e = self.n_e
        r = self.r

        positions = trajectory["positions"]
        pursuer_pos = positions[:, :n_p, :]
        evader_pos = positions[:, n_p : n_p + n_e, :]
        diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)  # (T, n_p, n_e)
        any_capture = jnp.any(dists < r, axis=(1, 2))  # (T,)
        efficiency = jnp.mean(any_capture.astype(jnp.float32))
        base_desc = self.base(trajectory)
        return jnp.concatenate([base_desc, efficiency[None]])


def with_capture_efficiency(
    num_pursuers: int,
    num_evaders: int,
    capture_radius: float = 0.1,
) -> Callable:
    """Return a 3-D descriptor: 2-D base + capture efficiency.

    ``capture_efficiency`` is the fraction of timesteps in which at
    least one pursuer is within ``capture_radius`` of any evader
    (i.e. an empirical capture-rate proxy).

    Returned object is a picklable :class:`_WithCaptureEfficiency`
    instance.
    """
    return _WithCaptureEfficiency(
        num_pursuers, num_evaders, capture_radius=capture_radius
    )


__all__ = [
    "formation_spread_and_mean_distance",
    "with_capture_efficiency",
]

