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
"""

from __future__ import annotations

from typing import Callable


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
    """
    import jax.numpy as jnp

    n_p = int(num_pursuers)
    n_e = int(num_evaders)
    if n_p <= 0 or n_e <= 0:
        raise ValueError(
            f"num_pursuers and num_evaders must be positive, got "
            f"{n_p}, {n_e}."
        )
    norm = 2.0 * jnp.sqrt(2.0)

    def descriptor(trajectory):
        positions = trajectory["positions"]  # (T, N, 2)
        pursuer_pos = positions[:, :n_p, :]  # (T, n_p, 2)
        evader_pos = positions[:, n_p : n_p + n_e, :]  # (T, n_e, 2)

        centroid = jnp.mean(pursuer_pos, axis=1, keepdims=True)  # (T, 1, 2)
        spread = jnp.mean(
            jnp.linalg.norm(pursuer_pos - centroid, axis=-1)
        )  # scalar

        # Pairwise pursuer<->evader distances, averaged.
        diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)  # (T, n_p, n_e)
        mean_dist = jnp.mean(dists)

        return jnp.stack(
            [jnp.clip(spread / norm, 0.0, 1.0), jnp.clip(mean_dist / norm, 0.0, 1.0)]
        )

    descriptor.output_shape = 2  # type: ignore[attr-defined]
    descriptor.min_bd = (0.0, 0.0)  # type: ignore[attr-defined]
    descriptor.max_bd = (1.0, 1.0)  # type: ignore[attr-defined]
    return descriptor


def with_capture_efficiency(
    num_pursuers: int,
    num_evaders: int,
    capture_radius: float = 0.1,
) -> Callable:
    """Return a 3-D descriptor: 2-D base + capture efficiency.

    ``capture_efficiency`` is the fraction of timesteps in which at
    least one pursuer is within ``capture_radius`` of any evader
    (i.e. an empirical capture-rate proxy).
    """
    import jax.numpy as jnp

    base = formation_spread_and_mean_distance(num_pursuers, num_evaders)
    n_p = int(num_pursuers)
    n_e = int(num_evaders)
    r = float(capture_radius)

    def descriptor(trajectory):
        positions = trajectory["positions"]
        pursuer_pos = positions[:, :n_p, :]
        evader_pos = positions[:, n_p : n_p + n_e, :]
        diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
        dists = jnp.linalg.norm(diff, axis=-1)  # (T, n_p, n_e)
        any_capture = jnp.any(dists < r, axis=(1, 2))  # (T,)
        efficiency = jnp.mean(any_capture.astype(jnp.float32))
        base_desc = base(trajectory)
        return jnp.concatenate([base_desc, efficiency[None]])

    descriptor.output_shape = 3  # type: ignore[attr-defined]
    descriptor.min_bd = (0.0, 0.0, 0.0)  # type: ignore[attr-defined]
    descriptor.max_bd = (1.0, 1.0, 1.0)  # type: ignore[attr-defined]
    return descriptor


__all__ = [
    "formation_spread_and_mean_distance",
    "with_capture_efficiency",
]
