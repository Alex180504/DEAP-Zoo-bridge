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

    def __init__(
        self,
        num_pursuers: int,
        num_evaders: int,
        norm_scale: float = None,
    ) -> None:
        n_p = int(num_pursuers)
        n_e = int(num_evaders)
        if n_p <= 0:
            raise ValueError(
                f"num_pursuers must be positive, got {n_p}."
            )
        if n_e < 0:
            raise ValueError(
                f"num_evaders must be >= 0, got {n_e}."
            )
        self.n_p = n_p
        self.n_e = n_e
        self.norm_scale: float = (
            float(norm_scale) if norm_scale is not None else 2.0 * (2.0 ** 0.5)
        )

    def __call__(self, trajectory):
        import jax.numpy as jnp

        n_p = self.n_p
        n_e = self.n_e
        norm = self.norm_scale

        positions = trajectory["positions"]  # (T, N, 2)
        pursuer_pos = positions[:, :n_p, :]  # (T, n_p, 2)
        evader_pos = positions[:, n_p : n_p + n_e, :]  # (T, n_e, 2)

        centroid = jnp.mean(pursuer_pos, axis=1, keepdims=True)
        spread = jnp.mean(
            jnp.linalg.norm(pursuer_pos - centroid, axis=-1)
        )

        if n_e > 0:
            diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
            dists = jnp.linalg.norm(diff, axis=-1)
            mean_dist = jnp.mean(dists)
        else:
            mean_dist = jnp.zeros(())

        return jnp.stack(
            [
                jnp.clip(spread / norm, 0.0, 1.0),
                jnp.clip(mean_dist / norm, 0.0, 1.0),
            ]
        )


class _SingleTeamFormation:
    """Picklable 2-D descriptor for a single homogeneous team.

    Useful when the environment only exposes one role (e.g. evader-only
    training where pursuers are scripted and absent from
    ``env.possible_agents``).  The two axes are:

    * **spread** — mean distance of each agent from the team centroid
      (averaged over time), normalised by ``norm_scale``.
    * **pairwise** — mean pairwise inter-agent distance (averaged over
      time), normalised by ``norm_scale``.

    Both axes are clipped to ``[0, 1]``.
    """

    output_shape: int = 2
    min_bd: Tuple[float, ...] = (0.0, 0.0)
    max_bd: Tuple[float, ...] = (1.0, 1.0)

    def __init__(self, num_agents: int, norm_scale: float = 1.0) -> None:
        if int(num_agents) <= 0:
            raise ValueError(
                f"num_agents must be positive, got {num_agents}."
            )
        self.n = int(num_agents)
        self.norm_scale = float(norm_scale)

    def __call__(self, trajectory):
        import jax.numpy as jnp

        positions = trajectory["positions"]  # (T, N, 2)
        pos = positions[:, : self.n, :]  # (T, n, 2)
        norm = self.norm_scale

        centroid = jnp.mean(pos, axis=1, keepdims=True)  # (T, 1, 2)
        spread = jnp.mean(jnp.linalg.norm(pos - centroid, axis=-1))

        # Mean pairwise distance; guard n==1 to avoid division by zero.
        diff = pos[:, :, None, :] - pos[:, None, :, :]  # (T, n, n, 2)
        dists = jnp.linalg.norm(diff, axis=-1)  # (T, n, n)
        n_pairs = self.n * (self.n - 1)
        mean_pair = (
            jnp.sum(dists) / (n_pairs * positions.shape[0])
            if n_pairs > 0
            else jnp.zeros(())
        )

        return jnp.stack(
            [
                jnp.clip(spread / norm, 0.0, 1.0),
                jnp.clip(mean_pair / norm, 0.0, 1.0),
            ]
        )


def single_team_formation(num_agents: int, norm_scale: float = 1.0) -> Callable:
    """Return a 2-D descriptor ``(centroid_spread, mean_pairwise_dist)``.

    Use this when the env only exposes a single agent role (no
    pursuer/evader split in ``env.possible_agents``), e.g. evader-only
    training where the opposing team is scripted.

    Args:
        num_agents: Number of agents expected in ``positions``.
        norm_scale: Divisor to normalise distances to ``[0, 1]``.
            Set to the arena diagonal for aquarium envs.
    """
    return _SingleTeamFormation(num_agents, norm_scale=norm_scale)


def formation_spread_and_mean_distance(
    num_pursuers: int,
    num_evaders: int,
    norm_scale: float = None,
) -> Callable:
    """Return a 2-D descriptor ``(formation_spread, mean_dist_to_evader)``.

    ``formation_spread`` is the mean distance of each pursuer to the
    pursuer centroid (averaged over time). ``mean_dist_to_evader`` is
    the mean distance from every pursuer to every evader (averaged
    over time). Both are normalised to ``[0, 1]`` by dividing by
    ``norm_scale``.

    Args:
        norm_scale: Divisor used to normalise spread and distance into
            ``[0, 1]``. Defaults to ``2 * sqrt(2)`` (the diagonal of
            the 2×2 MPE world). Set to the arena diagonal length for
            non-MPE environments (e.g. ``3606.0`` for a 2000×3000
            pixel aquarium).

    The returned object is a picklable
    :class:`_FormationSpreadAndMeanDistance` instance, so it survives
    cross-process transport via :mod:`pickle` (used by the parallel
    scoring path).
    """
    return _FormationSpreadAndMeanDistance(num_pursuers, num_evaders, norm_scale=norm_scale)


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
        norm_scale: float = None,
    ) -> None:
        self.base = _FormationSpreadAndMeanDistance(
            num_pursuers, num_evaders, norm_scale=norm_scale
        )
        self.n_p = int(num_pursuers)
        self.n_e = int(num_evaders)
        self.r = float(capture_radius)

    def __call__(self, trajectory):
        import jax.numpy as jnp

        n_p = self.n_p
        n_e = self.n_e
        r = self.r

        if n_e > 0:
            positions = trajectory["positions"]
            pursuer_pos = positions[:, :n_p, :]
            evader_pos = positions[:, n_p : n_p + n_e, :]
            diff = pursuer_pos[:, :, None, :] - evader_pos[:, None, :, :]
            dists = jnp.linalg.norm(diff, axis=-1)  # (T, n_p, n_e)
            any_capture = jnp.any(dists < r, axis=(1, 2))  # (T,)
            efficiency = jnp.mean(any_capture.astype(jnp.float32))
        else:
            efficiency = jnp.zeros(())

        base_desc = self.base(trajectory)
        return jnp.concatenate([base_desc, efficiency[None]])


def with_capture_efficiency(
    num_pursuers: int,
    num_evaders: int,
    capture_radius: float = 0.1,
    norm_scale: float = None,
) -> Callable:
    """Return a 3-D descriptor: 2-D base + capture efficiency.

    ``capture_efficiency`` is the fraction of timesteps in which at
    least one pursuer is within ``capture_radius`` of any evader
    (i.e. an empirical capture-rate proxy).

    Args:
        norm_scale: Divisor for normalising spread and distance axes.
            See :func:`formation_spread_and_mean_distance` for details.

    Returned object is a picklable :class:`_WithCaptureEfficiency`
    instance.
    """
    return _WithCaptureEfficiency(
        num_pursuers, num_evaders, capture_radius=capture_radius, norm_scale=norm_scale
    )


__all__ = [
    "formation_spread_and_mean_distance",
    "single_team_formation",
    "with_capture_efficiency",
]

