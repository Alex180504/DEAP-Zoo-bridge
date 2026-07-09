"""Behavioural-descriptor primitives for QDax scoring functions.

 A descriptor is a pure function mapping a *trajectory summary* (a dict
 of ``jnp.ndarray``s produced inside :func:`~scoring_jax.make_jax_scoring_function`
or :func:`~scoring_numpy.make_numpy_scoring_function`) to a 1-D
:class:`jnp.ndarray` of fixed shape.

Descriptor functions must be **pure** and :func:`jit`-compatible (no
Python-side control flow depending on array values) so they can be
embedded inside the JAX rollout.

The trajectory summary dict conventionally carries:

* ``positions``: shape ``(T, N, 2)`` — per-step positions of all
  agents (pursuers first, then evaders).
* ``velocities``: shape ``(T, N, 2)`` — per-step velocities.
* ``rewards``: shape ``(T, N)`` — per-step rewards.
* ``captures``: shape ``(T,)`` or ``(T, n_evaders)`` — boolean or
  integer indicators of capture events (env-specific).

Concrete descriptor implementations are *experiment-side* code and live
outside the library (e.g. ``examples/qdax/descriptors.py``). The library
only requires the generic :class:`DescriptorProtocol` contract below.
"""

from __future__ import annotations

from typing import Callable, Protocol, Tuple


class DescriptorProtocol(Protocol):
    """Structural type for a descriptor function.

    A callable ``(trajectory_summary: Dict[str, Array]) -> Array`` with
    a fixed output shape declared via :attr:`output_shape`. The
    :attr:`min_bd` / :attr:`max_bd` bounds are used by the trainers to
    build the archive centroids.
    """

    output_shape: int  # length of the returned 1-D array
    min_bd: Tuple[float, ...]  # lower bounds of the behaviour-descriptor space
    max_bd: Tuple[float, ...]  # upper bounds of the behaviour-descriptor space

    def __call__(self, trajectory): ...


__all__ = ["DescriptorProtocol"]
