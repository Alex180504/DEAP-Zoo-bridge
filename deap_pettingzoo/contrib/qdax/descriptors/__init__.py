"""Behavioural-descriptor primitives for QDax scoring functions.

A descriptor is a pure function mapping a *trajectory summary* (a dict
of ``jnp.ndarray``\u200bs produced inside :func:`~scoring_jax.make_jax_scoring_function`
or :func:`~scoring_numpy.make_numpy_scoring_function`) to a 1-D
:class:`jnp.ndarray` of fixed shape.

Descriptor functions must be **pure** and :func:`jit`-compatible (no
Python-side control flow depending on array values) so they can be
embedded inside the JAX rollout.

The trajectory summary dict conventionally carries:

* ``positions``: shape ``(T, N, 2)`` \u2014 per-step positions of all
  agents (pursuers first, then evaders).
* ``velocities``: shape ``(T, N, 2)`` \u2014 per-step velocities.
* ``rewards``: shape ``(T, N)`` \u2014 per-step rewards.
* ``captures``: shape ``(T,)`` or ``(T, n_evaders)`` \u2014 boolean or
  integer indicators of capture events (env-specific).

Pursuit-evasion descriptors live in :mod:`.pursuit_evasion`; new
scenarios add new modules alongside it.
"""

from __future__ import annotations

from typing import Callable, Protocol


class DescriptorProtocol(Protocol):
    """Structural type for a descriptor function.

    A callable ``(trajectory_summary: Dict[str, Array]) -> Array`` with
    a fixed output shape declared via :attr:`output_shape`.
    """

    output_shape: int  # length of the returned 1-D array

    def __call__(self, trajectory): ...


__all__ = ["DescriptorProtocol"]
