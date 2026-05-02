"""Action heads for :class:`FeedforwardNeuralPolicy`.

An *action head* converts the raw output of a feedforward network into a
value that is valid for a given :mod:`gymnasium.spaces` action space.

Two heads are provided:

- :class:`DiscreteHead` — argmax over ``n`` logits, returns a Python ``int``.
- :class:`BoxHead` — tanh-squash raw output to ``[-1, 1]`` then rescale to
  ``[low, high]``, returns a ``np.float32`` array.

:class:`MultiDiscrete`, :class:`Tuple`, and :class:`Dict` action spaces are
explicitly **not supported** for EA-driven policies.  :func:`make_head`
raises :class:`NotImplementedError` with a message pointing callers at the
AgileRL trainers, which handle those spaces natively.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np


class ActionHead:
    """Base action-head protocol."""

    #: Whether :meth:`decode` accepts an ``action_mask`` keyword. 
    #: Defaults to ``False``; :class:`MaskedDiscreteHead` overrides to ``True``.
    supports_mask: bool = False

    def decode(self, raw_output: np.ndarray) -> Any:  # pragma: no cover - abstract
        """Convert raw network output to an env-valid action."""
        raise NotImplementedError

    @property
    def output_size(self) -> int:  # pragma: no cover - abstract
        """Number of network output units required by this head."""
        raise NotImplementedError


class DiscreteHead(ActionHead):
    """Argmax over ``n`` logits.

    Args:
        n: Number of discrete actions (size of the output vector).

    Returns Python ``int`` in ``[0, n)`` from :meth:`decode`.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"DiscreteHead requires n >= 1, got {n}")
        self.n = int(n)

    @property
    def output_size(self) -> int:
        return self.n

    def decode(self, raw_output: np.ndarray) -> int:
        arr = np.asarray(raw_output).ravel()
        if arr.size != self.n:
            raise ValueError(
                f"DiscreteHead expected output of size {self.n}, "
                f"got size {arr.size}"
            )
        return int(np.argmax(arr))


class MaskedDiscreteHead(DiscreteHead):
    """Mask-aware variant of :class:`DiscreteHead` for turn-based games.

    :meth:`decode` accepts an optional ``action_mask`` of length ``n``
    where ``0`` marks an illegal action; the corresponding logits are
    set to ``-inf`` before :func:`numpy.argmax` so that only legal
    actions can be selected. When ``action_mask`` is ``None`` the head
    behaves identically to :class:`DiscreteHead`.

    If the mask is entirely zero (no legal action — e.g. a "waiting"
    agent on the other side of a turn-based board under the masked
    adapter), :meth:`decode` returns ``0`` by convention; the adapter
    discards the action anyway.
    """

    supports_mask = True

    def decode(  # type: ignore[override]
        self,
        raw_output: np.ndarray,
        action_mask: "np.ndarray | None" = None,
    ) -> int:
        arr = np.asarray(raw_output, dtype=np.float64).ravel()
        if arr.size != self.n:
            raise ValueError(
                f"MaskedDiscreteHead expected output of size {self.n}, "
                f"got size {arr.size}"
            )
        if action_mask is None:
            return int(np.argmax(arr))
        mask = np.asarray(action_mask).ravel()
        if mask.size != self.n:
            raise ValueError(
                f"MaskedDiscreteHead expected action_mask of size {self.n}, "
                f"got size {mask.size}"
            )
        if not np.any(mask):
            # Nothing legal (waiting-turn agent); return 0 — adapter will discard this action anyway.
            return 0
        masked = np.where(mask.astype(bool), arr, -np.inf)
        return int(np.argmax(masked))


class BoxHead(ActionHead):
    """Tanh-squashed, rescaled Box action.

    The raw output is passed through ``tanh`` to produce values in
    ``[-1, 1]`` and then linearly rescaled to ``[low, high]`` element-wise.
    ``low``/``high`` may be scalar or array-like with matching shape.

    Arbitrary action shapes are supported: the network is expected to
    produce ``prod(shape)`` output units; :meth:`decode` reshapes the
    decoded flat vector back to ``shape`` before returning.

    Args:
        low: Lower action bound.
        high: Upper action bound.
        shape: Action shape; inferred from ``low``/``high`` when possible.

    Raises:
        ValueError: If ``low``/``high`` are scalars and no explicit
            ``shape`` is provided, or if ``shape`` is empty.

    Note:
        Non-finite bounds (``±inf``) are silently replaced with ``∓1.0``
        so that unbounded :class:`Box` spaces still produce finite actions.
    """

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: tuple = None,
    ) -> None:
        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)
        if shape is None:
            if low_arr.ndim == 0 and high_arr.ndim == 0:
                raise ValueError(
                    "BoxHead needs an explicit shape when low/high are scalars."
                )
            shape = low_arr.shape if low_arr.ndim > 0 else high_arr.shape
        if len(shape) == 0:
            raise ValueError(
                f"BoxHead requires a non-empty action shape, got {shape}."
            )
        self._shape = tuple(int(s) for s in shape)
        self._low = np.broadcast_to(low_arr, self._shape).astype(np.float32).copy()
        self._high = np.broadcast_to(high_arr, self._shape).astype(np.float32).copy()
        # Replace non-finite bounds with a sensible default (-1/+1) so
        # unbounded Box spaces still produce finite actions.
        self._low = np.where(np.isfinite(self._low), self._low, -1.0).astype(np.float32)
        self._high = np.where(np.isfinite(self._high), self._high, 1.0).astype(np.float32)
        self._flat_low = self._low.ravel()
        self._flat_high = self._high.ravel()

    @property
    def output_size(self) -> int:
        """Number of network output units required: ``prod(shape)``."""
        return int(np.prod(self._shape))

    @property
    def shape(self) -> tuple:
        return self._shape

    def decode(self, raw_output: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_output, dtype=np.float32).ravel()
        if arr.size != self.output_size:
            raise ValueError(
                f"BoxHead expected output of size {self.output_size}, "
                f"got size {arr.size}"
            )
        squashed = np.tanh(arr)
        # Affine rescale from [-1, 1] to [low, high].
        scale = (self._flat_high - self._flat_low) / 2.0
        bias = (self._flat_high + self._flat_low) / 2.0
        decoded = (squashed * scale + bias).astype(np.float32)
        return decoded.reshape(self._shape)


def required_output_size(action_space: Any) -> int:
    """Return the number of network output units required for ``action_space``.

    Mirrors :func:`make_head`'s space support: :class:`Discrete` needs
    ``n`` outputs, :class:`Box` needs ``prod(shape)`` outputs (1-D only
    in practice).  Unsupported spaces raise :class:`NotImplementedError`
    with the same message pattern used by :func:`make_head`.

    Args:
        action_space: A :class:`gymnasium.spaces.Space` instance.

    Returns:
        Required number of output units.

    Raises:
        NotImplementedError: For :class:`MultiDiscrete`, :class:`Tuple`,
            :class:`Dict`, and any other unsupported space type.
        TypeError: If ``action_space`` is ``None``.
    """
    if action_space is None:
        raise TypeError("required_output_size requires a gymnasium Space, got None.")

    if hasattr(action_space, "n") and not hasattr(action_space, "nvec"):
        return int(action_space.n)
    if hasattr(action_space, "nvec"):
        raise NotImplementedError(
            "MultiDiscrete action spaces are not supported by "
            "FeedforwardNeuralPolicy. Use one of the AgileRL trainers "
            "(MADDPG / MATD3 / IPPO) for envs with communication "
            "channels (e.g. simple_reference_v3, simple_world_comm_v3)."
        )
    if hasattr(action_space, "shape") and hasattr(action_space, "low"):
        shape = action_space.shape
        if shape is None or len(shape) == 0:
            raise ValueError(
                f"Box action space must have a non-empty shape, got {shape!r}."
            )
        size = 1
        for dim in shape:
            size *= int(dim)
        return size
    if hasattr(action_space, "spaces"):
        raise NotImplementedError(
            "Tuple/Dict action spaces are not supported by "
            "FeedforwardNeuralPolicy. Use one of the AgileRL trainers."
        )
    raise NotImplementedError(
        f"Unsupported action space type: {type(action_space).__name__}. "
        "Use one of the AgileRL trainers for this environment."
    )


def make_head(action_space: Any, *, masked: bool = False) -> ActionHead:
    """Construct the :class:`ActionHead` appropriate for ``action_space``.

    Args:
        action_space: A :class:`gymnasium.spaces.Space` instance.
        masked: If ``True`` and ``action_space`` is :class:`Discrete`,
            return a :class:`MaskedDiscreteHead` instead of the default
            :class:`DiscreteHead` so the caller can supply an
            ``action_mask`` to :meth:`ActionHead.decode`.

    Returns:
        :class:`DiscreteHead` (or :class:`MaskedDiscreteHead` when
        ``masked=True``) for :class:`Discrete`, :class:`BoxHead` for
        :class:`Box`.

    Raises:
        NotImplementedError: For :class:`MultiDiscrete`, :class:`Tuple`,
            :class:`Dict`, and any other unsupported space type.  The
            message points callers at the AgileRL trainers.
        TypeError: If ``action_space`` is ``None``.
    """
    if action_space is None:
        raise TypeError("make_head requires a gymnasium Space, got None.")

    # Use duck-typing so we don't force a hard gymnasium import here.
    if hasattr(action_space, "n") and not hasattr(action_space, "nvec"):
        cls = MaskedDiscreteHead if masked else DiscreteHead
        return cls(int(action_space.n))
    if hasattr(action_space, "nvec"):
        raise NotImplementedError(
            "MultiDiscrete action spaces are not supported by "
            "FeedforwardNeuralPolicy. Use one of the AgileRL trainers "
            "(MADDPG / MATD3 / IPPO) for envs with communication "
            "channels (e.g. simple_reference_v3, simple_world_comm_v3)."
        )
    if hasattr(action_space, "low") and hasattr(action_space, "high") and hasattr(
        action_space, "shape"
    ):
        return BoxHead(action_space.low, action_space.high, shape=action_space.shape)
    if hasattr(action_space, "spaces"):
        raise NotImplementedError(
            "Tuple/Dict action spaces are not supported by "
            "FeedforwardNeuralPolicy. Use one of the AgileRL trainers."
        )
    raise NotImplementedError(
        f"Unsupported action space type: {type(action_space).__name__}. "
        "Use one of the AgileRL trainers for this environment."
    )


__all__ = [
    "ActionHead",
    "DiscreteHead",
    "MaskedDiscreteHead",
    "BoxHead",
    "make_head",
    "required_output_size",
]
