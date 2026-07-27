"""Action heads for :class:`FeedforwardNeuralPolicy`.

An *action head* converts the raw output of a feedforward network into a
value that is valid for a given :mod:`gymnasium.spaces` action space.

Deterministic heads:

- :class:`DiscreteHead` — argmax over ``n`` logits, returns a Python ``int``.
- :class:`BoxHead` — tanh-squash raw output to ``[-1, 1]`` then rescale to
  ``[low, high]``, returns a ``np.float32`` array.
- :class:`MultiDiscreteHead` — split ``sum(nvec)`` logits into one chunk per
  sub-action and argmax each, returns a ``np.int64`` array of shape
  ``(len(nvec),)``.

Stochastic heads (for gradient-free search — argmax gives EA/CMA/PSO/QD a
step-function fitness landscape; softmax sampling smooths it):

- :class:`SoftmaxSamplingHead` — sample from ``softmax(logits / temperature)``
  over ``n`` logits, returns a Python ``int``.
- :class:`MultiDiscreteSoftmaxSamplingHead` — per-sub-action softmax sampling,
  returns a ``np.int64`` array of shape ``(len(nvec),)``.

:func:`make_head` selects a stochastic head via ``stochastic=True``.

:class:`Tuple` and :class:`Dict` action spaces are **not supported** for
EA-driven policies.  :func:`make_head` raises :class:`NotImplementedError`
with a message pointing callers at the AgileRL trainers, which handle those
spaces natively (relevant for custom enviroments only,
the native PettingZoo envs never use them).
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

    If the mask is entirely zero (no legal action, e.g. a "waiting"
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
            raise ValueError(f"BoxHead requires a non-empty action shape, got {shape}.")
        self._shape = tuple(int(s) for s in shape)
        self._low = np.broadcast_to(low_arr, self._shape).astype(np.float32).copy()
        self._high = np.broadcast_to(high_arr, self._shape).astype(np.float32).copy()
        # Replace non-finite bounds with a sensible default (-1/+1) so
        # unbounded Box spaces still produce finite actions.
        self._low = np.where(np.isfinite(self._low), self._low, -1.0).astype(np.float32)
        self._high = np.where(np.isfinite(self._high), self._high, 1.0).astype(
            np.float32
        )
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


class MultiDiscreteHead(ActionHead):
    """Per-sub-action argmax over concatenated logits.

    A :class:`gymnasium.spaces.MultiDiscrete` space with ``nvec = [n0, n1,
    ...]`` is decoded from a flat output of size ``sum(nvec)``: the output is
    split into one contiguous chunk per sub-action and :func:`numpy.argmax`
    is taken within each chunk.  This matches :func:`gymnasium.spaces.flatdim`
    and AgileRL's ``sample_multi_discrete`` layout, so a network can drive
    either backend with the same number of outputs.

    The per-chunk argmax is scale-invariant, so any output activation
    (e.g. NEAT's sigmoid, a raw linear layer) yields valid sub-actions.

    Args:
        nvec: Per-sub-action choice counts (array-like of ints, all >= 1).

    Returns a ``np.int64`` array of shape ``(len(nvec),)`` from :meth:`decode`,
    with entry ``i`` in ``[0, nvec[i])``.
    """

    def __init__(self, nvec: Any) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64).ravel()
        if self.nvec.size == 0:
            raise ValueError("MultiDiscreteHead requires a non-empty nvec.")
        if np.any(self.nvec < 1):
            raise ValueError(
                f"MultiDiscreteHead requires every nvec entry >= 1, got {self.nvec.tolist()}"
            )
        # Chunk boundaries into the flat logit vector: offsets[i]:offsets[i+1].
        self._offsets = np.concatenate(([0], np.cumsum(self.nvec))).astype(np.int64)

    @property
    def output_size(self) -> int:
        """Number of network output units required: ``sum(nvec)``."""
        return int(self.nvec.sum())

    def decode(self, raw_output: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_output).ravel()
        if arr.size != self.output_size:
            raise ValueError(
                f"MultiDiscreteHead expected output of size {self.output_size}, "
                f"got size {arr.size}"
            )
        return np.array(
            [
                int(np.argmax(arr[self._offsets[i] : self._offsets[i + 1]]))
                for i in range(self.nvec.size)
            ],
            dtype=np.int64,
        )


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable ``softmax(logits / temperature)``."""
    scaled = logits / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


class SoftmaxSamplingHead(ActionHead):
    """Stochastic softmax-sampling head for :class:`Discrete` spaces.

    The deterministic :class:`DiscreteHead` decodes with ``argmax``, which
    hands gradient-free search (EA / CMA-ES / PSO / MAP-Elites) a
    step-function fitness landscape: a small weight perturbation almost never
    flips the argmax, so most of the population shares a fitness and selection
    has no signal to follow.  Sampling from ``softmax(raw / temperature)``
    instead makes the *expected* action — and hence expected fitness — a
    smooth function of the logits, restoring a usable search gradient.
    Scored evaluation typically switches back to the deterministic
    :class:`DiscreteHead`.

    Args:
        n: Number of discrete actions (size of the output vector).
        temperature: Softmax temperature (> 0).  Lower sharpens toward argmax;
            higher flattens toward uniform.  Defaults to ``1.0``.
        rng: Optional :class:`numpy.random.Generator`.  Seed it
            deterministically (e.g. per agent id) to keep common-random-number
            evaluation reproducible; when ``None`` a fresh default generator
            is used.

    Returns Python ``int`` in ``[0, n)`` from :meth:`decode`.
    """

    def __init__(
        self, n: int, temperature: float = 1.0, rng: "np.random.Generator | None" = None
    ) -> None:
        if n < 1:
            raise ValueError(f"SoftmaxSamplingHead requires n >= 1, got {n}")
        if temperature <= 0:
            raise ValueError(
                f"SoftmaxSamplingHead requires temperature > 0, got {temperature}"
            )
        self.n = int(n)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def output_size(self) -> int:
        return self.n

    def decode(self, raw_output: np.ndarray) -> int:
        arr = np.asarray(raw_output, dtype=np.float64).ravel()
        if arr.size != self.n:
            raise ValueError(
                f"SoftmaxSamplingHead expected output of size {self.n}, "
                f"got size {arr.size}"
            )
        probs = _softmax(arr, self.temperature)
        return int(self.rng.choice(self.n, p=probs))


class MultiDiscreteSoftmaxSamplingHead(ActionHead):
    """Stochastic per-sub-action softmax-sampling head for :class:`MultiDiscrete`.

    The stochastic counterpart of :class:`MultiDiscreteHead`: each sub-action's
    contiguous logit chunk is turned into a ``softmax(chunk / temperature)``
    distribution and sampled independently, so gradient-free search sees a
    smooth landscape on every sub-action (see :class:`SoftmaxSamplingHead` for
    the rationale).

    Args:
        nvec: Per-sub-action choice counts (array-like of ints, all >= 1).
        temperature: Softmax temperature (> 0).  Defaults to ``1.0``.
        rng: Optional :class:`numpy.random.Generator` (see
            :class:`SoftmaxSamplingHead`).

    Returns a ``np.int64`` array of shape ``(len(nvec),)`` from :meth:`decode`,
    with entry ``i`` in ``[0, nvec[i])``.
    """

    def __init__(
        self,
        nvec: Any,
        temperature: float = 1.0,
        rng: "np.random.Generator | None" = None,
    ) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64).ravel()
        if self.nvec.size == 0:
            raise ValueError(
                "MultiDiscreteSoftmaxSamplingHead requires a non-empty nvec."
            )
        if np.any(self.nvec < 1):
            raise ValueError(
                "MultiDiscreteSoftmaxSamplingHead requires every nvec entry >= 1, "
                f"got {self.nvec.tolist()}"
            )
        if temperature <= 0:
            raise ValueError(
                "MultiDiscreteSoftmaxSamplingHead requires temperature > 0, "
                f"got {temperature}"
            )
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Chunk boundaries into the flat logit vector: offsets[i]:offsets[i+1].
        self._offsets = np.concatenate(([0], np.cumsum(self.nvec))).astype(np.int64)

    @property
    def output_size(self) -> int:
        return int(self.nvec.sum())

    def decode(self, raw_output: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_output, dtype=np.float64).ravel()
        if arr.size != self.output_size:
            raise ValueError(
                "MultiDiscreteSoftmaxSamplingHead expected output of size "
                f"{self.output_size}, got size {arr.size}"
            )
        out = np.empty(self.nvec.size, dtype=np.int64)
        for i in range(self.nvec.size):
            chunk = arr[self._offsets[i] : self._offsets[i + 1]]
            probs = _softmax(chunk, self.temperature)
            out[i] = int(self.rng.choice(chunk.size, p=probs))
        return out


def required_output_size(action_space: Any) -> int:
    """Return the number of network output units required for ``action_space``.

    Mirrors :func:`make_head`'s space support: :class:`Discrete` needs
    ``n`` outputs, :class:`MultiDiscrete` needs ``sum(nvec)`` outputs, and
    :class:`Box` needs ``prod(shape)`` outputs (1-D only in practice).
    Unsupported spaces raise :class:`NotImplementedError` with the same
    message pattern used by :func:`make_head`.

    Args:
        action_space: A :class:`gymnasium.spaces.Space` instance.

    Returns:
        Required number of output units.

    Raises:
        NotImplementedError: For :class:`Tuple`, :class:`Dict`, and any
            other unsupported space type.
        TypeError: If ``action_space`` is ``None``.
    """
    if action_space is None:
        raise TypeError("required_output_size requires a gymnasium Space, got None.")

    if hasattr(action_space, "n") and not hasattr(action_space, "nvec"):
        return int(action_space.n)
    if hasattr(action_space, "nvec"):
        return int(np.sum(action_space.nvec))
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


def make_head(
    action_space: Any,
    *,
    masked: bool = False,
    stochastic: bool = False,
    temperature: float = 1.0,
    rng: "np.random.Generator | None" = None,
) -> ActionHead:
    """Construct the :class:`ActionHead` appropriate for ``action_space``.

    Args:
        action_space: A :class:`gymnasium.spaces.Space` instance.
        masked: If ``True`` and ``action_space`` is :class:`Discrete`,
            return a :class:`MaskedDiscreteHead` instead of the default
            :class:`DiscreteHead` so the caller can supply an
            ``action_mask`` to :meth:`ActionHead.decode`.  Ignored for
            non-Discrete spaces (MPE2 MultiDiscrete envs are not turn-based).
        stochastic: If ``True``, return the softmax-sampling head for the
            space — :class:`SoftmaxSamplingHead` for :class:`Discrete`,
            :class:`MultiDiscreteSoftmaxSamplingHead` for
            :class:`MultiDiscrete` — which smooths the argmax fitness
            landscape for gradient-free search.  Ignored for :class:`Box`
            (already smooth).  Mutually exclusive with ``masked``.
        temperature: Softmax temperature for the stochastic heads (> 0).
            Ignored when ``stochastic=False``.
        rng: Optional :class:`numpy.random.Generator` for the stochastic
            heads (seed it for reproducible common-random-number evaluation).

    Returns:
        :class:`SoftmaxSamplingHead` / :class:`MultiDiscreteSoftmaxSamplingHead`
        when ``stochastic=True`` (Discrete / MultiDiscrete); otherwise
        :class:`DiscreteHead` (or :class:`MaskedDiscreteHead` when
        ``masked=True``) for :class:`Discrete`,
        :class:`MultiDiscreteHead` for :class:`MultiDiscrete`,
        :class:`BoxHead` for :class:`Box`.

    Raises:
        NotImplementedError: For :class:`Tuple`, :class:`Dict`, and any
            other unsupported space type.  The message points callers at
            the AgileRL trainers.
        TypeError: If ``action_space`` is ``None``.
        ValueError: If both ``masked`` and ``stochastic`` are requested.
    """
    if action_space is None:
        raise TypeError("make_head requires a gymnasium Space, got None.")
    if masked and stochastic:
        raise ValueError(
            "make_head: masked and stochastic are mutually exclusive — masking "
            "is for turn-based argmax, stochastic sampling is for gradient-free "
            "search."
        )

    # Use duck-typing so to avoid a hard gymnasium import here.
    if hasattr(action_space, "n") and not hasattr(action_space, "nvec"):
        if stochastic:
            return SoftmaxSamplingHead(
                int(action_space.n), temperature=temperature, rng=rng
            )
        cls = MaskedDiscreteHead if masked else DiscreteHead
        return cls(int(action_space.n))
    if hasattr(action_space, "nvec"):
        if stochastic:
            return MultiDiscreteSoftmaxSamplingHead(
                action_space.nvec, temperature=temperature, rng=rng
            )
        return MultiDiscreteHead(action_space.nvec)
    if (
        hasattr(action_space, "low")
        and hasattr(action_space, "high")
        and hasattr(action_space, "shape")
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
    "MultiDiscreteHead",
    "SoftmaxSamplingHead",
    "MultiDiscreteSoftmaxSamplingHead",
    "make_head",
    "required_output_size",
]
