"""Flax MLP policy + adapter compatible with ``FixedTopologyGenome``.

The public contract:

* :class:`FlaxMLPPolicy` — a :class:`flax.linen.Module` whose parameter
  layout is *topologically identical* to
  :class:`~deap_pettingzoo.genomes.FixedTopologyGenome`. For a genome
  with ``layer_sizes = [n_0, n_1, ..., n_L]`` there are ``L`` Dense
  layers with sizes ``n_1, ..., n_L`` and activations applied between
  the hidden layers (not the output) \u2014 matching ``FixedTopologyGenome``
  semantics.
* :func:`flat_to_pytree` / :func:`pytree_to_flat` \u2014 bijection between
  flat weight vectors (DEAP's native representation) and Flax pytrees.
  Concatenation order is ``(W_0, b_0, W_1, b_1, ..., W_{L-1}, b_{L-1})``
  to match ``FixedTopologyGenome``'s layout.
* :class:`FlaxPolicyAdapter` \u2014 satisfies
  :class:`~deap_pettingzoo.baselines.base.PolicyProtocol`
  (``get_action(obs, action_mask=None) -> np.ndarray | int``) and
  marshals numpy \u2194 JAX at the boundary. Used in numpy-boundary
  scoring and by :class:`QDaxRepertoirePool` when sampling an opponent
  from a repertoire cell.

All JAX / Flax imports live inside function/constructor bodies so the
core package stays JAX-free.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


_ACTIVATION_MAP = {
    "tanh": "tanh",
    "relu": "relu",
    "sigmoid": "sigmoid",
    "linear": "linear",
}


def _resolve_activation(name: str):
    """Return the Flax activation callable for ``name``."""
    import jax.numpy as jnp
    from flax import linen as nn

    name = name.lower()
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return nn.relu
    if name == "sigmoid":
        return nn.sigmoid
    if name == "linear":
        return lambda x: x
    raise ValueError(
        f"Unsupported activation {name!r}; expected one of "
        f"{list(_ACTIVATION_MAP)}."
    )


def make_flax_mlp(layer_sizes: Sequence[int], activation: str = "tanh"):
    """Build a :class:`flax.linen.Module` mirroring ``FixedTopologyGenome``.

    Args:
        layer_sizes: ``[input, hidden..., output]``. Must have at least
            two entries.
        activation: One of ``{'tanh', 'relu', 'sigmoid', 'linear'}``.
            Applied between hidden layers; the output layer is linear.

    Returns:
        An instantiated ``FlaxMLPPolicy`` module.
    """
    from flax import linen as nn

    if len(layer_sizes) < 2:
        raise ValueError(
            f"layer_sizes must have >=2 entries, got {list(layer_sizes)}."
        )

    activation_fn = _resolve_activation(activation)
    sizes = tuple(int(s) for s in layer_sizes)

    class FlaxMLPPolicy(nn.Module):
        """MLP with ``len(layer_sizes) - 1`` Dense layers."""

        @nn.compact
        def __call__(self, obs):
            x = obs
            for i, out in enumerate(sizes[1:]):
                x = nn.Dense(features=out, name=f"dense_{i}")(x)
                if i < len(sizes) - 2:
                    x = activation_fn(x)
            return x

    return FlaxMLPPolicy()


def init_flax_mlp_params(module, input_size: int, seed: int = 0):
    """Return ``module.init(key, dummy_obs)['params']`` for ``module``."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(int(seed))
    dummy = jnp.zeros((input_size,), dtype=jnp.float32)
    variables = module.init(key, dummy)
    return variables["params"]


def _layer_shapes(layer_sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """Return ``[(in_0, out_0), (in_1, out_1), ...]``."""
    pairs: List[Tuple[int, int]] = []
    for i in range(len(layer_sizes) - 1):
        pairs.append((int(layer_sizes[i]), int(layer_sizes[i + 1])))
    return pairs


def flat_count(layer_sizes: Sequence[int]) -> int:
    """Count of scalar weights + biases for ``layer_sizes``."""
    return sum(a * b + b for a, b in _layer_shapes(layer_sizes))


def flat_to_pytree(flat: np.ndarray, layer_sizes: Sequence[int]):
    """Convert a flat weight vector into a Flax params pytree.

    The flat layout matches :meth:`FixedTopologyGenome.get_weights`:
    ``[W_0.flatten(), b_0, W_1.flatten(), b_1, ...]``. Each ``W`` has
    shape ``(in, out)``; ``b`` has shape ``(out,)``.
    """
    import jax.numpy as jnp

    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    expected = flat_count(layer_sizes)
    if arr.size != expected:
        raise ValueError(
            f"flat vector has {arr.size} entries; expected {expected} "
            f"for layer_sizes={list(layer_sizes)}."
        )
    params: dict = {}
    offset = 0
    for i, (in_dim, out_dim) in enumerate(_layer_shapes(layer_sizes)):
        w_size = in_dim * out_dim
        W = arr[offset : offset + w_size].reshape(in_dim, out_dim)
        offset += w_size
        b = arr[offset : offset + out_dim]
        offset += out_dim
        params[f"dense_{i}"] = {
            "kernel": jnp.asarray(W, dtype=jnp.float32),
            "bias": jnp.asarray(b, dtype=jnp.float32),
        }
    return params


def pytree_to_flat(params, layer_sizes: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`flat_to_pytree`."""
    chunks: List[np.ndarray] = []
    for i, (in_dim, out_dim) in enumerate(_layer_shapes(layer_sizes)):
        layer = params[f"dense_{i}"]
        W = np.asarray(layer["kernel"]).reshape(in_dim, out_dim)
        b = np.asarray(layer["bias"]).reshape(out_dim)
        chunks.append(W.reshape(-1))
        chunks.append(b)
    return np.concatenate(chunks).astype(np.float32, copy=False)


class FlaxPolicyAdapter:
    """Numpy-facing :class:`PolicyProtocol` implementation.

    Wraps a Flax module + params pytree; marshals numpy observations to
    JAX, runs the forward pass, decodes the output through an optional
    :class:`~deap_pettingzoo.policies.action_heads.ActionHead`.
    """

    def __init__(
        self,
        module,
        params,
        *,
        action_space=None,
        head=None,
        deterministic: bool = True,
    ) -> None:
        self._module = module
        self._params = params
        self._deterministic = bool(deterministic)
        self._head = head
        if head is None and action_space is not None:
            from deap_pettingzoo.policies.action_heads import make_head

            self._head = make_head(action_space, masked=True)

    @property
    def params(self):
        return self._params

    def reset(self) -> None:
        """No-op; the policy is stateless."""

    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        import jax.numpy as jnp

        obs = jnp.asarray(np.asarray(observation, dtype=np.float32))
        raw = np.asarray(self._module.apply({"params": self._params}, obs))
        if self._head is not None:
            return self._head.decode(raw, action_mask=action_mask)
        return raw


__all__ = [
    "make_flax_mlp",
    "init_flax_mlp_params",
    "flat_count",
    "flat_to_pytree",
    "pytree_to_flat",
    "FlaxPolicyAdapter",
]
