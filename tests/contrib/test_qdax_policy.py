"""Tests for :mod:`deap_pettingzoo.contrib.qdax.policy` \u2014 MLP plumbing.

Gated via ``pytest.importorskip('jax')``; Flax/QDax pull in JAX so
these remain optional.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


def test_flat_count_matches_dense_weights_and_biases() -> None:
    from deap_pettingzoo.contrib.qdax.policy import flat_count

    # 2 \u2192 3 \u2192 1: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
    assert flat_count((2, 3, 1)) == 13


def test_flat_to_pytree_and_back_is_identity() -> None:
    from deap_pettingzoo.contrib.qdax.policy import (
        flat_count,
        flat_to_pytree,
        pytree_to_flat,
    )

    layer_sizes = (4, 5, 3)
    size = flat_count(layer_sizes)
    flat = np.arange(size, dtype=np.float32)
    tree = flat_to_pytree(flat, layer_sizes)
    back = pytree_to_flat(tree, layer_sizes)
    np.testing.assert_allclose(back, flat)


def test_init_flax_mlp_params_shape_is_consistent() -> None:
    from deap_pettingzoo.contrib.qdax.policy import (
        flat_count,
        init_flax_mlp_params,
        make_flax_mlp,
        pytree_to_flat,
    )

    layer_sizes = (3, 4, 2)
    module = make_flax_mlp(layer_sizes, activation="tanh")
    params = init_flax_mlp_params(module, input_size=3, seed=0)
    flat = pytree_to_flat(params, layer_sizes)
    assert flat.shape == (flat_count(layer_sizes),)
