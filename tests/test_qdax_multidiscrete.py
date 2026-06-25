"""MultiDiscrete support for the QDax numpy scoring path.

Exercises :func:`make_numpy_scoring_function` with ``action_type=
"multidiscrete"`` on the synthetic homogeneous MultiDiscrete env. A finite
fitness implies the per-sub-action argmax decode produced env-valid integer
action vectors (a wrong scalar decode would make ``env.step`` raise).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdax")
pytest.importorskip("flax")
pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from deap_pettingzoo.contrib.qdax.policy import (  # noqa: E402
    init_flax_mlp_params,
    make_flax_mlp,
)
from deap_pettingzoo.contrib.qdax.scoring_numpy import (  # noqa: E402
    close_scoring_function,
    make_numpy_scoring_function,
)
from tests.multidiscrete_env import MD_ENV_NAME, register_md_env  # noqa: E402


def _mean_xy_descriptor(trajectory):
    """Minimal 2-D descriptor: mean (x, y) position over the trajectory."""
    pos = trajectory["positions"]  # (T, N, 2)
    return jnp.mean(pos, axis=(0, 1))


_mean_xy_descriptor.output_shape = (2,)


def _build_genotypes(module, layer_sizes, batch):
    inits = [
        init_flax_mlp_params(module, input_size=layer_sizes[0], seed=k)
        for k in range(batch)
    ]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *inits)


@pytest.fixture(scope="module", autouse=True)
def _md_env():
    from deap_pettingzoo.environments.registry import is_registered, unregister_env

    register_md_env()
    yield MD_ENV_NAME
    if is_registered(MD_ENV_NAME):
        unregister_env(MD_ENV_NAME)


def test_numpy_scoring_multidiscrete_finite():
    layer_sizes = (4, 8, 7)  # obs=4, sum(nvec)=3+4=7
    module = make_flax_mlp(layer_sizes, activation="tanh")
    fn = make_numpy_scoring_function(
        MD_ENV_NAME,
        layer_sizes=layer_sizes,
        episode_length=10,
        descriptor_fn=_mean_xy_descriptor,
        action_type="multidiscrete",
        nvec=(3, 4),
        activation="tanh",
        num_workers=1,
    )
    try:
        geno = _build_genotypes(module, layer_sizes, batch=3)
        fit, desc, extra = fn(geno, jax.random.PRNGKey(0))
        assert fit.shape == (3,)
        assert np.all(np.isfinite(np.asarray(fit)))
        assert np.asarray(desc).shape[0] == 3
    finally:
        close_scoring_function(fn)


def test_numpy_scoring_multidiscrete_requires_nvec():
    layer_sizes = (4, 8, 7)
    module = make_flax_mlp(layer_sizes, activation="tanh")
    fn = make_numpy_scoring_function(
        MD_ENV_NAME,
        layer_sizes=layer_sizes,
        episode_length=5,
        descriptor_fn=_mean_xy_descriptor,
        action_type="multidiscrete",
        nvec=None,  # missing on purpose
        activation="tanh",
        num_workers=1,
    )
    try:
        geno = _build_genotypes(module, layer_sizes, batch=1)
        with pytest.raises(ValueError, match="nvec"):
            fn(geno, jax.random.PRNGKey(0))
    finally:
        close_scoring_function(fn)
