"""Focused tests for MAP-Elites init sampling in the QDax trainer."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Discrete

pytest.importorskip("qdax")
pytest.importorskip("flax")
pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from deap_pettingzoo.contrib.qdax.policy import (  # noqa: E402
    init_flax_mlp_params,
    make_flax_mlp,
)
from deap_pettingzoo.contrib.qdax.trainer import (  # noqa: E402
    _build_initial_genotypes,
    _extract_champion,
)


def _make_init_params():
    module = make_flax_mlp((3, 4, 2), activation="relu")
    return init_flax_mlp_params(module, input_size=3, seed=7)


def test_gaussian_init_with_zero_sigma_returns_zero_centered_batch():
    init_params = _make_init_params()
    genotypes = _build_initial_genotypes(
        init_params,
        key=jax.random.PRNGKey(0),
        init_batch_size=5,
        init_mode="gaussian",
        init_sigma=0.0,
    )

    leaves = jax.tree_util.tree_leaves(genotypes)
    assert leaves
    for leaf in leaves:
        assert leaf.shape[0] == 5
        np.testing.assert_allclose(np.asarray(leaf), 0.0)


def test_flax_perturb_with_zero_sigma_repeats_base_params():
    init_params = _make_init_params()
    genotypes = _build_initial_genotypes(
        init_params,
        key=jax.random.PRNGKey(1),
        init_batch_size=3,
        init_mode="flax_perturb",
        init_sigma=0.0,
    )

    for sampled, base in zip(
        jax.tree_util.tree_leaves(genotypes),
        jax.tree_util.tree_leaves(init_params),
    ):
        expected = np.repeat(np.asarray(base)[None, ...], 3, axis=0)
        np.testing.assert_allclose(np.asarray(sampled), expected)


def test_extract_champion_attaches_action_head_for_discrete_envs():
    from types import SimpleNamespace

    module = make_flax_mlp((3, 4, 2), activation="relu")
    params0 = init_flax_mlp_params(module, input_size=3, seed=3)
    params1 = init_flax_mlp_params(module, input_size=3, seed=4)
    repertoire = SimpleNamespace(
        fitnesses=np.asarray([-np.inf, 1.0], dtype=np.float32),
        genotypes=jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), params0, params1
        ),
    )

    policy, fitness = _extract_champion(
        repertoire,
        module,
        action_space=Discrete(2),
    )

    assert fitness == pytest.approx(1.0)
    action = policy.get_action(np.zeros(3, dtype=np.float32))
    assert isinstance(action, (int, np.integer))