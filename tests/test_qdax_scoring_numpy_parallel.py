"""Parallel-evaluation tests for the QDax numpy scoring path.

These tests do **not** require running an actual MAP-Elites loop;
they exercise :func:`make_numpy_scoring_function` directly to verify:

* The serial path returns finite fitness/descriptor arrays.
* The process-pool path returns identical results to the serial path
  for the same inputs (same key → same per-rollout seeds).
* The returned callable exposes ``close()``, ``_num_workers``, and
  ``_parallel_backend`` for pool lifecycle management.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("qdax")
pytest.importorskip("flax")
pytest.importorskip("jax")
pytest.importorskip("pettingzoo.mpe")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from examples.qdax import descriptors as pe
from deap_pettingzoo.contrib.qdax.policy import (
    init_flax_mlp_params,
    make_flax_mlp,
)
from deap_pettingzoo.contrib.qdax.scoring_numpy import (
    close_scoring_function,
    make_numpy_scoring_function,
)


_ENV = "simple_spread_v3"


def _build_genotypes(module, layer_sizes, batch: int):
    inits = []
    for k in range(batch):
        params = init_flax_mlp_params(module, input_size=layer_sizes[0], seed=k)
        inits.append(params)

    # Stack along leading axis to form a batched pytree.
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *inits)
    return stacked


@pytest.fixture(scope="module")
def _scoring_setup():
    layer_sizes = (18, 8, 5)  # simple_spread: obs=18, 5 actions
    module = make_flax_mlp(layer_sizes, activation="tanh")
    # 3 cooperative pursuers, no evaders; use 2 + 1 split just so the
    # descriptor has well-defined geometry.
    descriptor_fn = pe.formation_spread_and_mean_distance(
        num_pursuers=2, num_evaders=1
    )
    return layer_sizes, module, descriptor_fn


def _make_fn(setup, *, num_workers: int, eval_episodes: int = 1):
    layer_sizes, module, descriptor_fn = setup
    return make_numpy_scoring_function(
        _ENV,
        layer_sizes=layer_sizes,
        episode_length=10,
        descriptor_fn=descriptor_fn,
        action_type="discrete",
        activation="tanh",
        num_workers=num_workers,
        eval_episodes=eval_episodes,
    )


class TestNumpyScoringSerial:
    def test_returns_finite_fitness_and_descriptor(self, _scoring_setup):
        layer_sizes, module, _ = _scoring_setup
        fn = _make_fn(_scoring_setup, num_workers=1)
        try:
            assert fn._num_workers == 1
            assert fn._parallel_backend == "process"
            geno = _build_genotypes(module, layer_sizes, batch=2)
            key = jax.random.PRNGKey(0)
            fit, desc, extra = fn(geno, key)
            assert fit.shape == (2,)
            assert desc.shape[0] == 2
            assert np.all(np.isfinite(np.asarray(fit)))
            assert isinstance(extra, dict)
        finally:
            close_scoring_function(fn)


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs at least 2 CPUs"
)
class TestNumpyScoringParallel:
    def test_serial_and_parallel_match(self, _scoring_setup):
        layer_sizes, module, _ = _scoring_setup
        geno = _build_genotypes(module, layer_sizes, batch=4)
        key = jax.random.PRNGKey(123)

        serial_fn = _make_fn(_scoring_setup, num_workers=1, eval_episodes=2)
        try:
            assert serial_fn._num_eval_episodes == 2
            sf, sd, _ = serial_fn(geno, key)
            sf = np.asarray(sf)
            sd = np.asarray(sd)
        finally:
            close_scoring_function(serial_fn)

        parallel_fn = _make_fn(_scoring_setup, num_workers=2, eval_episodes=2)
        try:
            assert parallel_fn._num_workers == 2
            assert parallel_fn._num_eval_episodes == 2
            pf, pd, _ = parallel_fn(geno, key)
            pf = np.asarray(pf)
            pd = np.asarray(pd)
        finally:
            close_scoring_function(parallel_fn)

        # Same key → same seeds → identical fitness/descriptor.
        np.testing.assert_allclose(sf, pf, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sd, pd, rtol=1e-5, atol=1e-5)

    def test_close_is_idempotent(self, _scoring_setup):
        fn = _make_fn(_scoring_setup, num_workers=2)
        # Force pool spin-up.
        layer_sizes, module, _ = _scoring_setup
        geno = _build_genotypes(module, layer_sizes, batch=2)
        fn(geno, jax.random.PRNGKey(0))

        close_scoring_function(fn)
        # Second close must not raise.
        close_scoring_function(fn)


class TestNumpyScoringEnvStepsConsumed:
    """The numpy scoring path tracks real consumed env ticks"""

    def test_counter_starts_zero_and_accumulates(self, _scoring_setup):
        layer_sizes, module, _ = _scoring_setup
        fn = _make_fn(_scoring_setup, num_workers=1)
        try:
            assert fn._env_steps_consumed == 0
            geno = _build_genotypes(module, layer_sizes, batch=3)
            fn(geno, jax.random.PRNGKey(0))
            after_one = fn._env_steps_consumed
            # episode_length=10, batch=3 -> at most 30 ticks, at least 1.
            assert 0 < after_one <= 3 * 10
            fn(geno, jax.random.PRNGKey(1))
            # Cumulative across scoring calls.
            assert fn._env_steps_consumed > after_one
        finally:
            close_scoring_function(fn)
