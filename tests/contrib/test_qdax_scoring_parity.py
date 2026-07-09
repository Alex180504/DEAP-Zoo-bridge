"""Regression tests for the QDax numpy/JAX scoring boundary.

Covers:

* The JAX scoring path must freeze ``positions``/``velocities``
  at their last pre-``done`` value once an episode ends, mirroring the numpy
  path's "pad with the last live frame" behaviour
  (:func:`~deap_pettingzoo.contrib.qdax.scoring_numpy._rollout_once`).
* The ``state_extractor`` (the ``(obs, agents) -> (positions,
  velocities)`` contract used by both scoring builders) must be validated on
  first use, rejecting the incompatible ``position_extractor`` contract
  (a single ``(N, 2)`` array) instead of silently mis-splitting rows or
  raising an opaque unpacking error.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
jnp = jax.numpy


class _FakeQDEnv:
    """Deterministic JaxMARL-shaped env: obs/reward depend only on step count.

    ``done`` fires once ``step >= max_steps``; the env keeps incrementing
    (no JaxMARL-style auto-reset) so a scoring path that fails to mask
    post-done frames would observe ever-growing values there.
    """

    agents = ["agent_0", "agent_1"]

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, max_steps: int = 2):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.max_steps = max_steps

    def action_space(self, agent):
        import gymnasium.spaces as spaces

        return spaces.Discrete(self.n_actions)

    def reset(self, key):
        obs = {a: jnp.zeros((self.obs_dim,), dtype=jnp.float32) for a in self.agents}
        state = {"step": jnp.asarray(0)}
        return obs, state

    def step(self, key, state, actions):
        step = state["step"] + 1
        val = step.astype(jnp.float32)
        obs = {a: jnp.full((self.obs_dim,), val, dtype=jnp.float32) for a in self.agents}
        rewards = {a: jnp.asarray(1.0) for a in self.agents}
        done_flag = step >= self.max_steps
        dones = {a: done_flag for a in self.agents}
        dones["__all__"] = done_flag
        infos = {a: {} for a in self.agents}
        return obs, {"step": step}, rewards, dones, infos


def _agent0_positions_descriptor(trajectory):
    """Return the full positions trajectory of agent 0's x-coordinate."""
    return trajectory["positions"][:, 0, 0]


def _build_scoring_function(**kwargs):
    from deap_pettingzoo.contrib.qdax.policy import (
        init_flax_mlp_params,
        make_flax_mlp,
    )
    from deap_pettingzoo.contrib.qdax.scoring_jax import make_jax_scoring_function

    env = _FakeQDEnv(max_steps=2)
    module = make_flax_mlp((env.obs_dim, env.n_actions), activation="tanh")
    init_params = init_flax_mlp_params(module, input_size=env.obs_dim, seed=0)
    genotypes = jax.tree_util.tree_map(lambda x: jnp.stack([x]), init_params)

    scoring_fn = make_jax_scoring_function(
        env,
        module,
        episode_length=5,
        descriptor_fn=_agent0_positions_descriptor,
        action_type="discrete",
        **kwargs,
    )
    return scoring_fn, genotypes


class TestPostDoneMasking:
    def test_jax_freezes_positions_after_done(self):
        """F-2.12-01: frames after ``done`` must equal the last live frame."""
        scoring_fn, genotypes = _build_scoring_function()
        _fitnesses, descriptors, _extra = scoring_fn(
            genotypes, jax.random.PRNGKey(0)
        )
        trajectory = np.asarray(descriptors[0])
        # pos0 (reset) = 0; step 1 = 1 (still alive); step 2 = 2 (the
        # terminal step, still recorded per numpy's "terminal step counts"
        # semantics); steps 3-5 frozen at the step-2 value.
        np.testing.assert_allclose(trajectory, [0.0, 1.0, 2.0, 2.0, 2.0, 2.0])

    def test_unmasked_would_keep_growing(self):
        """Sanity check that the fake env *would* diverge without masking
        (proves the test actually exercises the masking branch)."""
        env = _FakeQDEnv(max_steps=2)
        state = {"step": jnp.asarray(0)}
        for _ in range(5):
            _, state, _, _, _ = env.step(None, state, {})
        # Without masking the raw obs value at the final tick is 5, not 2.
        assert int(state["step"]) == 5


class TestStateExtractorValidation:
    def test_jax_rejects_position_extractor_contract(self):
        """A single-array position_extractor must fail loudly, not silently
        mis-split rows into (positions, velocities)."""

        def _bad_extractor(obs_dict, agents):
            # position_extractor-style: a single (N, 2) array, not a 2-tuple.
            return jnp.stack([obs_dict[a][:2] for a in agents], axis=0)

        scoring_fn, genotypes = _build_scoring_function(state_extractor=_bad_extractor)
        with pytest.raises(ValueError, match="state_extractor"):
            scoring_fn(genotypes, jax.random.PRNGKey(0))

    def test_numpy_rejects_position_extractor_contract(self):
        from deap_pettingzoo.contrib.qdax.scoring_numpy import _RolloutSpec, _rollout_once

        def _bad_extractor(obs_dict, agents):
            return np.stack([np.asarray(obs_dict[a])[:2] for a in agents], axis=0)

        def _descriptor_fn(trajectory):
            return np.asarray([0.0, 0.0])

        _descriptor_fn.min_bd = (0.0, 0.0)
        _descriptor_fn.max_bd = (1.0, 1.0)
        _descriptor_fn.output_shape = 2

        spec = _RolloutSpec(
            env_name="simple_spread_v3",
            env_config={"max_cycles": 10, "continuous_actions": False},
            layer_sizes=(18, 8, 5),
            activation="tanh",
            episode_length=10,
            descriptor_fn=_descriptor_fn,
            action_type="discrete",
            state_extractor=_bad_extractor,
        )
        flat_params = np.zeros(
            18 * 8 + 8 + 8 * 5 + 5, dtype=np.float32
        )
        with pytest.raises(ValueError, match="state_extractor"):
            _rollout_once(spec, flat_params, seed=0)


def _good_state_extractor(obs_dict, agents):
    poses = np.stack([np.asarray(obs_dict[a])[2:4] for a in agents], axis=0)
    vels = np.stack([np.asarray(obs_dict[a])[0:2] for a in agents], axis=0)
    return poses, vels


def _bad_position_extractor(obs_dict, agents):
    return np.stack([np.asarray(obs_dict[a])[:2] for a in agents], axis=0)


class TestCacheHitRevalidation:
    def test_swapped_extractor_on_cache_hit_is_revalidated(self):
        """Regression test: the worker-context cache key is
        ``type(state_extractor).__qualname__``, which is ``"function"`` for
        *any* plain function — it cannot tell two different module-level
        extractors apart. A cache "hit" that swaps in a different
        ``state_extractor`` must reset ``_checked_extractor`` rather than
        trusting a stale ``True`` left over from the previous (different)
        extractor.
        """
        from deap_pettingzoo.contrib.qdax.scoring_numpy import (
            _NUMPY_WORKER_CACHE,
            _RolloutSpec,
            _call_state_extractor,
            _ensure_worker_context,
        )

        def _descriptor_fn(trajectory):
            return np.asarray([0.0, 0.0])

        _descriptor_fn.min_bd = (0.0, 0.0)
        _descriptor_fn.max_bd = (1.0, 1.0)
        _descriptor_fn.output_shape = 2

        common = dict(
            env_name="simple_spread_v3",
            env_config={"max_cycles": 10, "continuous_actions": False},
            layer_sizes=(18, 8, 5),
            activation="tanh",
            episode_length=10,
            descriptor_fn=_descriptor_fn,
            action_type="discrete",
        )
        obs = {"agent_0": np.arange(4, dtype=np.float32)}

        try:
            spec_good = _RolloutSpec(**common, state_extractor=_good_state_extractor)
            ctx = _ensure_worker_context(spec_good)
            _call_state_extractor(ctx["extract"], obs, ["agent_0"], ctx)
            assert ctx["_checked_extractor"] is True
            
            spec_bad = _RolloutSpec(
                **common, state_extractor=_bad_position_extractor
            )
            ctx2 = _ensure_worker_context(spec_bad)
            assert ctx2["extract"] is _bad_position_extractor
            assert ctx2["_checked_extractor"] is False
            with pytest.raises(ValueError, match="state_extractor"):
                _call_state_extractor(ctx2["extract"], obs, ["agent_0"], ctx2)
        finally:
            _NUMPY_WORKER_CACHE.clear()
