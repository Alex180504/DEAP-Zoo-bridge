"""Numpy-boundary scoring function builder for QDax (Goal 2).

Lets QDax drive a **PettingZoo** env through the slow per-step
numpy<->JAX boundary. Positioned as a validation / debugging tool: it
trades the ~100x speed advantage of JAX-native rollouts for the
ability to reuse any registered PettingZoo env (MPE2, aquarium,
pursuit-evasion, OpenSpiel...).

Contract mirrors :func:`make_jax_scoring_function` so QDax trainers
can dispatch between the two implementations on
``env_spec.api == "jax_parallel"``.

Because this scoring function runs a Python loop per individual
inside a ``jax.jit`` boundary, it is wrapped with
:func:`jax.pure_callback` via a :func:`jax.experimental.host_callback`-
style escape hatch: the outer QDax loop still receives JAX arrays but
the per-individual rollout is plain numpy.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence


def _extract_positions_velocities_np(obs_dict, agents):
    """Numpy equivalent of the JAX extractor in :mod:`scoring_jax`."""
    import numpy as np

    vels = np.stack([np.asarray(obs_dict[a])[0:2] for a in agents], axis=0)
    poses = np.stack([np.asarray(obs_dict[a])[2:4] for a in agents], axis=0)
    return poses, vels


def make_numpy_scoring_function(
    env_name: str,
    module,
    layer_sizes: Sequence[int],
    episode_length: int,
    descriptor_fn: Callable,
    *,
    env_config: Optional[dict] = None,
    action_type: str = "discrete",
    extract_state_fn: Optional[Callable] = None,
) -> Callable:
    """Build a QDax-shaped scoring function backed by a PettingZoo env.

    Args:
        env_name: Registered env name. The env is constructed once
            per scoring call via
            :class:`~deap_pettingzoo.environments.PettingZooWrapper`.
        module: Flax module returned by :func:`make_flax_mlp`
            (used purely for forward passes \u2014 no JAX rollout).
        layer_sizes: MLP topology; used to decode flat genotypes.
        episode_length: Max steps per rollout.
        descriptor_fn: Pure JAX descriptor function. The numpy
            trajectory is converted to :class:`jnp.ndarray`\u200bs before
            calling.
        env_config: Optional env kwargs.
        action_type: ``"discrete"`` (argmax) or ``"continuous"`` (tanh).
        extract_state_fn: Optional observation \u2192 (positions, velocities)
            extractor.

    Returns:
        A callable ``scoring_function(batched_params, key) ->
        (fitness, descriptor, extra_scores)``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from deap_pettingzoo.environments import PettingZooWrapper
    from deap_pettingzoo.contrib.qdax.policy import pytree_to_flat

    extract = extract_state_fn or _extract_positions_velocities_np

    def _rollout_single(flat_params, seed: int):
        from deap_pettingzoo.contrib.qdax.policy import flat_to_pytree

        params = flat_to_pytree(flat_params, layer_sizes)
        env = PettingZooWrapper(env_name, env_config=env_config or {})
        obs = env.reset(seed=int(seed))
        agents = list(env.agents)
        pos0, vel0 = extract(obs, agents)
        pos_hist: List[np.ndarray] = [pos0]
        vel_hist: List[np.ndarray] = [vel0]
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < episode_length and env.agents:
            actions = {}
            for a in env.agents:
                raw = np.asarray(
                    module.apply({"params": params}, jnp.asarray(obs[a]))
                )
                if action_type == "discrete":
                    actions[a] = int(np.argmax(raw))
                else:
                    actions[a] = np.tanh(raw)
            obs, rewards, terms, truncs, _infos = env.step(actions)
            total_reward += float(sum(rewards.values()))
            if env.agents:
                p, v = extract(obs, list(env.agents))
                # Pad if some agents dropped to keep fixed shape.
                if p.shape[0] < len(agents):
                    pad = len(agents) - p.shape[0]
                    p = np.concatenate(
                        [p, np.zeros((pad, p.shape[1]))], axis=0
                    )
                    v = np.concatenate(
                        [v, np.zeros((pad, v.shape[1]))], axis=0
                    )
                pos_hist.append(p)
                vel_hist.append(v)
            done = all(terms.values()) or all(truncs.values())
            steps += 1
        # Pad trajectory to length episode_length + 1 (matches JAX rollout).
        while len(pos_hist) < episode_length + 1:
            pos_hist.append(pos_hist[-1])
            vel_hist.append(vel_hist[-1])
        trajectory = {
            "positions": jnp.asarray(np.stack(pos_hist[: episode_length + 1])),
            "velocities": jnp.asarray(np.stack(vel_hist[: episode_length + 1])),
        }
        descriptor = np.asarray(descriptor_fn(trajectory))
        return float(total_reward), descriptor

    def scoring_function(genotypes, key):
        # ``genotypes`` is a batched Flax pytree. Flatten each entry and
        # run a Python loop per individual.
        leaves = jax.tree_util.tree_leaves(genotypes)
        batch_size = int(leaves[0].shape[0])
        seeds = np.asarray(jax.random.randint(key, (batch_size,), 0, 2**30))
        # Convert each individual's pytree to flat weights (deterministic order).
        flats = []
        for i in range(batch_size):
            single_tree = jax.tree_util.tree_map(lambda x, i=i: x[i], genotypes)
            flats.append(pytree_to_flat(single_tree, layer_sizes))
        fitnesses = []
        descriptors = []
        for flat, seed in zip(flats, seeds):
            f, d = _rollout_single(flat, int(seed))
            fitnesses.append(f)
            descriptors.append(d)
        return (
            jnp.asarray(fitnesses, dtype=jnp.float32),
            jnp.asarray(np.stack(descriptors), dtype=jnp.float32),
            {},
        )

    return scoring_function


__all__ = ["make_numpy_scoring_function"]
