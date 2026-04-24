"""JAX-native scoring function builder for QDax (Goal 1).

Produces a :func:`jit`/:func:`vmap`-able scoring function satisfying the
QDax contract:

.. code-block:: python

    scoring_function(genotypes, key) -> (fitness, descriptor, extra_scores)

where ``genotypes`` is a batched Flax params pytree (leading axis is
the population dimension) and the return values are batched
correspondingly.

Assumptions:

* All agents share the same MLP policy (homogeneous team
  neuroevolution; mirrors :class:`SelfPlayEvaluator`). Heterogeneous
  teams must use :mod:`deap_pettingzoo.evaluators.mixed_policy`.
* The env implements :class:`~deap_pettingzoo.environments.jax_parallel.JaxEnvProtocol`.
* Observations are fixed-shape flat vectors across agents.

The rollout uses :func:`jax.lax.scan` over time and :func:`jax.vmap`
over the population axis, so a single ``scoring_function`` call with
``batch_size=128`` costs one XLA launch on GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def _extract_positions_velocities(obs_dict, agents):
    """Stack ``positions``/``velocities`` from MPE-style observations.

    MPE2 / JaxMARL MPE observations start with
    ``[self_vel(2), self_pos(2), ...]``. This helper extracts those
    first four entries per agent so descriptors depending on position
    can be computed from the observation dict alone (no privileged
    state access).
    """
    import jax.numpy as jnp

    vels = jnp.stack([obs_dict[a][0:2] for a in agents], axis=0)  # (N, 2)
    poses = jnp.stack([obs_dict[a][2:4] for a in agents], axis=0)  # (N, 2)
    return poses, vels


def make_jax_scoring_function(
    jax_env,
    module,
    episode_length: int,
    descriptor_fn: Callable,
    *,
    action_type: str = "discrete",
    agents: Optional[list] = None,
    extract_state_fn: Optional[Callable] = None,
) -> Callable:
    """Build a QDax-shaped scoring function over a JaxMARL-style env.

    Args:
        jax_env: A JaxMARL-style env (see :class:`JaxEnvProtocol`).
        module: Flax module returned by :func:`make_flax_mlp`.
        episode_length: Number of steps to roll out per episode.
        descriptor_fn: Pure function
            ``(trajectory_summary: Dict[str, Array]) -> Array`` with
            ``output_shape`` attribute. See
            :mod:`deap_pettingzoo.contrib.qdax.descriptors`.
        action_type: ``"discrete"`` (argmax over output logits) or
            ``"continuous"`` (tanh-rescale; env must accept Box actions).
        agents: Optional override for the agent ordering. Defaults to
            ``jax_env.agents``.
        extract_state_fn: Optional pure function
            ``(obs_dict, agents) -> (positions, velocities)`` for
            envs whose observation layout differs from the MPE
            convention. Defaults to :func:`_extract_positions_velocities`.

    Returns:
        A callable ``scoring_function(batched_params, key) ->
        (fitness, descriptor, extra_scores)`` with no side effects.
    """
    import jax
    import jax.numpy as jnp

    if action_type not in ("discrete", "continuous"):
        raise ValueError(
            f"action_type must be 'discrete' or 'continuous'; got {action_type!r}."
        )
    agent_list = list(agents if agents is not None else jax_env.agents)
    extract = extract_state_fn or _extract_positions_velocities

    def _action(raw):
        if action_type == "discrete":
            return jnp.argmax(raw).astype(jnp.int32)
        return jnp.tanh(raw)

    def _single_rollout(params, key):
        obs, state = jax_env.reset(key)
        pos0, vel0 = extract(obs, agent_list)

        def _step_fn(carry, _):
            state, obs, key, acc_reward, acc_done = carry
            key, sub = jax.random.split(key)
            actions = {}
            for a in agent_list:
                raw = module.apply({"params": params}, obs[a])
                actions[a] = _action(raw)
            new_obs, new_state, rewards, dones, _infos = jax_env.step(
                sub, state, actions
            )
            step_reward = jnp.sum(
                jnp.stack([rewards[a] for a in agent_list])
            )
            pos, vel = extract(new_obs, agent_list)
            done_flag = jnp.asarray(dones.get("__all__", False)).astype(jnp.bool_)
            alive_mask = (~acc_done).astype(step_reward.dtype)
            acc_reward = acc_reward + alive_mask * step_reward
            acc_done = acc_done | done_flag
            return (new_state, new_obs, key, acc_reward, acc_done), (pos, vel)

        init_carry = (state, obs, key, jnp.asarray(0.0), jnp.asarray(False))
        (_, _, _, total_reward, _), (pos_traj, vel_traj) = jax.lax.scan(
            _step_fn, init_carry, xs=None, length=int(episode_length)
        )
        # Prepend step-0 pos/vel so trajectory length == episode_length + 1.
        positions = jnp.concatenate([pos0[None], pos_traj], axis=0)
        velocities = jnp.concatenate([vel0[None], vel_traj], axis=0)
        trajectory = {
            "positions": positions,
            "velocities": velocities,
        }
        descriptor = descriptor_fn(trajectory)
        return total_reward, descriptor

    def scoring_function(genotypes, key):
        batch_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        keys = jax.random.split(key, batch_size)
        fitnesses, descriptors = jax.vmap(_single_rollout)(genotypes, keys)
        return fitnesses, descriptors, {}

    return scoring_function


__all__ = ["make_jax_scoring_function"]
