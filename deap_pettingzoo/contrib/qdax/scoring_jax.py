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


def _validate_state_extractor_output(raw: Any, num_agents: int) -> None:
    """Validate a ``state_extractor`` return value on its first (trace) call.

    A ``position_extractor``
    (the :mod:`deap_pettingzoo.contrib.qdax.position_extractors` contract) returns
    a single ``(N, 2)`` array, not a ``(positions, velocities)`` 2-tuple.
    Passing one where a ``state_extractor`` is expected either raises an
    unpacking error or silently mis-splits rows into "positions" and
    "velocities" — catch it immediately with a clear message instead.
    """
    if not (isinstance(raw, tuple) and len(raw) == 2):
        raise ValueError(
            "state_extractor must return a (positions, velocities) 2-tuple, "
            f"got {type(raw).__name__}. This looks like a "
            "position_extractor (a single (N, 2) array) -- see "
            "deap_pettingzoo.contrib.qdax.position_extractors, which is a "
            "different contract used by QDaxSideState."
        )
    positions, velocities = raw
    for name, arr in (("positions", positions), ("velocities", velocities)):
        shape = getattr(arr, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != num_agents:
            raise ValueError(
                f"state_extractor must return (positions, velocities) each "
                f"shaped (num_agents={num_agents}, D); got {name} with shape "
                f"{shape}."
            )


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
    state_extractor: Optional[Callable] = None,
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
        action_type: ``"discrete"`` (argmax over output logits),
            ``"multidiscrete"`` (split ``sum(nvec)`` logits into one chunk
            per sub-action and argmax each; env must accept MultiDiscrete
            actions), or ``"continuous"`` (tanh-rescale; env must accept Box
            actions).
        agents: Optional override for the agent ordering. Defaults to
            ``jax_env.agents``.
        state_extractor: Optional pure function
            ``(obs_dict, agents) -> (positions, velocities)`` for envs whose
            observation layout differs from the MPE convention. Defaults to
            :func:`_extract_positions_velocities`. This is a **different**
            contract from :mod:`deap_pettingzoo.contrib.qdax.position_extractors`'
            ``position_extractor`` (used by ``QDaxSideState``), which returns
            a single ``(N, 2)`` array rather than a ``(positions,
            velocities)`` pair — mismatch is caught by a first-trace shape check.

    Returns:
        A callable ``scoring_function(batched_params, key) ->
        (fitness, descriptor, extra_scores)`` with no side effects.
    """
    import jax
    import jax.numpy as jnp

    if action_type not in ("discrete", "continuous", "multidiscrete"):
        raise ValueError(
            "action_type must be 'discrete', 'multidiscrete' or 'continuous'; "
            f"got {action_type!r}."
        )
    agent_list = list(agents if agents is not None else jax_env.agents)
    # Agents present in the env but NOT in agent_list need noop actions so
    # jax_env.step() receives a complete action dict.
    all_env_agents = list(jax_env.agents)
    policy_set = set(agent_list)
    noop_agents = [a for a in all_env_agents if a not in policy_set]

    def _get_action_space(agent):
        if hasattr(jax_env, "action_space"):
            return jax_env.action_space(agent)
        action_spaces = getattr(jax_env, "action_spaces", None)
        if action_spaces is not None:
            return action_spaces[agent]
        return None

    def _noop_action_for_agent(agent):
        space = _get_action_space(agent)
        if space is None:
            raise ValueError(
                "make_jax_scoring_function cannot infer a noop action for "
                f"omitted agent {agent!r}; include all env agents in `agents` "
                "or provide an environment exposing action_space metadata."
            )
        if action_type == "continuous":
            shape = getattr(space, "shape", None)
            if shape is None:
                raise ValueError(
                    "Continuous action_type requires action spaces with a "
                    f"defined shape; could not infer noop action for agent {agent!r}."
                )
            dtype = getattr(space, "dtype", jnp.float32)
            if shape == ():
                return jnp.array(0, dtype=dtype)
            return jnp.zeros(shape, dtype=dtype)
        # Discrete-like spaces: preserve shape for MultiDiscrete-style spaces.
        if hasattr(space, "nvec"):
            return jnp.zeros(jnp.shape(space.nvec), dtype=jnp.int32)
        shape = getattr(space, "shape", ())
        if shape in (None, ()):
            return jnp.array(0, dtype=jnp.int32)
        return jnp.zeros(shape, dtype=jnp.int32)

    if noop_agents:
        _noop_action = _noop_action_for_agent(noop_agents[0])
        for _noop_agent in noop_agents[1:]:
            _agent_noop = _noop_action_for_agent(_noop_agent)
            if _agent_noop.shape != _noop_action.shape or _agent_noop.dtype != _noop_action.dtype:
                raise ValueError(
                    "Omitted agents must share a compatible action shape/dtype "
                    f"for noop actions; got {_noop_agent!r} with "
                    f"{_agent_noop.shape}/{_agent_noop.dtype} versus "
                    f"{noop_agents[0]!r} with {_noop_action.shape}/{_noop_action.dtype}. "
                    "Include all env agents in `agents` for heterogeneous action spaces."
                )
    else:
        _noop_action = jnp.array(0, dtype=jnp.int32)

    extract = state_extractor or _extract_positions_velocities

    # For MultiDiscrete, precompute static (start, stop) logit chunk bounds
    # (one per sub-action) from a policy agent's nvec (homogeneous team).
    _md_bounds: Optional[list] = None
    if action_type == "multidiscrete":
        md_space = _get_action_space(agent_list[0])
        if md_space is None or not hasattr(md_space, "nvec"):
            raise ValueError(
                "multidiscrete action_type requires a MultiDiscrete action "
                "space exposing 'nvec'."
            )
        _md_bounds = []
        _off = 0
        for _n in (int(x) for x in md_space.nvec):
            _md_bounds.append((_off, _off + _n))
            _off += _n

    def _action(raw):
        if action_type == "discrete":
            return jnp.argmax(raw).astype(jnp.int32)
        if action_type == "multidiscrete":
            return jnp.stack(
                [jnp.argmax(raw[lo:hi]).astype(jnp.int32) for lo, hi in _md_bounds]
            )
        return jnp.tanh(raw)

    _checked_extractor = [False]

    def _single_rollout(params, key):
        obs, state = jax_env.reset(key)
        raw0 = extract(obs, agent_list)
        if not _checked_extractor[0]:
            _validate_state_extractor_output(raw0, len(agent_list))
            _checked_extractor[0] = True
        pos0, vel0 = raw0

        def _step_fn(carry, _):
            state, obs, key, acc_reward, acc_done, last_pos, last_vel = carry
            key, sub = jax.random.split(key)
            actions = {}
            for a in agent_list:
                raw = module.apply({"params": params}, obs[a])
                actions[a] = _action(raw)
            for a in noop_agents:
                actions[a] = _noop_action
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
            # Freeze pos/vel at their last pre-done value once the episode
            # has ended, mirroring the numpy path's "pad with the last live
            # frame" behaviour (scoring_numpy._rollout_once). Without this,
            # JaxMARL's auto-reset-on-done means post-done frames come from a
            # reset/continuing episode, landing trajectory-averaged
            # descriptors in a different archive cell than the numpy path
            # would compute for the identical rollout.
            alive = ~acc_done
            pos = jnp.where(alive, pos, last_pos)
            vel = jnp.where(alive, vel, last_vel)
            acc_done = acc_done | done_flag
            return (new_state, new_obs, key, acc_reward, acc_done, pos, vel), (pos, vel)

        init_carry = (
            state, obs, key, jnp.asarray(0.0), jnp.asarray(False), pos0, vel0,
        )
        (_, _, _, total_reward, _, _, _), (pos_traj, vel_traj) = jax.lax.scan(
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
