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

Process-pool parallelism
------------------------
Passing ``num_workers > 1`` to :func:`make_numpy_scoring_function`
spawns a persistent :class:`concurrent.futures.ProcessPoolExecutor`
on first call and reuses it across QD iterations. Each worker
lazily builds its own ``PettingZooWrapper`` + Flax ``module`` on
first invocation (module-global cache), so only small numpy arrays
and a picklable :class:`_RolloutSpec` cross the process boundary.
The executor is torn down when the caller closes the scoring
function (handled automatically by QDax trainers via ``try / finally``
around the QD loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence


__all__ = ["make_numpy_scoring_function", "close_scoring_function"]


# --------------------------------------------------------------------------
# Worker-side helpers (must live at module top level to be picklable).
# --------------------------------------------------------------------------

@dataclass
class _RolloutSpec:
    """Picklable spec shipped to worker processes.

    Everything a worker needs to reconstruct the rollout context on
    first use. We ship **metadata**, not live JAX/Flax/env objects:

    * ``env_name`` + ``env_config`` — worker rebuilds
      :class:`PettingZooWrapper` on first call.
    * ``layer_sizes`` + ``activation`` — worker rebuilds the Flax MLP
      module on first call.
    * ``descriptor_fn`` — must itself be a module-top-level callable
      (pickleable). The factories in
      :mod:`deap_pettingzoo.contrib.qdax.descriptors.pursuit_evasion`
      return such callables (closures over numeric constants only).
    * ``extract_state_fn`` — same pickle constraint; default is the
      numpy extractor in this module.
    """

    env_name: str
    env_config: Optional[dict]
    layer_sizes: tuple
    activation: str
    episode_length: int
    descriptor_fn: Callable
    action_type: str
    extract_state_fn: Optional[Callable] = None


# Module-global cache populated lazily on each worker. Same pattern
# as ``deap_pettingzoo.integration.deap_integration._WORKER_CACHE`` —
# works uniformly for ProcessPoolExecutor, SCOOP (no init hook), and
# Dask workers.
_NUMPY_WORKER_CACHE: dict = {}


def _extract_positions_velocities_np(obs_dict, agents):
    """Numpy equivalent of the JAX extractor in :mod:`scoring_jax`."""
    import numpy as np

    vels = np.stack([np.asarray(obs_dict[a])[0:2] for a in agents], axis=0)
    poses = np.stack([np.asarray(obs_dict[a])[2:4] for a in agents], axis=0)
    return poses, vels


def _ensure_worker_context(spec: _RolloutSpec):
    """Build & cache per-worker env + module on first call."""
    cached_sig = _NUMPY_WORKER_CACHE.get("sig")
    sig = (
        spec.env_name,
        id(spec.descriptor_fn),  # cheap identity check; specs reuse the same fn
        tuple(spec.layer_sizes),
        spec.activation,
    )
    if cached_sig == sig:
        return _NUMPY_WORKER_CACHE

    from deap_pettingzoo.contrib.qdax.policy import make_flax_mlp
    from deap_pettingzoo.environments import PettingZooWrapper

    module = make_flax_mlp(spec.layer_sizes, activation=spec.activation)
    env = PettingZooWrapper(spec.env_name, env_config=spec.env_config or {})
    extract = spec.extract_state_fn or _extract_positions_velocities_np

    _NUMPY_WORKER_CACHE.clear()
    _NUMPY_WORKER_CACHE.update(
        sig=sig,
        module=module,
        env=env,
        extract=extract,
        descriptor_fn=spec.descriptor_fn,
    )
    return _NUMPY_WORKER_CACHE


def _numpy_rollout(args):
    """Worker entry point: one (flat_params, seed) → (fitness, descriptor).

    Args:
        args: ``(spec, flat_params, seed)`` tuple. ``spec`` is a
            :class:`_RolloutSpec`; ``flat_params`` is a flat numpy
            weight vector; ``seed`` is an int.

    Returns:
        ``(fitness: float, descriptor: np.ndarray)``.
    """
    import jax.numpy as jnp
    import numpy as np

    from deap_pettingzoo.contrib.qdax.policy import flat_to_pytree

    spec, flat_params, seed = args
    ctx = _ensure_worker_context(spec)
    module = ctx["module"]
    env = ctx["env"]
    extract = ctx["extract"]
    descriptor_fn = ctx["descriptor_fn"]

    params = flat_to_pytree(flat_params, spec.layer_sizes)
    obs = env.reset(seed=int(seed))
    agents = list(env.agents)
    pos0, vel0 = extract(obs, agents)
    pos_hist: List[np.ndarray] = [pos0]
    vel_hist: List[np.ndarray] = [vel0]
    total_reward = 0.0
    done = False
    steps = 0
    episode_length = int(spec.episode_length)
    action_type = spec.action_type

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

    while len(pos_hist) < episode_length + 1:
        pos_hist.append(pos_hist[-1])
        vel_hist.append(vel_hist[-1])

    trajectory = {
        "positions": jnp.asarray(np.stack(pos_hist[: episode_length + 1])),
        "velocities": jnp.asarray(np.stack(vel_hist[: episode_length + 1])),
    }
    descriptor = np.asarray(descriptor_fn(trajectory))
    return float(total_reward), descriptor


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------

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
    activation: str = "tanh",
    num_workers: int = 1,
    parallel_backend: str = "process",
) -> Callable:
    """Build a QDax-shaped scoring function backed by a PettingZoo env.

    Args:
        env_name: Registered env name. The env is constructed once
            per scoring call via
            :class:`~deap_pettingzoo.environments.PettingZooWrapper`.
        module: Flax module returned by :func:`make_flax_mlp`
            (used purely for forward passes — no JAX rollout).
        layer_sizes: MLP topology; used to decode flat genotypes.
        episode_length: Max steps per rollout.
        descriptor_fn: Pure JAX descriptor function. The numpy
            trajectory is converted to :class:`jnp.ndarray`\u200bs before
            calling. Must be a **top-level** callable (or a closure
            over picklable state) when ``num_workers > 1``.
        env_config: Optional env kwargs.
        action_type: ``"discrete"`` (argmax) or ``"continuous"`` (tanh).
        extract_state_fn: Optional observation → (positions, velocities)
            extractor. Must be picklable when ``num_workers > 1``.
        num_workers: Number of worker processes for per-individual
            rollouts. ``0`` / ``1`` → serial in-process. ``>1`` spawns
            a persistent :class:`ProcessPoolExecutor` on first call
            and reuses it across QD iterations. Call
            :func:`close_scoring_function` on the returned callable to
            tear down the pool explicitly; trainers do this in a
            ``try / finally``.
        parallel_backend: One of ``"process"``, ``"scoop"``, ``"dask"``.
            See :mod:`deap_pettingzoo.utils.parallel` for details.

    Returns:
        A callable ``scoring_function(batched_params, key) ->
        (fitness, descriptor, extra_scores)``. The callable has two
        extra attributes for pool lifecycle:

        * ``scoring_function.close()`` — close the persistent executor.
        * ``scoring_function._executor`` — lazily-created executor ref
          (None until first call; ``_SerialExecutor`` for serial).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from deap_pettingzoo.contrib.qdax.policy import pytree_to_flat
    from deap_pettingzoo.utils.parallel import (
        make_executor,
        resolve_num_workers,
    )

    effective_workers = resolve_num_workers(num_workers)
    rollout_spec = _RolloutSpec(
        env_name=env_name,
        env_config=dict(env_config) if env_config else None,
        layer_sizes=tuple(int(x) for x in layer_sizes),
        activation=str(activation),
        episode_length=int(episode_length),
        descriptor_fn=descriptor_fn,
        action_type=action_type,
        extract_state_fn=extract_state_fn,
    )

    # Local (in-process) fallback for the serial path — avoids the
    # lazy-worker-cache indirection when we're not crossing processes.
    def _local_rollout(flat_params, seed: int):
        return _numpy_rollout((rollout_spec, flat_params, int(seed)))

    # Mutable closure state so the returned scoring_function can lazily
    # spin up the executor on first call and release it on close().
    state: dict = {"executor": None, "ctx": None}

    def _ensure_executor():
        if state["executor"] is not None:
            return state["executor"]
        ctx = make_executor(
            effective_workers, backend=parallel_backend
        )
        state["ctx"] = ctx
        state["executor"] = ctx.__enter__()
        return state["executor"]

    def scoring_function(genotypes, key):
        leaves = jax.tree_util.tree_leaves(genotypes)
        batch_size = int(leaves[0].shape[0])
        seeds = np.asarray(
            jax.random.randint(key, (batch_size,), 0, 2**30)
        )
        flats = []
        for i in range(batch_size):
            single_tree = jax.tree_util.tree_map(
                lambda x, i=i: x[i], genotypes
            )
            flats.append(pytree_to_flat(single_tree, layer_sizes))

        if effective_workers <= 1:
            # Serial: avoid executor overhead entirely.
            results = [
                _local_rollout(flat, int(s))
                for flat, s in zip(flats, seeds)
            ]
        else:
            executor = _ensure_executor()
            payloads = [
                (rollout_spec, np.asarray(flat), int(s))
                for flat, s in zip(flats, seeds)
            ]
            results = list(executor.map(_numpy_rollout, payloads))

        fitnesses = [r[0] for r in results]
        descriptors = [r[1] for r in results]
        return (
            jnp.asarray(fitnesses, dtype=jnp.float32),
            jnp.asarray(np.stack(descriptors), dtype=jnp.float32),
            {},
        )

    def _close():
        ctx = state["ctx"]
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            finally:
                state["executor"] = None
                state["ctx"] = None

    scoring_function.close = _close  # type: ignore[attr-defined]
    scoring_function._executor = lambda: state["executor"]  # type: ignore[attr-defined]
    scoring_function._num_workers = effective_workers  # type: ignore[attr-defined]
    scoring_function._parallel_backend = parallel_backend  # type: ignore[attr-defined]
    return scoring_function


def close_scoring_function(scoring_function: Callable) -> None:
    """Close the persistent pool of a scoring function, if any."""
    closer = getattr(scoring_function, "close", None)
    if closer is not None:
        closer()
