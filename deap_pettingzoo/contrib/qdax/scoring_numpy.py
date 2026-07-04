"""Numpy-boundary scoring function builder for QDax.

Lets QDax drive a **PettingZoo** env through the slow per-step
numpy<->JAX boundary. Positioned as a validation / debugging tool: it
trades the ~100x speed advantage of JAX-native rollouts for the
ability to reuse any registered PettingZoo env (MPE2, OpenSpiel...).

Contract mirrors :func:`make_jax_scoring_function` so QDax trainers
can dispatch between the two implementations on
``env_spec.api == "jax_parallel"``.

This scoring path performs the per-individual rollout in a plain
Python loop using numpy/PettingZoo operations, then converts the
collected results back to JAX arrays for the caller. It is therefore
intended for validation/debugging and general environment reuse, not
as a ``jax.jit``-compatible or ``jax.pure_callback``-wrapped execution
path.

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

import importlib
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
      (pickleable), i.e. a top-level class instance rather than a
      lambda/closure, so it survives cross-process transport.
        * ``state_extractor`` — same pickle constraint; default is the
            numpy extractor in this module. Distinct from the
            ``position_extractor`` contract in
            :mod:`deap_pettingzoo.contrib.qdax.position_extractors` (used by
            ``QDaxSideState``) — this one returns a ``(positions,
            velocities)`` 2-tuple, not a single ``(N, 2)`` array.
        * ``worker_imports`` — optional module names imported in the worker
            before env construction so import-time ``register_env(...)`` side
            effects are replayed under ``spawn``.
    """

    env_name: str
    env_config: Optional[dict]
    layer_sizes: tuple
    activation: str
    episode_length: int
    descriptor_fn: Callable
    action_type: str
    eval_episodes: int = 1
    state_extractor: Optional[Callable] = None
    worker_imports: tuple[str, ...] = field(default_factory=tuple)
    #: Per-sub-action choice counts for ``action_type == "multidiscrete"``.
    nvec: Optional[tuple] = None
    #: Registry snapshot replayed in the worker so Dask (and the
    #: initializer-less process path here) can resolve runtime-registered envs.
    registry_snapshot: Optional[list] = None


# Module-global cache populated lazily on each worker. Same pattern
# as ``deap_pettingzoo.integration.deap_integration._WORKER_CACHE`` —
# works uniformly for ProcessPoolExecutor and Dask workers (no init hook).
_NUMPY_WORKER_CACHE: dict = {}


def _extract_positions_velocities_np(obs_dict, agents):
    """Numpy equivalent of the JAX extractor in :mod:`scoring_jax`."""
    import numpy as np

    vels = np.stack([np.asarray(obs_dict[a])[0:2] for a in agents], axis=0)
    poses = np.stack([np.asarray(obs_dict[a])[2:4] for a in agents], axis=0)
    return poses, vels


def _import_worker_modules(module_names: Sequence[str]) -> None:
    """Import helper modules that replay env-registration side effects."""
    for module_name in module_names:
        importlib.import_module(str(module_name))


def _ensure_worker_context(spec: _RolloutSpec):
    """Build & cache per-worker env + module on first call.

    The cache is keyed on stable structural fields that fully identify
    the rollout context: ``env_name``, ``env_config``, ``layer_sizes``,
    ``activation``, ``action_type``, and the *type* of the
    ``descriptor_fn`` / ``state_extractor`` callables.
    The object ``id()`` id not used for callable fields because pickling/unpickling
    across the process boundary produces a new object with a different
    ``id`` on every task — which would invalidate the cache every call.
    Instead refresh the callable handles in-place when the signature
    matches, so the cached env + module are reused across tasks while
    the callable references are kept up-to-date.
    """
    cached_sig = _NUMPY_WORKER_CACHE.get("sig")
    _env_cfg = spec.env_config or {}
    _env_cfg_frozen = tuple(sorted(_env_cfg.items()))
    _extract_key = (
        type(spec.state_extractor).__qualname__
        if spec.state_extractor is not None
        else None
    )
    _worker_imports = tuple(str(name) for name in (spec.worker_imports or ()))
    sig = (
        spec.env_name,
        _env_cfg_frozen,
        type(spec.descriptor_fn).__qualname__,
        tuple(spec.layer_sizes),
        spec.activation,
        spec.action_type,
        spec.nvec,
        _extract_key,
        _worker_imports,
    )
    if cached_sig == sig:
        # Refresh callable handles so the cached ones match the
        # freshly-unpickled spec on each task.
        _NUMPY_WORKER_CACHE["descriptor_fn"] = spec.descriptor_fn
        if spec.state_extractor is not None:
            _NUMPY_WORKER_CACHE["extract"] = spec.state_extractor
            # `_extract_key` above is `type(...).__qualname__`, which is
            # "function" for ANY plain function — it cannot tell two
            # different module-level extractors apart. A persistent worker
            # pool (dask) reused across scoring-function instances
            # could therefore swap in a genuinely different extractor on a
            # cache "hit"; re-validate it rather than trusting a stale
            # `_checked_extractor=True` left over from the previous one.
            _NUMPY_WORKER_CACHE["_checked_extractor"] = False
        return _NUMPY_WORKER_CACHE

    from deap_pettingzoo.contrib.qdax.policy import make_flax_mlp
    from deap_pettingzoo.environments import PettingZooWrapper
    from deap_pettingzoo.environments.registry import (
        replay_registry_snapshot_once,
    )

    _import_worker_modules(_worker_imports)
    # Re-register runtime-registered custom envs (no initializer hook here).
    replay_registry_snapshot_once(spec.registry_snapshot)
    module = make_flax_mlp(spec.layer_sizes, activation=spec.activation)
    env = PettingZooWrapper(spec.env_name, env_config=spec.env_config or {})
    extract = spec.state_extractor or _extract_positions_velocities_np

    _NUMPY_WORKER_CACHE.clear()
    _NUMPY_WORKER_CACHE.update(
        sig=sig,
        module=module,
        env=env,
        extract=extract,
        _checked_extractor=False,
        descriptor_fn=spec.descriptor_fn,
    )
    return _NUMPY_WORKER_CACHE


def _call_state_extractor(extract, obs, agents, ctx: dict):
    """Call ``extract`` and validate its output shape on the first call.

    A ``position_extractor`` (the :mod:`deap_pettingzoo.contrib.qdax.position_extractors`
    contract) returns a single ``(N, 2)`` array, not a ``(positions, velocities)``
    2-tuple. Passing one where a ``state_extractor`` is expected either
    raises an opaque unpacking error or silently mis-splits rows into
    "positions" and "velocities" — catch it immediately instead. Checked
    once per worker context (not every call) to keep the hot rollout loop
    cheap.

    Extractors that expose ``needs_env = True`` are called as
    ``extract(obs, agents, env)`` with the live worker env so they can read
    true entity state (e.g. absolute positions the observation vectors do
    not encode).  Such extractors may return more rows than ``len(agents)``
    — scripted opponents hidden from the public agent list are a legitimate
    part of two-team descriptors — so only the 2-D shape is enforced.
    """
    needs_env = bool(getattr(extract, "needs_env", False))
    raw = extract(obs, agents, ctx.get("env")) if needs_env else extract(obs, agents)
    if ctx.get("_checked_extractor", False):
        return raw
    if not (isinstance(raw, tuple) and len(raw) == 2):
        raise ValueError(
            "state_extractor must return a (positions, velocities) 2-tuple, "
            f"got {type(raw).__name__}. This looks like a position_extractor "
            "(a single (N, 2) array) — see "
            "deap_pettingzoo.contrib.qdax.position_extractors, which is a "
            "different contract used by QDaxSideState."
        )
    positions, velocities = raw
    import numpy as np

    num_agents = len(agents)
    for name, arr in (("positions", positions), ("velocities", velocities)):
        shape = np.asarray(arr).shape
        row_count_ok = shape and (shape[0] == num_agents or needs_env)
        if len(shape) != 2 or not row_count_ok:
            raise ValueError(
                f"state_extractor must return (positions, velocities) each "
                f"shaped (num_agents={num_agents}, D); got {name} with shape "
                f"{shape}."
            )
    ctx["_checked_extractor"] = True
    return raw


def _pad_agent_state(positions, velocities, agent_count: int):
    """Pad per-agent state arrays so descriptor inputs keep a fixed shape."""
    import numpy as np

    if positions.shape[0] >= agent_count:
        return positions, velocities

    pad = agent_count - positions.shape[0]
    return (
        np.concatenate(
            [positions, np.zeros((pad, positions.shape[1]), dtype=positions.dtype)],
            axis=0,
        ),
        np.concatenate(
            [
                velocities,
                np.zeros((pad, velocities.shape[1]), dtype=velocities.dtype),
            ],
            axis=0,
        ),
    )


def _rollout_once(
    spec: _RolloutSpec,
    flat_params,
    seed: int,
    *,
    ctx: Optional[dict] = None,
):
    """Run one rollout episode and return ``(fitness, descriptor, steps)``.

    ``steps`` is the real number of env ticks this episode consumed (it
    truncates below ``episode_length`` whenever the env terminates early).

    ``ctx`` is the lazily-built worker context returned by
    :func:`_ensure_worker_context`; passing it lets a caller average multiple
    episodes without rebuilding the env/module cache.
    """
    import jax.numpy as jnp
    import numpy as np

    from deap_pettingzoo.contrib.qdax.policy import flat_to_pytree

    if ctx is None:
        ctx = _ensure_worker_context(spec)
    module = ctx["module"]
    env = ctx["env"]
    extract = ctx["extract"]
    descriptor_fn = ctx["descriptor_fn"]

    params = flat_to_pytree(flat_params, spec.layer_sizes)
    obs = env.reset(seed=int(seed))
    agents = list(env.agents)
    pos0, vel0 = _call_state_extractor(extract, obs, agents, ctx)
    pos_hist: List[np.ndarray] = [pos0]
    vel_hist: List[np.ndarray] = [vel0]
    total_reward = 0.0
    done = False
    steps = 0
    episode_length = int(spec.episode_length)
    action_type = spec.action_type

    # Precompute static (start, stop) logit chunk bounds for MultiDiscrete.
    md_bounds = None
    if action_type == "multidiscrete":
        if not spec.nvec:
            raise ValueError("multidiscrete action_type requires spec.nvec to be set.")
        md_bounds = []
        _off = 0
        for _n in (int(x) for x in spec.nvec):
            md_bounds.append((_off, _off + _n))
            _off += _n

    while not done and steps < episode_length and env.agents:
        actions = {}
        for a in env.agents:
            raw = np.asarray(module.apply({"params": params}, jnp.asarray(obs[a])))
            if action_type == "discrete":
                actions[a] = int(np.argmax(raw))
            elif action_type == "multidiscrete":
                actions[a] = np.array(
                    [int(np.argmax(raw[lo:hi])) for lo, hi in md_bounds],
                    dtype=np.int64,
                )
            else:
                actions[a] = np.tanh(raw)
        obs, rewards, terms, truncs, _infos = env.step(actions)
        total_reward += float(sum(rewards.values()))
        if env.agents:
            p, v = _pad_agent_state(
                *_call_state_extractor(extract, obs, list(env.agents), ctx),
                len(agents),
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
    return float(total_reward), descriptor, int(steps)


def _numpy_rollout(args):
    """Worker entry point: one genotype → averaged fitness/descriptor.

    Args:
        args: ``(spec, flat_params, seed)`` tuple. ``spec`` is a
            :class:`_RolloutSpec`; ``flat_params`` is a flat numpy
            weight vector; ``seed`` is an int.

    Returns:
        ``(fitness: float, descriptor: np.ndarray, steps: int)`` —
        fitness/descriptor averaged over ``spec.eval_episodes`` rollouts and
        the total real env ticks summed across those rollouts.
    """
    import numpy as np

    spec, flat_params, seed = args
    ctx = _ensure_worker_context(spec)
    eval_episodes = max(1, int(spec.eval_episodes))
    fitnesses = []
    descriptors = []
    total_steps = 0
    base_seed = int(seed)

    for episode_idx in range(eval_episodes):
        fitness, descriptor, steps = _rollout_once(
            spec,
            flat_params,
            base_seed + episode_idx,
            ctx=ctx,
        )
        fitnesses.append(float(fitness))
        descriptors.append(np.asarray(descriptor, dtype=np.float32))
        total_steps += int(steps)

    return (
        float(np.mean(np.asarray(fitnesses, dtype=np.float64))),
        np.mean(np.stack(descriptors, axis=0), axis=0),
        total_steps,
    )


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------


def make_numpy_scoring_function(
    env_name: str,
    layer_sizes: Sequence[int],
    episode_length: int,
    descriptor_fn: Callable,
    *,
    env_config: Optional[dict] = None,
    action_type: str = "discrete",
    nvec: Optional[Sequence[int]] = None,
    state_extractor: Optional[Callable] = None,
    activation: str = "tanh",
    num_workers: int = 1,
    parallel_backend: str = "process",
    scheduler_address: Optional[str] = None,
    eval_episodes: int = 1,
    worker_imports: Sequence[str] = (),
) -> Callable:
    """Build a QDax-shaped scoring function backed by a PettingZoo env.

    Args:
        env_name: Registered env name. The env is constructed once
            per scoring call via
            :class:`~deap_pettingzoo.environments.PettingZooWrapper`.
        layer_sizes: MLP topology; the worker rebuilds the Flax module
            from this + ``activation`` on first call. Used to decode
            flat genotypes.
        episode_length: Max steps per rollout.
        descriptor_fn: Pure JAX descriptor function. The numpy
            trajectory is converted to :class:`jnp.ndarray`\u200bs before
            calling. Must be a **top-level** callable (or a closure
            over picklable state) when ``num_workers > 1``.
        env_config: Optional env kwargs.
        action_type: ``"discrete"`` (argmax), ``"multidiscrete"`` (per
            sub-action argmax over ``sum(nvec)`` logits), or
            ``"continuous"`` (tanh).
        nvec: Per-sub-action choice counts; required when
            ``action_type == "multidiscrete"``.
        state_extractor: Optional observation -> (positions, velocities)
            extractor. Must be picklable when ``num_workers > 1``. Distinct
            from the ``position_extractor`` contract in
            :mod:`deap_pettingzoo.contrib.qdax.position_extractors` (used by
            ``QDaxSideState``).
        eval_episodes: Number of rollout episodes to average per genotype.
            Descriptors are averaged elementwise across those rollouts.
        num_workers: Number of worker processes for per-individual
            rollouts. ``0`` / ``1`` -> serial in-process. ``>1`` spawns
            a persistent :class:`ProcessPoolExecutor` on first call
            and reuses it across QD iterations. Call
            :func:`close_scoring_function` on the returned callable to
            tear down the pool explicitly; trainers do this in a
            ``try / finally``.
        parallel_backend: One of ``"process"``, ``"dask"``.
            See :mod:`deap_pettingzoo.utils.parallel` for details.
        scheduler_address: Remote Dask scheduler (``tcp://...`` URL or
            scheduler-file path); requires ``parallel_backend='dask'``.
            ``None`` auto-builds a local cluster unless the
            ``DEAP_DASK_SCHEDULER`` env var supplies an address. When a
            remote scheduler is in play, ``num_workers`` is advisory
            metadata and rollouts always dispatch through the executor.
        worker_imports: Optional module names imported inside each worker
            before env construction. Use this when the env is registered by
            importing a module for side effects rather than being part of the
            built-in registry.

    Returns:
        A callable ``scoring_function(batched_params, key) ->
        (fitness, descriptor, extra_scores)``. The callable has two
        extra attributes for pool lifecycle:

        * ``scoring_function.close()`` — close the persistent executor.
        * ``scoring_function._executor`` — lazily-created executor ref
          (``None`` until first parallel call; the actual executor
          object thereafter; reset to ``None`` after ``close()``).
                * ``scoring_function._last_batch_fitness`` — last fitness batch
                    scored by the numpy path, or ``None`` before the first call.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from deap_pettingzoo.contrib.qdax.policy import pytree_to_flat
    from deap_pettingzoo.environments.registry import get_registry_snapshot
    from deap_pettingzoo.utils.parallel import (
        make_executor,
        resolve_num_workers,
        resolve_scheduler_address,
    )

    # Remote scheduler (arg or DEAP_DASK_SCHEDULER env var): the workers live
    # elsewhere, so skip the driver CPU cap and always dispatch through the
    # executor even for num_workers <= 1 (advisory there).
    remote = (
        parallel_backend == "dask"
        and resolve_scheduler_address(scheduler_address) is not None
    )
    effective_workers = resolve_num_workers(num_workers, cap_to_cpu=not remote)
    use_parallel = effective_workers > 1 or remote
    rollout_spec = _RolloutSpec(
        env_name=env_name,
        env_config=dict(env_config) if env_config else None,
        layer_sizes=tuple(int(x) for x in layer_sizes),
        activation=str(activation),
        episode_length=int(episode_length),
        eval_episodes=max(1, int(eval_episodes)),
        descriptor_fn=descriptor_fn,
        action_type=action_type,
        nvec=tuple(int(x) for x in nvec) if nvec is not None else None,
        state_extractor=state_extractor,
        worker_imports=tuple(str(name) for name in worker_imports),
        # Propagate runtime-registered custom envs to workers (no init hook).
        registry_snapshot=get_registry_snapshot(target_remote=remote),
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
            effective_workers,
            backend=parallel_backend,
            scheduler_address=scheduler_address,
        )
        state["ctx"] = ctx
        state["executor"] = ctx.__enter__()
        # Mirror the live ref onto the public attribute so callers
        # (and tests) can introspect the actual executor object.
        scoring_function._executor = state["executor"]  # type: ignore[attr-defined]
        return state["executor"]

    def scoring_function(genotypes, key):
        leaves = jax.tree_util.tree_leaves(genotypes)
        batch_size = int(leaves[0].shape[0])
        seeds = np.asarray(jax.random.randint(key, (batch_size,), 0, 2**30))
        flats = []
        for i in range(batch_size):
            single_tree = jax.tree_util.tree_map(lambda x, i=i: x[i], genotypes)
            flats.append(pytree_to_flat(single_tree, layer_sizes))

        if not use_parallel:
            # Serial: avoid executor overhead entirely.
            results = [_local_rollout(flat, int(s)) for flat, s in zip(flats, seeds)]
        else:
            executor = _ensure_executor()
            payloads = [
                (rollout_spec, np.asarray(flat), int(s))
                for flat, s in zip(flats, seeds)
            ]
            results = list(executor.map(_numpy_rollout, payloads))

        fitnesses = [r[0] for r in results]
        descriptors = [r[1] for r in results]
        scoring_function._last_batch_fitness = np.asarray(  # type: ignore[attr-defined]
            fitnesses, dtype=np.float64
        )
        # Accumulate real env ticks across iterations. The trainer reads this
        # back as the run's consumed-steps total on the numpy/PettingZoo path.
        scoring_function._env_steps_consumed += int(  # type: ignore[attr-defined]
            sum(int(r[2]) for r in results)
        )
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
                scoring_function._executor = None  # type: ignore[attr-defined]
                scoring_function._last_batch_fitness = None  # type: ignore[attr-defined]

    scoring_function.close = _close  # type: ignore[attr-defined]
    scoring_function._executor = None  # type: ignore[attr-defined]
    scoring_function._last_batch_fitness = None  # type: ignore[attr-defined]
    # Cumulative real env ticks across all scoring calls (numpy path only).
    # The JAX-native scoring function does not define this attribute, so the
    # trainer reports env_steps_consumed=None there (a documented structural limit).
    scoring_function._env_steps_consumed = 0  # type: ignore[attr-defined]
    scoring_function._num_workers = effective_workers  # type: ignore[attr-defined]
    scoring_function._num_eval_episodes = rollout_spec.eval_episodes  # type: ignore[attr-defined]
    scoring_function._parallel_backend = parallel_backend  # type: ignore[attr-defined]
    return scoring_function


def close_scoring_function(scoring_function: Callable) -> None:
    """Close the persistent pool of a scoring function, if any."""
    closer = getattr(scoring_function, "close", None)
    if closer is not None:
        closer()
