"""QDax trainers for deap-bridge: :class:`MAPElitesTrainer`, :class:`CMAMETrainer`.

Both subclass :class:`~deap_pettingzoo.baselines.base.BaselineTrainer` so
they drop into :func:`compare_baselines` and share the common
:class:`TrainingConfig` / :class:`TrainingResult` API.

Dispatch rule: when the env's registered ``api`` is ``"jax_parallel"``
(i.e. a JaxMARL env registered via :func:`register_jaxmarl_env`), the
fast JAX-native scoring path is used
(:func:`make_jax_scoring_function`); otherwise the numpy-boundary path
(:func:`make_numpy_scoring_function`) runs, emitting a warning.

Hyperparameters are consumed from :attr:`TrainingConfig.algorithm_params`.
See each trainer's ``_DEFAULT_PARAMS`` for the keys and defaults.
"""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.contrib.qdax import _require_qdax
from deap_pettingzoo.contrib.qdax.progress import (
    format_qdax_progress_fields,
    summarise_fitness_batch,
    summarise_qdax_archive,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import get_env_spec

logger = logging.getLogger(__name__)


# Module-level flag so we only emit the "frame recording is unsupported
# on the QDax/JAX path" warning once per process even if the user starts
# multiple QDax trainers in the same script.
_QDAX_RECORDING_WARNED = False


def _warn_recording_unsupported_once(config: TrainingConfig) -> None:
    """Emit a one-shot RuntimeWarning when recording is requested.

    Frame recording requires running policy rollouts inside the
    PettingZoo wrapper (so :meth:`PettingZooWrapper.render` can return
    RGB arrays). The QDax path runs scoring entirely in JAX-compiled
    Brax-style scoring functions and never touches the PettingZoo
    wrapper at training time, so there is no env to render.
    """
    global _QDAX_RECORDING_WARNED
    if _QDAX_RECORDING_WARNED:
        return
    if not getattr(config, "recording", None):
        return
    if not config.recording.enabled:
        return
    warnings.warn(
        "RecordingConfig.enabled=True is ignored on the QDax/JAX path "
        "(MAPElitesTrainer / CMAMETrainer): training scoring runs "
        "entirely in jit-compiled JAX with no live PettingZoo env to "
        "render. To record QDax behaviours, post-train with "
        "UnifiedEvaluator on the best individual instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    _QDAX_RECORDING_WARNED = True


_DEFAULT_MAP_ELITES_PARAMS: Dict[str, Any] = {
    "layer_sizes": None,  # inferred from env if None
    "hidden_sizes": (32, 32),
    "activation": "tanh",
    "init_batch_size": 64,
    "init_mode": "flax_perturb",
    "init_sigma": 0.01,
    "batch_size": 32,
    "eval_episodes": 1,
    "grid_shape": (10, 10),
    "num_centroids": None,  # if set, use CVT centroids instead of grid
    "iso_sigma": 0.005,
    "line_sigma": 0.05,
    "episode_length": 25,
    "descriptor": "pursuit_evasion_2d",
    "descriptor_kwargs": {},
    "seed": 0,
    "save_repertoire": True,
    "num_workers": 1,
    "parallel_backend": "process",
    "worker_imports": (),
    "verbose": True,
    "progress_every": 1,
}

_DEFAULT_CMA_ME_PARAMS: Dict[str, Any] = {
    "layer_sizes": None,
    "hidden_sizes": (16, 16),
    "activation": "tanh",
    "batch_size": 32,
    "eval_episodes": 1,
    "grid_shape": (10, 10),
    "num_centroids": None,
    "sigma_g": 0.1,
    "num_emitters": 1,
    "episode_length": 25,
    "descriptor": "pursuit_evasion_2d",
    "descriptor_kwargs": {},
    "seed": 0,
    "save_repertoire": True,
    "num_workers": 1,
    "parallel_backend": "process",
    "worker_imports": (),
    "verbose": True,
    "progress_every": 1,
}


def _resolve_descriptor(name: str, env: PettingZooWrapper, kwargs: Dict[str, Any]):
    """Instantiate a descriptor factory by short name."""
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe_desc

    agents = list(env.possible_agents)

    # --- single_team: no pursuer/evader split needed ----------------------
    if name == "single_team":
        num_agents = int(kwargs.pop("num_agents", len(agents)))
        norm_scale = float(kwargs.pop("norm_scale", 1.0))
        return pe_desc.single_team_formation(num_agents=num_agents, norm_scale=norm_scale)

    # --- pursuit_evasion_*: resolve pursuer/evader counts -----------------
    pursuers = [a for a in agents if a.startswith("adversary")]
    evaders = [a for a in agents if not a.startswith("adversary")]

    if pursuers and evaders:
        default_kwargs = {
            "num_pursuers": len(pursuers),
            "num_evaders": len(evaders),
        }
    else:
        default_kwargs = {}
        if "num_pursuers" not in kwargs or "num_evaders" not in kwargs:
            raise ValueError(
                "Could not infer a consistent pursuit/evasion agent partition "
                f"from env.possible_agents={agents!r}. Automatic inference only "
                "works for tag-style envs with agent names prefixed by "
                "'adversary'. Please pass explicit descriptor_kwargs with both "
                "'num_pursuers' and 'num_evaders'."
            )

    merged = {**default_kwargs, **kwargs}
    num_pursuers = int(merged["num_pursuers"])
    num_evaders = int(merged["num_evaders"])
    # Only validate the agent-count sum when the counts were *auto-inferred*
    # from agent-name prefixes.  When the user passes explicit kwargs they
    # know their environment (e.g. scripted opponents may hide some agents
    # from env.possible_agents).
    if default_kwargs and (num_pursuers + num_evaders != len(agents)):
        raise ValueError(
            "Descriptor agent counts must sum to the number of possible "
            f"agents: num_pursuers={num_pursuers}, num_evaders={num_evaders}, "
            f"len(env.possible_agents)={len(agents)}."
        )
    merged["num_pursuers"] = num_pursuers
    merged["num_evaders"] = num_evaders

    if name == "pursuit_evasion_2d":
        return pe_desc.formation_spread_and_mean_distance(**merged)
    if name == "pursuit_evasion_3d":
        return pe_desc.with_capture_efficiency(**merged)
    raise ValueError(
        f"Unknown descriptor {name!r}. Supported: "
        "'pursuit_evasion_2d', 'pursuit_evasion_3d', 'single_team'. "
        "Pass a callable directly via algorithm_params['descriptor_fn'] "
        "to use a custom one."
    )


def _infer_layer_sizes(env: PettingZooWrapper, hidden: tuple) -> tuple:
    """Return ``(input, hidden..., output)`` from env spaces."""
    from deap_pettingzoo.policies.action_heads import required_output_size

    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    in_dim = int(np.prod(obs_space.shape))
    out_dim = required_output_size(env.action_space(agent))
    return (in_dim, *hidden, out_dim)


def _compute_centroids(min_bd, max_bd, grid_shape, num_centroids, key):
    import jax
    from qdax.core.containers.mapelites_repertoire import (
        compute_cvt_centroids,
        compute_euclidean_centroids,
    )

    if num_centroids is not None:
        key, sub = jax.random.split(key)
        return compute_cvt_centroids(
            num_descriptors=len(min_bd),
            num_init_cvt_samples=max(50_000, num_centroids * 50),
            num_centroids=int(num_centroids),
            minval=list(min_bd),
            maxval=list(max_bd),
            key=sub,
        ), key
    return compute_euclidean_centroids(
        grid_shape=tuple(int(x) for x in grid_shape),
        minval=list(min_bd),
        maxval=list(max_bd),
    ), key


def _sample_initial_leaf(leaf, key, *, batch_size: int, mode: str, sigma: float):
    """Sample one batched parameter leaf for the MAP-Elites init population."""
    import jax

    noise = sigma * jax.random.normal(
        key,
        (batch_size, *leaf.shape),
        dtype=leaf.dtype,
    )
    if mode == "gaussian":
        return noise
    if mode == "flax_perturb":
        return leaf[None, ...] + noise
    raise ValueError(
        "algorithm_params['init_mode'] must be one of "
        "{'flax_perturb', 'gaussian'}; got "
        f"{mode!r}."
    )


def _build_initial_genotypes(
    init_params,
    *,
    key,
    init_batch_size: int,
    init_mode: str,
    init_sigma: float,
):
    """Return a batched pytree of MAP-Elites init genotypes."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(init_params)
    leaf_keys = jax.random.split(key, len(leaves))
    batched_leaves = [
        _sample_initial_leaf(
            leaf,
            leaf_key,
            batch_size=init_batch_size,
            mode=init_mode,
            sigma=init_sigma,
        )
        for leaf, leaf_key in zip(leaves, leaf_keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, batched_leaves)


def _budget_to_iterations(config: TrainingConfig, params: Dict[str, Any]) -> int:
    """Translate ``config.budget_value`` into a QDax iteration count."""
    batch = int(params["batch_size"])
    ep_len = int(params["episode_length"])
    eval_episodes = max(1, int(params.get("eval_episodes", 1)))
    if config.budget_mode == BudgetMode.ENV_STEPS:
        return max(1, config.budget_value // (batch * ep_len * eval_episodes))
    if config.budget_mode == BudgetMode.EVAL_EPISODES:
        return max(1, config.budget_value // (batch * eval_episodes))
    raise ValueError(
        f"QDax trainers support BudgetMode.ENV_STEPS and "
        f"BudgetMode.EVAL_EPISODES; got {config.budget_mode}."
    )


def _print_qdax_plan(
    label: str,
    *,
    init_batch_size: int,
    batch_size: int,
    eval_episodes: int,
    episode_length: int,
    num_iters: int,
    num_workers: int,
) -> None:
    msg = (
        f"[{label}] init_batch={init_batch_size}  batch={batch_size}  "
        f"eval_eps={eval_episodes}  ep_len={episode_length}  "
        f"iters={num_iters}  workers={num_workers}"
    )
    print(msg, flush=True)
    logger.info("%s", msg)


def _validate_jax_parallel_settings(num_workers: int, eval_episodes: int) -> None:
    """Reject QDax settings that the JAX fast path cannot honor."""
    if num_workers > 1:
        raise ValueError(
            "algorithm_params['num_workers'] > 1 is not supported on "
            "the JAX fast path (env.api='jax_parallel'). The fast "
            "path already batches the whole population through a "
            "single jax.vmap; use 'batch_size' to scale instead. "
            "To use process-pool parallelism, register the env "
            "without the jax_parallel adapter (numpy scoring path)."
        )

    if eval_episodes > 1:
        raise ValueError(
            "algorithm_params['eval_episodes'] > 1 is not supported on "
            "the JAX fast path (env.api='jax_parallel'). Use the numpy "
            "scoring path for multi-rollout averaging."
        )


def _print_qdax_metrics(
    label: str,
    stage: str,
    repertoire: Any,
    elapsed: float,
    batch_fitness: Optional[Any] = None,
) -> None:
    archive_best, qd_score, coverage, filled_niches, total_niches = (
        summarise_qdax_archive(repertoire.fitnesses)
    )
    batch_best, batch_mean = summarise_fitness_batch(batch_fitness)
    summary = format_qdax_progress_fields(
        archive_best=archive_best,
        qd_score=qd_score,
        coverage=coverage,
        filled_niches=filled_niches,
        total_niches=total_niches,
        batch_best=batch_best,
        batch_mean=batch_mean,
    )
    msg = f"[{label}] {stage}  {summary}  ({elapsed:.1f}s)"
    print(msg, flush=True)
    logger.info("%s", msg)


def _build_env_and_scoring(config: TrainingConfig, params: Dict[str, Any]):
    """Construct env + scoring function, auto-dispatching fast vs slow path."""
    _require_qdax()

    import jax

    from deap_pettingzoo.contrib.qdax.policy import (
        make_flax_mlp,
        init_flax_mlp_params,
    )
    from deap_pettingzoo.contrib.qdax.scoring_jax import make_jax_scoring_function
    from deap_pettingzoo.contrib.qdax.scoring_numpy import (
        make_numpy_scoring_function,
    )

    env = PettingZooWrapper(
        env_name=config.env_name, env_config=config.env_config
    )
    hidden = tuple(params["hidden_sizes"])
    layer_sizes = params["layer_sizes"] or _infer_layer_sizes(env, hidden)
    module = make_flax_mlp(layer_sizes, activation=params["activation"])

    descriptor_fn = params.get("descriptor_fn") or _resolve_descriptor(
        params["descriptor"], env, params.get("descriptor_kwargs", {})
    )

    num_workers = int(params.get("num_workers", 1))
    parallel_backend = str(params.get("parallel_backend", "process"))
    eval_episodes = max(1, int(params.get("eval_episodes", 1)))

    spec = get_env_spec(config.env_name)
    if spec.api == "jax_parallel":
        _validate_jax_parallel_settings(num_workers, eval_episodes)
        # Raw JAX env for fast scoring.
        if getattr(spec, "wrappers", None):
            warnings.warn(
                f"Env {config.env_name!r} is registered with wrappers, but "
                "the QDax JAX fast path constructs the env directly from "
                "spec.factory(...) and does not apply EnvSpec.wrappers. "
                "Fast-path scoring may therefore use a different "
                "observation/action interface than wrapper-based trainers. "
                "Use the numpy scoring path if wrapper application is "
                "required.",
                UserWarning,
                stacklevel=2,
            )
        raw_jax_env = spec.factory(**(config.env_config or {}))
        # The JAX env may have heterogeneous agents (e.g. simple_tag has
        # adversaries with obs_size=16 and a good agent with obs_size=14).
        # Filter agent_list to only agents whose obs size matches the policy
        # input (layer_sizes[0]) so the shared network applies cleanly.
        import jax as _jax
        _k0 = _jax.random.PRNGKey(0)
        _obs0, _ = raw_jax_env.reset(_k0)
        _all_agents = list(_obs0.keys())
        _matching = [a for a in _all_agents
                     if int(_obs0[a].shape[0]) == layer_sizes[0]]
        if not _matching:
            # No agent matches — rebuild with the first agent's actual obs size.
            _agent0 = _all_agents[0]
            actual_in_dim = int(_obs0[_agent0].shape[0])
            layer_sizes = (actual_in_dim, *layer_sizes[1:])
            module = make_flax_mlp(layer_sizes, activation=params["activation"])
            _matching = [a for a in _all_agents
                         if int(_obs0[a].shape[0]) == actual_in_dim]
        scoring_fn = make_jax_scoring_function(
            raw_jax_env,
            module,
            episode_length=int(params["episode_length"]),
            descriptor_fn=descriptor_fn,
            action_type="discrete" if not config.continuous else "continuous",
            agents=_matching,
        )
    else:
        warnings.warn(
            f"Env {config.env_name!r} is not api='jax_parallel'; using the "
            "slow numpy-boundary scoring path. Register the env with "
            "register_jaxmarl_env(...) for the fast path.",
            UserWarning,
            stacklevel=2,
        )
        scoring_fn = make_numpy_scoring_function(
            config.env_name,
            layer_sizes=layer_sizes,
            episode_length=int(params["episode_length"]),
            descriptor_fn=descriptor_fn,
            env_config=config.env_config,
            action_type="discrete" if not config.continuous else "continuous",
            activation=str(params["activation"]),
            num_workers=num_workers,
            parallel_backend=parallel_backend,
            eval_episodes=eval_episodes,
            worker_imports=params.get("worker_imports", ()),
        )

    init_params = init_flax_mlp_params(
        module, input_size=layer_sizes[0], seed=params["seed"]
    )
    return env, module, layer_sizes, descriptor_fn, scoring_fn, init_params


def _save_repertoire(repertoire, output_dir: Path) -> Path:
    """Pickle the repertoire pytree to ``output_dir/repertoire.pkl``."""
    import jax

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "repertoire.pkl"
    with target.open("wb") as f:
        pickle.dump(jax.device_get(repertoire), f)
    return target


def _resolve_shared_action_space(env: PettingZooWrapper):
    """Return the shared action space across all agents in ``env``."""
    agents = list(env.possible_agents)
    reference = env.action_space(agents[0])
    divergent = [a for a in agents[1:] if env.action_space(a) != reference]
    if divergent:
        raise ValueError(
            "QDax trainers require a single shared action space across all "
            "agents (homogeneous-team policies). Agents diverging from "
            f"{agents[0]!r}'s action space: {divergent}."
        )
    return reference


def _extract_champion(repertoire, module, *, action_space=None):
    """Pick the highest-fitness genotype from ``repertoire`` as a policy."""
    import jax
    import numpy as _np

    from deap_pettingzoo.contrib.qdax.policy import FlaxPolicyAdapter

    fitnesses = _np.asarray(repertoire.fitnesses)
    # Ignore empty cells (QDax marks those with -inf).
    best_idx = int(_np.argmax(_np.where(_np.isfinite(fitnesses), fitnesses, -_np.inf)))
    champion_params = jax.tree_util.tree_map(
        lambda leaf: leaf[best_idx], repertoire.genotypes
    )
    return (
        FlaxPolicyAdapter(module, champion_params, action_space=action_space),
        float(fitnesses[best_idx]),
    )


class MAPElitesTrainer(BaselineTrainer):
    """MAP-Elites trainer over a Flax MLP policy.

    Supported :class:`BudgetMode`\u200bs: ``ENV_STEPS``, ``EVAL_EPISODES``.
    ``WALL_CLOCK`` is intentionally not supported: the QDax loop is
    unpredictable in wall time because the first iteration incurs the
    ``jit`` compile cost.
    """

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in (BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": True,
            "supports_turn_based": False,
            "produces_repertoire": True,
        }

    def train(self, config: TrainingConfig) -> TrainingResult:
        _require_qdax()

        import jax
        from qdax.core.map_elites import MAPElites
        from qdax.core.emitters.mutation_operators import isoline_variation
        from qdax.core.emitters.standard_emitters import MixingEmitter
        from qdax.utils.metrics import default_qd_metrics

        params = {**_DEFAULT_MAP_ELITES_PARAMS, **(config.algorithm_params or {})}
        env, module, layer_sizes, descriptor_fn, scoring_fn, init_params = (
            _build_env_and_scoring(config, params)
        )

        min_bd = tuple(getattr(descriptor_fn, "min_bd", (0.0, 0.0)))
        max_bd = tuple(getattr(descriptor_fn, "max_bd", (1.0, 1.0)))

        key = jax.random.PRNGKey(int(params["seed"]))
        centroids, key = _compute_centroids(
            min_bd, max_bd, params["grid_shape"], params.get("num_centroids"), key
        )

        # Build the initial repertoire population. Default preserves the old
        # QDax-style "Flax init + small perturbation" behaviour, but callers
        # can opt into a zero-mean Gaussian init to match CMA-like startup.
        key, sub = jax.random.split(key)
        init_genotypes = _build_initial_genotypes(
            init_params,
            key=sub,
            init_batch_size=int(params["init_batch_size"]),
            init_mode=str(params.get("init_mode", "flax_perturb")),
            init_sigma=float(params.get("init_sigma", 0.01)),
        )

        variation_fn = lambda x1, x2, k: isoline_variation(  # noqa: E731
            x1,
            x2,
            k,
            iso_sigma=float(params["iso_sigma"]),
            line_sigma=float(params["line_sigma"]),
        )
        mutation_fn = lambda x, k: (x, k)  # identity, mutation handled via variation
        emitter = MixingEmitter(
            mutation_fn=mutation_fn,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=int(params["batch_size"]),
        )
        metrics_fn = lambda rep: default_qd_metrics(  # noqa: E731
            rep, qd_offset=0.0
        )

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

        num_iters = _budget_to_iterations(config, params)
        verbose = bool(params.get("verbose", True))
        progress_every = max(1, int(params.get("progress_every", 1)))
        if verbose:
            _print_qdax_plan(
                "MAP-Elites",
                init_batch_size=int(params["init_batch_size"]),
                batch_size=int(params["batch_size"]),
                eval_episodes=max(1, int(params.get("eval_episodes", 1))),
                episode_length=int(params["episode_length"]),
                num_iters=num_iters,
                num_workers=int(params.get("num_workers", 1)),
            )

        try:
            init_start = time.time()
            key, sub = jax.random.split(key)
            repertoire, emitter_state, init_metrics = map_elites.init(
                init_genotypes, centroids, sub
            )
            if verbose:
                _print_qdax_metrics(
                    "MAP-Elites",
                    "init",
                    repertoire,
                    time.time() - init_start,
                    getattr(scoring_fn, "_last_batch_fitness", None),
                )

            history: list = [init_metrics]
            start = time.time()
            for iter_idx in range(1, num_iters + 1):
                key, sub = jax.random.split(key)
                repertoire, emitter_state, metrics = map_elites.update(
                    repertoire, emitter_state, sub
                )
                history.append(metrics)
                if verbose and (
                    iter_idx == 1
                    or iter_idx == num_iters
                    or iter_idx % progress_every == 0
                ):
                    _print_qdax_metrics(
                        "MAP-Elites",
                        f"iter {iter_idx}/{num_iters}",
                        repertoire,
                        time.time() - start,
                        getattr(scoring_fn, "_last_batch_fitness", None),
                    )
            elapsed = time.time() - start
        finally:
            # Tear down the scoring function's worker pool, if any.
            closer = getattr(scoring_fn, "close", None)
            if closer is not None:
                closer()

        champion_policy, champion_fitness = _extract_champion(
            repertoire,
            module,
            action_space=_resolve_shared_action_space(env),
        )

        metadata: Dict[str, Any] = {
            "descriptor": params["descriptor"],
            "grid_shape": tuple(params["grid_shape"]),
            "num_iterations": num_iters,
            "layer_sizes": tuple(layer_sizes),
            "init_batch_size": int(params["init_batch_size"]),
            "init_mode": str(params.get("init_mode", "flax_perturb")),
            "init_sigma": float(params.get("init_sigma", 0.01)),
            "eval_episodes": max(1, int(params.get("eval_episodes", 1))),
            "num_workers": int(params.get("num_workers", 1)),
            "parallel_backend": str(params.get("parallel_backend", "process")),
        }
        if params.get("save_repertoire") and config.output_dir is not None:
            path = _save_repertoire(repertoire, Path(config.output_dir))
            metadata["repertoire_path"] = str(path)

        # Summary metrics: take the last entry of each QDax metric
        last = history[-1]
        metrics_out = {
            "best_fitness": champion_fitness,
            "qd_score": float(np.asarray(last["qd_score"])),
            "coverage": float(np.asarray(last["coverage"])),
            "max_fitness": float(np.asarray(last["max_fitness"])),
            "num_iterations": num_iters,
        }

        _persist_qdax_results(
            config,
            algorithm="map_elites",
            history=history,
            summary=metrics_out,
            training_time=elapsed,
            extra_summary={"descriptor": params["descriptor"]},
            champion_policy=champion_policy,
            recording_max_steps=int(params["episode_length"]),
            repertoire_path=Path(metadata["repertoire_path"])
            if "repertoire_path" in metadata
            else None,
        )

        return TrainingResult(
            algorithm="map_elites",
            policy=champion_policy,
            metrics=metrics_out,
            metadata=metadata,
            seed=int(params["seed"]),
            training_time=elapsed,
        )


class CMAMETrainer(BaselineTrainer):
    """CMA-ME trainer (CMA-Improvement emitter) over a flat Flax MLP.

    Uses flat (1-D) genotypes internally because
    :class:`CMAImprovementEmitter` expects an ``Array`` population, not
    a pytree. The champion is decoded back into a Flax pytree via
    :func:`flat_to_pytree` for the returned policy.
    """

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in (BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": True,
            "supports_turn_based": False,
            "produces_repertoire": True,
        }

    def train(self, config: TrainingConfig) -> TrainingResult:
        _require_qdax()

        import jax
        import jax.numpy as jnp
        from qdax.core.map_elites import MAPElites
        from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
        from qdax.utils.metrics import default_qd_metrics

        from deap_pettingzoo.contrib.qdax.policy import (
            flat_count,
            flat_to_pytree,
            FlaxPolicyAdapter,
        )

        params = {**_DEFAULT_CMA_ME_PARAMS, **(config.algorithm_params or {})}
        env, module, layer_sizes, descriptor_fn, scoring_fn, _ = (
            _build_env_and_scoring(config, params)
        )

        genotype_dim = flat_count(layer_sizes)
        batch_size = int(params["batch_size"])

        # CMA-ME operates on flat arrays; adapt the scoring function.
        def _flat_scoring(flat_population, key):
            def _to_tree(flat_one):
                # Materialise a pytree inside the JIT; shape-stable.
                return flat_to_pytree(np.asarray(flat_one), layer_sizes)

            # Since flat_to_pytree uses numpy, we must fall back to a host loop
            # for CMA-ME with JAX envs. This works but is slower than pure-JAX.
            flats_np = np.asarray(flat_population)
            genotype_batch = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs),
                *(_to_tree(row) for row in flats_np),
            )
            return scoring_fn(genotype_batch, key)

        min_bd = tuple(getattr(descriptor_fn, "min_bd", (0.0, 0.0)))
        max_bd = tuple(getattr(descriptor_fn, "max_bd", (1.0, 1.0)))

        key = jax.random.PRNGKey(int(params["seed"]))
        centroids, key = _compute_centroids(
            min_bd, max_bd, params["grid_shape"], params.get("num_centroids"), key
        )

        emitter = CMAImprovementEmitter(
            batch_size=batch_size,
            genotype_dim=genotype_dim,
            centroids=centroids,
            sigma_g=float(params["sigma_g"]),
        )
        metrics_fn = lambda rep: default_qd_metrics(rep, qd_offset=0.0)  # noqa: E731

        map_elites = MAPElites(
            scoring_function=_flat_scoring,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

        num_iters = _budget_to_iterations(config, params)
        verbose = bool(params.get("verbose", True))
        progress_every = max(1, int(params.get("progress_every", 1)))
        if verbose:
            _print_qdax_plan(
                "CMA-ME",
                init_batch_size=batch_size,
                batch_size=batch_size,
                eval_episodes=max(1, int(params.get("eval_episodes", 1))),
                episode_length=int(params["episode_length"]),
                num_iters=num_iters,
                num_workers=int(params.get("num_workers", 1)),
            )

        # Initial batch: Gaussian around zero.
        key, sub = jax.random.split(key)
        init_genotypes = 0.01 * jax.random.normal(sub, (batch_size, genotype_dim))

        try:
            init_start = time.time()
            key, sub = jax.random.split(key)
            repertoire, emitter_state, init_metrics = map_elites.init(
                init_genotypes, centroids, sub
            )
            if verbose:
                _print_qdax_metrics(
                    "CMA-ME",
                    "init",
                    repertoire,
                    time.time() - init_start,
                    getattr(scoring_fn, "_last_batch_fitness", None),
                )

            history = [init_metrics]
            start = time.time()
            for iter_idx in range(1, num_iters + 1):
                key, sub = jax.random.split(key)
                repertoire, emitter_state, metrics = map_elites.update(
                    repertoire, emitter_state, sub
                )
                history.append(metrics)
                if verbose and (
                    iter_idx == 1
                    or iter_idx == num_iters
                    or iter_idx % progress_every == 0
                ):
                    _print_qdax_metrics(
                        "CMA-ME",
                        f"iter {iter_idx}/{num_iters}",
                        repertoire,
                        time.time() - start,
                        getattr(scoring_fn, "_last_batch_fitness", None),
                    )
            elapsed = time.time() - start
        finally:
            closer = getattr(scoring_fn, "close", None)
            if closer is not None:
                closer()

        # Champion extraction: repertoire genotypes are flat arrays here.
        fitnesses = np.asarray(repertoire.fitnesses)
        best_idx = int(
            np.argmax(np.where(np.isfinite(fitnesses), fitnesses, -np.inf))
        )
        flat_champion = np.asarray(repertoire.genotypes[best_idx])
        champion_params = flat_to_pytree(flat_champion, layer_sizes)
        champion_policy = FlaxPolicyAdapter(
            module,
            champion_params,
            action_space=_resolve_shared_action_space(env),
        )

        metadata: Dict[str, Any] = {
            "descriptor": params["descriptor"],
            "grid_shape": tuple(params["grid_shape"]),
            "num_iterations": num_iters,
            "layer_sizes": tuple(layer_sizes),
            "genotype_dim": genotype_dim,
            "eval_episodes": max(1, int(params.get("eval_episodes", 1))),
            "num_workers": int(params.get("num_workers", 1)),
            "parallel_backend": str(params.get("parallel_backend", "process")),
        }
        if params.get("save_repertoire") and config.output_dir is not None:
            path = _save_repertoire(repertoire, Path(config.output_dir))
            metadata["repertoire_path"] = str(path)

        last = history[-1]
        metrics_out = {
            "best_fitness": float(fitnesses[best_idx]),
            "qd_score": float(np.asarray(last["qd_score"])),
            "coverage": float(np.asarray(last["coverage"])),
            "max_fitness": float(np.asarray(last["max_fitness"])),
            "num_iterations": num_iters,
        }

        _persist_qdax_results(
            config,
            algorithm="cma_me",
            history=history,
            summary=metrics_out,
            training_time=elapsed,
            extra_summary={"descriptor": params["descriptor"]},
            champion_policy=champion_policy,
            recording_max_steps=int(params["episode_length"]),
            repertoire_path=Path(metadata["repertoire_path"])
            if "repertoire_path" in metadata
            else None,
        )

        return TrainingResult(
            algorithm="cma_me",
            policy=champion_policy,
            metrics=metrics_out,
            metadata=metadata,
            seed=int(params["seed"]),
            training_time=elapsed,
        )


__all__ = ["MAPElitesTrainer", "CMAMETrainer"]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _maybe_record_qdax_champion(
    log,
    config: TrainingConfig,
    *,
    champion_policy,
    recording_max_steps: Optional[int],
    iteration: Optional[int] = None,
) -> None:
    """Record one final champion rollout when recording is enabled.

    ``iteration`` is the QDax iteration the recorded champion corresponds
    to (typically the final iteration). It is written to the recorder's
    ``frames/recordings.json`` sidecar so the UI can associate the artefact
    with its metrics row (carrying ``qd_score``, ``coverage``, etc.).
    """
    if champion_policy is None:
        return

    recording_cfg = config.recording
    if not recording_cfg.enabled:
        return

    from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator

    recorder = log.get_frame_recorder(
        fps=recording_cfg.fps,
        fmt=recording_cfg.fmt,
        max_frames_per_episode=recording_cfg.max_frames_per_episode,
        max_dim=recording_cfg.max_dim,
        episode_prefix=recording_cfg.episode_prefix or "train_",
    )
    if not recorder.enabled:
        return

    evaluator = UnifiedEvaluator(
        max_steps=int(recording_max_steps or 500),
        frame_recorder=recorder,
    )
    try:
        rollout_seed = config.algorithm_params.get("seed")
        evaluator.evaluate(
            policy=champion_policy,
            env_name=config.env_name,
            num_episodes=1,
            seed=int(rollout_seed) if rollout_seed is not None else None,
            env_config=dict(config.env_config) if config.env_config else None,
            iteration=iteration,
        )
    except Exception:
        pass


def _persist_qdax_results(
    config: TrainingConfig,
    *,
    algorithm: str,
    history,
    summary: Dict[str, Any],
    training_time: float,
    extra_summary: Optional[Dict[str, Any]] = None,
    champion_policy=None,
    recording_max_steps: Optional[int] = None,
    repertoire_path: Optional[Path] = None,
) -> None:
    """Write per-iteration QDax metrics + final summary via ExperimentLogger.

    Each entry in ``history`` is a dict-like of QDax scalars (typically
    ``qd_score``, ``coverage``, ``max_fitness``, ``min_fitness``,
    ``mean_fitness``). Numeric scalars are coerced to float; non-numeric
    fields are skipped. No-op when ``config.output_dir`` is ``None``.

    If ``repertoire_path`` points to an existing file, it is registered
    in ``manifest.artifacts['policy']`` so the analysis loader can
    rebuild a policy for the UI's Live Rollout page.
    """
    from deap_pettingzoo.baselines._logging_helpers import (
        build_logger,
        write_summary,
    )
    from deap_pettingzoo.utils.logging import (
        KIND_QDAX_REPERTOIRE,
        register_existing_artifact,
    )

    with build_logger(config, algorithm=algorithm) as log:
        for i, entry in enumerate(history):
            row: Dict[str, Any] = {
                "iteration": i,
                "env_steps": 0,
                "wall_time": 0.0,
            }
            try:
                items = entry.items()
            except AttributeError:
                continue
            for k, v in items:
                try:
                    row[str(k)] = float(np.asarray(v))
                except (TypeError, ValueError):
                    continue
            log.row("metrics", row)

        try:
            n_iters = len(history)
        except TypeError:
            n_iters = 0
        _maybe_record_qdax_champion(
            log,
            config,
            champion_policy=champion_policy,
            recording_max_steps=recording_max_steps,
            iteration=(n_iters - 1) if n_iters else None,
        )

        full_summary = {"algorithm": algorithm, **summary}
        if extra_summary:
            full_summary.update(extra_summary)
        if repertoire_path is not None:
            register_existing_artifact(
                log,
                kind=KIND_QDAX_REPERTOIRE,
                path=Path(repertoire_path),
                extra={"algorithm": algorithm},
            )
        write_summary(log, full_summary, training_time_s=training_time)
