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
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import get_env_spec


_DEFAULT_MAP_ELITES_PARAMS: Dict[str, Any] = {
    "layer_sizes": None,  # inferred from env if None
    "hidden_sizes": (32, 32),
    "activation": "tanh",
    "init_batch_size": 64,
    "batch_size": 32,
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
}

_DEFAULT_CMA_ME_PARAMS: Dict[str, Any] = {
    "layer_sizes": None,
    "hidden_sizes": (16, 16),
    "activation": "tanh",
    "batch_size": 32,
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
}


def _resolve_descriptor(name: str, env: PettingZooWrapper, kwargs: Dict[str, Any]):
    """Instantiate a descriptor factory by short name."""
    from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe_desc

    agents = list(env.possible_agents)
    pursuers = [a for a in agents if a.startswith("adversary")]
    evaders = [a for a in agents if not a.startswith("adversary")]
    default_kwargs = {
        "num_pursuers": len(pursuers) or len(agents) // 2,
        "num_evaders": len(evaders) or len(agents) - len(pursuers),
    }
    merged = {**default_kwargs, **kwargs}
    if name == "pursuit_evasion_2d":
        return pe_desc.formation_spread_and_mean_distance(**merged)
    if name == "pursuit_evasion_3d":
        return pe_desc.with_capture_efficiency(**merged)
    raise ValueError(
        f"Unknown descriptor {name!r}. Supported: "
        "'pursuit_evasion_2d', 'pursuit_evasion_3d'. Pass a callable "
        "directly via algorithm_params['descriptor_fn'] to use a custom one."
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


def _budget_to_iterations(config: TrainingConfig, params: Dict[str, Any]) -> int:
    """Translate ``config.budget_value`` into a QDax iteration count."""
    batch = int(params["batch_size"])
    ep_len = int(params["episode_length"])
    if config.budget_mode == BudgetMode.ENV_STEPS:
        return max(1, config.budget_value // (batch * ep_len))
    if config.budget_mode == BudgetMode.EVAL_EPISODES:
        return max(1, config.budget_value // batch)
    raise ValueError(
        f"QDax trainers support BudgetMode.ENV_STEPS and "
        f"BudgetMode.EVAL_EPISODES; got {config.budget_mode}."
    )


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

    spec = get_env_spec(config.env_name)
    if spec.api == "jax_parallel":
        if num_workers > 1:
            raise ValueError(
                "algorithm_params['num_workers'] > 1 is not supported on "
                "the JAX fast path (env.api='jax_parallel'). The fast "
                "path already batches the whole population through a "
                "single jax.vmap; use 'batch_size' to scale instead. "
                "To use process-pool parallelism, register the env "
                "without the jax_parallel adapter (numpy scoring path)."
            )
        # Raw JAX env for fast scoring.
        raw_jax_env = spec.factory(**(config.env_config or {}))
        scoring_fn = make_jax_scoring_function(
            raw_jax_env,
            module,
            episode_length=int(params["episode_length"]),
            descriptor_fn=descriptor_fn,
            action_type="discrete" if not config.continuous else "continuous",
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
            module,
            layer_sizes=layer_sizes,
            episode_length=int(params["episode_length"]),
            descriptor_fn=descriptor_fn,
            env_config=config.env_config,
            action_type="discrete" if not config.continuous else "continuous",
            activation=str(params["activation"]),
            num_workers=num_workers,
            parallel_backend=parallel_backend,
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


def _extract_champion(repertoire, module, layer_sizes):
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
    return FlaxPolicyAdapter(module, champion_params), float(fitnesses[best_idx])


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

        # Build a population of size init_batch_size by sampling small
        # Gaussian perturbations around init_params (standard QDax pattern).
        key, sub = jax.random.split(key)

        def _perturb(seed_key):
            leaves, treedef = jax.tree_util.tree_flatten(init_params)

            def _one(k, leaf):
                return leaf + 0.01 * jax.random.normal(k, leaf.shape)

            sub_keys = jax.random.split(seed_key, len(leaves))
            new_leaves = [_one(k, leaf) for k, leaf in zip(sub_keys, leaves)]
            return jax.tree_util.tree_unflatten(treedef, new_leaves)

        init_keys = jax.random.split(sub, int(params["init_batch_size"]))
        init_genotypes = jax.vmap(_perturb)(init_keys)

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

        try:
            key, sub = jax.random.split(key)
            repertoire, emitter_state, init_metrics = map_elites.init(
                init_genotypes, centroids, sub
            )

            num_iters = _budget_to_iterations(config, params)
            history: list = [init_metrics]
            start = time.time()
            for _ in range(num_iters):
                key, sub = jax.random.split(key)
                repertoire, emitter_state, metrics = map_elites.update(
                    repertoire, emitter_state, sub
                )
                history.append(metrics)
            elapsed = time.time() - start
        finally:
            # Tear down the scoring function's worker pool, if any.
            closer = getattr(scoring_fn, "close", None)
            if closer is not None:
                closer()

        champion_policy, champion_fitness = _extract_champion(
            repertoire, module, layer_sizes
        )

        metadata: Dict[str, Any] = {
            "descriptor": params["descriptor"],
            "grid_shape": tuple(params["grid_shape"]),
            "num_iterations": num_iters,
            "layer_sizes": tuple(layer_sizes),
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
        )

        params = {**_DEFAULT_CMA_ME_PARAMS, **(config.algorithm_params or {})}
        env, module, layer_sizes, descriptor_fn, scoring_fn, init_params = (
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

        # Initial batch: Gaussian around zero.
        key, sub = jax.random.split(key)
        init_genotypes = 0.01 * jax.random.normal(sub, (batch_size, genotype_dim))

        try:
            key, sub = jax.random.split(key)
            repertoire, emitter_state, init_metrics = map_elites.init(
                init_genotypes, centroids, sub
            )

            num_iters = _budget_to_iterations(config, params)
            history = [init_metrics]
            start = time.time()
            for _ in range(num_iters):
                key, sub = jax.random.split(key)
                repertoire, emitter_state, metrics = map_elites.update(
                    repertoire, emitter_state, sub
                )
                history.append(metrics)
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
        from deap_pettingzoo.contrib.qdax.policy import FlaxPolicyAdapter

        champion_params = flat_to_pytree(flat_champion, layer_sizes)
        champion_policy = FlaxPolicyAdapter(module, champion_params)

        metadata: Dict[str, Any] = {
            "descriptor": params["descriptor"],
            "grid_shape": tuple(params["grid_shape"]),
            "num_iterations": num_iters,
            "layer_sizes": tuple(layer_sizes),
            "genotype_dim": genotype_dim,
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

        return TrainingResult(
            algorithm="cma_me",
            policy=champion_policy,
            metrics=metrics_out,
            metadata=metadata,
            seed=int(params["seed"]),
            training_time=elapsed,
        )


__all__ = ["MAPElitesTrainer", "CMAMETrainer"]
