"""QDax \u2194 co-evolution bridge (experimental).

Adds two concrete plug-ins that slot into
:mod:`deap_pettingzoo.baselines.coevolution`:

* :class:`QDaxRepertoirePool` \u2014 an :class:`OpponentPool` backed by a
  QDax :class:`MapElitesRepertoire`. Sampling draws filled cells
  uniformly (default) or weighted by fitness, then decodes the stored
  genotype into a :class:`FlaxPolicyAdapter`. This class is
  production-ready.

* :class:`QDaxSideState` \u2014 a :class:`SideState` that runs one QDax
  MAP-Elites (or CMA-ME) iteration per co-evolution generation,
  scoring each proposed genotype by playing episodes against the
  *opponent sides' current pool* via the injected evaluator. This
  class is **experimental**: the current scoring path uses a
  placeholder (centre-of-range) descriptor because the upstream
  :class:`CoEvolutionEvaluator` does not expose trajectory
  observations. Subclass :meth:`QDaxSideState._evaluate_one_genotype`
  (or supply a custom evaluator with a trajectory hook) to use real
  descriptors.

Only the numpy-boundary scoring path is used here \u2014 co-evolution
drives episodes through :class:`PettingZooWrapper`, so mixing
opponents and QDax individuals is straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.coevolution.pools import (
    OpponentPool,
    OpponentSnapshot,
    _BasePool,
)
from deap_pettingzoo.baselines.coevolution.samplers import (
    Sampler,
    uniform_sampler,
)
from deap_pettingzoo.baselines.coevolution.sides import (
    GenerationStats,
    Side,
    SideState,
    _resolve_side_action_space,
)
from deap_pettingzoo.contrib.qdax import _require_qdax
from deap_pettingzoo.contrib.qdax.progress import summarise_qdax_archive
from deap_pettingzoo.environments import PettingZooWrapper


def _compute_qdax_descriptor(
    positions_list: List[np.ndarray],
    descriptor_fn: Any,
) -> np.ndarray:
    """Compute a QDax descriptor from collected positions, or return midpoint."""
    positions = _stack_consistent_positions(positions_list)
    if positions is not None:
        try:
            return np.asarray(
                descriptor_fn({"positions": positions}), dtype=np.float32
            )
        except Exception:
            pass
    mn = np.asarray(descriptor_fn.min_bd, dtype=np.float32)
    mx = np.asarray(descriptor_fn.max_bd, dtype=np.float32)
    return 0.5 * (mn + mx)


def _stack_consistent_positions(
    positions_list: List[np.ndarray],
) -> Optional[np.ndarray]:
    """Stack frames with the most common shape, skipping variable-size frames."""
    if not positions_list:
        return None
    shape_counts: Dict[tuple[int, ...], int] = {}
    for frame in positions_list:
        shape = tuple(np.asarray(frame).shape)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    target_shape = max(shape_counts, key=shape_counts.get)
    frames = [
        np.asarray(frame, dtype=np.float32)
        for frame in positions_list
        if tuple(np.asarray(frame).shape) == target_shape
    ]
    if not frames:
        return None
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Worker-side helpers for parallel genotype evaluation
# ---------------------------------------------------------------------------


@dataclass
class _QDaxGenotypePayload:
    """Picklable payload sent to each worker process for one genotype."""

    env_name: str
    env_config: Optional[Dict[str, Any]]
    flat_weights: np.ndarray              # shape (n_params,) float32
    layer_sizes: Tuple[int, ...]
    activation: str
    learner_role_prefixes: Tuple[str, ...]
    side_role_prefixes: Dict[str, Tuple[str, ...]]
    per_episode_snapshots: List[Dict[str, Any]]
    k_opponents: int
    aggregation: str
    max_steps: int
    position_extractor: Optional[Any]
    descriptor_fn: Any


@dataclass
class _QDaxGenotypeResult:
    """Return value of :func:`_qdax_genotype_rollout`."""

    fitness: float
    descriptor: np.ndarray   # shape (behavior_dim,) float32


def _qdax_genotype_rollout(payload: "_QDaxGenotypePayload") -> "_QDaxGenotypeResult":
    """Worker entry point: rebuild env+policy, run k episodes, return stats."""
    from deap_pettingzoo.baselines.coevolution.evaluator import (
        _invoke_get_action,
        _resolve_worker_env,
    )
    from deap_pettingzoo.baselines.coevolution.pools import rebuild_opponent_policy
    from deap_pettingzoo.contrib.qdax.policy import (
        FlaxPolicyAdapter,
        flat_to_pytree,
        make_flax_mlp,
    )

    env = _resolve_worker_env(payload.env_name, payload.env_config)
    module = make_flax_mlp(payload.layer_sizes, payload.activation)
    params = flat_to_pytree(payload.flat_weights, payload.layer_sizes)

    learner_agents = [
        a for a in env.possible_agents
        if any(a.startswith(p) for p in payload.learner_role_prefixes)
    ]
    action_space = env.action_space(learner_agents[0])
    policy = FlaxPolicyAdapter(module, params, action_space=action_space)
    learner_set = set(learner_agents)

    side_agents: Dict[str, List[str]] = {
        name: [
            a for a in env.possible_agents
            if any(a.startswith(p) for p in prefixes)
        ]
        for name, prefixes in payload.side_role_prefixes.items()
    }

    rewards: List[float] = []
    all_positions: List[np.ndarray] = []

    for ep_idx, ep_snaps in enumerate(payload.per_episode_snapshots):
        opp_policies: Dict[str, Any] = {}
        for side_name, snap in ep_snaps.items():
            members = side_agents.get(side_name, [])
            if members:
                opp_pol = rebuild_opponent_policy(
                    snap, action_space=env.action_space(members[0])
                )
                for a in members:
                    opp_policies[a] = opp_pol

        ep_positions: List[np.ndarray] = []
        total = _run_qdax_worker_episode(
            env=env,
            policy=policy,
            opp_policies=opp_policies,
            learner_set=learner_set,
            max_steps=payload.max_steps,
            position_extractor=payload.position_extractor,
            positions_out=ep_positions,
            seed=ep_idx,
        )
        rewards.append(total)
        all_positions.extend(ep_positions)

    arr = np.asarray(rewards, dtype=np.float64)
    if payload.aggregation == "mean":
        fitness = float(arr.mean())
    elif payload.aggregation == "min":
        fitness = float(arr.min())
    elif payload.aggregation == "max":
        fitness = float(arr.max())
    else:
        fitness = float(arr.mean())

    descriptor = _compute_qdax_descriptor(all_positions, payload.descriptor_fn)
    return _QDaxGenotypeResult(fitness=fitness, descriptor=descriptor)


def _run_qdax_worker_episode(
    *,
    env: Any,
    policy: Any,
    opp_policies: Dict[str, Any],
    learner_set: set,
    max_steps: int,
    position_extractor: Optional[Any],
    positions_out: List[np.ndarray],
    seed: Optional[int],
) -> float:
    """Run one episode in a worker; append position arrays to *positions_out*."""
    from deap_pettingzoo.baselines.coevolution.evaluator import _invoke_get_action

    observations = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    for p in opp_policies.values():
        if hasattr(p, "reset"):
            p.reset()

    total = 0.0
    steps = 0
    while steps < max_steps and observations:
        if position_extractor is not None:
            try:
                pos = position_extractor(observations, env)
                if pos is not None:
                    positions_out.append(np.asarray(pos, dtype=np.float32))
            except Exception:
                pass

        infos = getattr(env, "last_infos", {}) or {}
        actions: Dict[str, Any] = {}
        for agent, obs in observations.items():
            mask = infos.get(agent, {}).get("action_mask")
            if agent in learner_set:
                actions[agent] = _invoke_get_action(
                    policy, obs, mask, deterministic=True
                )
            elif agent in opp_policies:
                actions[agent] = _invoke_get_action(opp_policies[agent], obs, mask)

        observations, rew, terms, truncs, _ = env.step(actions)
        for agent in learner_set:
            total += float(rew.get(agent, 0.0))
        steps += 1
        if terms and all(terms.values()):
            break
        if truncs and all(truncs.values()):
            break
    return total


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class QDaxRepertoirePool(_BasePool):
    """Opponent pool backed by a QDax :class:`MapElitesRepertoire`.

    Empty cells (fitness ``-inf``) are skipped. ``mode='uniform'``
    samples filled cells uniformly; ``mode='fitness'`` samples with
    probability proportional to ``softmax(fitness / temperature)``.

    Args:
        repertoire: A QDax :class:`MapElitesRepertoire` (must already
            be populated with at least one filled cell).
        policy_factory: Callable ``params -> PolicyProtocol``. Typically
            a closure that wraps ``params`` in a
            :class:`FlaxPolicyAdapter` sharing the host Flax module.
        mode: ``'uniform'`` or ``'fitness'``. Default ``'uniform'``.
        temperature: Softmax temperature for ``mode='fitness'``.
            Ignored otherwise.
    """

    def __init__(
        self,
        repertoire: Any,
        policy_factory: Callable[[Any], PolicyProtocol],
        *,
        mode: str = "uniform",
        temperature: float = 1.0,
        layer_sizes: Optional[Sequence[int]] = None,
        activation: str = "tanh",
    ) -> None:
        if mode not in {"uniform", "fitness"}:
            raise ValueError(
                f"QDaxRepertoirePool: mode must be 'uniform' or 'fitness'; "
                f"got {mode!r}."
            )
        self._repertoire = repertoire
        self._policy_factory = policy_factory
        self._mode = mode
        self._temperature = float(temperature)
        # Snapshot metadata \u2014 required for parallel evaluation. When
        # left as ``None`` the pool falls back to the default
        # (live-policy) snapshot path, which fails to pickle Flax
        # objects and is therefore only useful in serial mode.
        self._layer_sizes: Optional[Tuple[int, ...]] = (
            tuple(int(s) for s in layer_sizes) if layer_sizes is not None else None
        )
        self._activation = activation
        self._refresh_filled()

    def _refresh_filled(self) -> None:
        fitnesses = np.asarray(self._repertoire.fitnesses).reshape(-1)
        filled = np.nonzero(np.isfinite(fitnesses))[0]
        if filled.size == 0:
            raise ValueError(
                "QDaxRepertoirePool: repertoire is empty (no filled cells). "
                "Run at least one MAP-Elites iteration before wrapping it."
            )
        self._filled_idx = filled
        self._filled_fitness = fitnesses[filled]

    def update(self, population_like: Any) -> None:
        """Swap in a new repertoire (e.g. after a QDax iteration)."""
        self._repertoire = population_like
        self._refresh_filled()

    def __len__(self) -> int:
        return int(self._filled_idx.size)

    def _sample_cell(self, rng: np.random.Generator) -> int:
        if self._mode == "uniform":
            return int(rng.choice(self._filled_idx))
        logits = self._filled_fitness / max(self._temperature, 1e-8)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return int(rng.choice(self._filled_idx, p=probs))


    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        """Sample ``k`` cells and ship flat-params payloads for workers.

        Each snapshot carries the per-cell flat parameter vector plus
        ``layer_sizes`` / ``activation`` so workers can rebuild the
        :class:`FlaxPolicyAdapter` without touching the live
        repertoire (which holds non-picklable JAX device arrays).
        """
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}.")
        if self._layer_sizes is None:
            raise ValueError(
                "QDaxRepertoirePool.snapshot_opponents: pool was constructed "
                "without 'layer_sizes='. Co-evolution parallelism requires "
                "layer_sizes so workers can rebuild the Flax policy. "
                "Pass layer_sizes=... to the QDaxRepertoirePool constructor "
                "(typically the same tuple used to build the host module)."
            )
        import jax

        from deap_pettingzoo.contrib.qdax.policy import pytree_to_flat

        snaps: List[OpponentSnapshot] = []
        for _ in range(k):
            idx = self._sample_cell(rng)
            params = jax.tree_util.tree_map(
                lambda leaf, j=idx: np.asarray(leaf[j])
                if hasattr(leaf, "__getitem__") else leaf,
                self._repertoire.genotypes,
            )
            flat = pytree_to_flat(params, self._layer_sizes)
            snaps.append(
                OpponentSnapshot(
                    kind="qdax_flax",
                    payload={
                        "flat_params": np.asarray(flat, dtype=np.float32),
                        "layer_sizes": self._layer_sizes,
                        "activation": self._activation,
                    },
                )
            )
        return snaps
    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        import jax  # noqa: F401 \u2014 required for device_get below

        idx = self._sample_cell(rng)
        params = jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf[idx]) if hasattr(leaf, "__getitem__")
            else leaf,
            self._repertoire.genotypes,
        )
        return self._policy_factory(params)


# ---------------------------------------------------------------------------
# Side state
# ---------------------------------------------------------------------------


@dataclass
class _QDaxSideConfig:
    """Normalised view of ``Side.optimizer_config`` for QDax sides."""

    hidden_sizes: tuple
    activation: str
    batch_size: int
    grid_shape: tuple
    num_centroids: Optional[int]
    iso_sigma: float
    line_sigma: float
    sigma_g: float
    episode_length: int
    descriptor_name: str
    descriptor_kwargs: dict
    seed: int
    pool_mode: str
    pool_temperature: float


_DEFAULT_QDAX_SIDE_CONFIG: dict = {
    "hidden_sizes": (16, 16),
    "activation": "tanh",
    "batch_size": 16,
    "grid_shape": (8, 8),
    "num_centroids": None,
    "iso_sigma": 0.005,
    "line_sigma": 0.05,
    "sigma_g": 0.1,
    "episode_length": 25,
    "descriptor": "pursuit_evasion_2d",
    "descriptor_kwargs": {},
    # Optional callable ``(obs_dict, env_wrapper) -> Optional[np.ndarray]``
    # returning an ``(N, 2)`` position array for one timestep.
    # When ``None`` (default), descriptors fall back to the zero-centre
    # placeholder (all genotypes map to the same archive cell).  Supply a
    # position extractor — e.g. from
    # :mod:`deap_pettingzoo.contrib.qdax.position_extractors` — to enable
    # real MAP-Elites archive diversity.
    "position_extractor": None,
    "seed": 0,
    "pool_mode": "uniform",
    "pool_temperature": 1.0,
}


class QDaxSideState(SideState):
    """Co-evolving QDax MAP-Elites / CMA-ME side.

    Each :meth:`step_generation`:

    1. Emits ``batch_size`` candidate genotypes via the QDax emitter.
    2. Scores each by playing episodes against the opponent pool via
       the supplied ``evaluator_factory``.
    3. Adds them back into the repertoire (``map_elites.add``).
    4. Returns :class:`GenerationStats` summarising the batch.

    The opponent pool (:class:`QDaxRepertoirePool` by default) is
    refreshed with the new repertoire at the end of each generation
    so other sides always see an up-to-date opponent distribution.

    Descriptors are computed inside ``evaluator_factory`` by calling
    the configured descriptor on the captured trajectory. Because the
    co-evolution scaffold runs episodes sequentially, this path
    intentionally uses the numpy boundary (no ``jax.vmap`` /
    ``jax.lax.scan``).
    """

    def __init__(self, side: Side) -> None:
        _require_qdax()
        super().__init__(side)
        cfg = {**_DEFAULT_QDAX_SIDE_CONFIG, **(side.optimizer_config or {})}
        self._cfg_raw = cfg
        self._algorithm = side.optimizer  # "qdax_map_elites" or "qdax_cma_me"

        self._env: Optional[PettingZooWrapper] = None
        self._module = None
        self._layer_sizes: tuple = ()
        self._descriptor_fn = None
        self._map_elites = None
        self._repertoire = None
        self._emitter_state = None
        self._centroids = None
        self._init_genotypes = None
        self._key = None
        self._pool: Optional[QDaxRepertoirePool] = None
        self._last_batch_fitness = np.array([0.0])
        # Optional position extractor: (obs_dict, env_wrapper) -> Optional[np.ndarray(N,2)]
        self._position_extractor = cfg.get("position_extractor", None)

        # Placeholder evaluator used only during MAPElites.init() before the
        # first step_generation() call supplies a real evaluator factory.
        class _NullEvaluator:
            def evaluate(self, _policy) -> float:
                return 0.0

        self._current_evaluator: Any = _NullEvaluator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        import jax

        from qdax.core.map_elites import MAPElites
        from qdax.utils.metrics import default_qd_metrics

        from deap_pettingzoo.contrib.qdax.policy import (
            FlaxPolicyAdapter,
            flat_count,
            flat_to_pytree,
            init_flax_mlp_params,
            make_flax_mlp,
        )
        from deap_pettingzoo.contrib.qdax.descriptors import pursuit_evasion as pe
        from deap_pettingzoo.contrib.qdax.trainer import _compute_centroids

        self._env = env
        # Shared action space across this side's agents.
        action_space = _resolve_side_action_space(env, self.side.role_prefixes)
        cfg = self._cfg_raw
        agent = [
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        ][0]
        obs_space = env.observation_space(agent)
        in_dim = int(np.prod(obs_space.shape))
        from deap_pettingzoo.policies.action_heads import required_output_size

        out_dim = required_output_size(action_space)
        layer_sizes = (in_dim, *cfg["hidden_sizes"], out_dim)
        self._layer_sizes = layer_sizes
        self._module = make_flax_mlp(layer_sizes, activation=cfg["activation"])
        self._action_space = action_space

        # Descriptor
        side_agents = [
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        ]
        desc_kwargs = {
            "num_pursuers": len(side_agents),
            "num_evaders": max(1, len(env.possible_agents) - len(side_agents)),
            **cfg["descriptor_kwargs"],
        }
        if cfg["descriptor"] == "pursuit_evasion_2d":
            self._descriptor_fn = pe.formation_spread_and_mean_distance(**desc_kwargs)
        elif cfg["descriptor"] == "pursuit_evasion_3d":
            self._descriptor_fn = pe.with_capture_efficiency(**desc_kwargs)
        else:
            raise ValueError(
                f"QDaxSideState: unknown descriptor {cfg['descriptor']!r}."
            )

        min_bd = tuple(self._descriptor_fn.min_bd)
        max_bd = tuple(self._descriptor_fn.max_bd)

        self._key = jax.random.PRNGKey(int(cfg["seed"]))
        centroids, self._key = _compute_centroids(
            min_bd, max_bd, cfg["grid_shape"], cfg["num_centroids"], self._key
        )
        self._centroids = centroids

        # Build emitter
        if self._algorithm == "qdax_map_elites":
            from qdax.core.emitters.mutation_operators import isoline_variation
            from qdax.core.emitters.standard_emitters import MixingEmitter

            variation_fn = lambda x1, x2, k: isoline_variation(  # noqa: E731
                x1, x2, k,
                iso_sigma=float(cfg["iso_sigma"]),
                line_sigma=float(cfg["line_sigma"]),
            )
            emitter = MixingEmitter(
                mutation_fn=lambda x, k: (x, k),
                variation_fn=variation_fn,
                variation_percentage=1.0,
                batch_size=int(cfg["batch_size"]),
            )
            self._genotype_layout = "pytree"
        else:  # qdax_cma_me
            from qdax.core.emitters.cma_improvement_emitter import (
                CMAImprovementEmitter,
            )

            emitter = CMAImprovementEmitter(
                batch_size=int(cfg["batch_size"]),
                genotype_dim=flat_count(layer_sizes),
                centroids=centroids,
                sigma_g=float(cfg["sigma_g"]),
            )
            self._genotype_layout = "flat"

        self._map_elites = MAPElites(
            scoring_function=self._jit_safe_scoring,
            emitter=emitter,
            metrics_function=lambda rep: default_qd_metrics(rep, qd_offset=0.0),
        )

        # Build initial population.
        self._key, sub = jax.random.split(self._key)
        init_params = init_flax_mlp_params(
            self._module, input_size=in_dim, seed=cfg["seed"]
        )
        if self._genotype_layout == "pytree":
            leaves, treedef = jax.tree_util.tree_flatten(init_params)

            def _perturb_one(k, leaf):
                return leaf + 0.01 * jax.random.normal(k, leaf.shape)

            batch = int(cfg["batch_size"])
            keys_batch = jax.random.split(sub, batch)

            def _one(k):
                leaf_keys = jax.random.split(k, len(leaves))
                return jax.tree_util.tree_unflatten(
                    treedef,
                    [
                        _perturb_one(lk, leaf)
                        for lk, leaf in zip(leaf_keys, leaves)
                    ],
                )

            init_genotypes = jax.vmap(_one)(keys_batch)
        else:
            init_genotypes = 0.01 * jax.random.normal(
                sub, (int(cfg["batch_size"]), flat_count(layer_sizes))
            )
        self._init_genotypes = init_genotypes

        self._key, sub = jax.random.split(self._key)
        (
            self._repertoire,
            self._emitter_state,
            _metrics,
        ) = self._map_elites.init(init_genotypes, centroids, sub)

        # Opponent pool for other sides
        def _factory(params):
            if self._genotype_layout == "flat":
                params = flat_to_pytree(np.asarray(params), layer_sizes)
            return FlaxPolicyAdapter(
                self._module, params, action_space=action_space
            )

        self._policy_factory = _factory
        self._pool = QDaxRepertoirePool(
            self._repertoire,
            _factory,
            mode=cfg["pool_mode"],
            temperature=cfg["pool_temperature"],
            layer_sizes=layer_sizes,
            activation=cfg["activation"],
        )

    def rebind_evaluator(self, evaluator: Any) -> None:
        """Bind the real coevolution evaluator for bootstrap/generation scoring."""
        self._current_evaluator = evaluator

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        """Re-evaluate the initial QDax repertoire with the real evaluator.

        ``MAPElites.init()`` has to run during :meth:`initialise` so other
        sides can sample an initial QDax opponent pool. At that point the
        coevolution evaluator does not exist yet, so the temporary repertoire
        is seeded through a null evaluator. The trainer calls this bootstrap
        hook once real opponent pools are available; replacing the temporary
        repertoire here prevents unevaluated ``0.0`` cells from contaminating
        archive statistics and opponent sampling.
        """
        del batch_eval_fn
        import jax

        if self._map_elites is None or self._init_genotypes is None:
            return
        if self._centroids is None:
            return
        self._key, sub = jax.random.split(self._key)
        (
            self._repertoire,
            self._emitter_state,
            _metrics,
        ) = self._map_elites.init(self._init_genotypes, self._centroids, sub)
        if self._pool is not None:
            self._pool.update(self._repertoire)

    # ------------------------------------------------------------------
    # Scoring closure
    # ------------------------------------------------------------------

    def _evaluate_one_genotype(self, params_or_flat) -> tuple:
        """Play one episode against the opponent pool; return (fitness, descriptor).

        When the injected evaluator supports :attr:`trajectory_hook` **and**
        a ``position_extractor`` has been configured, the descriptor is
        computed from real per-step position data.  Otherwise the method
        falls back to the zero-centre placeholder (all genotypes land in the
        same archive cell, which degenerates MAP-Elites to a ``(1+batch)``
        evolution strategy).

        To enable real descriptors, set ``position_extractor`` in the side's
        ``optimizer_config``::

            from deap_pettingzoo.contrib.qdax.position_extractors import obs_first_two
            Side(
                optimizer="qdax_map_elites",
                optimizer_config={"position_extractor": obs_first_two, ...},
            )

        For environments whose observation vectors do not encode raw
        positions (e.g. marl-aquarium), use
        :func:`~deap_pettingzoo.contrib.qdax.position_extractors.make_aquarium_extractor`
        or supply a custom callable.
        """
        import jax

        from deap_pettingzoo.contrib.qdax.policy import (
            FlaxPolicyAdapter,
            flat_to_pytree,
        )

        if self._genotype_layout == "flat":
            params = flat_to_pytree(np.asarray(params_or_flat), self._layer_sizes)
        else:
            params = jax.tree_util.tree_map(
                lambda x: np.asarray(x), params_or_flat
            )
        policy = FlaxPolicyAdapter(
            self._module,
            params,
            action_space=self._action_space,
        )

        # Trajectory collection: accumulate per-step (N, 2) position arrays
        # if the evaluator exposes a trajectory_hook slot.
        positions_list: List[np.ndarray] = []
        has_hook = (
            self._position_extractor is not None
            and hasattr(self._current_evaluator, "trajectory_hook")
        )
        if has_hook:
            extractor = self._position_extractor

            def _hook(obs_dict, env_wrapper):
                try:
                    pos = extractor(obs_dict, env_wrapper)
                    if pos is not None:
                        positions_list.append(np.asarray(pos, dtype=np.float32))
                except Exception:
                    pass

            self._current_evaluator.trajectory_hook = _hook

        try:
            fitness = float(self._current_evaluator.evaluate(policy))
        finally:
            if has_hook:
                self._current_evaluator.trajectory_hook = None

        # Compute descriptor from collected positions, or fall back.
        positions = _stack_consistent_positions(positions_list)
        if positions is not None:
            try:
                descriptor = np.asarray(
                    self._descriptor_fn({"positions": positions}),
                    dtype=np.float32,
                )
            except Exception:
                descriptor = self._zero_descriptor()
        else:
            descriptor = self._zero_descriptor()

        return fitness, descriptor

    def _zero_descriptor(self) -> np.ndarray:
        """Return the centre of the descriptor range as a placeholder."""
        mn = np.asarray(self._descriptor_fn.min_bd, dtype=np.float32)
        mx = np.asarray(self._descriptor_fn.max_bd, dtype=np.float32)
        return 0.5 * (mn + mx)

    def _jit_safe_scoring(self, genotypes, key):
        import jax
        import jax.numpy as jnp

        # Host-side loop; QDax tolerates this when scoring is not jit-wrapped.
        leaves, _ = jax.tree_util.tree_flatten(genotypes)
        batch = int(leaves[0].shape[0])

        executor = getattr(self._current_evaluator, "_executor", None)
        if executor is not None and batch > 0:
            fitnesses, descs = self._parallel_scoring(genotypes, batch)
            # Must mirror the serial branch: update _last_batch_fitness so
            # step_generation() reports the current batch in its stats.
            self._last_batch_fitness = np.asarray(fitnesses, dtype=np.float64)
        else:
            fitnesses = []
            descs = []
            for i in range(batch):
                one = jax.tree_util.tree_map(lambda leaf, j=i: leaf[j], genotypes)
                f, d = self._evaluate_one_genotype(one)
                fitnesses.append(f)
                descs.append(d)
            self._last_batch_fitness = np.asarray(fitnesses, dtype=np.float64)

        return jnp.asarray(fitnesses), jnp.asarray(np.stack(descs)), {}

    def _parallel_scoring(
        self, genotypes: Any, batch: int
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Dispatch all genotypes to the evaluator's ProcessPoolExecutor.

        Returns ``(fitnesses, descriptors)`` — two plain Python lists with one
        entry per genotype — so ``_jit_safe_scoring`` can forward them to JAX.
        """
        import jax

        from deap_pettingzoo.contrib.qdax.policy import pytree_to_flat

        evaluator = self._current_evaluator
        executor = evaluator._executor
        rng = evaluator._rng_factory()
        k = evaluator._k

        # Convert each genotype in the JAX batch to a flat numpy weight vector.
        flat_weights_list: List[np.ndarray] = []
        for i in range(batch):
            one = jax.tree_util.tree_map(lambda leaf, j=i: leaf[j], genotypes)
            if self._genotype_layout == "flat":
                flat = np.asarray(one, dtype=np.float32).ravel()
            else:
                flat = pytree_to_flat(
                    jax.tree_util.tree_map(np.asarray, one),
                    self._layer_sizes,
                )
            flat_weights_list.append(flat)

        # Sample opponents from each side's pool up-front (requires picklable
        # snapshots; EAPool → "feedforward", QDaxRepertoirePool → "qdax_flax").
        side_role_prefixes: Dict[str, Tuple[str, ...]] = {}
        per_side_snaps: Dict[str, List[Any]] = {}
        for side_name, prefixes, pool in evaluator._opponent_sides:
            side_role_prefixes[side_name] = tuple(prefixes)
            per_side_snaps[side_name] = pool.snapshot_opponents(rng, batch * k)

        # Build one payload per genotype.
        payloads: List[_QDaxGenotypePayload] = []
        for i in range(batch):
            per_episode: List[Dict[str, Any]] = []
            for ep in range(k):
                global_ep = i * k + ep
                per_episode.append(
                    {name: per_side_snaps[name][global_ep] for name in per_side_snaps}
                )
            payloads.append(
                _QDaxGenotypePayload(
                    env_name=evaluator._env_name,
                    env_config=evaluator._env_config,
                    flat_weights=flat_weights_list[i],
                    layer_sizes=self._layer_sizes,
                    activation=self._cfg_raw["activation"],
                    learner_role_prefixes=tuple(self.side.role_prefixes),
                    side_role_prefixes=side_role_prefixes,
                    per_episode_snapshots=per_episode,
                    k_opponents=k,
                    aggregation=evaluator._aggregation,
                    max_steps=evaluator._max_steps,
                    position_extractor=self._position_extractor,
                    descriptor_fn=self._descriptor_fn,
                )
            )

        results = list(executor.map(_qdax_genotype_rollout, payloads))
        return [r.fitness for r in results], [r.descriptor for r in results]

    # ------------------------------------------------------------------
    # SideState contract
    # ------------------------------------------------------------------

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        import jax

        self._current_evaluator = evaluator_factory(self)
        self._key, sub = jax.random.split(self._key)
        self._repertoire, self._emitter_state, _metrics = self._map_elites.update(
            self._repertoire, self._emitter_state, sub
        )
        # Refresh pool for other sides.
        try:
            self._pool.update(self._repertoire)
        except ValueError:
            pass  # still no filled cells; leave previous pool untouched

        self.generation += 1
        batch = np.asarray(self._last_batch_fitness, dtype=np.float64).reshape(-1)
        finite = batch[np.isfinite(batch)]
        if finite.size == 0:
            finite = np.array([0.0])
        archive_best, qd_score, coverage, filled_niches, total_niches = (
            summarise_qdax_archive(self._repertoire.fitnesses)
        )
        if not np.isfinite(archive_best):
            archive_best = float(finite.max())
        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(finite.mean()),
            std=float(finite.std()),
            min=float(finite.min()),
            max=float(finite.max()),
            best_fitness=float(finite.max()),
            qd_score=qd_score,
            coverage=coverage,
            archive_best_fitness=archive_best,
            filled_niches=filled_niches,
            total_niches=total_niches,
        )
        self.best_fitness = max(self.best_fitness, archive_best)
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        assert self._pool is not None, "call initialise() first"
        return self._pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        import jax

        fitnesses = np.asarray(self._repertoire.fitnesses).reshape(-1)
        idx = int(np.argmax(np.where(np.isfinite(fitnesses), fitnesses, -np.inf)))
        params = jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf[idx]), self._repertoire.genotypes
        )
        return self._policy_factory(params)

    def best_snapshot(self) -> Any:
        import jax

        fitnesses = np.asarray(self._repertoire.fitnesses).reshape(-1)
        idx = int(np.argmax(np.where(np.isfinite(fitnesses), fitnesses, -np.inf)))
        return jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf[idx]), self._repertoire.genotypes
        )


__all__ = ["QDaxRepertoirePool", "QDaxSideState"]
