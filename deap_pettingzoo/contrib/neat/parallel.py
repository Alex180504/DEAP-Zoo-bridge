"""Parallel evaluation for NEAT-family trainers.

Mirrors the EA :class:`~deap_pettingzoo.integration.deap_integration._EvalWorkerSpec`
pattern: ship picklable genome + environment metadata to worker processes,
rebuild env + policy on each worker, run episodes, return fitness.

Data flow
---------
Driver:  genome ──pickle──→ Worker: rebuild genome via pickle
         _NeatWorkerSpec ──→        rebuild neat.Config from input/output sizes
                                    rebuild action space from hint
                                    build PettingZooWrapper from env_name
                                    build NeatPolicy / HyperNeatPolicy / ESHyperNeatPolicy
                                    run eval_eps episodes → return mean fitness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class _NeatWorkerSpec:
    """Picklable spec enabling a worker process to evaluate one NEAT genome.

    All fields are shared across every genome in a population — only the
    genome itself varies per worker call.

    Attributes
    ----------
    env_name:
        Registered env name (resolved via registry snapshot on worker).
    env_config:
        Env kwargs forwarded to ``PettingZooWrapper``.
    input_size:
        Observation dimension for ``build_neat_config``.
    output_size:
        Action dimension for ``build_neat_config``.
    action_space_type:
        ``"discrete"`` or ``"continuous"``.
    action_space_args:
        Type-specific args: ``{"n": int}`` for discrete,
        ``{"shape": tuple, "low": list, "high": list}`` for continuous.
    eval_episodes_per_genome:
        Number of episodes to average over.
    seed:
        Base seed for episode RNG (varied per episode in worker).
    substrate_spec:
        If non-``None``, worker builds a HyperNEAT phenotype policy
        instead of a plain NeatPolicy.  Keys:
        ``input_coordinates``, ``output_coordinates``,
        ``hidden_coordinates`` (optional).
    es_params:
        If non-``None``, worker builds an ES-HyperNEAT phenotype.
        Forwarded directly to ``ESHyperNeatPolicy``.
    phenotype_activation:
        Phenotype activation function name (HyperNEAT only, default ``"sigmoid"``).
    weight_threshold:
        Minimum absolute CPPN output below which phenotype connection
        weights are zeroed.  Default 0.1 (HyperNEAT / ES-HyperNEAT).
    """

    env_name: str
    env_config: Dict[str, Any]
    input_size: int
    output_size: int
    action_space_type: str
    action_space_args: Dict[str, Any] = field(default_factory=dict)
    eval_episodes_per_genome: int = 1
    seed: int = 0
    substrate_spec: Optional[Dict[str, Any]] = None
    es_params: Optional[Dict[str, Any]] = None
    phenotype_activation: str = "sigmoid"
    weight_threshold: float = 0.1
    #: Registry snapshot replayed once per worker so SCOOP/Dask workers (and
    #: any worker missing a runtime registration) can resolve ``env_name``.
    registry_snapshot: Optional[List] = None


def _rebuild_action_space(spec: _NeatWorkerSpec) -> Any:
    """Rebuild a gymnasium Space from *spec*'s action-space metadata."""
    import gymnasium.spaces as spaces

    if spec.action_space_type == "discrete":
        n = int(spec.action_space_args.get("n", spec.output_size))
        return spaces.Discrete(n)
    if spec.action_space_type == "multidiscrete":
        nvec = spec.action_space_args.get("nvec")
        if nvec is None:
            raise ValueError(
                "MultiDiscrete action_space_args must include 'nvec'."
            )
        return spaces.MultiDiscrete(np.asarray(nvec, dtype=np.int64))
    if spec.action_space_type == "continuous":
        shape = tuple(spec.action_space_args.get("shape", (spec.output_size,)))
        low = np.asarray(spec.action_space_args.get("low", -1.0), dtype=np.float32)
        high = np.asarray(spec.action_space_args.get("high", 1.0), dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    raise ValueError(
        f"Unknown action_space_type {spec.action_space_type!r}"
    )


def _rebuild_substrate(spec: _NeatWorkerSpec) -> Any:
    """Rebuild a :class:`~._pureples.substrate.Substrate` from *spec*."""
    from deap_pettingzoo.contrib.neat._pureples.substrate import Substrate

    inputs = [tuple(c) for c in spec.substrate_spec["input_coordinates"]]
    outputs = [tuple(c) for c in spec.substrate_spec["output_coordinates"]]
    hidden = spec.substrate_spec.get("hidden_coordinates", None)
    if hidden is not None:
        hidden = [[tuple(c) for c in layer] for layer in hidden]
    return Substrate(inputs, outputs, hidden)


def _neat_worker_evaluate(payload: tuple) -> float:
    """Worker-side genome evaluator for ``executor.map``.

    Parameters
    ----------
    payload:
        ``(genome: neat.DefaultGenome, spec: _NeatWorkerSpec)`` tuple.

    Returns
    -------
    float
        Mean episode reward across ``spec.eval_episodes_per_genome``
        episodes.
    """
    from deap_pettingzoo.contrib.neat.config import build_neat_config, build_cppn_config
    from deap_pettingzoo.contrib.neat.policy import NeatPolicy
    from deap_pettingzoo.environments import PettingZooWrapper
    from deap_pettingzoo.environments.registry import (
        replay_registry_snapshot_once,
    )

    genome, spec = payload

    # Re-register custom envs so SCOOP/Dask workers can resolve env_name; no-op
    # on repeat calls and for built-in-only snapshots.
    replay_registry_snapshot_once(spec.registry_snapshot)

    # Rebuild config — use CPPN config for HyperNEAT / ES-HyperNEAT,
    # standard config for plain NEAT.
    if spec.substrate_spec is not None:
        neat_cfg = build_cppn_config(
            input_size=spec.input_size,
            output_size=spec.output_size,
            seed=spec.seed,
        )
    else:
        neat_cfg = build_neat_config(
            input_size=spec.input_size,
            output_size=spec.output_size,
            seed=spec.seed,
        )

    # Rebuild action space and policy.
    action_space = _rebuild_action_space(spec)

    # HyperNEAT / ES-HyperNEAT logic.
    if spec.substrate_spec is not None:
        from deap_pettingzoo.contrib.neat.hyperneat import (
            HyperNeatPolicy,
            ESHyperNeatPolicy,
        )

        substrate = _rebuild_substrate(spec)
        if spec.es_params is not None:
            policy = ESHyperNeatPolicy(
                genome, neat_cfg, substrate,
                action_space=action_space,
                es_params=dict(spec.es_params),
            )
        else:
            policy = HyperNeatPolicy(
                genome, neat_cfg, substrate,
                action_space=action_space,
                activation=spec.phenotype_activation,
                weight_threshold=spec.weight_threshold,
            )
    else:
        policy = NeatPolicy(genome, neat_cfg, action_space=action_space)

    # Build env and evaluate.
    env = PettingZooWrapper(
        spec.env_name,
        env_config=spec.env_config,
    )
    try:
        rewards: list[float] = []
        for ep_i in range(spec.eval_episodes_per_genome):
            ep_seed = spec.seed * 10_000 + ep_i
            observations = env.reset(seed=ep_seed)
            ep_rew = 0.0
            done = False
            _steps = 0
            while not done and _steps < 500:
                actions = {}
                for agent in observations:
                    actions[agent] = policy.get_action(observations[agent])
                observations, step_rewards, dones, truncateds, _infos = (
                    env.step(actions)
                )
                ep_rew += float(sum(step_rewards.values()))
                done = all(dones.values()) or all(truncateds.values())
                _steps += 1
            rewards.append(ep_rew)

        return float(np.mean(rewards))
    finally:
        try:
            env.close()
        except Exception:
            pass


def make_neat_parallel_evaluate(
    executor: Any,
    worker_spec: _NeatWorkerSpec,
) -> "Any":
    """Return a callable suitable as a neat fitness function.

    The returned callable has the signature expected by
    :meth:`neat.Population.run`'s ``fitness_function`` parameter:
    ``(genomes_and_ids: list, config: neat.Config) -> None``.
    Genomes are shipped to workers via *executor*; the driver's
    ``neat.Config`` is ignored (workers rebuild their own).

    Parameters
    ----------
    executor:
        A context-managed executor from :func:`~deap_pettingzoo.utils.parallel.make_executor`.
    worker_spec:
        Shared spec describing the env, action space, and eval budget.

    Returns
    -------
    callable
        ``fitness_function(genomes_and_ids, neat_config) -> None``
    """

    def _parallel_fitness(
        genomes_and_ids: list, _neat_cfg: Any,
    ) -> None:
        """neat fitness-function contract — parallelised."""
        # Build payloads: one (genome, spec) per genome.
        payloads = [(genome, worker_spec) for _genome_id, genome in genomes_and_ids]

        # Evaluate in parallel.  executor.map preserves order.
        fitnesses: list = list(executor.map(_neat_worker_evaluate, payloads))

        # Assign fitnesses back to genomes.
        for (_genome_id, genome), fitness in zip(genomes_and_ids, fitnesses):
            genome.fitness = float(fitness)

    return _parallel_fitness
