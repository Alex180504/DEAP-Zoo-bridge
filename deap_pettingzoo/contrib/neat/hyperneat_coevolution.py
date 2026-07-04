"""HyperNEAT and ES-HyperNEAT co-evolution side states.

:class:`HyperNeatSideState` and :class:`ESHyperNeatSideState` extend
:class:`NeatSideState` — the only difference is the phenotype
construction step (HyperNEAT substrate-query vs ES-HyperNEAT quadtree).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.coevolution.pools import (
    OpponentPool,
    OpponentSnapshot,
)
from deap_pettingzoo.baselines.coevolution.sides import (
    GenerationStats,
    Side,
    SideState,
    _resolve_side_action_space,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS
from ._pureples.substrate import Substrate
from .coevolution import (
    NeatSideState,
    _NeatPopulationPool,
    _restore_random_state,
    _save_random_state,
)
from .config import build_cppn_config
from .hyperneat import ESHyperNeatPolicy, HyperNeatPolicy
from .substrate import substrate_for_env


class _HyperNeatPopulationPool(_NeatPopulationPool):
    """Population pool for HyperNEAT genomes.

    Overrides :meth:`snapshot_opponents` to emit ``kind='hyperneat'``
    snapshots carrying ``env_obs_size`` / ``env_act_size`` so workers
    call :func:`~deap_pettingzoo.baselines.coevolution.pools.rebuild_opponent_policy`
    with the correct substrate dimensions rather than treating them as
    plain NEAT policies (which would cause an obs-size mismatch on
    networks with CPPN geometry).
    """

    def __init__(
        self,
        genomes_by_id,
        policy_factory,
        neat_config,
        env_obs_size: int,
        env_act_size: int,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__(genomes_by_id, policy_factory, neat_config)
        self._env_obs_size = env_obs_size
        self._env_act_size = env_act_size
        self._activation = activation

    def snapshot_opponents(self, rng, k):
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}")
        snapshots = []
        for _ in range(k):
            idx = rng.integers(0, len(self._genomes))
            genome = self._genomes[idx]
            snapshots.append(
                OpponentSnapshot(
                    kind="hyperneat",
                    payload={
                        "genome": genome,
                        "env_obs_size": self._env_obs_size,
                        "env_act_size": self._env_act_size,
                        "activation": self._activation,
                    },
                )
            )
        return snapshots


class _ESHyperNeatPopulationPool(_NeatPopulationPool):
    """Population pool for ES-HyperNEAT genomes.

    Same rationale as :class:`_HyperNeatPopulationPool` but emits
    ``kind='es_hyperneat'`` snapshots that also carry the ES-HyperNEAT
    parameter dict.
    """

    def __init__(
        self,
        genomes_by_id,
        policy_factory,
        neat_config,
        env_obs_size: int,
        env_act_size: int,
        es_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(genomes_by_id, policy_factory, neat_config)
        self._env_obs_size = env_obs_size
        self._env_act_size = env_act_size
        self._es_params = dict(es_params) if es_params else {}

    def snapshot_opponents(self, rng, k):
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}")
        snapshots = []
        for _ in range(k):
            idx = rng.integers(0, len(self._genomes))
            genome = self._genomes[idx]
            snapshots.append(
                OpponentSnapshot(
                    kind="es_hyperneat",
                    payload={
                        "genome": genome,
                        "env_obs_size": self._env_obs_size,
                        "env_act_size": self._env_act_size,
                        "es_params": self._es_params,
                    },
                )
            )
        return snapshots


class HyperNeatSideState(NeatSideState):
    """HyperNEAT co-evolution side.

    Same as :class:`NeatSideState` but uses :class:`HyperNeatPolicy`
    for phenotype construction.

    Extra ``optimizer_config`` keys (vs NeatSideState):

    * ``substrate_hidden_layers`` — optional ``list[int]`` of hidden
      layer sizes for the HyperNEAT substrate.
    * ``phenotype_activation`` — activation for the phenotype network
      (default ``"sigmoid"``).
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self._hidden_layers = cfg.get("substrate_hidden_layers", None)
        self._phenotype_activation = cfg.get("phenotype_activation", "sigmoid")
        self._substrate: Optional[Substrate] = None

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._env = env
        if rng is not None:
            self._rng = rng
        elif self.seed is not None:
            self._rng = np.random.default_rng(int(self.seed))

        self._action_space = _resolve_side_action_space(env, self.side.role_prefixes)

        agents = [
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        ]
        obs_shape = env.observation_space(agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(self._action_space)

        # Stored so current_pool() can embed them in snapshots shipped to workers.
        self._env_obs_size = input_size
        self._env_act_size = output_size

        # Build substrate.
        self._substrate = substrate_for_env(
            input_size,
            output_size,
            layout="uniform",
            hidden_layers=self._hidden_layers,
        )

        # Build CPPN config (5 inputs, 1 output weight).
        genome_overrides = dict(self.neat_overrides.get("DefaultGenome", {}))
        genome_overrides["num_hidden"] = self.num_hidden
        overrides = dict(self.neat_overrides)
        overrides["DefaultGenome"] = genome_overrides

        self._neat_config = build_cppn_config(
            input_size=5,
            output_size=1,
            pop_size=self.pop_size,
            seed=self.seed,
            overrides=overrides if overrides else None,
        )

        self._population_obj = neat.Population(
            self._neat_config,
            seed=self.seed if self.seed is not None else None,
        )
        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        if self._population_obj is None:
            raise RuntimeError("HyperNeatSideState: initialise() must be called first.")

        evaluator = evaluator_factory(self)
        env: PettingZooWrapper = getattr(evaluator, "_env", self._env)

        self._gen_env_steps = 0

        def _fitness_func(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
            for _genome_id, genome in genomes_and_ids:
                policy = HyperNeatPolicy(
                    genome,
                    neat_cfg,
                    self._substrate,  # type: ignore[arg-type]
                    action_space=self._action_space,
                    activation=self._phenotype_activation,
                )
                rewards: list[float] = []
                for _ in range(self.eval_eps):
                    observations = env.reset()
                    ep_rew = 0.0
                    done = False
                    _steps = 0
                    while not done and _steps < 500:
                        actions = {}
                        for agent in observations:
                            actions[agent] = policy.get_action(observations[agent])
                        observations, s_rewards, dones, truncateds, _ = env.step(
                            actions
                        )
                        ep_rew += float(sum(s_rewards.values()))
                        done = all(dones.values()) or all(truncateds.values())
                        _steps += 1
                    self._gen_env_steps += _steps
                    rewards.append(ep_rew)
                genome.fitness = float(np.mean(rewards))

        old_random_state = _save_random_state()
        try:
            for _ in range(self.n_gens_per_step):
                self._population_obj.run(_fitness_func, n=1)
                self._n_generations_run += 1
        except neat.CompleteExtinctionException:
            pass
        finally:
            _restore_random_state(old_random_state)

        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

        fitnesses = [
            g.fitness for g in self._genomes_by_id.values() if g.fitness is not None
        ]
        if not fitnesses:
            fitnesses = [0.0]

        best_fit = float(np.max(fitnesses))
        self.best_fitness = max(self.best_fitness, best_fit)
        self.generation += 1

        return GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(np.mean(fitnesses)),
            std=float(np.std(fitnesses)),
            min=float(np.min(fitnesses)),
            max=float(np.max(fitnesses)),
            best_fitness=best_fit,
            env_steps=int(self._gen_env_steps),
        )

    def current_pool(self) -> OpponentPool:
        if not self._genomes_by_id or self._neat_config is None:
            raise RuntimeError("HyperNeatSideState: initialise() must be called first.")

        def _factory(genome: neat.DefaultGenome) -> PolicyProtocol:
            return HyperNeatPolicy(
                genome,
                self._neat_config,  # type: ignore[arg-type]
                self._substrate,  # type: ignore[arg-type]
                action_space=self._action_space,
                activation=self._phenotype_activation,
            )

        # Use _HyperNeatPopulationPool so snapshot_opponents emits
        # kind='hyperneat' (not 'neat'), giving parallel workers the
        # correct substrate dimensions to rebuild the policy.
        return _HyperNeatPopulationPool(
            genomes_by_id=dict(self._genomes_by_id),
            policy_factory=_factory,
            neat_config=self._neat_config,  # type: ignore[arg-type]
            env_obs_size=self._env_obs_size,
            env_act_size=self._env_act_size,
            activation=self._phenotype_activation,
        )

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        if self._best_genome is None or self._neat_config is None:
            raise RuntimeError("HyperNeatSideState: no best genome available.")
        return HyperNeatPolicy(
            self._best_genome,
            self._neat_config,
            self._substrate,  # type: ignore[arg-type]
            action_space=self._action_space,
            activation=self._phenotype_activation,
        )


class ESHyperNeatSideState(NeatSideState):
    """ES-HyperNEAT co-evolution side.

    Same as :class:`NeatSideState` but uses :class:`ESHyperNeatPolicy`.

    Extra ``optimizer_config`` keys:

    * ``es_params`` — ES-HyperNEAT parameter dict (default
      :data:`~._pureples.es_hyperneat.ES_HYPERNEAT_PARAMS_DEFAULTS`).
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self._es_params = dict(cfg.get("es_params", ES_HYPERNEAT_PARAMS_DEFAULTS))
        self._substrate: Optional[Substrate] = None

    def initialise(
        self,
        env: PettingZooWrapper,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._env = env
        if rng is not None:
            self._rng = rng
        elif self.seed is not None:
            self._rng = np.random.default_rng(int(self.seed))

        self._action_space = _resolve_side_action_space(env, self.side.role_prefixes)

        agents = [
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        ]
        obs_shape = env.observation_space(agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(self._action_space)

        # Stored so current_pool() can embed them in snapshots shipped to workers.
        self._env_obs_size = input_size
        self._env_act_size = output_size

        # ES-HyperNEAT substrate: no hidden nodes.
        self._substrate = substrate_for_env(input_size, output_size, layout="uniform")

        genome_overrides = dict(self.neat_overrides.get("DefaultGenome", {}))
        genome_overrides["num_hidden"] = self.num_hidden
        overrides = dict(self.neat_overrides)
        overrides["DefaultGenome"] = genome_overrides

        self._neat_config = build_cppn_config(
            input_size=5,
            output_size=1,
            pop_size=self.pop_size,
            seed=self.seed,
            overrides=overrides if overrides else None,
        )

        self._population_obj = neat.Population(
            self._neat_config,
            seed=self.seed if self.seed is not None else None,
        )
        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        if self._population_obj is None:
            raise RuntimeError(
                "ESHyperNeatSideState: initialise() must be called first."
            )

        evaluator = evaluator_factory(self)
        env: PettingZooWrapper = getattr(evaluator, "_env", self._env)

        self._gen_env_steps = 0

        def _fitness_func(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
            for _genome_id, genome in genomes_and_ids:
                policy = ESHyperNeatPolicy(
                    genome,
                    neat_cfg,
                    self._substrate,  # type: ignore[arg-type]
                    action_space=self._action_space,
                    es_params=self._es_params,
                )
                rewards: list[float] = []
                for _ in range(self.eval_eps):
                    observations = env.reset()
                    ep_rew = 0.0
                    done = False
                    _steps = 0
                    while not done and _steps < 500:
                        actions = {}
                        for agent in observations:
                            actions[agent] = policy.get_action(observations[agent])
                        observations, s_rewards, dones, truncateds, _ = env.step(
                            actions
                        )
                        ep_rew += float(sum(s_rewards.values()))
                        done = all(dones.values()) or all(truncateds.values())
                        _steps += 1
                    self._gen_env_steps += _steps
                    rewards.append(ep_rew)
                genome.fitness = float(np.mean(rewards))

        old_random_state = _save_random_state()
        try:
            for _ in range(self.n_gens_per_step):
                self._population_obj.run(_fitness_func, n=1)
                self._n_generations_run += 1
        except neat.CompleteExtinctionException:
            pass
        finally:
            _restore_random_state(old_random_state)

        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

        fitnesses = [
            g.fitness for g in self._genomes_by_id.values() if g.fitness is not None
        ]
        if not fitnesses:
            fitnesses = [0.0]

        best_fit = float(np.max(fitnesses))
        self.best_fitness = max(self.best_fitness, best_fit)
        self.generation += 1

        return GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=float(np.mean(fitnesses)),
            std=float(np.std(fitnesses)),
            min=float(np.min(fitnesses)),
            max=float(np.max(fitnesses)),
            best_fitness=best_fit,
            env_steps=int(self._gen_env_steps),
        )

    def current_pool(self) -> OpponentPool:
        if not self._genomes_by_id or self._neat_config is None:
            raise RuntimeError(
                "ESHyperNeatSideState: initialise() must be called first."
            )

        def _factory(genome: neat.DefaultGenome) -> PolicyProtocol:
            return ESHyperNeatPolicy(
                genome,
                self._neat_config,  # type: ignore[arg-type]
                self._substrate,  # type: ignore[arg-type]
                action_space=self._action_space,
                es_params=self._es_params,
            )

        # Use _ESHyperNeatPopulationPool so snapshot_opponents emits
        # kind='es_hyperneat' with correct substrate dimensions and es_params.
        return _ESHyperNeatPopulationPool(
            genomes_by_id=dict(self._genomes_by_id),
            policy_factory=_factory,
            neat_config=self._neat_config,  # type: ignore[arg-type]
            env_obs_size=self._env_obs_size,
            env_act_size=self._env_act_size,
            es_params=self._es_params,
        )

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        if self._best_genome is None or self._neat_config is None:
            raise RuntimeError("ESHyperNeatSideState: no best genome available.")
        return ESHyperNeatPolicy(
            self._best_genome,
            self._neat_config,
            self._substrate,  # type: ignore[arg-type]
            action_space=self._action_space,
            es_params=self._es_params,
        )
