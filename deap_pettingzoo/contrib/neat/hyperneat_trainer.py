"""HyperNEAT and ES-HyperNEAT trainers.

:class:`HyperNEATTrainer` and :class:`ESHyperNEATTrainer` subclass
:class:`NEATTrainer` — the only difference is the phenotype construction
step.  The underlying NEAT population still evolves CPPN genomes; what
changes is how those genomes are turned into executable policies.

Both trainers accept the same ``algorithm_params`` as
:class:`NEATTrainer`, plus:

* ``substrate`` — a pre-built :class:`~.substrate.Substrate`, or
  ``None`` to auto-build one from the env's observation/action sizes.
* ``substrate_layout`` — ``"uniform"`` (default) or ``"2d_grid"``.
* ``substrate_hidden_layers`` — optional list of int layer sizes.
* ``es_params`` (ES-HyperNEAT only) — ES parameter dict.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

import neat  # guarded by _require_neat

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines._logging_helpers import (
    build_logger,
    dump_fitness_history,
    write_summary,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

from .config import build_cppn_config
from .substrate import Substrate, substrate_for_env
from .hyperneat import HyperNeatPolicy, ESHyperNeatPolicy
from .trainer import NEATTrainer
from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS


class HyperNEATTrainer(NEATTrainer):
    """NEAT trainer that builds HyperNEAT phenotype policies."""

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["algorithm"] = "HyperNEAT"
        return caps

    def train(self, config: TrainingConfig) -> TrainingResult:
        t0 = time.perf_counter()
        params = dict(config.algorithm_params)

        pop_size = int(params.get("pop_size", 50))
        n_generations = params.get("n_generations", None)
        if n_generations is None:
            n_generations = self._budget_to_generations(config, pop_size)

        # --- Build env ---------------------------------------------------
        env = PettingZooWrapper(
            config.env_name,
            env_config=config.env_config,
            render_mode="rgb_array" if config.render else None,
        )
        shared_action_space = env.action_space(env.possible_agents[0])
        obs_shape = env.observation_space(env.possible_agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(shared_action_space)

        # --- Substrate ---------------------------------------------------
        substrate = params.get("substrate", None)
        if substrate is None:
            hidden_layers = params.get("substrate_hidden_layers", None)
            substrate = substrate_for_env(
                input_size,
                output_size,
                layout=params.get("substrate_layout", "uniform"),
                hidden_layers=hidden_layers,
            )

        # --- CPPN config (5 inputs: x1,y1,x2,y2,bias; 1 output: weight) -
        cppn_input_size = 5
        cppn_output_size = 1
        overrides = dict(params.get("neat_config_overrides", {}))
        seed = params.get("seed", None)
        neat_config = build_cppn_config(
            input_size=cppn_input_size,
            output_size=cppn_output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        # --- Create population -------------------------------------------
        pop = neat.Population(neat_config, seed=seed)

        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        activation = params.get("phenotype_activation", "sigmoid")

        def _eval_genomes(
            genomes_and_ids: list, neat_cfg: neat.Config
        ) -> None:
            self._generation_num += 1
            gen_fitnesses: list[float] = []

            for _genome_id, genome in genomes_and_ids:
                policy = HyperNeatPolicy(
                    genome,
                    neat_cfg,
                    substrate,
                    action_space=shared_action_space,
                    activation=activation,
                )
                rewards: list[float] = []
                for _ in range(eval_eps):
                    observations = env.reset(seed=seed)
                    ep_rew = 0.0
                    done = False
                    _steps = 0
                    while not done and _steps < 500:
                        actions = {}
                        for agent in observations:
                            actions[agent] = policy.get_action(
                                observations[agent]
                            )
                        observations, s_rewards, dones, truncateds, _ = (
                            env.step(actions)
                        )
                        ep_rew += float(sum(s_rewards.values()))
                        done = all(dones.values()) or all(
                            truncateds.values()
                        )
                        _steps += 1
                    rewards.append(ep_rew)
                genome.fitness = float(np.mean(rewards))
                gen_fitnesses.append(genome.fitness)

            if gen_fitnesses:
                self._fitness_history.append(float(np.max(gen_fitnesses)))

        pop.add_reporter(neat.StdOutReporter(show_species_detail=False))

        try:
            best_genome = pop.run(_eval_genomes, n=n_generations)
        except neat.CompleteExtinctionException:
            warnings.warn("NEAT population went extinct.", stacklevel=2)
            best_genome = pop.best_genome

        champion = HyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            activation=activation,
        )

        elapsed = time.perf_counter() - t0
        self._log(config, best_genome, pop_size, elapsed)

        return TrainingResult(
            algorithm="HyperNEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
            },
            metadata={"genome": best_genome, "config": neat_config},
            seed=seed,
            training_time=elapsed,
        )


class ESHyperNEATTrainer(NEATTrainer):
    """NEAT trainer that builds ES-HyperNEAT phenotype policies."""

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["algorithm"] = "ES-HyperNEAT"
        return caps

    def train(self, config: TrainingConfig) -> TrainingResult:
        t0 = time.perf_counter()
        params = dict(config.algorithm_params)

        pop_size = int(params.get("pop_size", 50))
        n_generations = params.get("n_generations", None)
        if n_generations is None:
            n_generations = self._budget_to_generations(config, pop_size)

        env = PettingZooWrapper(
            config.env_name,
            env_config=config.env_config,
            render_mode="rgb_array" if config.render else None,
        )
        shared_action_space = env.action_space(env.possible_agents[0])
        obs_shape = env.observation_space(env.possible_agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(shared_action_space)

        # ES-HyperNEAT substrate: no hidden nodes.
        substrate = params.get("substrate", None)
        if substrate is None:
            substrate = substrate_for_env(
                input_size, output_size, layout="uniform"
            )

        cppn_input_size = 5
        cppn_output_size = 1
        overrides = dict(params.get("neat_config_overrides", {}))
        seed = params.get("seed", None)
        neat_config = build_cppn_config(
            input_size=cppn_input_size,
            output_size=cppn_output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        pop = neat.Population(neat_config, seed=seed)
        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        es_params = dict(
            params.get("es_params", ES_HYPERNEAT_PARAMS_DEFAULTS)
        )

        def _eval_genomes(
            genomes_and_ids: list, neat_cfg: neat.Config
        ) -> None:
            self._generation_num += 1
            gen_fitnesses: list[float] = []

            for _genome_id, genome in genomes_and_ids:
                policy = ESHyperNeatPolicy(
                    genome,
                    neat_cfg,
                    substrate,
                    action_space=shared_action_space,
                    es_params=es_params,
                )
                rewards: list[float] = []
                for _ in range(eval_eps):
                    observations = env.reset(seed=seed)
                    ep_rew = 0.0
                    done = False
                    _steps = 0
                    while not done and _steps < 500:
                        actions = {}
                        for agent in observations:
                            actions[agent] = policy.get_action(
                                observations[agent]
                            )
                        observations, s_rewards, dones, truncateds, _ = (
                            env.step(actions)
                        )
                        ep_rew += float(sum(s_rewards.values()))
                        done = all(dones.values()) or all(
                            truncateds.values()
                        )
                        _steps += 1
                    rewards.append(ep_rew)
                genome.fitness = float(np.mean(rewards))
                gen_fitnesses.append(genome.fitness)

            if gen_fitnesses:
                self._fitness_history.append(float(np.max(gen_fitnesses)))

        pop.add_reporter(neat.StdOutReporter(show_species_detail=False))

        try:
            best_genome = pop.run(_eval_genomes, n=n_generations)
        except neat.CompleteExtinctionException:
            warnings.warn("NEAT population went extinct.", stacklevel=2)
            best_genome = pop.best_genome

        champion = ESHyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            es_params=es_params,
        )

        elapsed = time.perf_counter() - t0
        self._log(config, best_genome, pop_size, elapsed)

        return TrainingResult(
            algorithm="ES-HyperNEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
            },
            metadata={"genome": best_genome, "config": neat_config},
            seed=seed,
            training_time=elapsed,
        )
