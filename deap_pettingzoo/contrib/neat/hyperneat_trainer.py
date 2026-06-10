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
from typing import Any, Dict, Optional

import numpy as np

import neat  # guarded by _require_neat

from deap_pettingzoo.baselines.base import (
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

from .config import build_cppn_config
from .substrate import substrate_for_env
from .hyperneat import HyperNeatPolicy, ESHyperNeatPolicy
from .trainer import NEATTrainer
from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS


def _substrate_to_spec(substrate: Any) -> Dict[str, Any]:
    """Serialize a :class:`~._pureples.substrate.Substrate` to a picklable dict."""
    return {
        "input_coordinates": [list(c) for c in substrate.input_coordinates],
        "output_coordinates": [list(c) for c in substrate.output_coordinates],
        "hidden_coordinates": [
            [list(c) for c in layer] for layer in substrate.hidden_coordinates
        ] if substrate.hidden_coordinates else None,
    }


def _run_neat(
    pop: neat.Population,
    fitness_func: Any,
    n_generations: int,
) -> None:
    """Run the NEAT population and handle extinction.

    Mutates *pop* in-place; ``pop.best_genome`` is set after completion.
    """
    import warnings

    pop.add_reporter(neat.StdOutReporter(show_species_detail=False))
    try:
        pop.run(fitness_func, n=n_generations)
    except neat.CompleteExtinctionException:
        warnings.warn("NEAT population went extinct.", stacklevel=2)


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

        # --- Parallel or sequential eval -------------------------------------
        if self._num_workers > 1:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor
            from .trainer import _action_space_hint
            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate

            _act_hint = _action_space_hint(shared_action_space)
            snapshot = get_registry_snapshot()
            worker_spec = _NeatWorkerSpec(
                env_name=config.env_name,
                env_config=dict(config.env_config),
                input_size=cppn_input_size,
                output_size=cppn_output_size,
                action_space_type=_act_hint["type"],
                action_space_args=_act_hint.get("args", {}),
                eval_episodes_per_genome=eval_eps,
                seed=seed if seed is not None else 0,
                substrate_spec=_substrate_to_spec(substrate),
                phenotype_activation=activation,
            )

            with make_executor(
                self._num_workers,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
            ) as executor:
                _eval_genomes = make_neat_parallel_evaluate(executor, worker_spec)
                _run_neat(pop, _eval_genomes, n_generations)
                if not self._fitness_history:
                    self._fitness_history.append(0.0)
        else:
            _base_seed = seed if seed is not None else 0

            def _eval_genomes(
                genomes_and_ids: list, neat_cfg: neat.Config
            ) -> None:
                self._generation_num += 1
                gen_fitnesses: list[float] = []

                for _genome_id, genome in genomes_and_ids:
                    try:
                        policy = HyperNeatPolicy(
                            genome,
                            neat_cfg,
                            substrate,
                            action_space=shared_action_space,
                            activation=activation,
                        )
                        rewards: list[float] = []
                        for ep_i in range(eval_eps):
                            ep_seed = _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            observations = env.reset(seed=ep_seed)
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
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        genome.fitness = -1e9
                    gen_fitnesses.append(genome.fitness)

                if gen_fitnesses:
                    self._fitness_history.append(float(np.max(gen_fitnesses)))

            _run_neat(pop, _eval_genomes, n_generations)

        best_genome = pop.best_genome

        champion = HyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            activation=activation,
        )

        elapsed = time.perf_counter() - t0

        try:
            env.close()
        except Exception:
            pass

        self._log(config, best_genome, pop_size, elapsed, algorithm="HyperNEAT")

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

        # --- Parallel or sequential eval -------------------------------------
        if self._num_workers > 1:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor
            from .trainer import _action_space_hint
            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate

            _act_hint = _action_space_hint(shared_action_space)
            snapshot = get_registry_snapshot()
            worker_spec = _NeatWorkerSpec(
                env_name=config.env_name,
                env_config=dict(config.env_config),
                input_size=cppn_input_size,
                output_size=cppn_output_size,
                action_space_type=_act_hint["type"],
                action_space_args=_act_hint.get("args", {}),
                eval_episodes_per_genome=eval_eps,
                seed=seed if seed is not None else 0,
                substrate_spec=_substrate_to_spec(substrate),
                es_params=dict(es_params),
            )

            with make_executor(
                self._num_workers,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
            ) as executor:
                _eval_genomes = make_neat_parallel_evaluate(executor, worker_spec)
                _run_neat(pop, _eval_genomes, n_generations)
                if not self._fitness_history:
                    self._fitness_history.append(0.0)
        else:
            _base_seed = seed if seed is not None else 0

            def _eval_genomes(
                genomes_and_ids: list, neat_cfg: neat.Config
            ) -> None:
                self._generation_num += 1
                gen_fitnesses: list[float] = []

                for _genome_id, genome in genomes_and_ids:
                    try:
                        policy = ESHyperNeatPolicy(
                            genome,
                            neat_cfg,
                            substrate,
                            action_space=shared_action_space,
                            es_params=es_params,
                        )
                        rewards: list[float] = []
                        for ep_i in range(eval_eps):
                            ep_seed = _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            observations = env.reset(seed=ep_seed)
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
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        genome.fitness = -1e9
                    gen_fitnesses.append(genome.fitness)

                if gen_fitnesses:
                    self._fitness_history.append(float(np.max(gen_fitnesses)))

            _run_neat(pop, _eval_genomes, n_generations)

        best_genome = pop.best_genome

        champion = ESHyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            es_params=es_params,
        )

        elapsed = time.perf_counter() - t0

        try:
            env.close()
        except Exception:
            pass

        self._log(config, best_genome, pop_size, elapsed, algorithm="ES-HyperNEAT")

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
