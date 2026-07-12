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

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines._logging_helpers import build_logger
from deap_pettingzoo.baselines.base import (
    BudgetMode,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.recording import RecordingController
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS
from .config import build_cppn_config
from .hyperneat import ESHyperNeatPolicy, HyperNeatPolicy
from .substrate import substrate_for_env
from .trainer import NEATTrainer, _RecordingReporter


def _substrate_to_spec(substrate: Any) -> Dict[str, Any]:
    """Serialize a :class:`~._pureples.substrate.Substrate` to a picklable dict."""
    return {
        "input_coordinates": [list(c) for c in substrate.input_coordinates],
        "output_coordinates": [list(c) for c in substrate.output_coordinates],
        "hidden_coordinates": (
            [[list(c) for c in layer] for layer in substrate.hidden_coordinates]
            if substrate.hidden_coordinates
            else None
        ),
    }


def _run_neat(
    pop: neat.Population,
    fitness_func: Any,
    n_generations: int,
    consumed_budget: Optional[int] = None,
    consumed_fn: Optional[Any] = None,
    max_steps: int = 1,
) -> None:
    """Run the NEAT population and handle extinction.

    Mutates *pop* in-place; ``pop.best_genome`` is set after completion.

    Under opt-in consumed-step budgeting (``consumed_budget`` set,
    ``consumed_fn`` returning the real env ticks so far) the population is
    stepped one generation at a time until the budget is exhausted:
    ``Population`` carries all state across ``run()`` calls and the NEAT
    config pins ``no_fitness_termination=True``, so n=1 stepping is
    semantically identical to a single n=N call. The safety cap mirrors
    the EA loop — each generation consumes at least one tick per genome,
    so ``n_generations * max_steps`` iterations always suffice.
    """
    import warnings

    pop.add_reporter(neat.StdOutReporter(show_species_detail=False))
    try:
        if consumed_budget is None or consumed_fn is None:
            pop.run(fitness_func, n=n_generations)
        else:
            safety_cap = n_generations * max(1, max_steps) + 10
            for _ in range(safety_cap):
                if consumed_fn() >= consumed_budget:
                    break
                pop.run(fitness_func, n=1)
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
        # Opt-in consumed-step budgeting: run until the real env-tick budget
        # is exhausted instead of the nominal generation count.
        consumed_budget: Optional[int] = None
        if config.budget_on_consumed and config.budget_mode == BudgetMode.ENV_STEPS:
            consumed_budget = int(config.budget_value)
        self._env_steps_consumed = 0
        params = dict(config.algorithm_params)
        max_steps = self._resolve_max_steps(config)

        pop_size = int(params.get("pop_size", 50))
        n_generations = params.get("n_generations", None)
        if n_generations is None:
            n_generations = self._budget_to_generations(config, pop_size)

        # Build env
        env = PettingZooWrapper(
            config.env_name,
            env_config=config.env_config,
            render_mode="rgb_array" if config.render else None,
        )
        shared_action_space = env.action_space(env.possible_agents[0])
        obs_shape = env.observation_space(env.possible_agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(shared_action_space)

        # Substrate
        substrate = params.get("substrate", None)
        if substrate is None:
            hidden_layers = params.get("substrate_hidden_layers", None)
            if hidden_layers is None:
                # HyperNEAT MUST have at least one hidden layer; without
                # hidden nodes the CPPN can only produce direct input->output
                # weights and the phenotype effectively does nothing.
                hidden_layers = [4, 4]
            substrate = substrate_for_env(
                input_size,
                output_size,
                layout=params.get("substrate_layout", "uniform"),
                hidden_layers=hidden_layers,
            )

        # CPPN config (5 inputs: x1,y1,x2,y2,bias; 1 output: weight) -
        cppn_input_size = 5
        cppn_output_size = 1
        overrides = dict(params.get("neat_config_overrides", {}))
        seed = config.seed if config.seed is not None else params.get("seed", None)
        neat_config = build_cppn_config(
            input_size=cppn_input_size,
            output_size=cppn_output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        # Logger + unified champion-rollout recording
        log = build_logger(config, algorithm="hyperneat")
        self._recorder = RecordingController(
            config,
            log,
            max_steps=max_steps,
            seed=seed,
        )

        # Create population
        pop = neat.Population(neat_config, seed=seed)

        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        activation = params.get("phenotype_activation", "sigmoid")
        weight_threshold = float(params.get("weight_threshold", 0.1))

        # Recording reporter
        def _hn_champion(best_genome):
            return HyperNeatPolicy(
                best_genome,
                neat_config,
                substrate,
                action_space=shared_action_space,
                activation=activation,
                weight_threshold=weight_threshold,
            )

        if self._recorder.enabled:

            def _record_champion(gen: int, best_genome) -> None:
                self._recorder.observe(
                    ProgressEvent(
                        unit="generation",
                        current=gen,
                        total=n_generations,
                        env_steps=gen * max_steps,
                    ),
                    lambda: _hn_champion(best_genome),
                )

            pop.add_reporter(_RecordingReporter(_record_champion))

        effective_workers, backend, scheduler_address, remote = (
            self._resolve_parallelism(config)
        )

        # Parallel or sequential eval
        if effective_workers > 1 or remote:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor

            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate
            from .trainer import _action_space_hint

            _act_hint = _action_space_hint(shared_action_space)
            snapshot = get_registry_snapshot(target_remote=remote)
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
                weight_threshold=weight_threshold,
                max_steps=max_steps,
                registry_snapshot=snapshot,
            )

            def _on_generation(best_fitness: float, _gen_steps: int) -> None:
                self._generation_num += 1
                self._fitness_history.append(best_fitness)

            _parallel_steps = [0]
            _consumed_fn = lambda: (  # noqa: E731
                int(self._env_steps_consumed) + int(_parallel_steps[0])
            )
            with make_executor(
                effective_workers,
                backend=backend,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
                scheduler_address=scheduler_address,
            ) as executor:
                _eval_genomes = make_neat_parallel_evaluate(
                    executor,
                    worker_spec,
                    step_sink=_parallel_steps,
                    on_generation=_on_generation,
                )
                _run_neat(
                    pop,
                    _eval_genomes,
                    n_generations,
                    consumed_budget=consumed_budget,
                    consumed_fn=_consumed_fn,
                    max_steps=max_steps,
                )
            self._env_steps_consumed += int(_parallel_steps[0])
        else:
            _base_seed = seed if seed is not None else 0

            def _eval_genomes(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
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
                            weight_threshold=weight_threshold,
                        )
                        rewards: list[float] = []
                        for ep_i in range(eval_eps):
                            ep_seed = (
                                _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            )
                            observations = env.reset(seed=ep_seed)
                            ep_rew = 0.0
                            done = False
                            _steps = 0
                            while not done and _steps < max_steps:
                                actions = {}
                                for agent in observations:
                                    actions[agent] = policy.get_action(
                                        observations[agent]
                                    )
                                observations, s_rewards, dones, truncateds, _ = (
                                    env.step(actions)
                                )
                                ep_rew += float(sum(s_rewards.values()))
                                done = all(dones.values()) or all(truncateds.values())
                                _steps += 1
                                self._env_steps_consumed += 1
                            rewards.append(ep_rew)
                        genome.fitness = float(np.mean(rewards))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        genome.fitness = -1e9
                    gen_fitnesses.append(genome.fitness)

                if gen_fitnesses:
                    self._fitness_history.append(float(np.max(gen_fitnesses)))

            _run_neat(
                pop,
                _eval_genomes,
                n_generations,
                consumed_budget=consumed_budget,
                consumed_fn=lambda: int(self._env_steps_consumed),
                max_steps=max_steps,
            )

        best_genome = pop.best_genome

        champion = HyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            activation=activation,
            weight_threshold=weight_threshold,
        )

        elapsed = time.perf_counter() - t0

        # Realized count: equals n_generations under nominal budgeting, but
        # consumed budgeting (and extinction) can realize a different number.
        realized_gens = int(self._generation_num)

        self._recorder.finalize(
            lambda: _hn_champion(best_genome),
            iteration=realized_gens,
            env_steps=realized_gens * max_steps,
        )

        try:
            env.close()
        except Exception:
            pass
        self._recorder.close()

        self._log(
            config,
            best_genome,
            pop_size,
            elapsed,
            algorithm="HyperNEAT",
            nominal_env_steps_per_iteration=pop_size * eval_eps * max_steps,
        )
        if config.recording.enabled:
            log.close()

        return TrainingResult(
            algorithm="HyperNEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
                "env_steps_nominal": int(
                    realized_gens * pop_size * eval_eps * max_steps
                ),
                "env_steps_consumed": int(self._env_steps_consumed),
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
        # Opt-in consumed-step budgeting: run until the real env-tick budget
        # is exhausted instead of the nominal generation count.
        consumed_budget: Optional[int] = None
        if config.budget_on_consumed and config.budget_mode == BudgetMode.ENV_STEPS:
            consumed_budget = int(config.budget_value)
        self._env_steps_consumed = 0
        params = dict(config.algorithm_params)
        max_steps = self._resolve_max_steps(config)

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
        # IMPORTANT: must use uniform 1D layout (inputs at y=-1, outputs at
        # y=+1).  The ES-HyperNEAT pruning_extraction function enforces a
        # strict "con.y1 < con.y2 (upward only)" constraint, so inputs must
        # be at the bottom and outputs at the top.  A 2D grid substrate
        # places inputs at mixed y levels, disconnecting most of them.
        substrate = params.get("substrate", None)
        if substrate is None:
            substrate = substrate_for_env(
                input_size,
                output_size,
                layout="uniform",
                hidden_layers=params.get("substrate_hidden_layers", None),
            )

        cppn_input_size = 5
        cppn_output_size = 1
        overrides = dict(params.get("neat_config_overrides", {}))
        seed = config.seed if config.seed is not None else params.get("seed", None)
        neat_config = build_cppn_config(
            input_size=cppn_input_size,
            output_size=cppn_output_size,
            pop_size=pop_size,
            seed=seed,
            overrides=overrides,
        )

        # Logger + unified champion-rollout recording
        log = build_logger(config, algorithm="es-hyperneat")
        self._recorder = RecordingController(
            config,
            log,
            max_steps=max_steps,
            seed=seed,
        )

        pop = neat.Population(neat_config, seed=seed)

        eval_eps = int(params.get("eval_episodes_per_genome", 1))
        es_params = dict(params.get("es_params", ES_HYPERNEAT_PARAMS_DEFAULTS))

        # Recording reporter
        def _esn_champion(best_genome):
            return ESHyperNeatPolicy(
                best_genome,
                neat_config,
                substrate,
                action_space=shared_action_space,
                es_params=es_params,
            )

        if self._recorder.enabled:

            def _record_champion(gen: int, best_genome) -> None:
                self._recorder.observe(
                    ProgressEvent(
                        unit="generation",
                        current=gen,
                        total=n_generations,
                        env_steps=gen * max_steps,
                    ),
                    lambda: _esn_champion(best_genome),
                )

            pop.add_reporter(_RecordingReporter(_record_champion))

        effective_workers, backend, scheduler_address, remote = (
            self._resolve_parallelism(config)
        )

        # Parallel or sequential eval
        if effective_workers > 1 or remote:
            from deap_pettingzoo.environments.registry import (
                get_registry_snapshot,
                replay_registry_snapshot,
            )
            from deap_pettingzoo.utils.parallel import make_executor

            from .parallel import _NeatWorkerSpec, make_neat_parallel_evaluate
            from .trainer import _action_space_hint

            _act_hint = _action_space_hint(shared_action_space)
            snapshot = get_registry_snapshot(target_remote=remote)
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
                max_steps=max_steps,
                registry_snapshot=snapshot,
            )

            def _on_generation(best_fitness: float, _gen_steps: int) -> None:
                self._generation_num += 1
                self._fitness_history.append(best_fitness)

            _parallel_steps = [0]
            _consumed_fn = lambda: (  # noqa: E731
                int(self._env_steps_consumed) + int(_parallel_steps[0])
            )
            with make_executor(
                effective_workers,
                backend=backend,
                initializer=replay_registry_snapshot,
                initargs=(snapshot,),
                scheduler_address=scheduler_address,
            ) as executor:
                _eval_genomes = make_neat_parallel_evaluate(
                    executor,
                    worker_spec,
                    step_sink=_parallel_steps,
                    on_generation=_on_generation,
                )
                _run_neat(
                    pop,
                    _eval_genomes,
                    n_generations,
                    consumed_budget=consumed_budget,
                    consumed_fn=_consumed_fn,
                    max_steps=max_steps,
                )
            self._env_steps_consumed += int(_parallel_steps[0])
        else:
            _base_seed = seed if seed is not None else 0

            def _eval_genomes(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
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
                            ep_seed = (
                                _base_seed * 10_000 + self._generation_num * 100 + ep_i
                            )
                            observations = env.reset(seed=ep_seed)
                            ep_rew = 0.0
                            done = False
                            _steps = 0
                            while not done and _steps < max_steps:
                                actions = {}
                                for agent in observations:
                                    actions[agent] = policy.get_action(
                                        observations[agent]
                                    )
                                observations, s_rewards, dones, truncateds, _ = (
                                    env.step(actions)
                                )
                                ep_rew += float(sum(s_rewards.values()))
                                done = all(dones.values()) or all(truncateds.values())
                                _steps += 1
                                self._env_steps_consumed += 1
                            rewards.append(ep_rew)
                        genome.fitness = float(np.mean(rewards))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        genome.fitness = -1e9
                    gen_fitnesses.append(genome.fitness)

                if gen_fitnesses:
                    self._fitness_history.append(float(np.max(gen_fitnesses)))

            _run_neat(
                pop,
                _eval_genomes,
                n_generations,
                consumed_budget=consumed_budget,
                consumed_fn=lambda: int(self._env_steps_consumed),
                max_steps=max_steps,
            )

        best_genome = pop.best_genome

        champion = ESHyperNeatPolicy(
            best_genome,
            neat_config,
            substrate,
            action_space=shared_action_space,
            es_params=es_params,
        )

        elapsed = time.perf_counter() - t0

        # Realized count: equals n_generations under nominal budgeting, but
        # consumed budgeting (and extinction) can realize a different number.
        realized_gens = int(self._generation_num)

        self._recorder.finalize(
            lambda: _esn_champion(best_genome),
            iteration=realized_gens,
            env_steps=realized_gens * max_steps,
        )

        try:
            env.close()
        except Exception:
            pass
        self._recorder.close()

        self._log(
            config,
            best_genome,
            pop_size,
            elapsed,
            algorithm="ES-HyperNEAT",
            nominal_env_steps_per_iteration=pop_size * eval_eps * max_steps,
        )
        if config.recording.enabled:
            log.close()

        return TrainingResult(
            algorithm="ES-HyperNEAT",
            policy=champion,
            metrics={
                "fitness_history": list(self._fitness_history),
                "final_generation": self._generation_num,
                "population_size": pop_size,
                "env_steps_nominal": int(
                    realized_gens * pop_size * eval_eps * max_steps
                ),
                "env_steps_consumed": int(self._env_steps_consumed),
            },
            metadata={"genome": best_genome, "config": neat_config},
            seed=seed,
            training_time=elapsed,
        )
