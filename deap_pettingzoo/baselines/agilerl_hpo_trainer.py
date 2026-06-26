"""
AgileRL evolutionary hyperparameter-optimisation trainer.

Wraps AgileRL's built-in ``Mutations``, ``TournamentSelection``, and
``EvolvableAlgorithm.population()`` facilities behind the unified
:class:`BaselineTrainer` API.  A population of AgileRL agents is
evolved generation-by-generation, with periodic architecture / weight /
hyperparameter mutations and tournament-driven survival.

This module depends on the optional ``rl-baselines`` extra
(``pip install 'deap-pettingzoo[rl-baselines]'``).  Imports are lazy
inside :meth:`train` so the core package remains importable without
AgileRL or PyTorch.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from deap_pettingzoo.baselines.agilerl_trainer import AgileRLTrainer
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.baselines.base import (
    BudgetMode,
    ProgressCallback,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
)

logger = logging.getLogger(__name__)

#: Mutation-probability keys understood by AgileRL's ``Mutations`` and the
#: order in which the flat ``prob_*`` form-knobs map onto them.
_MUTATION_PROB_KEYS = (
    "no_mutation",
    "architecture",
    "new_layer_prob",
    "parameters",
    "activation",
    "rl_hp",
)

#: Form-group name for the per-mutation-probability knobs.
_MUT_GROUP = "HPO / Mutation"


class EvolveHPOTrainer(AgileRLTrainer):
    """AgileRL trainer with evolutionary HPO (population-level mutations).

    Builds a *population* of AgileRL agents, runs each through a
    rollout-and-learn loop, evaluates fitness, then applies tournament
    selection and mutations to form the next generation.

    Constructor parameters mirror :class:`AgileRLTrainer` with the
    addition of:

    * ``population_size`` — number of agents in the HPO population.
    * ``evo_every_n_steps`` — number of env steps between evaluations.
    * ``mutation_probs`` — dict of mutation probabilities passed to
      AgileRL's :class:`~agilerl.hpo.mutation.Mutations`.

    See the AgileRL documentation for full details on the mutation and
    tournament framework:
    https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html
    """

    def __init__(
        self,
        population_size: int = 6,
        evo_every_n_steps: int = 50_000,
        mutation_probs: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        super().__init__(device=device, verbose=verbose)
        if population_size < 2:
            raise ValueError(
                f"population_size must be >= 2, got {population_size}"
            )
        self._hpo_population_size = int(population_size)
        self._evo_every_n_steps = int(evo_every_n_steps)
        self._mutation_probs = dict(mutation_probs or {})
        self._mutation_probs.setdefault("no_mutation", 0.4)
        self._mutation_probs.setdefault("architecture", 0.2)
        self._mutation_probs.setdefault("new_layer_prob", 0.2)
        self._mutation_probs.setdefault("parameters", 0.2)
        self._mutation_probs.setdefault("activation", 0.0)
        self._mutation_probs.setdefault("rl_hp", 0.2)

    # ------------------------------------------------------------------
    # Catalog hyperparameter surface
    # ------------------------------------------------------------------

    @staticmethod
    def describe_hyperparameters():
        """Declarative hyperparameter overlay consumed by the catalog.

        Extends the inherited AgileRL surface
        (:meth:`AgileRLTrainer.describe_hyperparameters`) with the
        population-level HPO knobs. ``population_size`` and
        ``evo_every_n_steps`` are routed to the constructor
        (:class:`~deap_pettingzoo.catalog.schema.ParamTarget.TRAINER`) so the
        constructor's ``population_size >= 2`` validation runs; the remaining
        knobs flow through ``algorithm_params`` and are read in :meth:`train`.
        The six ``prob_*`` floats are reassembled into the AgileRL
        ``Mutations`` probability dict by :meth:`_assemble_mutation_probs`.
        """
        from deap_pettingzoo.catalog.schema import (
            ParamSpec,
            ParamTarget,
            ParamType,
        )

        I, F = ParamType.INT, ParamType.FLOAT
        TR, AP = ParamTarget.TRAINER, ParamTarget.ALGORITHM_PARAMS
        base = list(AgileRLTrainer.describe_hyperparameters())
        hpo = (
            ParamSpec("population_size", I, 6, minimum=2, step=1, target=TR,
                      group="HPO",
                      help="Number of AgileRL agents evolved in parallel."),
            ParamSpec("evo_every_n_steps", I, 50_000, minimum=1, step=1000,
                      target=TR, group="HPO",
                      help="Env steps each agent trains between generations."),
            ParamSpec("tournament_size", I, 2, minimum=1, step=1, target=AP,
                      group="HPO",
                      help="Tournament size for survivor selection."),
            ParamSpec("mutation_sd", F, 0.1, minimum=0.0, step=0.01, target=AP,
                      group="HPO", advanced=True,
                      help="Standard deviation of parameter mutations."),
            ParamSpec("prob_no_mutation", F, 0.4, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of leaving an agent unmutated."),
            ParamSpec("prob_architecture", F, 0.2, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of an architecture mutation."),
            ParamSpec("prob_new_layer_prob", F, 0.2, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of adding/removing a layer."),
            ParamSpec("prob_parameters", F, 0.2, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of a weight-parameter mutation."),
            ParamSpec("prob_activation", F, 0.0, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of an activation-function mutation."),
            ParamSpec("prob_rl_hp", F, 0.2, minimum=0.0, maximum=1.0,
                      step=0.05, target=AP, group=_MUT_GROUP, advanced=True,
                      help="Relative probability of an RL-hyperparameter mutation."),
        )
        return tuple(base) + hpo

    def _assemble_mutation_probs(
        self, algorithm_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Merge constructor defaults, a nested ``mutation_probs`` dict and the
        flat ``prob_*`` form-knobs into the AgileRL ``Mutations`` probability dict.

        Precedence (lowest → highest): constructor defaults <
        ``algorithm_params['mutation_probs']`` (programmatic) <
        ``algorithm_params['prob_<key>']`` (UI form).
        """
        mut_probs = dict(self._mutation_probs)
        mut_probs.update(algorithm_params.get("mutation_probs", {}))
        for key in _MUTATION_PROB_KEYS:
            flat = algorithm_params.get(f"prob_{key}")
            if flat is not None:
                mut_probs[key] = float(flat)
        return mut_probs

    # ------------------------------------------------------------------
    # BaselineTrainer
    # ------------------------------------------------------------------

    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Run evolutionary HPO training.

        Args:
            config: Training configuration.  ``algorithm_params`` may
                contain HPO-specific keys:

                * ``hp_config`` — dict mapping parameter names to
                  ``[min, max]`` ranges, forwarded to
                  :class:`~agilerl.algorithms.core.registry.HyperparameterConfig`.
                * ``net_config`` — ``{encoder_config, head_config}``
                  dict for AgileRL network construction.
                * ``population_size`` — overrides the constructor default.
                * ``tournament_size`` — tournament selection size.
                * ``mutation_probs`` — overrides constructor defaults.
                * ``seed`` — RNG seed for reproducibility.

        Returns:
            :class:`TrainingResult` with champion policy and metrics.
        """
        start_time = time.time()

        # Lazy imports — AgileRL is an optional dependency.
        try:
            from agilerl.hpo.mutation import Mutations
            from agilerl.hpo.tournament import TournamentSelection
            from agilerl.algorithms.core.registry import (
                HyperparameterConfig,
                RLParameter,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EvolveHPOTrainer requires the 'rl-baselines' extra. "
                "Install with: pip install 'deap-pettingzoo[rl-baselines]'"
            ) from exc

        # ---- resolve algorithm class and environment ----
        alg_cls = self._import_algorithm()
        env, obs_spaces, act_spaces, agent_ids = self._build_env(config)
        is_vec = config.algorithm_params.get("num_envs", 1) > 1
        max_steps = config.algorithm_params.get("max_steps_per_episode", 200)

        # ---- HPO hyperparameters ----
        pop_size = config.algorithm_params.get(
            "population_size", self._hpo_population_size
        )
        evo_every = config.algorithm_params.get(
            "evo_every_n_steps", self._evo_every_n_steps
        )
        tournament_size = config.algorithm_params.get("tournament_size", 2)
        seed = config.algorithm_params.get("seed", None)
        if seed is not None:
            self._seed_everything(int(seed))

        # Build NetConfig and HyperparameterConfig.
        net_config = config.algorithm_params.get("net_config", {})
        hp_config = self._build_hp_config(
            config.algorithm_params.get("hp_config", {}),
            HyperparameterConfig,
            RLParameter,
        )

        # ---- build population ----
        ctor_kwargs: Dict[str, Any] = {}
        ctor_kwargs["device"] = config.algorithm_params.get(
            "device", self.device
        )
        for key in (
            "batch_size", "memory_size", "learning_delay", "learn_step",
            "lr_actor", "lr_critic", "lr", "gamma", "tau", "policy_freq",
            "expl_noise", "O_U_noise", "ent_coef",
        ):
            if key in config.algorithm_params:
                ctor_kwargs[key] = config.algorithm_params[key]

        # Algorithm-specific pre-flight (e.g. grouped-IPPO patch + validation).
        self._prepare_population_build(obs_spaces, act_spaces, agent_ids)

        population = self._build_population(
            alg_cls, pop_size, obs_spaces, act_spaces, agent_ids,
            net_config, hp_config, ctor_kwargs,
        )

        # ---- HPO machinery ----
        mut_probs = self._assemble_mutation_probs(config.algorithm_params)
        mutations = Mutations(
            no_mutation=mut_probs["no_mutation"],
            architecture=mut_probs["architecture"],
            new_layer_prob=mut_probs["new_layer_prob"],
            parameters=mut_probs["parameters"],
            activation=mut_probs["activation"],
            rl_hp=mut_probs["rl_hp"],
            mutation_sd=config.algorithm_params.get("mutation_sd", 0.1),
            rand_seed=seed,
            device=self.device,
        )
        tournament = TournamentSelection(
            tournament_size=tournament_size,
            elitism=True,
            population_size=pop_size,
            eval_loop=1,
        )

        # ---- training loop ----
        rng = np.random.default_rng(seed)
        budget_steps = self._resolve_budget_steps(config)
        steps_done = 0
        generation = 0
        fitness_history: List[Dict[str, Any]] = []

        from deap_pettingzoo.baselines._logging_helpers import build_logger, write_summary
        from deap_pettingzoo.baselines.recording import RecordingController
        from deap_pettingzoo.utils.logging import save_agilerl_agent

        algo_tag = f"{self.ALGORITHM_NAME}-hpo"
        with build_logger(config, algorithm=algo_tag) as log:
            _env_cfg = config.env_config or {}
            _rec_max_steps = int(
                config.recording.max_frames_per_episode
                or _env_cfg.get("max_cycles")
                or _env_cfg.get("max_time_steps")
                or 500
            )
            # Champion tracks the live best-agent wrapper (updated each gen).
            _champion_agent: Any = population[0] if population else None
            _champion_policy: Any = AgileRLPolicyWrapper(
                _champion_agent, agent_ids, algorithm_name=algo_tag,
            ) if _champion_agent is not None else None

            def _champion():
                return _champion_policy

            self._recorder = RecordingController(
                config, log,
                max_steps=_rec_max_steps,
                seed=seed,
            )

            # Main loop: evolve for budget_steps.
            while steps_done < budget_steps:
                generation += 1
                gen_fitnesses: List[float] = []
                gen_progress = 0

                # Train each agent for evo_every steps. On-policy (IPPO) and
                # off-policy (MADDPG/MATD3) take different rollout/learn paths.
                for agent in population:
                    if self.ON_POLICY:
                        gen_steps = self._train_one_agent_on_policy(
                            agent, env, agent_ids, evo_every, rng,
                        )
                    else:
                        gen_steps = self._train_one_agent(
                            agent, env, agent_ids, is_vec, max_steps,
                            evo_every, seed, rng,
                        )
                    steps_done += gen_steps
                    gen_progress += gen_steps

                # Guard against a non-progressing population (e.g. an agent that
                # is not actually trainable) so the budget loop can never spin
                # forever.
                if gen_progress == 0:
                    logger.warning(
                        "[%s] generation %d consumed 0 env steps; stopping early",
                        algo_tag, generation,
                    )
                    warnings.warn(
                        f"{algo_tag}: HPO generation made no progress; "
                        "stopping early.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

                # Evaluate all agents.
                for agent in population:
                    fit = self._evaluate_agent(
                        agent, env, agent_ids, max_steps,
                        seed, rng, num_episodes=3,
                    )
                    gen_fitnesses.append(fit)

                # Record stats.
                best_fit = max(gen_fitnesses) if gen_fitnesses else 0.0
                mean_fit = float(np.mean(gen_fitnesses)) if gen_fitnesses else 0.0
                std_fit = float(np.std(gen_fitnesses)) if gen_fitnesses else 0.0
                min_fit = min(gen_fitnesses) if gen_fitnesses else 0.0
                fitness_history.append({
                    "generation": generation,
                    "steps": steps_done,
                    "mean": mean_fit,
                    "max": best_fit,
                    "min": min_fit,
                    "std": std_fit,
                })

                elapsed = time.time() - start_time
                log.row("metrics", {
                    "iteration": generation,
                    "env_steps": steps_done,
                    "wall_time_s": elapsed,
                    "mean": mean_fit,
                    "std": std_fit,
                    "min": min_fit,
                    "max": best_fit,
                    "mean_loss": None,
                })

                if self.verbose:
                    print(
                        f"[{algo_tag}] gen {generation} "
                        f"steps={steps_done:,} mean={mean_fit:+.2f} "
                        f"max={best_fit:+.2f}"
                    )

                # Tournament selection + mutation.
                _, population = tournament.select(population)
                population = mutations.mutation(population)

                # Update champion reference for the recording controller.
                best_idx_gen = int(np.argmax(gen_fitnesses)) if gen_fitnesses else 0
                _champion_agent = population[best_idx_gen % len(population)]
                _champion_policy = AgileRLPolicyWrapper(
                    _champion_agent, agent_ids, algorithm_name=algo_tag,
                )
                self._on_progress(
                    progress_callback,
                    ProgressEvent(
                        unit="env_step",
                        current=min(steps_done, budget_steps),
                        total=budget_steps,
                        env_steps=min(steps_done, budget_steps),
                        best_fitness=best_fit,
                        mean_fitness=mean_fit,
                        wall_time_s=elapsed,
                    ),
                    champion=_champion,
                )

            # Champion = best agent by fitness.
            champ_fits = [
                self._evaluate_agent(a, env, agent_ids, max_steps, seed, rng, num_episodes=5)
                for a in population
            ]
            champ_idx = int(np.argmax(champ_fits))
            champion = population[champ_idx]

            policy = AgileRLPolicyWrapper(
                champion, agent_ids, algorithm_name=algo_tag,
            )
            training_time = time.time() - start_time

            self._recorder.finalize(_champion, iteration=generation, env_steps=steps_done)
            self._recorder.close()

            save_agilerl_agent(
                log,
                champion,
                extra={
                    "algorithm": algo_tag,
                    "env_name": config.env_name,
                    "agent_ids": list(agent_ids),
                    "num_generations": generation,
                    "population_size": pop_size,
                },
            )
            write_summary(
                log,
                {
                    "algorithm": algo_tag,
                    "num_generations": generation,
                    "population_size": pop_size,
                    "final_fitness": champ_fits[champ_idx],
                    "total_env_steps": steps_done,
                    "mutation_probs": mut_probs,
                },
                training_time_s=training_time,
            )

        return TrainingResult(
            algorithm=algo_tag,
            policy=policy,
            metrics={
                "fitness_history": fitness_history,
                "final_fitness": champ_fits[champ_idx],
                "num_generations": generation,
                "population_size": pop_size,
                "total_env_steps": steps_done,
            },
            metadata={
                "algorithm_name": self.ALGORITHM_NAME,
                "population_size": pop_size,
                "mutation_probs": mut_probs,
            },
            training_time=training_time,
        )

    # ------------------------------------------------------------------
    # BaselineTrainer — capabilities
    # ------------------------------------------------------------------

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # Only the on-policy (StochasticActor / PPO) base supports
            # MultiDiscrete; the deterministic off-policy bases do not.
            "multidiscrete_action": self.ON_POLICY,
            "needs_global_state": False,
            "on_policy": self.ON_POLICY,
            "supports_turn_based": False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hp_config(
        hp_spec: Dict[str, Any],
        HyperparameterConfig: type,
        RLParameter: type,
    ) -> Any:
        """Build a ``HyperparameterConfig`` from a simple ``{name: [lo,hi]}`` dict."""
        if not hp_spec:
            return HyperparameterConfig()
        params = {}
        for name, spec in hp_spec.items():
            if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo, hi = float(spec[0]), float(spec[1])
                grow = float(spec[2]) if len(spec) > 2 else 1.5
                shrink = float(spec[3]) if len(spec) > 3 else 0.75
                params[name] = RLParameter(
                    min=lo, max=hi, grow_factor=grow, shrink_factor=shrink,
                )
            else:
                params[name] = spec  # already an RLParameter
        return HyperparameterConfig(**params)

    def _build_population(
        self,
        alg_cls: type,
        size: int,
        obs_spaces: Any,
        act_spaces: Any,
        agent_ids: List[str],
        net_config: Dict[str, Any],
        hp_config: Any,
        ctor_kwargs: Dict[str, Any],
    ) -> List[Any]:
        """Create a population of AgileRL agents.

        Uses ``alg_cls.population(...)`` when available (AgileRL >= 2.6);
        otherwise falls back to manual construction.
        """
        kwargs = dict(ctor_kwargs)
        if net_config:
            kwargs["net_config"] = net_config
        if hp_config:
            kwargs["hp_config"] = hp_config

        try:
            return list(alg_cls.population(
                size=size,
                observation_space=obs_spaces,
                action_space=act_spaces,
                agent_ids=agent_ids,
                **kwargs,
            ))
        except TypeError:
            # Older AgileRL — construct manually.
            pop = []
            for i in range(size):
                agent = alg_cls(
                    observation_spaces=obs_spaces,
                    action_spaces=act_spaces,
                    agent_ids=agent_ids,
                    index=i,
                    **kwargs,
                )
                pop.append(agent)
            return pop

    def _train_one_agent(
        self,
        agent: Any,
        env: Any,
        agent_ids: List[str],
        is_vec: bool,
        max_steps: int,
        target_steps: int,
        seed: Optional[int],
        rng: np.random.Generator,
    ) -> int:
        """Run off-policy rollout-and-learn for one agent, returning steps consumed."""
        steps = 0
        learn_step = getattr(agent, "learn_step", 100)
        batch_size = getattr(agent, "batch_size", 64)
        memory = getattr(agent, "memory", None)
        learning_delay = getattr(agent, "learning_delay", 500)
        total_transitions = getattr(agent, "_total_transitions", 0)
        discrete = bool(getattr(agent, "discrete_actions", False))

        while steps < target_steps:
            ep_seed = int(rng.integers(0, 2**31)) if seed is not None else None
            obs, info = env.reset(seed=ep_seed)
            ep_steps = 0

            while ep_steps < max_steps and steps < target_steps:
                try:
                    actions = agent.get_action(obs, training=True)
                except TypeError:
                    actions = agent.get_action(obs)
                result = actions[0] if isinstance(actions, tuple) else actions
                # Unwrap batched actions for raw PettingZoo envs.
                unwrapped: Dict[str, Any] = {}
                for a, act in result.items():
                    arr = np.asarray(act)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                        if discrete and arr.shape == ():
                            arr = arr.item()
                    unwrapped[a] = arr
                next_obs, rewards, terms, truncs, infos = env.step(unwrapped)
                done = all(terms.get(a, False) for a in agent_ids) or all(
                    truncs.get(a, False) for a in agent_ids
                )

                # Store transition.
                if memory is not None:
                    for a in agent_ids:
                        memory.save_to_memory(
                            obs.get(a), result.get(a),
                            float(rewards.get(a, 0.0)),
                            next_obs.get(a), done,
                        )
                total_transitions += 1
                steps += 1
                ep_steps += 1
                obs = next_obs

                if (
                    total_transitions >= learning_delay
                    and steps % learn_step == 0
                    and memory is not None
                    and len(memory) >= batch_size
                ):
                    try:
                        experiences = memory.sample(batch_size)
                        agent.learn(experiences)
                    except Exception:  # pragma: no cover - version drift
                        logger.warning(
                            "[%s-hpo] _train_one_agent: agent.learn() failed",
                            self.ALGORITHM_NAME, exc_info=True,
                        )

                if done:
                    break

        if hasattr(agent, "_total_transitions"):
            object.__setattr__(agent, "_total_transitions", total_transitions)
        return steps

    def _train_one_agent_on_policy(
        self,
        agent: Any,
        env: Any,
        agent_ids: List[str],
        target_steps: int,
        rng: np.random.Generator,
    ) -> int:
        """Run on-policy (IPPO) rollout-and-learn for one agent.

        Delegates to the shared
        :func:`~deap_pettingzoo.baselines.agilerl_onpolicy.train_agent_on_policy`
        helper (also used by
        :class:`~deap_pettingzoo.baselines.coevolution.rl_hpo_side.RLEvolveHPOSideState`)
        so the on-policy population path cannot drift between the two.
        """
        from deap_pettingzoo.baselines.agilerl_onpolicy import (
            train_agent_on_policy,
        )

        return train_agent_on_policy(
            agent, env, agent_ids, target_steps, logger=logger, rng=rng,
        )

    def _prepare_population_build(
        self, obs_spaces: Any, act_spaces: Any, agent_ids: List[str]
    ) -> None:
        """Algorithm-specific pre-flight before the population is built.

        Default no-op. :class:`GroupedIPPOHPOTrainer` overrides this to install
        AgileRL's grouped-IPPO flattening patch and validate the agent grouping.
        """

    @staticmethod
    def _evaluate_agent(
        agent: Any,
        env: Any,
        agent_ids: List[str],
        max_steps: int,
        seed: Optional[int],
        rng: np.random.Generator,
        num_episodes: int = 3,
    ) -> float:
        """Evaluate an agent's deterministic policy and return mean reward."""
        discrete = bool(getattr(agent, "discrete_actions", False))
        rewards: List[float] = []
        for ep in range(num_episodes):
            ep_seed = int(rng.integers(0, 2**31)) if seed is not None else None
            obs, _ = env.reset(seed=ep_seed)
            total = 0.0
            for _ in range(max_steps):
                try:
                    actions = agent.get_action(obs, training=False)
                except TypeError:
                    actions = agent.get_action(obs)
                result = actions[0] if isinstance(actions, tuple) else actions
                unwrapped: Dict[str, Any] = {}
                for a, act in result.items():
                    arr = np.asarray(act)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                        if discrete and arr.shape == ():
                            arr = arr.item()
                    unwrapped[a] = arr
                next_obs, rews, terms, truncs, _ = env.step(unwrapped)
                total += sum(float(rews.get(a, 0.0)) for a in agent_ids)
                obs = next_obs
                if all(terms.get(a, False) for a in agent_ids) or all(
                    truncs.get(a, False) for a in agent_ids
                ):
                    break
            rewards.append(total)
        return float(np.mean(rewards)) if rewards else 0.0

    def _resolve_budget_steps(self, config: TrainingConfig) -> int:
        """Translate config budget to env-step count."""
        if config.budget_mode == BudgetMode.ENV_STEPS:
            return config.budget_value
        if config.budget_mode == BudgetMode.EVAL_EPISODES:
            max_steps = config.algorithm_params.get("max_steps_per_episode", 200)
            return config.budget_value * max_steps
        raise ValueError(
            f"EvolveHPOTrainer does not support budget mode {config.budget_mode}"
        )

    def _build_env(self, config: TrainingConfig):
        """Build env + extract spaces — reuses AgileRL adapter."""
        from deap_pettingzoo.baselines.agilerl_adapter import (
            build_parallel_env,
            extract_space_metadata,
        )
        env = build_parallel_env(
            config.env_name,
            env_config=config.env_config or None,
        )
        obs_spaces, act_spaces, agent_ids = extract_space_metadata(env)
        return env, obs_spaces, act_spaces, agent_ids

    # ------------------------------------------------------------------
    # Concrete subclasses per algorithm
    # ------------------------------------------------------------------


class MADDPGHPOTrainer(EvolveHPOTrainer):
    ALGORITHM_NAME = "maddpg"
    ALGORITHM_IMPORT = ("agilerl.algorithms.maddpg", "MADDPG")
    ON_POLICY = False


class MATD3HPOTrainer(EvolveHPOTrainer):
    ALGORITHM_NAME = "matd3"
    ALGORITHM_IMPORT = ("agilerl.algorithms.matd3", "MATD3")
    ON_POLICY = False


class IPPOHPOTrainer(EvolveHPOTrainer):
    """IPPO with evolutionary HPO (on-policy population evolution).

    Uses the shared on-policy rollout/learn path
    (:meth:`EvolveHPOTrainer._train_one_agent_on_policy`) instead of the
    replay-buffer path, since PPO updates must be computed from freshly
    collected on-policy rollouts.
    """

    ALGORITHM_NAME = "ippo"
    ALGORITHM_IMPORT = ("agilerl.algorithms.ippo", "IPPO")
    ON_POLICY = True


class GroupedIPPOHPOTrainer(IPPOHPOTrainer):
    """Prefix-grouped IPPO with evolutionary HPO.

    Mirrors :class:`~deap_pettingzoo.baselines.agilerl_trainer.GroupedIPPOTrainer`:
    agents whose ids share a prefix share one actor/critic network. The
    grouped-IPPO flattening patch is installed and the grouping is validated
    up-front (via :meth:`_prepare_population_build`) before the population is
    built — no silent fallback to per-agent networks.
    """

    ALGORITHM_NAME = "ippo-grouped"

    def _prepare_population_build(
        self, obs_spaces: Any, act_spaces: Any, agent_ids: List[str]
    ) -> None:
        """Install AgileRL's grouped-IPPO patch and validate the grouping.

        Reuses :class:`GroupedIPPOTrainer`'s patch + validation helpers so the
        standalone and HPO grouped-IPPO paths enforce the same rules. Raises
        loudly when the environment's agents cannot be sensibly grouped.
        """
        from deap_pettingzoo.baselines.agilerl_trainer import (
            GroupedIPPOTrainer,
            _apply_agilerl_grouping_patch,
        )

        _apply_agilerl_grouping_patch()
        # extract_space_metadata may return lists or dicts; the validator wants
        # per-agent-id dicts (same coercion GroupedIPPOTrainer._build_agent uses).
        val_obs = (
            dict(zip(agent_ids, obs_spaces))
            if isinstance(obs_spaces, (list, tuple))
            else obs_spaces
        )
        val_act = (
            dict(zip(agent_ids, act_spaces))
            if isinstance(act_spaces, (list, tuple))
            else act_spaces
        )
        GroupedIPPOTrainer._validate_grouping(agent_ids, val_obs, val_act)
