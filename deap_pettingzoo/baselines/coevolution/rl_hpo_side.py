"""RL HPO side state — champion-as-opponent co-evolution for AgileRL HPO.

Wraps an AgileRL evolutionary-HPO population (:class:`EvolveHPOTrainer`)
behind the :class:`SideState` protocol so it can participate in
bidirectional EA/CMA/PSO vs RL co-evolution.  

The population evolves internally; only the **current champion** is
exposed to opponent sides each generation (via a frozen CPU snapshot).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper

from .pools import OpponentPool, RLPolicyPool
from .sides import GenerationStats, Side, SideState

SUPPORTED_HPO_RL_ALGORITHMS: Dict[str, Tuple[str, str, bool]] = {
    "maddpg_hpo": ("agilerl.algorithms.maddpg", "MADDPG", False),
    "matd3_hpo":  ("agilerl.algorithms.matd3",  "MATD3",  False),
}


class RLEvolveHPOSideState(SideState):
    """RL side that evolves an AgileRL population internally.

    Accepted ``optimizer_config`` keys:

    * ``algorithm`` (required) — ``"maddpg_hpo"`` or ``"matd3_hpo"``.
    * ``population_size`` (default 6)
    * ``evo_every_n_steps`` (default 20_000 — coevolution generations
      are typically shorter than standalone HPO)
    * ``agent_ctor_kwargs`` — forwarded to the underlying AgileRL
      algorithm constructor (lr_actor, batch_size, ...).
    * ``net_config`` — ``{encoder_config, head_config}`` dict.
    * ``hp_config`` — ``{name: [min, max]}`` dict for
      :class:`~agilerl.algorithms.core.registry.HyperparameterConfig`.
    * ``mutation_probs`` — dict of mutation probabilities.
    * ``seed``, ``device``.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        algo = cfg.get("algorithm")
        if algo not in SUPPORTED_HPO_RL_ALGORITHMS:
            raise ValueError(
                f"RLEvolveHPOSideState: algorithm must be one of "
                f"{sorted(SUPPORTED_HPO_RL_ALGORITHMS)}, got {algo!r}"
            )
        self._algo_key = algo
        self._algo_module, self._algo_class, self._on_policy = (
            SUPPORTED_HPO_RL_ALGORITHMS[algo]
        )
        if self._on_policy:  # pragma: no cover — safety net
            raise NotImplementedError(
                "On-policy HPO co-evolution is not yet supported."
            )

        self.population_size = int(cfg.get("population_size", 6))
        self.evo_every = int(cfg.get("evo_every_n_steps", 20_000))
        from deap_pettingzoo.utils.device_check import resolve_device
        self.device = resolve_device(cfg.get("device", "cpu"))
        self.seed = cfg.get("seed", None)
        self.agent_ctor_kwargs = dict(cfg.get("agent_ctor_kwargs", {}))
        self.net_config = cfg.get("net_config", {})
        self.hp_config_spec = cfg.get("hp_config", {})
        self.mutation_probs = cfg.get("mutation_probs", {})

        self._population: List[Any] = []
        self._champion_wrapper: Optional[AgileRLPolicyWrapper] = None
        self._action_space: Any = None
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # SideState protocol
    # ------------------------------------------------------------------

    def initialise(
        self, env: PettingZooWrapper, rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is not None:
            self._rng = rng
        elif self.seed is not None:
            self._rng = np.random.default_rng(int(self.seed))

        # Resolve learner agents.
        self._learner_ids: List[str] = [
            a for a in env.possible_agents
            if a.startswith(self.side.role_prefixes)
        ]
        if not self._learner_ids:
            raise ValueError(
                f"RLEvolveHPOSideState: no agents match prefixes "
                f"{self.side.role_prefixes!r}"
            )
        self._action_space = env.action_space(self._learner_ids[0])

        # Build HPO population.
        try:
            from agilerl.hpo.mutation import Mutations
            from agilerl.hpo.tournament import TournamentSelection
            from agilerl.algorithms.core.registry import (
                HyperparameterConfig,
                RLParameter,
            )
            import importlib
            mod = importlib.import_module(self._algo_module)
            alg_cls = getattr(mod, self._algo_class)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RLEvolveHPOSideState requires the 'rl-baselines' extra."
            ) from exc

        obs_spaces = [env.observation_space(a) for a in self._learner_ids]
        act_spaces = [env.action_space(a) for a in self._learner_ids]

        hp_config = self._build_hp_config(hp_config_spec=self.hp_config_spec)
        pop = self._build_population(
            alg_cls=alg_cls,
            size=self.population_size,
            obs_spaces=obs_spaces,
            act_spaces=act_spaces,
            agent_ids=self._learner_ids,
            net_config=self.net_config,
            hp_config=hp_config,
        )
        self._population = pop
        self._champion_wrapper = AgileRLPolicyWrapper(
            self._population[0], self._learner_ids,
            algorithm_name=self._algo_key,
        )

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        self.generation += 1

        # 1. Train each agent for evo_every steps against current opponents.
        evaluator = evaluator_factory(self)
        env = evaluator._env  # type: ignore[attr-defined]
        for agent in self._population:
            self._train_agent(
                agent=agent,
                env=env,
                learner_ids=self._learner_ids,
                target_steps=self.evo_every,
            )

        # 2. Evaluate fitnesses.
        fitnesses: List[float] = []
        for agent in self._population:
            fit = self._eval_agent(
                agent=agent,
                env=env,
                learner_ids=self._learner_ids,
                num_episodes=3,
            )
            fitnesses.append(fit)

        # 3. Mutation.
        try:
            from agilerl.hpo.mutation import Mutations
        except ImportError:  # pragma: no cover
            raise ImportError("AgileRL required for HPO co-evolution.")

        mutations = Mutations(
            no_mutation=self.mutation_probs.get("no_mutation", 0.4),
            architecture=self.mutation_probs.get("architecture", 0.2),
            new_layer_prob=self.mutation_probs.get("new_layer_prob", 0.2),
            parameters=self.mutation_probs.get("parameters", 0.2),
            activation=self.mutation_probs.get("activation", 0.0),
            rl_hp=self.mutation_probs.get("rl_hp", 0.2),
            mutation_sd=self.mutation_probs.get("mutation_sd", 0.1),
            rand_seed=self.seed,
            device=self.device,
        )
        # Select by fitness: keep top 1/4 elites, mutate the rest.
        ranked = sorted(
            zip(self._population, fitnesses),
            key=lambda x: x[1], reverse=True,
        )
        elite_count = max(1, self.population_size // 4)
        elites = [a for a, _ in ranked[:elite_count]]
        others = [a for a, _ in ranked[elite_count:]]
        mutated = mutations.mutation(others)
        self._population = elites + mutated

        best = max(fitnesses)
        mean = float(np.mean(fitnesses))
        self.best_fitness = max(self.best_fitness, best)

        # Champion = best pre-mutation elite (fitnesses are pre-mutation;
        # using ranked[0][0] avoids pairing stale fitnesses with mutated agents).
        best_agent = ranked[0][0]
        self._champion_wrapper = AgileRLPolicyWrapper(
            best_agent, self._learner_ids, algorithm_name=self._algo_key,
        )

        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=mean,
            std=float(np.std(fitnesses)),
            min=min(fitnesses),
            max=best,
            best_fitness=best,
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        """Expose the current champion as a picklable opponent pool."""
        if self._champion_wrapper is None:
            return RLPolicyPool([])
        pool = RLPolicyPool([self._champion_wrapper])
        return pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        assert self._champion_wrapper is not None
        return self._champion_wrapper

    def best_snapshot(self) -> Any:
        """Return a frozen CPU snapshot of the champion."""
        assert self._champion_wrapper is not None
        return self._champion_wrapper.make_frozen_cpu_snapshot()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hp_config(hp_config_spec: dict) -> Any:
        from agilerl.algorithms.core.registry import (
            HyperparameterConfig,
            RLParameter,
        )
        if not hp_config_spec:
            return HyperparameterConfig()
        params = {}
        for name, spec in hp_config_spec.items():
            if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo, hi = float(spec[0]), float(spec[1])
                grow = float(spec[2]) if len(spec) > 2 else 1.5
                shrink = float(spec[3]) if len(spec) > 3 else 0.75
                params[name] = RLParameter(
                    min=lo, max=hi, grow_factor=grow, shrink_factor=shrink,
                )
            else:
                params[name] = spec
        return HyperparameterConfig(**params)

    @staticmethod
    def _build_population(
        alg_cls: type,
        size: int,
        obs_spaces: Any,
        act_spaces: Any,
        agent_ids: List[str],
        net_config: Dict[str, Any],
        hp_config: Any,
    ) -> List[Any]:
        kwargs: Dict[str, Any] = {}
        if net_config:
            kwargs["net_config"] = net_config
        if hp_config:
            kwargs["hp_config"] = hp_config

        try:
            return list(alg_cls.population(
                size=size,
                observation_spaces=obs_spaces,
                action_spaces=act_spaces,
                agent_ids=agent_ids,
                **kwargs,
            ))
        except TypeError:
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

    @staticmethod
    def _train_agent(
        agent: Any,
        env: Any,
        learner_ids: List[str],
        target_steps: int,
    ) -> None:
        steps = 0
        learn_step = getattr(agent, "learn_step", 100)
        batch_size = getattr(agent, "batch_size", 64)
        memory = getattr(agent, "memory", None)
        learning_delay = getattr(agent, "learning_delay", 500)
        total_transitions = getattr(agent, "_total_transitions", 0)

        while steps < target_steps:
            obs, info = env.reset()
            while steps < target_steps:
                try:
                    actions = agent.get_action(obs, training=True)
                except TypeError:
                    actions = agent.get_action(obs)
                result = actions[0] if isinstance(actions, tuple) else actions
                next_obs, rewards, terms, truncs, _ = env.step(result)

                if memory is not None:
                    for a in learner_ids:
                        r = float(rewards.get(a, 0.0))
                        o = obs.get(a)
                        no = next_obs.get(a)
                        if o is not None and no is not None:
                            memory.save_to_memory(o, result.get(a), r, no,
                                terms.get(a, False) or truncs.get(a, False))
                total_transitions += 1
                steps += 1
                obs = next_obs

                if total_transitions >= learning_delay and steps % learn_step == 0:
                    if memory is not None and len(memory) >= batch_size:
                        try:
                            experiences = memory.sample(batch_size)
                            agent.learn(experiences)
                        except Exception:
                            pass

                if all(terms.get(a, False) for a in learner_ids) or all(
                    truncs.get(a, False) for a in learner_ids
                ):
                    break

    @staticmethod
    def _eval_agent(
        agent: Any,
        env: Any,
        learner_ids: List[str],
        num_episodes: int = 3,
    ) -> float:
        rewards: List[float] = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            total = 0.0
            for _ in range(200):
                try:
                    actions = agent.get_action(obs, training=False)
                except TypeError:
                    actions = agent.get_action(obs)
                result = actions[0] if isinstance(actions, tuple) else actions
                next_obs, rews, terms, truncs, _ = env.step(result)
                total += sum(float(rews.get(a, 0.0)) for a in learner_ids)
                obs = next_obs
                if all(terms.get(a, False) for a in learner_ids) or all(
                    truncs.get(a, False) for a in learner_ids
                ):
                    break
            rewards.append(total)
        return float(np.mean(rewards)) if rewards else 0.0
