"""RL HPO side state — champion-as-opponent co-evolution for AgileRL HPO.

Wraps an AgileRL evolutionary-HPO population (:class:`EvolveHPOTrainer`)
behind the :class:`SideState` protocol so it can participate in
bidirectional EA/CMA/PSO vs RL co-evolution.  

The population evolves internally; only the **current champion** is
exposed to opponent sides each generation (via a frozen CPU snapshot).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper

from .pools import FrozenPolicyPool, OpponentPool, RLPolicyPool
from .sides import GenerationStats, Side, SideState

SUPPORTED_HPO_RL_ALGORITHMS: Dict[str, Tuple[str, str, bool]] = {
    # Underscore keys (historical / programmatic).
    "maddpg_hpo": ("agilerl.algorithms.maddpg", "MADDPG", False),
    "matd3_hpo":  ("agilerl.algorithms.matd3",  "MATD3",  False),
    "ippo_hpo":   ("agilerl.algorithms.ippo",   "IPPO",   True),
    "grouped_ippo_hpo": ("agilerl.algorithms.ippo", "IPPO", True),
    # Hyphenated catalog names (injected into ``optimizer_config['algorithm']``
    # by ``experiment_spec._build_coevolution_plan``). Mirrors the dual-key
    # pattern ``rl_side.SUPPORTED_RL_ALGORITHMS`` uses for grouped IPPO.
    "maddpg-hpo": ("agilerl.algorithms.maddpg", "MADDPG", False),
    "matd3-hpo":  ("agilerl.algorithms.matd3",  "MATD3",  False),
    "ippo-hpo":   ("agilerl.algorithms.ippo",   "IPPO",   True),
    "ippo-grouped-hpo": ("agilerl.algorithms.ippo", "IPPO", True),
}


class RLEvolveHPOSideState(SideState):
    """RL side that evolves an AgileRL population internally.

    Accepted ``optimizer_config`` keys:

    * ``algorithm`` (required) — ``"maddpg_hpo"`` / ``"matd3_hpo"``
      (off-policy) or ``"ippo_hpo"`` / ``"grouped_ippo_hpo"`` (on-policy).
    * ``population_size`` (default 6)
    * ``evo_every_n_steps`` (default 20_000 — coevolution generations
      are typically shorter than standalone HPO)
    * ``agent_ctor_kwargs`` — forwarded **verbatim** to the underlying
      AgileRL algorithm constructor. Valid keys are algorithm-specific:
      off-policy MADDPG/MATD3 take ``lr_actor`` / ``lr_critic`` /
      ``batch_size`` / ``learn_step`` / ``gamma`` / ``tau`` / ...; on-policy
      IPPO takes ``lr`` (not ``lr_actor``) and has no ``tau``. Non-constructor
      knobs (e.g. ``memory_size`` / ``learning_delay``) are not accepted here
      and raise ``TypeError`` from the constructor.
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

        self.population_size = int(cfg.get("population_size", 6))
        self.evo_every = int(cfg.get("evo_every_n_steps", 20_000))
        from deap_pettingzoo.utils.device_check import resolve_device
        self.device = resolve_device(cfg.get("device", "cpu"))
        self.seed = cfg.get("seed", None)
        self.agent_ctor_kwargs = dict(cfg.get("agent_ctor_kwargs", {}))
        self.net_config = cfg.get("net_config", {})
        self.hp_config_spec = cfg.get("hp_config", {})
        self.mutation_probs = cfg.get("mutation_probs", {})

        self.is_rl_side: bool = True
        # Mirror the public name RLSideState uses so the trainer's on-policy detector fires.
        self.is_on_policy: bool = self._on_policy
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
        self.resolved_device = self.device
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
        # Forward the user's per-agent ctor knobs (lr_actor, batch_size, ...)
        # verbatim, defaulting device to the side's resolved device so the
        # population lives on the same device as Mutations (mirrors RLSideState).
        ctor_kwargs = dict(self.agent_ctor_kwargs)
        ctor_kwargs.setdefault("device", self.resolved_device)
        pop = self._build_population(
            alg_cls=alg_cls,
            size=self.population_size,
            obs_spaces=obs_spaces,
            act_spaces=act_spaces,
            agent_ids=self._learner_ids,
            net_config=self.net_config,
            hp_config=hp_config,
            ctor_kwargs=ctor_kwargs,
        )
        self._population = pop
        if self._on_policy:
            self._maybe_apply_grouping_patch()
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
        _gen_t0 = time.time()

        # 1. Train each agent for evo_every steps against current opponents.
        evaluator = evaluator_factory(self)
        env = evaluator.env_wrapper

        # Sample one policy per opponent side for non-learner agents.
        # CoEvolutionEvaluator exposes .opponent_sides (list of
        # (name, prefixes, pool) triples) and .side_agents (name → agent list).
        learner_id_set = set(self._learner_ids)
        opponent_policies: Dict[str, Any] = {}
        for side_name, _prefixes, pool in getattr(evaluator, "opponent_sides", []):
            opp_policy = pool.sample_opponent(self._rng)
            for opp_agent in getattr(evaluator, "side_agents", {}).get(side_name, []):
                if opp_agent not in learner_id_set:
                    opponent_policies[opp_agent] = opp_policy

        gen_steps = 0
        for agent in self._population:
            if self._on_policy:
                gen_steps += self._train_agent_on_policy(
                    agent=agent,
                    env=env,
                    learner_ids=self._learner_ids,
                    target_steps=self.evo_every,
                )
            else:
                gen_steps += self._train_agent(
                    agent=agent,
                    env=env,
                    learner_ids=self._learner_ids,
                    target_steps=self.evo_every,
                    opponent_policies=opponent_policies,
                )

        # 2. Evaluate fitnesses.
        fitnesses: List[float] = []
        for agent in self._population:
            fit, eval_steps = self._eval_agent(
                agent=agent,
                env=env,
                learner_ids=self._learner_ids,
                num_episodes=3,
                opponent_policies=opponent_policies,
            )
            fitnesses.append(fit)
            gen_steps += eval_steps

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
            env_steps=int(gen_steps),
            wall_time_s=time.time() - _gen_t0,
        )
        self.history.append(stats)
        return stats

    def current_pool(self) -> OpponentPool:
        """Expose the current champion as a picklable opponent pool."""
        if self._champion_wrapper is None:
            return FrozenPolicyPool([])
        return RLPolicyPool(self._champion_wrapper)

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
        ctor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Create the AgileRL population.

        ``ctor_kwargs`` (e.g. ``lr_actor``, ``batch_size``, ``learn_step``,
        ``gamma``, ``tau``, ``device``) is forwarded verbatim to the
        algorithm constructor alongside ``net_config`` / ``hp_config``.
        """
        kwargs: Dict[str, Any] = dict(ctor_kwargs or {})
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
        opponent_policies: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Off-policy rollout-and-learn for one population member.

        AgileRL agents do not expose ``agent.memory``, so this builds a
        dedicated :class:`~agilerl.components.multi_agent_replay_buffer.MultiAgentReplayBuffer`
        per call, stores one agent-keyed transition per env step, and gates
        ``agent.learn`` on the real buffer. The co-evolution env
        is a single, non-vectorised parallel env, so ``is_vectorised=False``.
        """
        from agilerl.components.multi_agent_replay_buffer import (
            MultiAgentReplayBuffer,
        )
        from deap_pettingzoo.baselines.agilerl_trainer import AgileRLTrainer

        steps = 0
        learn_step = max(1, int(getattr(agent, "learn_step", 100)))
        batch_size = int(getattr(agent, "batch_size", 64))
        # Default to 0: co-evolution generations are short and total_transitions
        # persists across generations (written back below), so warm-up accrues.
        learning_delay = int(getattr(agent, "learning_delay", 0))
        total_transitions = int(getattr(agent, "_total_transitions", 0))
        discrete = bool(getattr(agent, "discrete_actions", False))
        learner_id_set = set(learner_ids)

        memory = MultiAgentReplayBuffer(
            memory_size=int(getattr(agent, "memory_size", 100_000)),
            field_names=["obs", "action", "reward", "next_obs", "done"],
            agent_ids=list(learner_ids),
            device=getattr(agent, "device", "cpu"),
        )

        # Precompute NaN placeholders once (dict.get defaults are eager).
        obs_placeholders = {
            a: AgileRLTrainer._nan_placeholder_for_space(env.observation_space(a))
            for a in learner_ids
        }
        act_placeholders = {
            a: AgileRLTrainer._nan_placeholder_for_space(env.action_space(a))
            for a in learner_ids
        }

        while steps < target_steps:
            obs = env.reset()
            # Normalise: raw mpe2/PettingZoo parallel envs return (obs, info);
            # PettingZooWrapper returns the obs dict directly.
            if isinstance(obs, tuple):
                obs, _info = obs
            while steps < target_steps:
                # Filter to only the agents this side controls; zero-fill any
                # that have already terminated so AgileRL sees a full roster.
                learner_obs = {a: ob for a, ob in obs.items() if a in learner_id_set}
                if len(learner_obs) < len(learner_ids):
                    _ref = next(iter(learner_obs.values())) if learner_obs else obs_placeholders[learner_ids[0]]
                    for a in learner_ids:
                        if a not in learner_obs:
                            learner_obs[a] = np.zeros_like(_ref)
                try:
                    action_result = agent.get_action(learner_obs, training=True)
                except TypeError:
                    action_result = agent.get_action(learner_obs)
                if isinstance(action_result, tuple):
                    actions, raw_actions = action_result[0], action_result[1]
                else:
                    actions = raw_actions = action_result

                # Unwrap the leading (1,) batch dim so the single env receives
                # scalar discrete / unbatched continuous actions.
                env_actions: Dict[str, Any] = {}
                for a, act in actions.items():
                    arr = np.asarray(act)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                        if discrete and arr.shape == ():
                            arr = arr.item()
                    env_actions[a] = arr
                # Use sampled opponent policies for non-learner active agents
                # when available; fall back to random for any missing agent.
                for a in env.agents:
                    if a not in env_actions:
                        if opponent_policies and a in opponent_policies and a in obs:
                            try:
                                opp_act = np.asarray(
                                    opponent_policies[a].get_action(obs[a])
                                )
                                if opp_act.ndim > 0 and opp_act.shape[0] == 1:
                                    opp_act = opp_act[0]
                                env_actions[a] = (
                                    opp_act.item()
                                    if opp_act.shape == ()
                                    else opp_act
                                )
                            except Exception:
                                env_actions[a] = env.action_space(a).sample()
                        else:
                            env_actions[a] = env.action_space(a).sample()
                next_obs, rewards, terms, truncs, _ = env.step(env_actions)

                # Build agent-keyed transition dicts; one save per transition.
                full_obs = {a: obs.get(a, obs_placeholders[a]) for a in learner_ids}
                full_raw_actions: Dict[str, Any] = {}
                for a in learner_ids:
                    if a in raw_actions:
                        arr = np.asarray(raw_actions[a])
                        if arr.ndim > 0 and arr.shape[0] == 1:
                            arr = arr[0]
                        full_raw_actions[a] = arr
                    else:
                        full_raw_actions[a] = act_placeholders[a]
                full_reward = {
                    a: float(rewards[a]) if a in rewards else np.nan
                    for a in learner_ids
                }
                full_next_obs = {
                    a: next_obs.get(a, obs_placeholders[a]) for a in learner_ids
                }
                done_dict = AgileRLTrainer._build_done_dict(
                    terms, truncs, list(learner_ids), is_vec=False, num_envs=1,
                )

                memory.save_to_memory(
                    full_obs, full_raw_actions, full_reward, full_next_obs,
                    done_dict, is_vectorised=False,
                )

                total_transitions += 1
                steps += 1
                obs = next_obs

                if (
                    total_transitions >= learning_delay
                    and (total_transitions - learning_delay) % learn_step == 0
                    and len(memory) >= batch_size
                ):
                    try:
                        experiences = memory.sample(batch_size)
                        agent.learn(experiences)
                    except Exception:  # pragma: no cover - version drift
                        logging.getLogger(__name__).warning(
                            "RLEvolveHPOSideState._train_agent: agent.learn() failed",
                            exc_info=True,
                        )

                if all(terms.get(a, False) for a in learner_ids) or all(
                    truncs.get(a, False) for a in learner_ids
                ):
                    break

        if hasattr(agent, "_total_transitions"):
            object.__setattr__(agent, "_total_transitions", total_transitions)
        return steps

    def _maybe_apply_grouping_patch(self) -> None:
        """Install AgileRL's grouped-IPPO advantage/state flattening fix.

        Same rationale as :meth:`RLSideState._maybe_apply_grouping_patch`:
        AgileRL shares one actor/critic across same-role agents, which
        triggers a flattening bug in ``IPPO._learn_individual``. No-op unless
        the population's agents are actually grouped.
        """
        grouped = self._algo_key == "grouped_ippo_hpo"
        if not grouped and self._population:
            has = getattr(self._population[0], "has_grouped_agents", None)
            try:
                grouped = bool(has()) if callable(has) else False
            except Exception:
                grouped = False
        if not grouped:
            return
        try:
            from deap_pettingzoo.baselines.agilerl_trainer import (
                _apply_agilerl_grouping_patch,
            )

            _apply_agilerl_grouping_patch()
        except Exception:  # pragma: no cover - defensive
            logger.warning("RLEvolveHPOSideState: could not apply grouped-IPPO patch", exc_info=True)

    @staticmethod
    def _train_agent_on_policy(
        agent: Any,
        env: Any,
        learner_ids: List[str],
        target_steps: int,
    ) -> int:
        """On-policy (IPPO) training for one population member.

        Thin wrapper over the shared
        :func:`~deap_pettingzoo.baselines.agilerl_onpolicy.train_agent_on_policy`
        helper, so this side and the standalone
        :class:`~deap_pettingzoo.baselines.agilerl_hpo_trainer.EvolveHPOTrainer`
        share one rollout-and-learn implementation.
        """
        from deap_pettingzoo.baselines.agilerl_onpolicy import (
            train_agent_on_policy,
        )

        return train_agent_on_policy(
            agent, env, learner_ids, target_steps, logger=logger,
        )

    @staticmethod
    def _eval_agent(
        agent: Any,
        env: Any,
        learner_ids: List[str],
        num_episodes: int = 3,
        opponent_policies: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int]:
        rewards: List[float] = []
        total_steps = 0
        learner_id_set = set(learner_ids)
        for _ in range(num_episodes):
            obs = env.reset()
            # Normalise: raw mpe2/PettingZoo parallel envs return (obs, info);
            # PettingZooWrapper returns the obs dict directly.
            if isinstance(obs, tuple):
                obs, _info = obs
            total = 0.0
            ep_steps = 0
            for _ in range(200):
                # Guard against an env that returns an empty obs dict on reset
                # (all agents immediately done) to avoid StopIteration below.
                if not obs:
                    break
                learner_obs = {a: ob for a, ob in obs.items() if a in learner_id_set}
                if len(learner_obs) < len(learner_ids):
                    _ref = (
                        next(iter(learner_obs.values()))
                        if learner_obs
                        else np.zeros(
                            env.observation_space(learner_ids[0]).shape,
                            dtype=np.float32,
                        )
                    )
                    for a in learner_ids:
                        if a not in learner_obs:
                            learner_obs[a] = np.zeros_like(_ref)
                try:
                    actions = agent.get_action(learner_obs, training=False)
                except TypeError:
                    actions = agent.get_action(learner_obs)
                result = actions[0] if isinstance(actions, tuple) else actions
                # Squeeze the leading (1,) batch dim so discrete envs receive
                # scalar actions (on-policy get_action returns (1,)-shaped
                # arrays); harmless for already-unbatched off-policy actions.
                step_actions: Dict[str, Any] = {}
                for a, act in result.items():
                    arr = np.asarray(act)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                    step_actions[a] = arr.item() if arr.shape == () else arr
                # Use sampled opponent policies for non-learner active agents
                # when available; fall back to random for any missing agent.
                for a in env.agents:
                    if a not in step_actions:
                        if opponent_policies and a in opponent_policies and a in obs:
                            try:
                                opp_act = np.asarray(
                                    opponent_policies[a].get_action(obs[a])
                                )
                                if opp_act.ndim > 0 and opp_act.shape[0] == 1:
                                    opp_act = opp_act[0]
                                step_actions[a] = (
                                    opp_act.item()
                                    if opp_act.shape == ()
                                    else opp_act
                                )
                            except Exception:
                                step_actions[a] = env.action_space(a).sample()
                        else:
                            step_actions[a] = env.action_space(a).sample()
                next_obs, rews, terms, truncs, _ = env.step(step_actions)
                total += sum(float(rews.get(a, 0.0)) for a in learner_ids)
                obs = next_obs
                ep_steps += 1
                if all(terms.get(a, False) for a in learner_ids) or all(
                    truncs.get(a, False) for a in learner_ids
                ):
                    break
            rewards.append(total)
            total_steps += ep_steps
        return (float(np.mean(rewards)) if rewards else 0.0), total_steps
