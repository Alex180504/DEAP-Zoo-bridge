"""Co-evolution evaluator.

Drives one episode where agents are partitioned across multiple
"sides". The learner side builds its policies from the evaluated
individual's genome; every other side contributes one sampled opponent
policy per agent, drawn from that side's
:class:`~deap_pettingzoo.baselines.coevolution.pools.OpponentPool`.

Rewards are partitioned per-side; the learner's fitness defaults to the
sum of its agents' rewards (symmetric competitive / cooperative with
additive fitness). Aggregation over ``k_opponents`` samples is
configurable (``'mean' | 'min' | 'max' | 'sum'``).

This evaluator reuses the same tolerant ``get_action`` invocation
semantics as :class:`MixedPolicyEvaluator` and consumes the env via
:class:`PettingZooWrapper`'s parallel API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.evaluators.base_evaluator import BaseEvaluator
from deap_pettingzoo.evaluators.mixed_policy import _invoke_get_action

from .pools import OpponentPool, OpponentSnapshot, rebuild_opponent_policy

#: Map ``(genome, agent_id, action_space) -> PolicyProtocol``. Matches the
#: learner factory signature used by :class:`MixedPolicyEvaluator` so the
#: same factory works in both evaluators.
LearnerPolicyFactory = Callable[[Any, str, Any], PolicyProtocol]


VALID_AGGREGATIONS = ("mean", "min", "max", "sum")


@dataclass
class GenomeIndividual:
    """Minimal picklable carrier satisfying the evaluator's ``.genome`` contract.

    :meth:`CoEvolutionEvaluator.run` requires an object exposing ``.genome``
    (DEAP individuals do so naturally). Paradigms whose population members
    are bare genomes — NEAT's ``neat.DefaultGenome``, for instance — wrap
    each one in this holder so they can use the shared evaluator.
    """

    genome: Any


# Worker-side state for parallel co-evolution


_COEV_WORKER_ENV_CACHE: Dict[Tuple[Any, ...], PettingZooWrapper] = {}


def _hashable_env_config(env_config: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
    if env_config is None:
        return ()
    # Build a stable, hashable tuple from the dict. We restrict to
    # primitive values; co-evolution env_configs are small and only
    # ever carry primitives in practice.
    return tuple(sorted((str(k), v) for k, v in env_config.items()))


def _resolve_worker_env(
    env_name: str,
    env_config: Optional[Dict[str, Any]],
    registry_snapshot: Optional[list] = None,
) -> PettingZooWrapper:
    """Return a process-local :class:`PettingZooWrapper`, cached by key.

    Note:
        ``_COEV_WORKER_ENV_CACHE`` is module-global and keyed on
        ``(env_name, _hashable(env_config))``.  Workers build an env
        lazily on first use and reuse it for the rest of the worker's
        lifetime.  Mirrors the ``_NUMPY_WORKER_CACHE`` pattern in
        ``contrib/qdax/scoring_numpy.py``.

        ``registry_snapshot`` is replayed once per worker before the env
        is built so Dask workers (no initializer hook) can resolve
        runtime-registered custom envs.
    """
    from deap_pettingzoo.environments.registry import (
        replay_registry_snapshot_once,
    )

    replay_registry_snapshot_once(registry_snapshot)
    key: Tuple[Any, ...] = (env_name, _hashable_env_config(env_config))
    cached = _COEV_WORKER_ENV_CACHE.get(key)
    if cached is not None:
        return cached
    env = PettingZooWrapper(env_name, env_config=env_config or None)
    _COEV_WORKER_ENV_CACHE[key] = env
    return env


def episode_seed_for(
    *,
    base_seed: Optional[int],
    episode_index: int,
    k_opponents: int,
    generation_index: int = 0,
    common_random_seeds: bool = False,
) -> Optional[int]:
    """Return the env reset seed for one evaluation episode.

    Single source of truth for episode seeding.  Every evaluation path —
    the serial loop, the per-individual parallel dispatch, the
    population-level batch dispatch, and the single-policy evaluator the
    QDax sides use — must call this, because they otherwise drift into
    different seeding rules that are invisible in the archive and silently
    change what a run measures.

    Two regimes:

    * ``common_random_seeds``: the seed depends only on the generation and
      the episode index, so every individual scored in that generation
      faces the same ``k`` starting states.  Between-individual fitness
      differences then reflect skill rather than which states each happened
      to draw.  The generation term keeps the run from training against one
      fixed set of initial conditions forever.
    * Otherwise: ``base_seed + episode_index`` when a base seed is given,
      and ``None`` (unseeded) when it is not.

    Args:
        base_seed: Caller-supplied base, or ``None`` for unseeded episodes.
        episode_index: Index of this episode within the individual's ``k``.
        k_opponents: Episodes per individual; the stride that keeps
            consecutive generations from reusing seeds.
        generation_index: Current generation, used only under CRN.
        common_random_seeds: Select the CRN regime.

    Returns:
        The seed to pass to ``env.reset``, or ``None`` to leave it unseeded.
    """
    if common_random_seeds:
        return generation_index * max(1, int(k_opponents)) + int(episode_index)
    if base_seed is None:
        return None
    return int(base_seed) + int(episode_index)


@dataclass
class _EpisodePayload:
    """Picklable payload for a single co-evolution episode.

    Driver-side: built once per ``(individual, episode_index)`` pair
    inside :meth:`CoEvolutionEvaluator.run`. Worker-side:
    :func:`_episode_rollout` consumes it.
    """

    env_name: str
    env_config: Optional[Dict[str, Any]]
    learner_genome: Any  # FixedTopologyGenome (picklable)
    learner_role_prefixes: Tuple[str, ...]
    learner_masked: bool
    #: Per-side opponent snapshot (one snapshot per side, not per agent).
    opponent_snapshots: Dict[str, OpponentSnapshot]
    #: Side metadata so the worker can rebuild ``side_agents`` without a driver round-trip.
    side_role_prefixes: Dict[str, Tuple[str, ...]]
    episode_seed: Optional[int]
    max_steps: int
    #: Registry snapshot replayed once per worker (Dask custom envs).
    registry_snapshot: Optional[list] = None
    #: Picklable descriptor of the learner phenotype for sides whose policy
    #: is not a plain MLP (NEAT family). ``None`` keeps the historical path:
    #: the worker builds a ``FeedforwardNeuralPolicy`` from ``learner_genome``.
    learner_snapshot: Optional[OpponentSnapshot] = None


@dataclass
class _EpisodeResult:
    """Return value of one :func:`_episode_rollout` call."""

    learner_total: float
    side_totals: Dict[str, float]
    steps: int


def _episode_rollout(payload: _EpisodePayload) -> _EpisodeResult:
    """Top-level worker function: run one co-evolution episode.

    Lives at module top level so it is picklable for
    :class:`ProcessPoolExecutor` and Dask. All state
    (env, genome, opponent policies) is rebuilt from the payload —
    nothing live crosses the process boundary.

    Note:
        Opponent policies are materialised fresh each episode: each
        side's :class:`OpponentSnapshot` is rebuilt into one shared
        policy covering all agents in that side — the canonical
        "one champion drives the whole team" contract, matching the
        driver-side serial path.
    """
    from deap_pettingzoo.policies import FeedforwardNeuralPolicy

    env = _resolve_worker_env(
        payload.env_name, payload.env_config, payload.registry_snapshot
    )
    agents = list(env.possible_agents)

    # Partition agents by role prefix (re-derived per episode; cheap).
    learner_agents: List[str] = [
        a for a in agents if any(a.startswith(p) for p in payload.learner_role_prefixes)
    ]
    side_agents: Dict[str, List[str]] = {
        name: [a for a in agents if any(a.startswith(p) for p in prefixes)]
        for name, prefixes in payload.side_role_prefixes.items()
    }
    agent_to_side: Dict[str, str] = {
        a: name for name, members in side_agents.items() for a in members
    }

    opponent_policies: Dict[str, PolicyProtocol] = {}
    for side_name, snapshot in payload.opponent_snapshots.items():
        members = side_agents.get(side_name, [])
        if not members:
            continue
        # All side members share an action space (validated by the
        # driver at side construction); use the first to resolve.
        action_space = env.action_space(members[0])
        side_policy = rebuild_opponent_policy(snapshot, action_space=action_space)
        for a in members:
            opponent_policies[a] = side_policy

    # Build learner policies fresh per episode (matches serial path). When
    # the side shipped a phenotype snapshot, build it *once* and share it
    # across the side's agents — the same contract opponents already use,
    # and the only affordable option for phenotypes whose construction is
    # expensive (ES-HyperNEAT's quadtree subdivision).
    learner_policies: Dict[str, PolicyProtocol]
    if payload.learner_snapshot is not None and learner_agents:
        learner_policy = rebuild_opponent_policy(
            payload.learner_snapshot,
            action_space=env.action_space(learner_agents[0]),
        )
        learner_policies = dict.fromkeys(learner_agents, learner_policy)
    else:
        learner_policies = {
            a: FeedforwardNeuralPolicy(
                payload.learner_genome,
                action_space=env.action_space(a),
                masked=payload.learner_masked,
            )
            for a in learner_agents
        }

    return _run_episode(
        env=env,
        learner_policies=learner_policies,
        opponent_policies=opponent_policies,
        agent_to_side=agent_to_side,
        side_names=list(payload.opponent_snapshots),
        seed=payload.episode_seed,
        max_steps=payload.max_steps,
    )


def _run_episode(
    *,
    env: PettingZooWrapper,
    learner_policies: Dict[str, PolicyProtocol],
    opponent_policies: Dict[str, PolicyProtocol],
    agent_to_side: Dict[str, str],
    side_names: Sequence[str],
    seed: Optional[int],
    max_steps: int,
) -> _EpisodeResult:
    """Inner episode loop shared by serial and worker paths."""
    observations = env.reset(seed=seed)
    for p in learner_policies.values():
        if hasattr(p, "reset"):
            p.reset()
    for p in opponent_policies.values():
        if hasattr(p, "reset"):
            p.reset()

    learner_total = 0.0
    side_totals: Dict[str, float] = {name: 0.0 for name in side_names}
    steps = 0
    while steps < max_steps and observations:
        infos = getattr(env, "last_infos", {}) or {}
        actions: Dict[str, Any] = {}
        for agent, obs in observations.items():
            mask = infos.get(agent, {}).get("action_mask")
            if agent in learner_policies:
                actions[agent] = _invoke_get_action(
                    learner_policies[agent], obs, mask, deterministic=True
                )
            elif agent in opponent_policies:
                actions[agent] = _invoke_get_action(opponent_policies[agent], obs, mask)

        observations, rewards, terms, truncs, _ = env.step(actions)

        for agent, reward in rewards.items():
            if agent in learner_policies:
                learner_total += float(reward)
            else:
                side = agent_to_side.get(agent)
                if side is not None:
                    side_totals[side] += float(reward)

        steps += 1
        if terms and all(terms.values()):
            break
        if truncs and all(truncs.values()):
            break

    return _EpisodeResult(
        learner_total=learner_total,
        side_totals=side_totals,
        steps=steps,
    )


@dataclass
class CoEvolutionEvalResult:
    """Per-call outcome returned by :meth:`CoEvolutionEvaluator.run`.

    Attributes:
        fitness: Aggregated learner fitness across the ``k_opponents``
            episodes (passed to DEAP).
        per_sample_learner: Per-sample learner rewards (length ``k``).
        per_sample_opponents: Per-sample opponent-side reward totals,
            keyed by side name (each value is length ``k``).
        steps: Total environment steps across all ``k`` episodes.
        num_samples: ``k_opponents`` actually run.
    """

    fitness: float
    per_sample_learner: List[float] = field(default_factory=list)
    per_sample_opponents: Dict[str, List[float]] = field(default_factory=dict)
    steps: int = 0
    num_samples: int = 0


class CoEvolutionEvaluator(BaseEvaluator):
    """Evaluate one individual against sampled opponents from opposing sides.

    Args:
        env_wrapper: :class:`PettingZooWrapper` around the target env.
        learner_role_prefixes: Agent-name prefixes controlled by the
            learner (e.g. ``('adversary',)`` to train pursuers against
            evaders). All other agents must be covered by
            ``opponent_sides``.
        learner_factory: Callable ``(genome, agent_id, action_space) ->
            PolicyProtocol`` building one fresh learner policy per
            learner agent per episode.
        opponent_sides: Ordered list of
            ``(side_name, role_prefixes, OpponentPool)`` tuples. Every
            non-learner agent must match exactly one side's prefixes.
        k_opponents: Number of opponent *tuples* to sample (one
            episode per tuple). Default ``3``.
        aggregation: How to collapse the ``k`` per-sample learner
            rewards into a single fitness. One of
            ``'mean' | 'min' | 'max' | 'sum'``. Default ``'mean'``.
        max_steps: Hard cap on env steps per episode.

    Raises:
        ValueError: On agent-coverage mismatch (unassigned or
            multi-assigned agents) or invalid ``aggregation``.
    """

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        learner_role_prefixes: Sequence[str],
        learner_factory: LearnerPolicyFactory,
        opponent_sides: Sequence[Tuple[str, Sequence[str], OpponentPool]],
        k_opponents: int = 3,
        aggregation: str = "mean",
        max_steps: int = 500,
        *,
        executor: Optional[Any] = None,
        env_name: Optional[str] = None,
        env_config: Optional[Dict[str, Any]] = None,
        learner_masked: bool = False,
        learner_snapshot_fn: Optional[Callable[[Any], Optional[OpponentSnapshot]]] = (
            None
        ),
        registry_snapshot: Optional[list] = None,
        common_random_seeds: bool = False,
        generation_index: int = 0,
    ) -> None:
        """Build a co-evolution evaluator.

        Parallel evaluation kwargs (all optional, all keyword-only):

        * ``executor`` — a context-managed executor matching the
          :func:`deap_pettingzoo.utils.parallel.make_executor` protocol.
          When provided **and** ``k_opponents > 1``, the
          ``k_opponents`` episodes inside one :meth:`evaluate` call are
          dispatched via ``executor.map``.
        * ``env_name`` / ``env_config`` — picklable env-construction
          metadata so workers can rebuild a process-local
          :class:`PettingZooWrapper`. Required when ``executor`` is
          set; ignored otherwise.
        * ``learner_masked`` — whether learner policies should attach
          a :class:`MaskedDiscreteHead` (turn-based envs). Workers
          use this to rebuild :class:`FeedforwardNeuralPolicy`
          locally; ignored on the serial path (which uses
          ``learner_factory`` directly).
        * ``learner_snapshot_fn`` — ``genome -> Optional[OpponentSnapshot]``
          describing the learner phenotype for workers. Returning ``None``
          (the default) keeps the :class:`FeedforwardNeuralPolicy` rebuild
          path; NEAT-family sides supply their own snapshot so the parallel
          path builds the same phenotype the driver would.
        * ``common_random_seeds`` / ``generation_index`` — episode-seeding
          policy, applied identically on the serial and parallel paths via
          :func:`episode_seed_for`. The trainer rebuilds the evaluator each
          generation and passes the current index.

        Raises:
            ValueError: on agent-coverage mismatch, invalid
                ``aggregation``, or ``executor`` set without
                ``env_name``.
        """
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation={aggregation!r}. "
                f"Must be one of {VALID_AGGREGATIONS}."
            )
        if k_opponents < 1:
            raise ValueError(f"k_opponents must be >= 1, got {k_opponents}.")
        if executor is not None and env_name is None:
            raise ValueError(
                "CoEvolutionEvaluator: executor= requires env_name= so "
                "workers can rebuild a process-local PettingZooWrapper."
            )

        self.env_wrapper = env_wrapper
        self.learner_role_prefixes: Tuple[str, ...] = tuple(learner_role_prefixes)
        self.learner_factory = learner_factory
        self.opponent_sides: List[Tuple[str, Tuple[str, ...], OpponentPool]] = [
            (name, tuple(prefixes), pool) for name, prefixes, pool in opponent_sides
        ]
        self.k_opponents = int(k_opponents)
        self.aggregation = aggregation
        self.max_steps = int(max_steps)
        self._executor = executor
        self._env_name = env_name
        self._env_config = env_config
        self._learner_masked = bool(learner_masked)
        self.learner_snapshot_fn = learner_snapshot_fn
        self._registry_snapshot = registry_snapshot
        self.common_random_seeds = bool(common_random_seeds)
        self.generation_index = int(generation_index)

        self._partition_agents()
        #: Populated after each :meth:`run` call for trainer logging.
        self.last_details: Optional[CoEvolutionEvalResult] = None
        #: Cumulative real env ticks across every :meth:`run` call made on
        #: this evaluator instance. Both the serial and the evaluator-side
        #: parallel paths funnel through :meth:`_finalise`, which advances
        #: this counter. The trainer builds one fresh evaluator per side per
        #: generation, so reading it after a population pass produces that
        #: side's real per-generation env-step count.
        self.env_steps_consumed: int = 0

    def _episode_seed(self, base_seed: Optional[int], episode_index: int) -> Optional[int]:
        """Seed for episode ``episode_index``; see :func:`episode_seed_for`."""
        return episode_seed_for(
            base_seed=base_seed,
            episode_index=episode_index,
            k_opponents=self.k_opponents,
            generation_index=self.generation_index,
            common_random_seeds=self.common_random_seeds,
        )

    # Agent partitioning

    def _partition_agents(self) -> None:
        """Validate that every env agent belongs to exactly one side."""
        agents = list(self.env_wrapper.possible_agents)
        learner_agents: List[str] = []
        side_agents: Dict[str, List[str]] = {
            name: [] for name, _, _ in self.opponent_sides
        }

        for agent in agents:
            matched: List[str] = []
            if any(agent.startswith(p) for p in self.learner_role_prefixes):
                matched.append("__learner__")
                learner_agents.append(agent)
            for side_name, prefixes, _ in self.opponent_sides:
                if any(agent.startswith(p) for p in prefixes):
                    matched.append(side_name)
                    side_agents[side_name].append(agent)

            if not matched:
                raise ValueError(
                    f"Agent {agent!r} is not claimed by the learner "
                    f"(prefixes={self.learner_role_prefixes!r}) or by any "
                    f"opponent side. Assign a role prefix that matches."
                )
            if len(matched) > 1:
                raise ValueError(
                    f"Agent {agent!r} is claimed by multiple sides "
                    f"({matched!r}); role prefixes must be disjoint."
                )

        if not learner_agents:
            raise ValueError(
                "CoEvolutionEvaluator: no agents matched "
                f"learner_role_prefixes={self.learner_role_prefixes!r}."
            )

        self.learner_agents: List[str] = learner_agents
        self.side_agents: Dict[str, List[str]] = side_agents

    # BaseEvaluator

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float]:
        """DEAP contract: return ``(fitness,)``.

        Args:
            individual: Must carry a ``.genome`` attribute.
            seed: Base seed for the underlying env (each of the ``k``
                sample episodes uses ``seed + i`` if ``seed`` is given).
            rng: Optional RNG used for opponent sampling. Defaults to a
                fresh ``np.random.default_rng(seed)`` — this keeps
                evaluations deterministic under a fixed ``seed``.
        """
        result = self.run(individual, seed=seed, rng=rng)
        return (result.fitness,)

    # Core loop

    def run(
        self,
        individual: Any,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> CoEvolutionEvalResult:
        """Run ``k_opponents`` episodes and aggregate learner fitness.

        When :attr:`_executor` is set and ``k_opponents > 1`` the
        episodes are dispatched in parallel: opponents are sampled on
        the driver (so pool RNGs / archive cursors mutate exactly once
        per :meth:`run` call regardless of worker count), then each
        episode is shipped as a picklable :class:`_EpisodePayload` to
        the executor. Aggregation always happens on the driver.

        Note:
            Parallel execution is enabled only when an executor is wired
            **and** ``k_opponents > 1``.  Single-episode evaluations skip
            the executor overhead entirely.
        """
        if (
            individual is None
            or not hasattr(individual, "genome")
            or individual.genome is None
        ):
            raise ValueError(
                "CoEvolutionEvaluator requires an individual with a 'genome'."
            )

        if rng is None:
            rng = np.random.default_rng(seed)

        use_parallel = self._executor is not None and self.k_opponents > 1

        if use_parallel:
            return self._run_parallel(individual, seed, rng)
        return self._run_serial(individual, seed, rng)

    def _run_serial(
        self,
        individual: Any,
        seed: Optional[int],
        rng: np.random.Generator,
    ) -> CoEvolutionEvalResult:
        """Run ``k_opponents`` episodes serially and aggregate results.

        Note:
            One opponent policy is sampled per side and shared across all
            agents in that side.  This matches the canonical self-play
            contract ("one champion drives the whole team") and reduces
            opponent-sampling variance compared to per-agent sampling.
        """
        learner_rewards: List[float] = []
        side_reward_samples: Dict[str, List[float]] = {
            name: [] for name, _, _ in self.opponent_sides
        }
        total_steps = 0

        for i in range(self.k_opponents):
            episode_seed = self._episode_seed(seed, i)
            opp_policies: Dict[str, PolicyProtocol] = {}
            for side_name, _prefixes, pool in self.opponent_sides:
                side_policy = pool.sample_opponent(rng)
                for agent in self.side_agents[side_name]:
                    opp_policies[agent] = side_policy

            learner_r, side_r, steps = self._run_one_episode(
                individual.genome, opp_policies, episode_seed
            )
            learner_rewards.append(learner_r)
            for side_name, total in side_r.items():
                side_reward_samples[side_name].append(total)
            total_steps += steps

        return self._finalise(learner_rewards, side_reward_samples, total_steps)

    def _run_parallel(
        self,
        individual: Any,
        seed: Optional[int],
        rng: np.random.Generator,
    ) -> CoEvolutionEvalResult:
        """Dispatch ``k_opponents`` episodes to :attr:`_executor` and aggregate.

        Note:
            All ``k`` opponent snapshots per side are taken on the driver
            before any worker is launched so pool RNGs and archive cursors
            mutate exactly once per call.  The per-side snapshot lists are
            then transposed to one ``{side: snapshot}`` dict per episode
            before building :class:`_EpisodePayload` objects.
        """
        per_side_snapshots: Dict[str, List[OpponentSnapshot]] = {}
        side_role_prefixes: Dict[str, Tuple[str, ...]] = {}
        for side_name, prefixes, pool in self.opponent_sides:
            per_side_snapshots[side_name] = pool.snapshot_opponents(
                rng, self.k_opponents
            )
            side_role_prefixes[side_name] = prefixes

        # Built once per call, not per episode: the descriptor is identical
        # across the k episodes and can be expensive to produce.
        learner_snapshot = (
            None
            if self.learner_snapshot_fn is None
            else self.learner_snapshot_fn(individual.genome)
        )

        payloads: List[_EpisodePayload] = []
        for i in range(self.k_opponents):
            episode_seed = self._episode_seed(seed, i)
            episode_snaps = {
                name: per_side_snapshots[name][i] for name in per_side_snapshots
            }
            payloads.append(
                _EpisodePayload(
                    env_name=str(self._env_name),
                    env_config=self._env_config,
                    learner_genome=individual.genome,
                    learner_role_prefixes=self.learner_role_prefixes,
                    learner_masked=self._learner_masked,
                    opponent_snapshots=episode_snaps,
                    side_role_prefixes=side_role_prefixes,
                    episode_seed=episode_seed,
                    max_steps=self.max_steps,
                    registry_snapshot=self._registry_snapshot,
                    learner_snapshot=learner_snapshot,
                )
            )

        results: List[_EpisodeResult] = list(
            self._executor.map(_episode_rollout, payloads)
        )

        learner_rewards = [r.learner_total for r in results]
        side_reward_samples: Dict[str, List[float]] = {
            name: [] for name, _, _ in self.opponent_sides
        }
        total_steps = 0
        for r in results:
            for side_name, total in r.side_totals.items():
                side_reward_samples[side_name].append(total)
            total_steps += r.steps

        return self._finalise(learner_rewards, side_reward_samples, total_steps)

    def _finalise(
        self,
        learner_rewards: List[float],
        side_reward_samples: Dict[str, List[float]],
        total_steps: int,
    ) -> CoEvolutionEvalResult:
        fitness = self._aggregate(learner_rewards)
        details = CoEvolutionEvalResult(
            fitness=fitness,
            per_sample_learner=learner_rewards,
            per_sample_opponents=side_reward_samples,
            steps=total_steps,
            num_samples=self.k_opponents,
        )
        self.last_details = details
        # Accumulate real env ticks so the trainer can surface a per-side,
        # per-generation consumed-steps figure (serial + evaluator-parallel
        # both reach here).
        self.env_steps_consumed += int(total_steps)
        return details

    def _aggregate(self, rewards: List[float]) -> float:
        arr = np.asarray(rewards, dtype=np.float64)
        if self.aggregation == "mean":
            return float(arr.mean())
        if self.aggregation == "min":
            return float(arr.min())
        if self.aggregation == "max":
            return float(arr.max())
        # "sum"
        return float(arr.sum())

    def _run_one_episode(
        self,
        genome: Any,
        opponent_policies: Dict[str, PolicyProtocol],
        seed: Optional[int],
    ) -> Tuple[float, Dict[str, float], int]:
        """Run one rollout with the pre-sampled opponent policies."""
        observations = self.env_wrapper.reset(seed=seed)

        # Build learner policies fresh per episode (stateful heads etc.).
        learner_policies: Dict[str, PolicyProtocol] = {
            agent: self.learner_factory(
                genome, agent, self.env_wrapper.action_space(agent)
            )
            for agent in self.learner_agents
        }
        for p in learner_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        for p in opponent_policies.values():
            if hasattr(p, "reset"):
                p.reset()

        learner_total = 0.0
        side_totals: Dict[str, float] = {
            name: 0.0 for name, _, _ in self.opponent_sides
        }
        agent_to_side: Dict[str, str] = {}
        for side_name, agents in self.side_agents.items():
            for a in agents:
                agent_to_side[a] = side_name

        steps = 0
        while steps < self.max_steps and observations:
            infos = getattr(self.env_wrapper, "last_infos", {}) or {}
            actions: Dict[str, Any] = {}
            for agent, obs in observations.items():
                mask = infos.get(agent, {}).get("action_mask")
                if agent in learner_policies:
                    actions[agent] = _invoke_get_action(
                        learner_policies[agent], obs, mask, deterministic=True
                    )
                elif agent in opponent_policies:
                    actions[agent] = _invoke_get_action(
                        opponent_policies[agent], obs, mask
                    )

            observations, rewards, terms, truncs, _ = self.env_wrapper.step(actions)

            for agent, reward in rewards.items():
                if agent in learner_policies:
                    learner_total += float(reward)
                else:
                    side = agent_to_side.get(agent)
                    if side is not None:
                        side_totals[side] += float(reward)

            steps += 1
            if terms and all(terms.values()):
                break
            if truncs and all(truncs.values()):
                break

        return learner_total, side_totals, steps


__all__ = [
    "CoEvolutionEvaluator",
    "CoEvolutionEvalResult",
    "GenomeIndividual",
    "LearnerPolicyFactory",
]
