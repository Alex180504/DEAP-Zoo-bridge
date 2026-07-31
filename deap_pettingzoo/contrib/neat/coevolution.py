"""NEAT co-evolution side — allows NEAT populations to participate in coevolution.

:class:`NeatSideState` wraps a :class:`neat.Population` behind the
:class:`SideState` protocol so NEAT can co-evolve alongside EA, CMA-ES,
PSO, RL, or other NEAT sides.

Key design notes:

* Each :meth:`step_generation` call advances the NEAT population by
  ``n_generations_per_step`` generations (default 1).
* Fitness comes from the trainer-supplied
  :class:`~deap_pettingzoo.baselines.coevolution.evaluator.CoEvolutionEvaluator`
  — the same evaluator every other side uses — so genomes are scored
  against opponents sampled from the *other* sides' pools, on the learner
  side's own rewards, under the configured episode cap and with the
  evaluator's common-random-numbers seeding.
* NEAT drives structural mutation from the **global** ``random`` module,
  so every call into ``neat`` runs inside :meth:`NeatSideState._isolated_rng`,
  which swaps in this side's own stream and hands the caller's back.
* :meth:`current_pool` returns a pool that samples genomes from the current
  generation (blended with the side's hall of fame when configured),
  rebuilding phenotypes on demand.
* Parallel-evaluation support is provided via :class:`OpponentSnapshot`
  with kind ``"neat"`` (``"hyperneat"`` / ``"es_hyperneat"`` in the
  subclasses), for opponents *and* for the learner itself.
"""

from __future__ import annotations

import copy
import random as _random
import time
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.coevolution.evaluator import GenomeIndividual
from deap_pettingzoo.baselines.coevolution.pools import (
    ArchivePool,
    CompositePool,
    OpponentPool,
    OpponentSnapshot,
)
from deap_pettingzoo.baselines.coevolution.samplers import Sampler, uniform_sampler
from deap_pettingzoo.baselines.coevolution.sides import (
    GenerationStats,
    Side,
    SideState,
    _resolve_side_action_space,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

from .config import build_neat_config
from .policy import NeatPolicy


class NeatSideState(SideState):
    """NEAT co-evolution side — wraps ``neat.Population``.

    Accepted ``optimizer_config`` keys (all optional):

    * ``pop_size`` (int, default 50).
    * ``num_hidden`` (int, default 0) — initial hidden nodes in the NEAT
      genome config; NEAT adds more via structural mutation regardless.
    * ``seed`` (int or None).
    * ``neat_config_overrides`` (dict) — section->key->value dict merged
      into :func:`build_neat_config`.
    * ``masked`` (bool, default ``False``) — attach a
      :class:`MaskedDiscreteHead` (turn-based envs).
    * ``n_generations_per_step`` (int, default 1) — how many NEAT
      generations run inside one :meth:`step_generation`. One
      ``step_generation`` therefore costs
      ``n_generations_per_step * pop_size * k_opponents`` episodes, all
      scored against the **same** opponent-pool snapshot — inner
      generations 2..N face increasingly stale opponents. Leave at 1
      unless that staleness is what you want.

    Note:
        ``eval_episodes_per_genome`` is deliberately *not* honoured here.
        Under co-evolution the number of episodes per genome is
        ``CoEvolutionConfig.k_opponents`` (which also decides *which*
        opponents each episode faces); a second multiplier would silently
        inflate the side's budget. The key stays live and load-bearing for
        the standalone NEAT / HyperNEAT trainers.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self.pop_size = int(cfg.get("pop_size", 50))
        self.num_hidden = int(cfg.get("num_hidden", 0))
        self.seed = cfg.get("seed", None)
        self.neat_overrides = dict(cfg.get("neat_config_overrides", {}))
        self.n_gens_per_step = int(cfg.get("n_generations_per_step", 1))
        if "eval_episodes_per_genome" in cfg:
            warnings.warn(
                f"Side {side.name!r} (optimizer={side.optimizer!r}): "
                "'eval_episodes_per_genome' is ignored by co-evolution sides. "
                "Episodes per genome are set by CoEvolutionConfig.k_opponents; "
                "drop the key here (it remains live for the standalone NEAT / "
                "HyperNEAT trainers).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._population_obj: Optional[neat.Population] = None
        self._neat_config: Optional[neat.Config] = None
        self._neat_seed: Optional[int] = None
        self._action_space: Any = None
        self._env: Optional[PettingZooWrapper] = None
        self._n_generations_run: int = 0

        self._genomes_by_id: Dict[int, neat.DefaultGenome] = {}
        self._best_genome: Optional[neat.DefaultGenome] = None
        self._best_genome_fitness: float = -np.inf
        self._evaluator: Any = None
        #: Single-slot phenotype memo (``genome.key``, policy) — see
        #: :meth:`_memoised_phenotype`.
        self._cached_phenotype_key: Any = None
        self._cached_phenotype: Optional[PolicyProtocol] = None

    # RNG isolation

    @contextmanager
    def _isolated_rng(self) -> Iterator[None]:
        """Run NEAT internals against this side's own RNG streams.

        ``neat`` mutates genomes from the **global** ``random`` module and
        ``neat.Population.__init__`` calls ``random.seed(...)`` outright, so
        without this two NEAT sides in one run would share — and clobber —
        a single stream. Entering swaps in this side's saved state; leaving
        stores where the side got to and hands the caller's stream back
        untouched. This is the continuity pattern EA / CMA-ES / PSO already
        use, and it is what keeps consecutive generations from replaying
        nearly the same mutation draws.
        """
        caller_py = _random.getstate()
        caller_np = np.random.get_state()
        if self._py_state is not None:
            _random.setstate(self._py_state)
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        try:
            yield
        finally:
            self._py_state = _random.getstate()
            self._np_state = np.random.get_state()
            _random.setstate(caller_py)
            np.random.set_state(caller_np)

    # SideState protocol

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
        self._neat_seed = self._resolve_neat_seed(rng)
        input_size, output_size = self._env_io_sizes(env)
        self._neat_config = self._build_config(input_size, output_size)
        self._create_population()

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        """Advance the NEAT population by ``n_generations_per_step`` generations.

        Follows the canonical side pattern: build this generation's
        evaluator, bind it, let NEAT's own ``Population.run`` drive
        speciation / reproduction, and report the real env ticks the
        evaluator consumed.
        """
        if self._population_obj is None:
            raise RuntimeError(
                f"{type(self).__name__}: initialise() must be called before "
                "step_generation()."
            )
        gen_t0 = time.time()
        self.generation += 1
        evaluator = evaluator_factory(self)
        self.rebind_evaluator(evaluator)

        captured: Dict[str, float] = {}

        def _fitness_func(genomes_and_ids: list, neat_cfg: neat.Config) -> None:
            del neat_cfg  # The side owns its config; NEAT hands back the same one.
            genomes = [genome for _genome_id, genome in genomes_and_ids]
            fitnesses = self._score_genomes(genomes, batch_eval_fn=batch_eval_fn)
            # Stats are captured here rather than read back from
            # Population.population afterwards: neat reproduces *after*
            # evaluating, and only the handful of carried-over elites keep
            # their fitness — reading back computes every statistic over the
            # top few percent of the population.
            arr = np.asarray(fitnesses, dtype=np.float64)
            captured.update(
                mean=float(arr.mean()),
                std=float(arr.std()),
                min=float(arr.min()),
                max=float(arr.max()),
            )
            self._n_generations_run += 1

        with self._isolated_rng():
            try:
                self._population_obj.run(_fitness_func, n=self.n_gens_per_step)
            except neat.CompleteExtinctionException:
                # Every species died out; keep whatever the last completed
                # generation scored rather than aborting the whole run.
                pass

        self._genomes_by_id = dict(self._population_obj.population)
        if not captured:  # Extinction before any generation completed.
            captured = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        # ``best_fitness`` (the running max of generation bests) is advanced by
        # ``_record_best`` inside the fitness function.

        stats = GenerationStats(
            generation=self.generation,
            side_name=self.side.name,
            mean=captured["mean"],
            std=captured["std"],
            min=captured["min"],
            max=captured["max"],
            best_fitness=captured["max"],
            # Real env ticks this side consumed. The trainer builds one
            # evaluator per side per generation, so the evaluator's
            # cumulative counter already totals every inner NEAT generation.
            env_steps=int(getattr(evaluator, "env_steps_consumed", 0)),
            wall_time_s=time.time() - gen_t0,
        )
        self.history.append(stats)
        return stats

    def rebind_evaluator(self, evaluator: Any) -> None:
        """Bind the evaluator this side scores its genomes against.

        Called by the trainer before every generation (and once during
        bootstrap) so opponents reflect the other sides' latest state.
        """
        self._evaluator = evaluator

    def evaluate_population(self, *, batch_eval_fn=None) -> None:
        """Score every genome in the current population.

        The trainer's bootstrap pass calls this so generation 1 starts from
        real fitnesses instead of ``None`` — without it ``_update_best``
        falls back to picking a genome arbitrarily and the initial
        hall-of-fame snapshot is meaningless.
        """
        if self._population_obj is None:
            raise RuntimeError(
                f"{type(self).__name__}: initialise() must be called first."
            )
        self._score_genomes(
            list(self._population_obj.population.values()),
            batch_eval_fn=batch_eval_fn,
        )
        self._genomes_by_id = dict(self._population_obj.population)

    def current_pool(self) -> OpponentPool:
        """Return the pool other sides sample this side's opponents from.

        Honours ``use_archive_in_opponent_pool`` / ``archive_mix`` /
        ``opponent_sampler`` exactly like the EA, CMA-ES and PSO sides.
        """
        if not self._genomes_by_id or self._neat_config is None:
            raise RuntimeError(
                f"{type(self).__name__}: initialise() must be called before "
                "current_pool()."
            )
        pop_pool: OpponentPool = _NeatPopulationPool(
            list(self._genomes_by_id.values()),
            self._opponent_policy_factory(),
            snapshot_fn=self._phenotype_snapshot,
            sampler=self.side.opponent_sampler,
        )
        if self.side.use_archive_in_opponent_pool and self.side.hof is not None:
            if len(self.side.hof) == 0:
                # Archive empty -> behave as if pure current-gen pool.
                return pop_pool
            archive_pool = ArchivePool(
                self.side.hof,
                self._opponent_policy_factory(),
                sampler=self.side.archive_sampler,
                snapshot_fn=self._phenotype_snapshot,
            )
            weights = (1.0 - self.side.archive_mix, self.side.archive_mix)
            return CompositePool([pop_pool, archive_pool], weights=weights)
        return pop_pool

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        del env  # Action space was resolved at initialise().
        if self._best_genome is None or self._neat_config is None:
            raise RuntimeError(
                f"{type(self).__name__}: no best genome available. "
                "Call initialise() first."
            )
        # Champion / probe rollouts decode deterministically even on sides
        # that train through a sampling head.
        return self._build_phenotype(
            self._best_genome, self._action_space, stochastic=False
        )

    def best_snapshot(self) -> Any:
        """Return a *deep-copied* genome of the current best (HoF-safe)."""
        if self._best_genome is None:
            raise RuntimeError(f"{type(self).__name__}: no best genome available.")
        return copy.deepcopy(self._best_genome)

    # Learner-policy hooks (SideState)

    def learner_policy_factory(self) -> Callable[[Any, str, Any], PolicyProtocol]:
        def _factory(genome: Any, agent_id: str, action_space: Any) -> PolicyProtocol:
            del agent_id  # Homogeneous side — one phenotype drives every agent.
            return self._memoised_phenotype(genome, action_space)

        return _factory

    def learner_snapshot(self, genome: Any) -> OpponentSnapshot:
        return self._phenotype_snapshot(genome)

    # Phenotype hooks — the only thing NEAT / HyperNEAT / ES-HyperNEAT differ in

    def _build_phenotype(
        self,
        genome: neat.DefaultGenome,
        action_space: Any,
        *,
        stochastic: bool = True,
    ) -> PolicyProtocol:
        """Construct the executable policy for one genome.

        Args:
            genome: The genome to express.
            action_space: Action space of the agent(s) this policy drives.
            stochastic: When ``False``, force deterministic decoding — used
                for champion and probe rollouts on sides configured with a
                sampling head. Plain NEAT has no such head, so the flag is
                a no-op here.
        """
        del stochastic
        return NeatPolicy(
            genome,
            self._neat_config,
            action_space=action_space,
            masked=self.masked,
        )

    def _phenotype_snapshot(self, genome: neat.DefaultGenome) -> OpponentSnapshot:
        """Picklable descriptor letting a worker rebuild :meth:`_build_phenotype`."""
        assert self._neat_config is not None
        return OpponentSnapshot(
            kind="neat",
            payload={
                "genome": genome,
                "config_input_size": self._neat_config.genome_config.num_inputs,
                "config_output_size": self._neat_config.genome_config.num_outputs,
                "masked": self.masked,
            },
        )

    def _build_config(self, input_size: int, output_size: int) -> neat.Config:
        """Build this side's NEAT config (CPPN variants override)."""
        return build_neat_config(
            input_size=input_size,
            output_size=output_size,
            pop_size=self.pop_size,
            seed=self._neat_seed,
            overrides=self._config_overrides(),
        )

    # Internal helpers

    def _env_io_sizes(self, env: PettingZooWrapper) -> Tuple[int, int]:
        """Return ``(observation_size, output_size)`` for this side's agents."""
        agents = [
            a for a in env.possible_agents if a.startswith(self.side.role_prefixes)
        ]
        obs_shape = env.observation_space(agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        return input_size, required_output_size(self._action_space)

    def _config_overrides(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Merge ``num_hidden`` into the caller's NEAT config overrides."""
        genome_overrides = dict(self.neat_overrides.get("DefaultGenome", {}))
        genome_overrides["num_hidden"] = self.num_hidden
        overrides = dict(self.neat_overrides)
        overrides["DefaultGenome"] = genome_overrides
        return overrides or None

    def _resolve_neat_seed(self, rng: Optional[np.random.Generator]) -> Optional[int]:
        """Derive this side's NEAT seed from the trainer-provided RNG.

        Deriving — rather than reusing ``optimizer_config['seed']`` directly
        — is what keeps two identically-configured NEAT sides from evolving
        perfectly correlated populations under one trainer seed. Mirrors
        what :class:`CMAESSideState` and :class:`PSOSideState` already do.
        """
        if rng is not None:
            return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        if self.seed is not None:
            return int(self.seed)
        return None

    def _create_population(self) -> None:
        """Instantiate ``neat.Population`` under this side's RNG stream."""
        with self._isolated_rng():
            self._population_obj = neat.Population(
                self._neat_config,
                seed=self._neat_seed,
            )
        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

    def _opponent_policy_factory(self) -> Callable[[Any], PolicyProtocol]:
        """``genome -> policy`` bound to this side's own action space."""
        return partial(self._build_phenotype, action_space=self._action_space)

    def _memoised_phenotype(
        self, genome: neat.DefaultGenome, action_space: Any
    ) -> PolicyProtocol:
        """Build — or reuse — the phenotype for *genome*.

        The evaluator asks for a learner policy per agent per episode, so a
        genome would otherwise be expressed ``k_opponents * n_agents`` times
        per evaluation; ES-HyperNEAT runs a full quadtree subdivision on
        every one of those. Sharing a single instance across the side's
        agents is safe — every phenotype resets its network on entry to
        ``get_action`` — and it mirrors the opponent contract, where one
        sampled policy already drives the whole side.

        A single slot suffices (and keeps memory flat): individuals are
        evaluated one at a time, and the side's agents all share one action
        space, so ``genome.key`` alone identifies the cached build.
        """
        key = getattr(genome, "key", id(genome))
        if self._cached_phenotype is None or self._cached_phenotype_key != key:
            self._cached_phenotype_key = key
            self._cached_phenotype = self._build_phenotype(genome, action_space)
        return self._cached_phenotype

    def _score_genomes(
        self,
        genomes: List[neat.DefaultGenome],
        *,
        batch_eval_fn=None,
    ) -> List[float]:
        """Evaluate *genomes* against the bound evaluator, assigning fitness."""
        if self._evaluator is None:
            raise RuntimeError(
                f"{type(self).__name__}: rebind_evaluator() must be called "
                "before genomes can be scored."
            )
        self._cached_phenotype_key = None
        self._cached_phenotype = None
        individuals = [GenomeIndividual(genome=g) for g in genomes]
        if batch_eval_fn is not None and individuals:
            fitnesses = [float(f[0]) for f in batch_eval_fn(individuals)]
        else:
            fitnesses = [float(self._evaluator.evaluate(ind)[0]) for ind in individuals]
        for genome, fitness in zip(genomes, fitnesses):
            genome.fitness = fitness
        self._record_best(genomes)
        return fitnesses

    def _record_best(self, genomes: Iterable[neat.DefaultGenome]) -> None:
        """Track the best-ever evaluated genome across generations.

        Also advances :attr:`best_fitness`, which is what
        :meth:`SideState.snapshot_best` pairs with the archived genome —
        including on the bootstrap pass, before any ``step_generation`` runs.
        """
        for genome in genomes:
            fitness = getattr(genome, "fitness", None)
            if fitness is not None and float(fitness) > self._best_genome_fitness:
                self._best_genome_fitness = float(fitness)
                self._best_genome = genome
        self.best_fitness = max(self.best_fitness, self._best_genome_fitness)

    def _update_best(self) -> None:
        """Seed :attr:`_best_genome` from a freshly created population.

        Before the first evaluation no genome carries a fitness, so
        :meth:`best_policy` would have nothing to return; pick the first
        genome so the side is usable from generation 0.
        """
        if self._best_genome is None and self._genomes_by_id:
            self._best_genome = next(iter(self._genomes_by_id.values()))


# Internal pool implementation


class _NeatPopulationPool(OpponentPool):
    """Population-pool adapter over a list of NEAT genomes.

    One class serves plain NEAT, HyperNEAT and ES-HyperNEAT: the owning side
    supplies both the live ``policy_factory`` and the ``snapshot_fn`` that
    produces the matching picklable payload for worker processes, so nothing
    about the phenotype leaks into the pool.
    """

    def __init__(
        self,
        genomes: Iterable[neat.DefaultGenome],
        policy_factory: Callable[[neat.DefaultGenome], PolicyProtocol],
        *,
        snapshot_fn: Callable[[neat.DefaultGenome], OpponentSnapshot],
        sampler: Sampler = uniform_sampler,
    ) -> None:
        self._genomes: List[neat.DefaultGenome] = list(genomes)
        self._factory = policy_factory
        self._snapshot_fn = snapshot_fn
        self._sampler = sampler

    def _fitnesses(self) -> Optional[np.ndarray]:
        """Per-genome fitness for the sampler; unevaluated genomes score -inf."""
        if not self._genomes:
            return None
        return np.asarray(
            [
                -np.inf if getattr(g, "fitness", None) is None else float(g.fitness)
                for g in self._genomes
            ],
            dtype=np.float64,
        )

    def _sample_index(
        self, rng: np.random.Generator, fitnesses: Optional[np.ndarray]
    ) -> int:
        return self._sampler(len(self._genomes), fitnesses, rng)

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        if not self._genomes:
            raise ValueError("_NeatPopulationPool.sample_opponent: pool is empty.")
        return self._factory(self._genomes[self._sample_index(rng, self._fitnesses())])

    def sample_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[PolicyProtocol]:
        if k < 1:
            raise ValueError(f"sample_opponents: k must be >= 1, got {k}")
        return [self.sample_opponent(rng) for _ in range(k)]

    def snapshot_opponents(
        self, rng: np.random.Generator, k: int
    ) -> List[OpponentSnapshot]:
        if k < 1:
            raise ValueError(f"snapshot_opponents: k must be >= 1, got {k}")
        if not self._genomes:
            raise ValueError("_NeatPopulationPool.snapshot_opponents: pool is empty.")
        fitnesses = self._fitnesses()
        return [
            self._snapshot_fn(self._genomes[self._sample_index(rng, fitnesses)])
            for _ in range(k)
        ]

    def update(self, population_like: Any) -> None:
        if isinstance(population_like, dict):
            self._genomes = list(population_like.values())
        elif hasattr(population_like, "__iter__"):
            self._genomes = list(population_like)
        else:
            raise TypeError(
                f"_NeatPopulationPool.update expects a dict or iterable, "
                f"got {type(population_like)}"
            )

    def __len__(self) -> int:
        return len(self._genomes)
