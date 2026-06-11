"""NEAT co-evolution side — allows NEAT populations to participate in coevolution.

:class:`NeatSideState` wraps a :class:`neat.Population` behind the
:class:`SideState` protocol so NEAT can co-evolve alongside EA, CMA-ES,
PSO, RL, or other NEAT sides.

Key design notes:

* Each :meth:`step_generation` call advances the NEAT population by
  **one generation** (one tournament / speciation cycle).
* The side saves/restores ``random.getstate()`` and ``numpy.random.get_state()``
  around NEAT internal calls so that different sides' RNG streams stay
  independent (NEAT uses the global ``random`` module internally).
* :meth:`current_pool` returns a :class:`PopulationPool` that samples
  genomes uniformly from the current generation, rebuilding
  :class:`NeatPolicy` instances on demand.
* Parallel-evaluation support is provided via
  :class:`OpponentSnapshot` with kind ``"neat"``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import neat  # guarded by _require_neat

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import required_output_size

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

from .config import build_neat_config
from .policy import NeatPolicy


class NeatSideState(SideState):
    """NEAT co-evolution side — wraps ``neat.Population``.

    Accepted ``optimizer_config`` keys (all optional):

    * ``pop_size`` (int, default 50).
    * ``hidden_size`` — renamed to ``num_hidden`` and set in the NEAT
      genome config (overrides default 0 which means no hidden nodes
      initially; NEAT will add nodes via mutation regardless).
    * ``seed`` (int or None).
    * ``neat_config_overrides`` (dict) — section→key→value dict merged
      into :func:`build_neat_config`.
    * ``eval_episodes_per_genome`` (int, default 1).
    * ``n_generations_per_step`` (int, default 1) — how many NEAT
      generations to run inside one :meth:`step_generation`.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self.pop_size = int(cfg.get("pop_size", 50))
        self.num_hidden = int(cfg.get("num_hidden", 0))
        self.seed = cfg.get("seed", None)
        self.neat_overrides = dict(cfg.get("neat_config_overrides", {}))
        self.eval_eps = int(cfg.get("eval_episodes_per_genome", 1))
        self.n_gens_per_step = int(cfg.get("n_generations_per_step", 1))

        self._population_obj: Optional[neat.Population] = None
        self._neat_config: Optional[neat.Config] = None
        self._action_space: Any = None
        self._env: Optional[PettingZooWrapper] = None
        self._n_generations_run: int = 0

        self._genomes_by_id: Dict[int, neat.DefaultGenome] = {}
        self._best_genome: Optional[neat.DefaultGenome] = None

    # ------------------------------------------------------------------
    # SideState protocol
    # ------------------------------------------------------------------

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

        self._action_space = _resolve_side_action_space(
            env, self.side.role_prefixes
        )

        agents = [
            a for a in env.possible_agents
            if a.startswith(self.side.role_prefixes)
        ]
        obs_shape = env.observation_space(agents[0]).shape
        input_size = int(obs_shape[0]) if obs_shape else 1
        output_size = required_output_size(self._action_space)

        # Apply genome overrides
        genome_overrides = dict(self.neat_overrides.get("DefaultGenome", {}))
        genome_overrides["num_hidden"] = self.num_hidden

        overrides = dict(self.neat_overrides)
        overrides["DefaultGenome"] = genome_overrides

        self._neat_config = build_neat_config(
            input_size=input_size,
            output_size=output_size,
            pop_size=self.pop_size,
            seed=self.seed,
            overrides=overrides if overrides else None,
        )

        self._population_obj = neat.Population(
            self._neat_config,
            seed=self.seed if self.seed is not None else None,
        )
        # Initialise the genome cache from the fresh population.
        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

    def step_generation(
        self,
        evaluator_factory: Callable[["SideState"], Any],
        *,
        batch_eval_fn=None,
    ) -> GenerationStats:
        """Advance the NEAT population by one or more generations.

        The *evaluator_factory* is used to build an evaluator that can
        compute fitness for each genome.  NEAT's internal generation
        loop is driven by :meth:`neat.Population.run`, but each
        co-evolution step calls it with ``n=self.n_gens_per_step``.
        """
        if self._population_obj is None:
            raise RuntimeError(
                "NeatSideState: initialise() must be called before "
                "step_generation()."
            )

        evaluator = evaluator_factory(self)
        env: PettingZooWrapper = getattr(evaluator, "_env", self._env)

        def _fitness_func(
            genomes_and_ids: list, neat_cfg: neat.Config
        ) -> None:
            gen_fitnesses: list[float] = []
            for _genome_id, genome in genomes_and_ids:
                policy = NeatPolicy(
                    genome, neat_cfg, action_space=self._action_space
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
                            actions[agent] = policy.get_action(
                                observations[agent]
                            )
                        observations, step_rewards, dones, truncateds, _infos = (
                            env.step(actions)
                        )
                        ep_rew += float(sum(step_rewards.values()))
                        done = all(dones.values()) or all(truncateds.values())
                        _steps += 1
                    rewards.append(ep_rew)
                genome.fitness = float(np.mean(rewards))
                gen_fitnesses.append(genome.fitness)

        # Save/restore RNG state for side isolation
        old_random_state = _save_random_state()
        try:
            for _ in range(self.n_gens_per_step):
                self._population_obj.run(
                    _fitness_func, n=1
                )
                self._n_generations_run += 1
        except neat.CompleteExtinctionException:
            pass  # Population will handle extinction per config
        finally:
            _restore_random_state(old_random_state)

        # Cache current genomes
        self._genomes_by_id = dict(self._population_obj.population)
        self._update_best()

        # Build stats
        fitnesses = [
            g.fitness for g in self._genomes_by_id.values()
            if g.fitness is not None
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
        )

    def current_pool(self) -> OpponentPool:
        """Return a :class:`PopulationPool` over the current NEAT population.

        The pool's policy factory rebuilds :class:`NeatPolicy` on
        demand, and :meth:`snapshot_opponents` produces kind ``"neat"``
        snapshots for parallel evaluation.
        """
        if self._genomes_by_id is None or self._neat_config is None:
            raise RuntimeError(
                "NeatSideState: initialise() must be called before "
                "current_pool()."
            )

        def _factory(genome: neat.DefaultGenome) -> PolicyProtocol:
            return NeatPolicy(
                genome, self._neat_config, action_space=self._action_space
            )

        # Build a PopulationPool-esque object.
        # We implement the pool protocol directly.
        return _NeatPopulationPool(
            genomes_by_id=dict(self._genomes_by_id),
            policy_factory=_factory,
            neat_config=self._neat_config,
        )

    def best_policy(self, env: PettingZooWrapper) -> PolicyProtocol:
        if self._best_genome is None or self._neat_config is None:
            raise RuntimeError(
                "NeatSideState: no best genome available. "
                "Call initialise() first."
            )
        return NeatPolicy(
            self._best_genome,
            self._neat_config,
            action_space=self._action_space,
        )

    def best_snapshot(self) -> Any:
        """Return a picklable representation of the best genome."""
        if self._best_genome is None:
            raise RuntimeError(
                "NeatSideState: no best genome available."
            )
        return self._best_genome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_best(self) -> None:
        """Update the cached best genome from the current population."""
        best: Optional[neat.DefaultGenome] = None
        best_fit = -np.inf
        for genome in self._genomes_by_id.values():
            fit = getattr(genome, "fitness", None)
            if fit is not None and fit > best_fit:
                best_fit = fit
                best = genome
        if best is not None:
            self._best_genome = best
        elif self._genomes_by_id:
            # No genomes evaluated yet (e.g. fresh population) —
            # pick the first one arbitrarily.
            self._best_genome = next(iter(self._genomes_by_id.values()))


# ---------------------------------------------------------------------------
# Internal pool implementation
# ---------------------------------------------------------------------------

class _NeatPopulationPool(OpponentPool):
    """Population-pool adapter for NEAT genomes.

    Satisfies :class:`OpponentPool` by holding a ``dict`` of
    ``genome_id → genome`` and a policy factory.

    For parallel evaluation, :meth:`snapshot_opponents` produces
    :class:`OpponentSnapshot` items with kind ``"neat"`` so workers can
    rebuild the policy via :func:`rebuild_opponent_policy`.
    """

    def __init__(
        self,
        genomes_by_id: Dict[int, neat.DefaultGenome],
        policy_factory: Callable[[neat.DefaultGenome], PolicyProtocol],
        neat_config: neat.Config,
    ) -> None:
        self._genomes = list(genomes_by_id.values())
        self._factory = policy_factory
        self._neat_config = neat_config

    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol:
        idx = rng.integers(0, len(self._genomes))
        return self._factory(self._genomes[idx])

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
            raise ValueError(
                f"snapshot_opponents: k must be >= 1, got {k}"
            )
        snapshots: List[OpponentSnapshot] = []
        for _ in range(k):
            idx = rng.integers(0, len(self._genomes))
            genome = self._genomes[idx]
            snapshots.append(
                OpponentSnapshot(
                    kind="neat",
                    payload={
                        "genome": genome,
                        # We store a minimal config dict so the worker
                        # can rebuild the config.  The worker already has
                        # neat imported; it calls build_neat_config.
                        "config_input_size": self._neat_config.genome_config.num_inputs,
                        "config_output_size": self._neat_config.genome_config.num_outputs,
                    },
                )
            )
        return snapshots

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


# ---------------------------------------------------------------------------
# RNG isolation helpers
# ---------------------------------------------------------------------------

import random as _random


def _save_random_state() -> Tuple[Any, Any]:
    """Snapshot ``random`` and ``numpy.random`` state.

    NEAT uses the global ``random`` module internally.  When multiple
    NEAT sides run in co-evolution, each must advance its own RNG
    stream independently.  Saving and restoring around
    :meth:`neat.Population.run` achieves this.
    """
    return _random.getstate(), np.random.get_state()


def _restore_random_state(state: Tuple[Any, Any]) -> None:
    """Restore ``random`` and ``numpy.random`` state from a tuple
    returned by :func:`_save_random_state`."""
    rng_state, np_state = state
    _random.setstate(rng_state)
    try:
        np.random.set_state(np_state)
    except (ValueError, TypeError):
        # If the numpy RNG has been reinitialised or the state is
        # incompatible, reseed rather than crashing.
        pass
