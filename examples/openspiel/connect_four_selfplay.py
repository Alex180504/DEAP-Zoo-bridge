"""Connect Four self-play demo via OpenSpiel + Shimmy.

Thin structural copy of ``examples/turn_based/tictactoe_selfplay.py``
that swaps the native PettingZoo ``tictactoe_v3`` for OpenSpiel's
``connect_four`` exposed through :mod:`shimmy`
(``OpenSpielCompatibilityV0``). It exercises the
:func:`deap_pettingzoo.contrib.openspiel.register_openspiel_game`
helper end-to-end under ``api='aec_masked'`` +
:class:`~deap_pettingzoo.evaluators.SelfPlayEvaluator`.

Requires the optional extra::

    pip install 'deap-pettingzoo[shimmy]'

Runtime target: under 60 seconds on CPU (tiny pop / 3 generations /
small network). Connect Four has 7 columns × 6 rows = 42 cells, a
Discrete(7) move space, and a 2-channel ternary board representation.

Usage::

    python examples/openspiel/connect_four_selfplay.py
"""

from __future__ import annotations

import random
import sys
import time
from typing import List

import numpy as np

try:  # noqa: SIM105 - explicit import guard with install hint
    import shimmy  # noqa: F401
except ImportError:
    print(
        "This demo requires the 'shimmy' optional extra. "
        "Install with: pip install 'deap-pettingzoo[shimmy]'"
    )
    sys.exit(0)

from deap import base, creator, tools

from deap_pettingzoo.contrib.openspiel import register_openspiel_game
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import is_registered
from deap_pettingzoo.evaluators import SelfPlayEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome


ENV_NAME = "openspiel_connect_four"

# Hyperparameters — kept tiny so the demo runs under a minute on CPU.
POPULATION = 8
GENERATIONS = 3
HIDDEN = 24
MUTATION_SIGMA = 0.2
MUTATION_PROB = 0.3
CROSSOVER_PROB = 0.5
SEED = 42


def _register_connect_four() -> None:
    # Always overwrite so re-runs pick up the intended api + episode length.
    register_openspiel_game(
        ENV_NAME,
        game_name="connect_four",
        api="aec_masked",
        default_episode_len=42,  # 7 cols × 6 rows
        overwrite=True,
        tags={"classic", "board"},
    )
    assert is_registered(ENV_NAME)


# --- Genome / DEAP plumbing ------------------------------------------------


def _probe_obs_dim() -> int:
    """Probe the env once to discover the flattened observation size."""
    tmp = PettingZooWrapper(env_name=ENV_NAME)
    try:
        obs = tmp.reset(seed=0)
        agent = next(iter(obs))
        return int(np.asarray(obs[agent]).size)
    finally:
        tmp.close()


def _make_genome(obs_dim: int) -> FixedTopologyGenome:
    # connect_four action space: Discrete(7) (one move per column)
    return FixedTopologyGenome(
        layer_sizes=[obs_dim, HIDDEN, 7],
        activation="tanh",
        random_seed=None,
    )


def _setup_deap(obs_dim: int):
    for name in ("FitnessMax_ConnectFour", "Individual_ConnectFour"):
        if hasattr(creator, name):
            delattr(creator, name)

    creator.create("FitnessMax_ConnectFour", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual_ConnectFour",
        object,
        fitness=creator.FitnessMax_ConnectFour,
        genome=None,
    )

    toolbox = base.Toolbox()

    def individual_factory():
        ind = creator.Individual_ConnectFour()
        ind.genome = _make_genome(obs_dim)
        return ind

    toolbox.register("individual", individual_factory)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def _mutate(ind) -> None:
    w = np.asarray(ind.genome.get_weights(), dtype=np.float64)
    mask = np.random.random(w.shape) < MUTATION_PROB
    w[mask] += np.random.normal(0.0, MUTATION_SIGMA, size=int(mask.sum()))
    ind.genome.set_weights(w.tolist())


def _crossover(parent1, parent2) -> None:
    w1 = np.asarray(parent1.genome.get_weights(), dtype=np.float64)
    w2 = np.asarray(parent2.genome.get_weights(), dtype=np.float64)
    alpha = np.random.random(w1.shape)
    c1 = alpha * w1 + (1.0 - alpha) * w2
    c2 = alpha * w2 + (1.0 - alpha) * w1
    parent1.genome.set_weights(c1.tolist())
    parent2.genome.set_weights(c2.tolist())


# --- Main loop -------------------------------------------------------------


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    _register_connect_four()
    obs_dim = _probe_obs_dim()

    wrapper = PettingZooWrapper(env_name=ENV_NAME)
    evaluator = SelfPlayEvaluator(
        wrapper,
        max_steps=42,
        aggregation="first_agent",
        masked=True,
        flatten_obs=True,
    )

    toolbox = _setup_deap(obs_dim)
    population: List = toolbox.population(n=POPULATION)

    t0 = time.time()
    for ind in population:
        ind.fitness.values = evaluator.evaluate(ind, seed=SEED)

    best_history: List[float] = [max(ind.fitness.values[0] for ind in population)]
    print(
        f"Gen 0 — best fitness: {best_history[0]:+.3f} "
        f"(elapsed {time.time() - t0:.2f}s)"
    )

    for gen in range(1, GENERATIONS + 1):
        parents = tools.selTournament(population, k=POPULATION, tournsize=3)
        offspring = [toolbox.individual() for _ in range(POPULATION)]
        for new, src in zip(offspring, parents):
            new.genome.set_weights(list(src.genome.get_weights()))
            new.fitness.values = src.fitness.values

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                _crossover(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for ind in offspring:
            if random.random() < MUTATION_PROB:
                _mutate(ind)
                if ind.fitness.valid:
                    del ind.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluator.evaluate(ind, seed=SEED + gen)

        population[:] = offspring
        best = max(ind.fitness.values[0] for ind in population)
        best_history.append(best)
        print(
            f"Gen {gen} — best fitness: {best:+.3f} "
            f"(elapsed {time.time() - t0:.2f}s)"
        )

    print()
    print("=" * 60)
    print("OpenSpiel connect_four self-play complete.")
    print(f"  Generations:     {GENERATIONS}")
    print(f"  Population:      {POPULATION}")
    print(f"  Best fitness:    {max(best_history):+.3f}")
    print(f"  Total time:      {time.time() - t0:.2f}s")
    print("=" * 60)

    wrapper.close()


if __name__ == "__main__":
    main()
