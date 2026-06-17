"""Tic-tac-toe self-play demo via the turn-based AEC adapter.

Runs a *tiny* EA self-play loop against PettingZoo's ``tictactoe_v3``,
which is a strictly turn-based AEC env (``is_parallelizable=False``).
The env is wrapped by
:class:`~deap_pettingzoo.environments.MaskedSimultaneousAdapter`
(``api='aec_masked'``) and every individual is evaluated by
:class:`~deap_pettingzoo.evaluators.SelfPlayEvaluator` — one shared-
weights policy plays *both* sides of the board, and fitness is the
player-1-side reward aggregated across a single game.

Runtime target: under 30 seconds on CPU (a handful of generations,
tiny population, tiny network).

Usage::

    python examples/turn_based/tictactoe_selfplay.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List

import numpy as np
from deap import base, creator, tools

from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import is_registered, register_env
from deap_pettingzoo.evaluators import SelfPlayEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.utils.logging import ExperimentLogger


ENV_NAME = "tictactoe_v3"

# Hyperparameters — kept tiny so the demo runs in under 30 seconds.
POPULATION = 8
GENERATIONS = 4
HIDDEN = 16
MUTATION_SIGMA = 0.2
MUTATION_PROB = 0.3
CROSSOVER_PROB = 0.5
SEED = 42
OUTPUT_DIR = Path("results/tictactoe_selfplay")


def _register_tictactoe() -> None:
    """Register ``tictactoe_v3`` with ``api='aec_masked'`` if not already."""
    if is_registered(ENV_NAME):
        # Overwrite so re-running the demo uses the turn-based adapter even
        # if a prior session registered a different api for this name.
        pass
    from pettingzoo.classic import tictactoe_v3

    register_env(
        ENV_NAME,
        tictactoe_v3,
        default_episode_len=9,  # 9 cells max
        api="aec_masked",
        overwrite=True,
        notes="Turn-based AEC; wrapped by MaskedSimultaneousAdapter.",
        tags={"classic", "turn_based"},
    )


# --- Genome / DEAP plumbing ------------------------------------------------


def _make_genome() -> FixedTopologyGenome:
    # tic-tac-toe obs: 3×3×2 = 18 inputs (after flatten by SelfPlayEvaluator)
    # Discrete(9) output head.
    return FixedTopologyGenome(
        layer_sizes=[18, HIDDEN, 9],
        activation="tanh",
        random_seed=None,
    )


def _setup_deap():
    # Reset any leftover fitness class to keep re-imports idempotent.
    if hasattr(creator, "FitnessMax_Tictactoe"):
        del creator.FitnessMax_Tictactoe
    if hasattr(creator, "Individual_Tictactoe"):
        del creator.Individual_Tictactoe

    creator.create("FitnessMax_Tictactoe", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual_Tictactoe",
        object,
        fitness=creator.FitnessMax_Tictactoe,
        genome=None,
    )

    toolbox = base.Toolbox()

    def individual_factory():
        ind = creator.Individual_Tictactoe()
        ind.genome = _make_genome()
        return ind

    toolbox.register("individual", individual_factory)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def _mutate(ind) -> None:
    w = np.asarray(ind.genome.get_weights(), dtype=np.float64)
    mask = np.random.random(w.shape) < MUTATION_PROB
    w[mask] += np.random.normal(0.0, MUTATION_SIGMA, size=mask.sum())
    ind.genome.set_weights(w.tolist())


def _crossover(parent1, parent2):
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

    _register_tictactoe()
    wrapper = PettingZooWrapper(env_name=ENV_NAME)
    evaluator = SelfPlayEvaluator(
        wrapper,
        max_steps=18,  # generous ceiling for a 9-cell board
        aggregation="first_agent",  # measure 'how good a player_1 are you'
        masked=True,
        flatten_obs=True,
    )

    toolbox = _setup_deap()
    population: List = toolbox.population(n=POPULATION)

    logger = ExperimentLogger.create(
        OUTPUT_DIR,
        algorithm="ea_selfplay",
        env_name=ENV_NAME,
        seed=SEED,
        budget_mode="eval_episodes",
        budget_value=POPULATION * (GENERATIONS + 1),
        hyperparams={
            "population": POPULATION,
            "generations": GENERATIONS,
            "hidden": HIDDEN,
            "mutation_sigma": MUTATION_SIGMA,
            "mutation_prob": MUTATION_PROB,
            "crossover_prob": CROSSOVER_PROB,
        },
        notes="Tic-tac-toe self-play (turn-based AEC).",
    )

    with logger:
        # Initial evaluation.
        t0 = time.time()
        for ind in population:
            ind.fitness.values = evaluator.evaluate(ind, seed=SEED)

        fits = [ind.fitness.values[0] for ind in population]
        best_history: List[float] = [max(fits)]
        logger.row("metrics", {
            "iteration": 0,
            "env_steps": POPULATION,
            "wall_time": time.time() - t0,
            "best": float(max(fits)),
            "mean": float(np.mean(fits)),
            "std": float(np.std(fits)),
            "min": float(min(fits)),
            "max": float(max(fits)),
        })
        print(
            f"Gen 0 — best fitness: {best_history[0]:+.3f} "
            f"(elapsed {time.time() - t0:.2f}s)"
        )

        for gen in range(1, GENERATIONS + 1):
            # Tournament selection.
            parents = tools.selTournament(population, k=POPULATION, tournsize=3)
            offspring = [toolbox.individual() for _ in range(POPULATION)]
            for new, src in zip(offspring, parents):
                new.genome.set_weights(list(src.genome.get_weights()))
                new.fitness.values = src.fitness.values

            # Crossover in pairs.
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PROB:
                    _crossover(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation.
            for ind in offspring:
                if random.random() < MUTATION_PROB:
                    _mutate(ind)
                    if ind.fitness.valid:
                        del ind.fitness.values

            # Re-evaluate dirty individuals.
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = evaluator.evaluate(ind, seed=SEED + gen)

            population[:] = offspring
            fits = [ind.fitness.values[0] for ind in population]
            best = max(fits)
            best_history.append(best)
            logger.row("metrics", {
                "iteration": gen,
                "env_steps": POPULATION * (gen + 1),
                "wall_time": time.time() - t0,
                "best": float(best),
                "mean": float(np.mean(fits)),
                "std": float(np.std(fits)),
                "min": float(min(fits)),
                "max": float(max(fits)),
            })
            print(
                f"Gen {gen} — best fitness: {best:+.3f} "
                f"(elapsed {time.time() - t0:.2f}s)"
            )

        total_time = time.time() - t0
        logger.summary({
            "final_fitness": float(max(best_history)),
            "best_fitness": float(max(best_history)),
            "num_generations": GENERATIONS,
            "population": POPULATION,
            "training_time_s": total_time,
        })

    print()
    print("=" * 60)
    print("Self-play training complete.")
    print(f"  Generations:     {GENERATIONS}")
    print(f"  Population:      {POPULATION}")
    print(f"  Best fitness:    {max(best_history):+.3f}")
    print(f"  Total time:      {time.time() - t0:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
