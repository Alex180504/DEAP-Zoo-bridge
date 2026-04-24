"""EA pursuers vs QDax MAP-Elites evaders — experimental co-evolution demo.

Uses :class:`CoEvolutionTrainer` with two sides:

* **Pursuers** (``adversary_*`` agents) driven by a DEAP EA
  (``optimizer='ea'``).
* **Evaders** (``agent_*``) driven by :class:`QDaxSideState` with
  MAP-Elites (``optimizer='qdax_map_elites'``).

The QDax side exposes its :class:`QDaxRepertoirePool` to the EA side
via the standard :class:`OpponentPool` protocol \u2014 no bespoke
plumbing needed.

**Experimental caveat.** The current :class:`QDaxSideState` scoring
path cannot pass trajectory observations through
:class:`CoEvolutionEvaluator`, so descriptors fall back to a
centre-of-range placeholder. Fitness is real; descriptor diversity
is *not* being exploited. This demo is here to exercise the plug-in
surface, not to produce publishable results.

Usage::

    python examples/qdax/ea_vs_qdax_coevolution.py --smoke
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--env", default="simple_tag_v3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
    from deap_pettingzoo.baselines.coevolution import (
        CoEvolutionConfig,
        CoEvolutionTrainer,
        Side,
    )

    pop = 8 if args.smoke else 32
    generations = 2 if args.smoke else 20
    episode_length = 15 if args.smoke else 50
    grid = (6, 6) if args.smoke else (8, 8)

    ea_side = Side(
        name="pursuers",
        role_prefixes=("adversary",),
        optimizer="ea",
        optimizer_config={
            "population_size": pop,
            "hidden_sizes": (16,) if args.smoke else (32, 32),
            "cx_pb": 0.5,
            "mut_pb": 0.2,
        },
    )
    qd_side = Side(
        name="evaders",
        role_prefixes=("agent",),
        optimizer="qdax_map_elites",
        optimizer_config={
            "batch_size": 8 if args.smoke else 32,
            "hidden_sizes": (16,) if args.smoke else (32, 32),
            "grid_shape": grid,
            "episode_length": episode_length,
            "descriptor": "pursuit_evasion_2d",
            "seed": args.seed,
        },
    )

    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[ea_side, qd_side],
        schedule="synchronous",
        k_opponents=2,
        max_steps_per_episode=episode_length,
        seed=args.seed,
    )

    config = TrainingConfig(
        env_name=args.env,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=generations * pop,
        algorithm_params={"coevolution": coev_cfg},
    )

    trainer = CoEvolutionTrainer()
    result = trainer.train(config)
    print(f"Algorithm:    {result.algorithm}")
    print(f"Best fitness: {result.metrics.get('best_fitness', float('nan')):.4f}")
    print(f"Wall time:    {result.training_time:.2f}s")
    print(f"Sides trained: {list(result.metadata.get('per_side_history', {}).keys())}")


if __name__ == "__main__":
    main()
