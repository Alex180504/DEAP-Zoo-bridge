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
import sys
from pathlib import Path

# Allow running as a plain script *or* as a module (-m) so the
# experiment-side descriptor package resolves either way.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--env", default="simple_tag_v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/ea_vs_qdax_coevolution")
    args = parser.parse_args()

    from pathlib import Path

    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
    from deap_pettingzoo.baselines.coevolution import (
        CoEvolutionConfig,
        CoEvolutionTrainer,
        Side,
    )

    from examples.qdax.descriptors import formation_spread_and_mean_distance

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
            # Learner side = evaders (1 ``agent_*``); opponents = 3
            # ``adversary_*`` pursuers. Descriptor receives learner
            # positions first (num_pursuers slot), then opponents.
            "descriptor_fn": formation_spread_and_mean_distance(
                num_pursuers=1, num_evaders=3
            ),
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
        algorithm_params={"coevolution": coev_cfg, "seed": args.seed},
        output_dir=Path(args.output),
    )

    trainer = CoEvolutionTrainer()
    result = trainer.train(config)
    per_side = result.metrics.get("per_side_history", {})
    final_best = result.metrics.get("final_best", {})
    best = max((v for v in final_best.values() if v is not None), default=float("nan"))
    print(f"Algorithm:    {result.algorithm}")
    print(f"Best fitness: {best:.4f}")
    print(f"Wall time:    {result.training_time:.2f}s")
    print(f"Sides trained: {list(per_side.keys())}")
    for side_name, history in per_side.items():
        if history:
            last = history[-1]
            print(f"  {side_name}: gen {last['generation']} best={last['best_fitness']:.3f}")


if __name__ == "__main__":
    main()
