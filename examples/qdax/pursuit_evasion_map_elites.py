"""MAP-Elites pursuit-evasion demo — 2-D descriptor, JAX-native fast path.

Runs :class:`MAPElitesTrainer` on a JaxMARL ``MPE_simple_tag_v3`` env
registered through :func:`register_jaxmarl_env`. The 2-D descriptor
``(formation_spread, mean_distance_to_evader)`` from
:mod:`examples.qdax.descriptors` carves the archive.

Usage::

    # Smoke test (<1 min on CPU):
    python examples/qdax/pursuit_evasion_map_elites.py --smoke

    # Full run (requires GPU for reasonable wall-clock):
    python examples/qdax/pursuit_evasion_map_elites.py

Requires the ``[qdax]`` extra: ``pip install 'deap-pettingzoo[qdax]'``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script *or* as a module (-m) so the
# experiment-side descriptor package resolves either way.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny-budget smoke run.")
    parser.add_argument("--output", default="results/qdax_map_elites_2d")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Importing here keeps --help working without the qdax extra installed.
    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
    from deap_pettingzoo.contrib.jaxmarl import register_jaxmarl_env
    from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer

    from examples.qdax.descriptors import formation_spread_and_mean_distance

    register_jaxmarl_env(
        "jaxmarl_simple_tag",
        env_id="MPE_simple_tag_v3",
        default_episode_len=25,
    )

    if args.smoke:
        iterations_budget = 2
        init_batch, batch_size = 16, 8
        grid_shape = (6, 6)
        episode_length = 15
        hidden = (16,)
    else:
        iterations_budget = 500
        init_batch, batch_size = 256, 128
        grid_shape = (20, 20)
        episode_length = 25
        hidden = (32, 32)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        env_name="jaxmarl_simple_tag",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=iterations_budget * batch_size,
        output_dir=output_dir,
        algorithm_params={
            "hidden_sizes": hidden,
            "init_batch_size": init_batch,
            "batch_size": batch_size,
            "grid_shape": grid_shape,
            "episode_length": episode_length,
            # simple_tag = 3 adversaries (pursuers) + 1 good agent (evader).
            "descriptor_fn": formation_spread_and_mean_distance(
                num_pursuers=3, num_evaders=1
            ),
            "seed": args.seed,
        },
    )

    trainer = MAPElitesTrainer()
    result = trainer.train(config)

    print(f"Algorithm:      {result.algorithm}")
    print(f"Best fitness:   {result.metrics['best_fitness']:.4f}")
    print(f"Archive coverage: {result.metrics['coverage']:.3%}")
    print(f"QD score:       {result.metrics['qd_score']:.4f}")
    print(f"Iterations:     {result.metrics['num_iterations']}")
    print(f"Wall time:      {result.training_time:.2f}s")
    print(f"Repertoire:     {result.metadata.get('repertoire_path', '<not saved>')}")


if __name__ == "__main__":
    main()
