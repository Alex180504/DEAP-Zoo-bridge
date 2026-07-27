"""MAP-Elites pursuit-evasion demo — 3-D descriptor variant.

Same JaxMARL env as :mod:`pursuit_evasion_map_elites` but uses the
:func:`with_capture_efficiency` descriptor to add a
*capture-efficiency* axis. The resulting archive separates "tight
formations that capture" from "tight formations that do not".
Usage::

    python examples/qdax/pursuit_evasion_map_elites_3d.py --smoke

Requires the ``[qdax]`` extra.
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
    parser.add_argument("--output", default="results/qdax_map_elites_3d")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
    from deap_pettingzoo.contrib.jaxmarl import register_jaxmarl_env
    from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer

    from examples.qdax.descriptors import with_capture_efficiency

    register_jaxmarl_env(
        "jaxmarl_simple_tag",
        env_id="MPE_simple_tag_v3",
        default_episode_len=25,
    )

    if args.smoke:
        iterations_budget = 2
        init_batch, batch_size = 16, 8
        grid_shape = (4, 4, 4)
        episode_length = 15
        hidden = (16,)
    else:
        iterations_budget = 500
        init_batch, batch_size = 256, 128
        grid_shape = (10, 10, 8)
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
            "descriptor_fn": with_capture_efficiency(
                num_pursuers=3, num_evaders=1, capture_radius=0.1
            ),
            "seed": args.seed,
        },
    )

    trainer = MAPElitesTrainer()
    result = trainer.train(config)

    print(f"Best fitness:    {result.metrics['best_fitness']:.4f}")
    print(f"Coverage:        {result.metrics['coverage']:.3%}")
    print(f"QD score:        {result.metrics['qd_score']:.4f}")
    print(f"Grid shape:      {result.metadata['grid_shape']}")
    print(f"Wall time:       {result.training_time:.2f}s")


if __name__ == "__main__":
    main()
