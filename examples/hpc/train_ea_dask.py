"""HPC driver: train :class:`EATrainer` on a remote Dask (or local process) backend.

Reads the Dask scheduler address from the ``DEAP_DASK_SCHEDULER`` environment
variable (exported by the compose / sbatch launch scripts) unless
``--scheduler-address`` is passed explicitly. Prints one JSON summary line on
stdout so callers (e.g., the Docker / Slurm rig tests) can parse the
result without scraping logs.

Seeding Python's global ``random`` module here (rather than relying on
``TrainingConfig.seed`` alone) makes an entire run — including DEAP's
otherwise-unseeded selection/crossover/mutation — reproducible: per-episode
evaluation seeds are already deterministic (assigned driver-side, independent
of worker count/backend), so seeding ``random`` before evolution starts gives
bit-identical fitness histories across separate invocations of this script
regardless of parallel backend.

Usage:
    python examples/hpc/train_ea_dask.py --backend dask --num-workers 2 \\
        --output-dir /shared/run1 --seed 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", default=None, help="Defaults to 'simple_spread_v3', "
                   "or the name returned by --register-module's register().")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--backend", choices=("process", "dask"), default="dask")
    p.add_argument(
        "--scheduler-address",
        default=None,
        help="Overrides DEAP_DASK_SCHEDULER; only meaningful with --backend dask.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--register-module",
        default=None,
        help="Dotted module (importable in this container) to import and "
        "call .register() on before training — proves the runtime-"
        "registered env survives the registry-snapshot replay on remote "
        "workers.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    env_name = args.env_name
    if args.register_module:
        module = importlib.import_module(args.register_module)
        registered_name = module.register()
        if env_name is None:
            env_name = registered_name
    if env_name is None:
        env_name = "simple_spread_v3"

    if args.seed is not None:
        random.seed(args.seed)

    cfg = TrainingConfig(
        env_name=env_name,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=args.population * args.generations,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
        output_dir=Path(args.output_dir),
    )
    trainer = EATrainer(
        hidden_size=16,
        population_size=args.population,
        num_workers=args.num_workers,
        parallel_backend=args.backend,
        scheduler_address=args.scheduler_address,
    )
    result = trainer.train(cfg)

    summary = {
        "final_fitness": float(result.metrics["final_fitness"]),
        "fitness_history": [
            float(row["mean"]) for row in result.metrics["fitness_history"]
        ],
        "num_workers": result.metadata.get("num_workers"),
        "parallel_backend": result.metadata.get("parallel_backend"),
        "scheduler_address": result.metadata.get("scheduler_address"),
        "dask_workers": result.metadata.get("dask_workers"),
        "run_dir": result.metadata.get("run_dir"),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
