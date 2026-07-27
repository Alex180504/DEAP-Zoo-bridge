"""HPC driver: bidirectional EA-vs-MADDPG co-evolution on ``simple_tag_v3``.

The Slurm-ready variant of ``examples/coevolution_ea_vs_maddpg.py``: EA-evolved
pursuers (``adversary_*``) against an AgileRL MADDPG evader (``agent_*``)
whose networks learn on the GPU against the current EA pool every generation.

Parallelism: with ``--num-workers > 1`` (or ``--backend dask`` against a
remote scheduler) the trainer dispatches the whole EA generation
(``population x k_opponents`` episodes) in one ``executor.map``, shipping
the live RL opponent as a per-generation ``FrozenRLPolicyWrapper`` CPU
snapshot; RL *learning* (replay buffer, gradients, CUDA) always stays in
this driver process. Evaluation workers must be able to unpickle that
snapshot, i.e. torch + AgileRL must be installed wherever they run (CPU
torch suffices -— the rig's CPU node images ship it for exactly this).
The companion sbatch template expresses the split as a Slurm heterogeneous
job: driver + MADDPG on the GPU node, Dask evaluation workers on the CPU
nodes.

Prints one JSON summary line on stdout (keyed by ``cuda_available``).
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=int, default=6, help="EA-side population.")
    p.add_argument(
        "--episodes", type=int, default=60, help="Total episode budget "
        "(generations are derived from it by the trainer)."
    )
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel evaluation workers for the EA side (advisory metadata "
        "when a remote Dask scheduler is used). Workers must have torch + "
        "AgileRL installed to unpickle the frozen RL-opponent snapshot.",
    )
    p.add_argument("--backend", choices=("process", "dask"), default="process")
    p.add_argument(
        "--scheduler-address",
        default=None,
        help="Overrides DEAP_DASK_SCHEDULER; only meaningful with --backend dask.",
    )
    p.add_argument("--device", default="cuda", help="Torch device for the RL side.")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail (instead of silently training the RL side on CPU) when "
        "CUDA is unavailable — the right behaviour inside a GPU allocation.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Torch/AgileRL only exist on the GPU node image (Dockerfile.gpu), so
    # keep every heavyweight import inside main().
    import torch

    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
    from deap_pettingzoo.baselines.coevolution import (
        CoEvolutionConfig,
        CoEvolutionTrainer,
        Side,
    )

    cuda_available = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        raise SystemExit(
            "CUDA required (--require-cuda) but torch.cuda.is_available() is "
            "False on this node — refusing to fall back to CPU inside a GPU "
            "allocation."
        )

    coev = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side(
                name="pursuers",
                role_prefixes=("adversary",),
                optimizer="ea",
                optimizer_config={
                    "population_size": args.population,
                    "hidden_size": 32,
                    "seed": args.seed,
                },
            ),
            Side(
                name="evaders",
                role_prefixes=("agent",),
                optimizer="rl",
                optimizer_config={
                    "algorithm": "maddpg",
                    "rollout_episodes_per_gen": 2,
                    "learning_steps_per_gen": 2,
                    "batch_size": 32,
                    "memory_size": 5_000,
                    "learning_delay": 64,
                    "device": args.device,
                    "seed": args.seed + 1,
                },
            ),
        ],
        # A live RL side is a degenerate size-1 opponent pool: sampling it
        # more than once per evaluation adds no diversity.
        k_opponents=1,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        parallel_backend=args.backend,
        scheduler_address=args.scheduler_address,
    )

    tcfg = TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True},  # MADDPG needs Box actions
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=args.episodes,
        seed=args.seed,
        coevolution_config=coev,
        output_dir=Path(args.output_dir),
    )
    result = CoEvolutionTrainer().train(tcfg)

    per_side_best = {
        name: [float(entry["best_fitness"]) for entry in hist]
        for name, hist in result.metrics["per_side_history"].items()
    }
    summary = {
        "hostname": socket.gethostname(),
        "cuda_available": cuda_available,
        "requested_device": args.device,
        "algorithm": "coevolution_ea_vs_maddpg",
        "num_generations": int(result.metrics["num_generations"]),
        "per_side_best": per_side_best,
        "num_workers": result.metadata.get("coevolution_num_workers"),
        "parallel_backend": result.metadata.get("coevolution_parallel_backend"),
        "scheduler_address": result.metadata.get("coevolution_scheduler_address"),
        "dask_workers": result.metadata.get("coevolution_dask_workers"),
        "run_dir": result.metadata.get("run_dir"),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
