"""HPC driver: train AgileRL :class:`MADDPGTrainer` on a GPU node.

Companion to ``train_ea_dask.py`` for the RL side of the library: EA/CMA/PSO
evaluation is pure numpy and scales out over Dask workers, while the AgileRL
learners are single-process torch training that scale *up* on one GPU node —
so this driver is submitted to a GPU partition (``--partition=gpu
--gres=gpu:1``, see ``examples/hpc/slurm/train_rl_gpu.sbatch``) instead of
spanning nodes.

:func:`~deap_pettingzoo.utils.device_check.resolve_device` deliberately falls
back to CPU when CUDA is unavailable rather than crashing; on a GPU
allocation that silent fallback would waste the reservation, so
``--require-cuda`` (set by the sbatch template) turns it into a hard error.

Prints one JSON summary line on stdout (keyed by ``cuda_available``) so the
Slurm rig test -- or a human reading the job's ``.out`` file — can verify
where and on which device the run actually happened.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", default="simple_spread_v3")
    p.add_argument("--steps", type=int, default=400, help="Env-step training budget.")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail (instead of silently training on CPU) when CUDA is "
        "unavailable -- the right behaviour inside a GPU allocation.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Torch/AgileRL only exist on the GPU node image (Dockerfile.gpu), so
    # keep every heavyweight import inside main().
    import torch

    from deap_pettingzoo.baselines.agilerl_trainer import MADDPGTrainer
    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig

    cuda_available = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        raise SystemExit(
            "CUDA required (--require-cuda) but torch.cuda.is_available() is "
            "False on this node -- refusing to fall back to CPU inside a GPU "
            "allocation."
        )

    cfg = TrainingConfig(
        env_name=args.env_name,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=args.steps,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
        output_dir=Path(args.output_dir),
        # Tiny off-policy knobs so a smoke-test budget still reaches
        # several learn() calls (i.e. real gradient steps on the device).
        algorithm_params={
            "batch_size": 32,
            "memory_size": 1_000,
            "learning_delay": 64,
            "learn_step": 16,
        },
    )
    trainer = MADDPGTrainer(device=args.device)
    result = trainer.train(cfg)

    summary = {
        "hostname": socket.gethostname(),
        "cuda_available": cuda_available,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "requested_device": args.device,
        "resolved_device": str(trainer.device),
        "algorithm": "maddpg",
        "metrics": {
            k: float(v)
            for k, v in result.metrics.items()
            if isinstance(v, (int, float))
        },
        "run_dir": result.metadata.get("run_dir"),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
