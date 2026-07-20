"""HPC driver for the aquarium pursuit-evasion experiment suite.

One CLI entry point for all four experiment families, sized so a Slurm array
job can fan out over (experiment, algorithm/pairing, seed, variant) cells.
The tested algorithms are EA, CMA-ES, PSO, MAP-Elites, HyperNEAT, and
MATD3:

* ``--experiment pursuer``      — the six-algorithm pursuer-training suite
  vs scripted APF prey
  (:mod:`examples.pursuit_evasion.aquarium_pursuer_training`).
* ``--experiment evader``       — the six-algorithm evader-training suite
  vs scripted APF pursuers
  (:mod:`examples.pursuit_evasion.aquarium_evader_training`).
* ``--experiment coevolution``  — one competitive coevolution pairing
  (:mod:`examples.pursuit_evasion.aquarium_coevolution`).
* ``--experiment sweep``        — the hyperparameter sweep
  (:mod:`examples.pursuit_evasion.aquarium_hparam_sweep`).

The experiment variant is selected via ``--action-mode`` (and ``--role``
for sweeps), which the driver exports as ``AQUARIUM_ACTION_MODE`` /
``AQUARIUM_ROLE`` *before* importing the experiment module — spawned
evaluation workers inherit the environment, so worker-side env
registration matches the driver.  Scalar knobs (budget, seeds, device,
workers, recording) are plain module-attribute overrides.

Examples::

    # one coevolution pairing, one seed, continuous actions
    python examples/hpc/train_aquarium_suite.py --experiment coevolution \
        --pairing ea_vs_matd3 --seed 3 --action-mode continuous

    # the pursuer comparison, EA only, 5 seeds
    python examples/hpc/train_aquarium_suite.py --experiment pursuer \
        --algos ea --seeds 5

    # evader-role sweep, continuous actions
    python examples/hpc/train_aquarium_suite.py --experiment sweep \
        --role evader --action-mode continuous

Prints one JSON summary line on stdout so array-job logs stay greppable.
See ``examples/hpc/slurm/train_aquarium_suite.sbatch`` for the Slurm
template and ``docs/enucc_campaign.md`` for the per-job resource requests.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

# Headless rendering + variant env vars must be set before the experiment
# modules (and, transitively, pygame) are imported.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

EXPERIMENTS = ("pursuer", "evader", "coevolution", "sweep")

# Algorithm toggle attribute per experiment module, keyed by CLI name.
# One campaign algorithm set for all three vs-APF/sweep families.
_CAMPAIGN_ALGOS = {
    "ea": "RUN_EA",
    "cmaes": "RUN_CMAES",
    "pso": "RUN_PSO",
    "map_elites": "RUN_MAP_ELITES",
    "hyperneat": "RUN_HYPERNEAT",
    "matd3": "RUN_MATD3",
}
_PURSUER_ALGOS = dict(_CAMPAIGN_ALGOS)
_EVADER_ALGOS = dict(_CAMPAIGN_ALGOS)
_SWEEP_ALGOS = dict(_CAMPAIGN_ALGOS)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    p.add_argument(
        "--algos",
        default="all",
        help="Comma-separated algorithm subset (pursuer/evader/sweep); "
        "'all' enables every toggle.",
    )
    p.add_argument(
        "--pairing",
        default=None,
        help="Coevolution pairing '<pred>_vs_<prey>' (e.g. ea_vs_matd3). "
        "Required for --experiment coevolution.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed.")
    p.add_argument("--seeds", type=int, default=1, help="Number of consecutive seeds.")
    p.add_argument("--budget", type=int, default=None, help="Env-steps budget override.")
    p.add_argument("--action-mode", default="discrete", choices=("discrete", "continuous"))
    p.add_argument("--role", default="pursuer", choices=("pursuer", "evader"),
                   help="Sweep only: which side learns.")
    p.add_argument(
        "--curriculum",
        action="store_true",
        help="Anneal the scripted opponent's speed toward the symmetric value "
        "(pursuer/evader experiments only; exported as AQUARIUM_CURRICULUM "
        "before the experiment-module import — ignored by coevolution and "
        "sweep).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of silently training on CPU when CUDA is "
        "unavailable — the right behaviour inside a GPU allocation.",
    )
    p.add_argument("--num-workers", type=int, default=0,
                   help="Evaluation worker processes (0 = all cores).")
    p.add_argument("--record", action="store_true", help="Enable video recording.")
    p.add_argument(
        "--archive-runs",
        action="store_true",
        help="After completion, pack each run directory into a UI-importable "
        "ZIP (deap_export_manifest.json + all artifacts) and remove the "
        "uncompressed source.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _apply_algo_toggles(module, mapping: dict, algos: str) -> list:
    selected = (
        list(mapping) if algos.strip().lower() == "all"
        else [a.strip() for a in algos.split(",") if a.strip()]
    )
    unknown = [a for a in selected if a not in mapping]
    if unknown:
        raise SystemExit(f"Unknown algorithms {unknown}; available: {sorted(mapping)}")
    for cli_name, attr in mapping.items():
        setattr(module, attr, cli_name in selected)
    return selected


def _check_cuda(args) -> bool:
    if args.device != "cuda":
        return False
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if args.require_cuda and not available:
        raise SystemExit(
            "CUDA required (--require-cuda) but unavailable on this node — "
            "refusing to fall back to CPU inside a GPU allocation."
        )
    return available


def main() -> None:
    args = _parse_args()

    # Variant selection must precede the experiment-module import.
    os.environ["AQUARIUM_ACTION_MODE"] = args.action_mode
    os.environ["AQUARIUM_ROLE"] = args.role
    os.environ["AQUARIUM_CURRICULUM"] = "1" if args.curriculum else "0"

    cuda_available = _check_cuda(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {
        "experiment": args.experiment,
        "action_mode": args.action_mode,
        "curriculum": bool(args.curriculum),
        "hostname": socket.gethostname(),
        "cuda_available": cuda_available,
        "output_dir": str(output_dir),
    }

    if args.experiment == "coevolution":
        if not args.pairing:
            raise SystemExit("--experiment coevolution requires --pairing")
        import examples.pursuit_evasion.aquarium_coevolution as coev

        coev.DEVICE = args.device
        coev.N_WORKERS = args.num_workers or (os.cpu_count() or 1)
        coev.RECORD_VIDEOS = args.record
        if args.budget:
            coev.COEV_TOTAL_STEPS = args.budget
        coev._register_envs()

        statuses = []
        for offset in range(args.seeds):
            seed = args.seed + offset
            run_dir = output_dir / args.pairing / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = coev.run_pairing(args.pairing, seed, run_dir)
            statuses.append({"seed": seed, "time_s": result["training_time_s"]})
        summary.update(pairing=args.pairing, budget=coev.COEV_TOTAL_STEPS, runs=statuses)

    elif args.experiment in ("pursuer", "evader"):
        if args.experiment == "pursuer":
            import examples.pursuit_evasion.aquarium_pursuer_training as exp

            mapping = _PURSUER_ALGOS
        else:
            import examples.pursuit_evasion.aquarium_evader_training as exp

            mapping = _EVADER_ALGOS

        selected = _apply_algo_toggles(exp, mapping, args.algos)
        exp.DEVICE = args.device
        exp.N_WORKERS = args.num_workers
        exp.RECORD = args.record
        exp.SEED_BASE = args.seed
        exp.NUM_SEEDS = args.seeds
        exp.OUTPUT_DIR = output_dir
        exp.PROBE_ENV = False
        if args.budget:
            exp.TOTAL_ENV_STEPS = args.budget
        exp.main()
        summary.update(algorithms=selected, budget=exp.TOTAL_ENV_STEPS,
                       seeds=list(range(args.seed, args.seed + args.seeds)))

    else:  # sweep
        import examples.pursuit_evasion.aquarium_hparam_sweep as sweep

        selected = _apply_algo_toggles(sweep, _SWEEP_ALGOS, args.algos)
        sweep.DEVICE = args.device
        sweep.N_WORKERS = args.num_workers
        sweep.SEED = args.seed
        sweep.OUTPUT_DIR = output_dir
        if args.budget:
            sweep.SWEEP_BUDGET_STEPS = args.budget
        sweep.main()
        summary.update(role=args.role, algorithms=selected,
                       budget=sweep.SWEEP_BUDGET_STEPS)

    if args.archive_runs:
        from deap_pettingzoo.service.export_import import archive_runs_under

        archives = archive_runs_under(output_dir, remove_source=True)
        summary["archived_runs"] = [str(p.relative_to(output_dir)) for p in archives]
        print(f"[archive] {len(archives)} run(s) zipped under {output_dir}", flush=True)

    summary["wall_time_s"] = round(time.time() - t0, 1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
