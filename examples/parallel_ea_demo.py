"""Demonstrate parallel fitness evaluation with :class:`EATrainer`.

Runs a tiny EA on ``simple_spread_v3`` twice — once serial, once with
a 4-worker process pool — and prints the best fitness and wall-clock
time for each. With ``seed`` defined in :class:`TrainingConfig`, global RNG
seeding is enabled (Python's ``random``, NumPy, and PyTorch), so the
per-individual evaluation seeds are deterministic and DEAP's selection /
mutation / crossover operators are reproducible. The **fitness arrays and
final champion policy match bit-for-bit** between the two runs.

Usage:
    python examples/parallel_ea_demo.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer


OUTPUT_ROOT = Path("results/parallel_ea_demo")


def _run(num_workers: int) -> tuple[float, float]:
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=64,  # 16 generations * 4 individuals
        seed=42,
        # Each invocation gets its own subdir so the UI shows two
        # comparable runs (serial vs process-pool).
        output_dir=OUTPUT_ROOT / f"workers_{num_workers}",
    )
    trainer = EATrainer(
        hidden_size=16,
        population_size=4,
        num_workers=num_workers,
        parallel_backend="process",
    )
    t0 = time.perf_counter()
    result = trainer.train(cfg)
    elapsed = time.perf_counter() - t0
    return float(result.metrics["final_fitness"]), elapsed


def main() -> None:
    print(f"CPU count: {os.cpu_count()}")
    print()
    print("Serial (num_workers=1)...")
    f1, t1 = _run(num_workers=1)
    print(f"  final_fitness={f1:.4f}   wall={t1:.2f}s")

    print("Process pool (num_workers=4)...")
    f4, t4 = _run(num_workers=4)
    print(f"  final_fitness={f4:.4f}   wall={t4:.2f}s")

    print()
    print(f"Speedup: {t1 / t4:.2f}x")
    print(
        "Both runs use global RNG seeding (seed=42), so fitness values and "
        "final champion policy should match bit-for-bit."
    )


if __name__ == "__main__":
    main()
