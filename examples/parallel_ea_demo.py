"""Demonstrate parallel fitness evaluation with :class:`EATrainer`.

Runs a tiny EA on ``simple_spread_v3`` twice — once serial, once with
a 4-worker process pool — and prints the best fitness and wall-clock
time for each. With ``algorithm_params["seed"]`` set, the per-individual
evaluation seeds are deterministic, so the **fitness arrays match
bit-for-bit** between the two runs (any divergence in the final
champion comes from DEAP's unseeded selection / mutation / crossover).

Usage:
    python examples/parallel_ea_demo.py
"""

from __future__ import annotations

import os
import time

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer


def _run(num_workers: int) -> tuple[float, float]:
    cfg = TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=64,  # 16 generations * 4 individuals
        algorithm_params={"seed": 0},
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
        "Note: differences in `final_fitness` come from DEAP's unseeded "
        "selection/mutation operators, not from evaluation noise — "
        "per-individual eval seeds are deterministic."
    )


if __name__ == "__main__":
    main()
