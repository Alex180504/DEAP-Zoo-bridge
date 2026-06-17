"""MATD3 coevolution hyperparameter sweep (cuda verification harness).

Runs the ``matd3_vs_ea`` pairing from
``aquarium_coevolution_ea_cma_matd3.py`` repeatedly under several MATD3
hyperparameter configurations and reports, for each config, the trend of
the RL *predator* champion's APF-scripted benchmark score (pure aquarium
catch reward, no shaping).  This is the only absolute, shaping-free signal
for whether the MATD3 predator is actually learning to pursue.

The harness mutates the experiment module's top-level constants in place
between runs (the ``Side`` factories read those globals at call time) and
monkeypatches ``_run_scripted_benchmark`` to capture every benchmark score
keyed by role, so no edits to the main experiment file are required.

To keep each verification run fast on cuda the harness shrinks the
EA opponent population, the per-candidate opponent count, the budget and
the benchmark episode count via overridable constants below.  These reduce
wall-clock without changing the MATD3 learning dynamics being measured.

Usage:
    .venv\\Scripts\\python.exe examples\\pursuit_evasion\\matd3_coev_sweep.py \
        --budget 250000 --ea-pop 16 --k-opponents 4 --configs baseline,gamma99
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a plain script: ensure the repo root is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import examples.pursuit_evasion.aquarium_coevolution_ea_cma_matd3 as exp


# ---------------------------------------------------------------------------
# Candidate MATD3 configurations to sweep.
#
# Each entry is a name -> dict of overrides.  Keys understood:
#   expl_noise, o_u_noise, lr_actor, lr_critic, gamma, policy_freq, tau,
#   batch_size, memory_size, learning_delay, learning_steps_per_gen,
#   rollout_episodes_per_gen.
# Anything in the agent-ctor set (expl_noise, o_u_noise, lr_*, gamma,
# policy_freq, tau) is forwarded to the MATD3 constructor; the rest set the
# RLSideState rollout/replay schedule.
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, Dict[str, Any]] = {
    # Current in-file settings (post Step-5 fix).
    "baseline": dict(
        expl_noise=0.4, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.95, learning_steps_per_gen=64,
    ),
    # Longer horizon credit assignment for sparse catches.
    "gamma99": dict(
        expl_noise=0.4, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.99, learning_steps_per_gen=64,
    ),
    # Stronger exploration of the 16-way argmax.
    "expl06": dict(
        expl_noise=0.6, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.95, learning_steps_per_gen=64,
    ),
    # More gradient steps + long horizon, bigger buffer/batch.
    "rich": dict(
        expl_noise=0.5, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.99, learning_steps_per_gen=128, batch_size=128,
        memory_size=100_000,
    ),
    # Higher actor LR to escape the constant-heading collapse faster,
    # paired with the long horizon.
    "fastactor": dict(
        expl_noise=0.5, o_u_noise=False, lr_actor=3e-4, lr_critic=1e-3,
        gamma=0.99, learning_steps_per_gen=96,
    ),
    # --- Confirmation-round refinements around the expl06 winner ---
    # Push exploration of the 16-way argmax even further.
    "expl08": dict(
        expl_noise=0.8, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.95, learning_steps_per_gen=64,
    ),
    # Winner exploration + long-horizon credit assignment for sparse catches.
    "expl06_g99": dict(
        expl_noise=0.6, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.99, learning_steps_per_gen=64,
    ),
    # Winner repeated verbatim (different seed via CLI) for robustness.
    "expl06_rep": dict(
        expl_noise=0.6, o_u_noise=False, lr_actor=1e-4, lr_critic=1e-3,
        gamma=0.95, learning_steps_per_gen=64,
    ),
}

_CTOR_KEYS = {"expl_noise", "o_u_noise", "lr_actor", "lr_critic", "gamma",
              "policy_freq", "tau"}
_CTOR_RENAME = {"o_u_noise": "O_U_noise"}


def _apply_config(cfg: Dict[str, Any]) -> None:
    """Mutate the experiment module globals for one MATD3 config."""
    ctor: Dict[str, Any] = {}
    for key, val in cfg.items():
        if key in _CTOR_KEYS:
            ctor[_CTOR_RENAME.get(key, key)] = val
    exp.MATD3_EXTRA_CTOR_KWARGS = ctor

    if "batch_size" in cfg:
        exp.MATD3_BATCH_SIZE = cfg["batch_size"]
    if "memory_size" in cfg:
        exp.MATD3_MEMORY_SIZE = cfg["memory_size"]
    if "learning_delay" in cfg:
        exp.MATD3_LEARNING_DELAY = cfg["learning_delay"]
    if "learning_steps_per_gen" in cfg:
        exp.MATD3_LEARNING_STEPS_PER_GEN = cfg["learning_steps_per_gen"]
    if "rollout_episodes_per_gen" in cfg:
        exp.MATD3_ROLLOUT_EPISODES_PER_GEN = cfg["rollout_episodes_per_gen"]


def _install_capture() -> List[Tuple[str, float]]:
    """Monkeypatch ``_run_scripted_benchmark`` to record (role, score)."""
    records: List[Tuple[str, float]] = []
    original = exp._run_scripted_benchmark

    def _wrapped(policy, role, env_cfg, n_episodes, seed=0):
        score = original(policy, role, env_cfg, n_episodes, seed=seed)
        records.append((role, float(score)))
        return score

    exp._run_scripted_benchmark = _wrapped  # type: ignore[assignment]
    return records


def _summarise(name: str, pred_scores: List[float], elapsed: float) -> Dict[str, Any]:
    if not pred_scores:
        print(f"[{name}] no RL-predator benchmark points captured "
              f"(elapsed {elapsed:.0f}s)")
        return {"name": name, "n": 0, "first": None, "last": None,
                "best": None, "mean_last3": None, "elapsed": elapsed}

    first = pred_scores[0]
    last = pred_scores[-1]
    best = max(pred_scores)
    mean_last3 = sum(pred_scores[-3:]) / len(pred_scores[-3:])
    trend = "UP" if last > first + 1 else ("DOWN" if last < first - 1 else "flat")
    traj = ", ".join(f"{s:.1f}" for s in pred_scores)
    print(f"\n[{name}] RL-predator APF benchmark "
          f"({len(pred_scores)} pts, {elapsed:.0f}s)")
    print(f"    first={first:.2f}  last={last:.2f}  best={best:.2f}  "
          f"mean_last3={mean_last3:.2f}  -> {trend}")
    print(f"    trajectory: {traj}")
    return {"name": name, "n": len(pred_scores), "first": first, "last": last,
            "best": best, "mean_last3": mean_last3, "elapsed": elapsed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=250_000,
                    help="env-step budget per config (lower = faster).")
    ap.add_argument("--ea-pop", type=int, default=16,
                    help="EA opponent population size (smaller = faster).")
    ap.add_argument("--k-opponents", type=int, default=4,
                    help="opponent episodes per candidate (smaller = faster).")
    ap.add_argument("--bench-every", type=int, default=3,
                    help="run the APF benchmark every N generations.")
    ap.add_argument("--bench-eps", type=int, default=6,
                    help="episodes averaged per benchmark.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--configs", type=str, default="all",
                    help="comma-separated config names, or 'all'.")
    args = ap.parse_args()

    # ---- Shared speed/scope overrides (applied once) ----
    exp.RECORD_VIDEOS = False
    exp.COEV_TOTAL_STEPS = args.budget
    exp.EA_POP = args.ea_pop
    exp.EA_OPTIMIZER_CONFIG = dict(exp.EA_OPTIMIZER_CONFIG)
    exp.EA_OPTIMIZER_CONFIG["population_size"] = args.ea_pop
    exp.K_OPPONENTS = args.k_opponents
    exp.BENCHMARK_EVERY_N = args.bench_every
    exp.BENCHMARK_N_EPISODES = args.bench_eps
    exp.DEVICE = args.device
    exp.MATD3_DEVICE = args.device

    names = (list(CONFIGS) if args.configs == "all"
             else [c.strip() for c in args.configs.split(",") if c.strip()])
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; known: {list(CONFIGS)}")

    print("=" * 70)
    print("MATD3 coevolution cuda sweep")
    print(f"budget={args.budget} ea_pop={args.ea_pop} k_opp={args.k_opponents} "
          f"bench_every={args.bench_every} bench_eps={args.bench_eps} "
          f"device={args.device}")
    print(f"configs: {names}")
    print("=" * 70)

    sweep_root = exp.OUTPUT_DIR / "_sweep"
    results: List[Dict[str, Any]] = []

    for name in names:
        print("\n" + "#" * 70)
        print(f"# CONFIG: {name}  ->  {CONFIGS[name]}")
        print("#" * 70)
        _apply_config(CONFIGS[name])
        exp._register_envs()  # re-register so shape alphas / env_config are fresh

        records = _install_capture()
        run_dir = sweep_root / name / f"seed_{args.seed}"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            exp.run_matd3_vs_ea(args.seed, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] RUN FAILED: {exc}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0

        pred_scores = [s for role, s in records if role == "predator"]
        results.append(_summarise(name, pred_scores, elapsed))

    # ---- Ranking ----
    print("\n" + "=" * 70)
    print("SWEEP RANKING (by best RL-predator APF benchmark, then mean_last3)")
    print("=" * 70)
    ranked = sorted(
        [r for r in results if r["best"] is not None],
        key=lambda r: (r["best"], r["mean_last3"]),
        reverse=True,
    )
    header = f"{'config':<12} {'best':>7} {'last':>7} {'mean_l3':>8} {'first':>7} {'pts':>4} {'secs':>6}"
    print(header)
    print("-" * len(header))
    for r in ranked:
        print(f"{r['name']:<12} {r['best']:>7.2f} {r['last']:>7.2f} "
              f"{r['mean_last3']:>8.2f} {r['first']:>7.2f} {r['n']:>4d} "
              f"{r['elapsed']:>6.0f}")
    empty = [r for r in results if r["best"] is None]
    for r in empty:
        print(f"{r['name']:<12} {'--':>7} {'--':>7} {'--':>8} {'--':>7} "
              f"{0:>4d} {r['elapsed']:>6.0f}")

    if ranked:
        winner = ranked[0]
        print(f"\nBEST CONFIG: {winner['name']}  "
              f"(best={winner['best']:.2f}, mean_last3={winner['mean_last3']:.2f})")
        print(f"  overrides: {CONFIGS[winner['name']]}")


if __name__ == "__main__":
    main()
