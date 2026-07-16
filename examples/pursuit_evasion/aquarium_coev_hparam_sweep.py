"""Coevolution hyperparameter sweep for the aquarium suite (symmetric env).

Sweeps one side's hyperparameters inside a *live* competitive coevolution
pairing, complementing ``aquarium_hparam_sweep.py`` (which tunes against a
fixed scripted APF opponent).  For every configuration the swept algorithm
drives the predator side against a fixed canonical EA prey side, so
configuration effects are measured under a moving-target opponent while
the opponent family stays controlled.

Because coevolution fitness is relative (it moves with the opponent), all
reported performance comes from the scripted-APF benchmark: every
``BENCH_EVERY_N`` generations both champions play a frozen APF opponent
(absolute progress probe), and the final predator champion is scored over
``FINAL_EVAL_EPISODES`` episodes.

Scaled sweep regime: population 32 / K_OPPONENTS 4 quarters the canonical
per-generation tick cost (48 x 8 would fund too few generations at sweep
budgets); the shipped canonical values stay untouched for full runs.  The
env is the perfectly symmetric config from ``experiment_config.py``.
Usage::

    python examples/pursuit_evasion/aquarium_coev_hparam_sweep.py --algos ea,cma,qdax
    python examples/pursuit_evasion/aquarium_coev_hparam_sweep.py --algos pso,hyperneat,matd3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig
from deap_pettingzoo.baselines.coevolution import CoEvolutionConfig, Side
from deap_pettingzoo.baselines.coevolution.archive import RingHallOfFame
from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

import examples.pursuit_evasion.aquarium_coevolution as coev
from examples.pursuit_evasion.aquarium import experiment_config as XC
from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    AquariumScriptedPredatorWrapper,
    AquariumScriptedPreyWrapper,
    _patch_aquarium_wrappers,
)

ENV_CFG: Dict[str, Any] = coev.ENV_CONFIG

SEED: int = 0
# CPU for the live RL learners: opponent evaluation runs against frozen
# CPU snapshots anyway (FrozenRLPolicyWrapper), the update batches are
# tiny, and this keeps the GPU free for concurrent standalone RL tracks.
DEVICE: str = "cpu"
N_WORKERS: int = 14

# Scaled sweep regime (canonical: pop 48, K 8 — see experiment_config.py).
# RL pairings use the same regime: EA-side evaluation against RL opponents
# parallelizes via frozen CPU policy snapshots, so their per-tick cost is
# comparable to the pure-evolutionary pairings.
SWEEP_POP: int = 32
K_OPPONENTS: int = 4
HOF_CAPACITY: int = XC.HOF_CAPACITY
ARCHIVE_MIX: float = XC.ARCHIVE_MIX

EVO_BUDGET: int = 600_000
QDAX_BUDGET: int = 500_000
RL_BUDGET: int = 600_000

BENCH_EVERY_N: int = 2
BENCH_EPISODES: int = 8
FINAL_EVAL_EPISODES: int = 20

OUTPUT_DIR: Path = Path("results/aquarium_symmetric_sweep/coev")


CONFIGS: List[Dict[str, Any]] = [
    # EA predator side vs canonical EA prey
    {"algo": "ea", "name": "coev_ea_center", "population_size": SWEEP_POP,
     "mut_sigma": 0.2, "mut_indpb": 0.3},
    {"algo": "ea", "name": "coev_ea_large_pop", "population_size": 48,
     "mut_sigma": 0.2, "mut_indpb": 0.3},
    {"algo": "ea", "name": "coev_ea_high_mutation", "population_size": SWEEP_POP,
     "mut_sigma": 0.4, "mut_indpb": 0.4},
    # CMA-ES predator side
    {"algo": "cma", "name": "coev_cma_center", "population_size": SWEEP_POP,
     "sigma0": 0.2},
    {"algo": "cma", "name": "coev_cma_low_sigma", "population_size": SWEEP_POP,
     "sigma0": 0.1},
    {"algo": "cma", "name": "coev_cma_high_sigma", "population_size": SWEEP_POP,
     "sigma0": 0.4},
    # PSO predator side (parallel evaluation like EA/CMA)
    {"algo": "pso", "name": "coev_pso_center", "population_size": SWEEP_POP,
     "phi1": 2.0, "phi2": 2.0, "weight": 0.7298},
    {"algo": "pso", "name": "coev_pso_constriction", "population_size": SWEEP_POP,
     "phi1": 1.49618, "phi2": 1.49618, "weight": 0.7298},
    {"algo": "pso", "name": "coev_pso_high_inertia", "population_size": SWEEP_POP,
     "phi1": 2.0, "phi2": 2.0, "weight": 0.9},
    # QDax MAP-Elites predator side
    {"algo": "qdax", "name": "coev_qdax_center", "batch_size": 32,
     "iso_sigma": 0.05, "line_sigma": 0.1},
    {"algo": "qdax", "name": "coev_qdax_high_iso", "batch_size": 32,
     "iso_sigma": 0.15, "line_sigma": 0.3},
    {"algo": "qdax", "name": "coev_qdax_large_batch", "batch_size": 64,
     "iso_sigma": 0.05, "line_sigma": 0.1},
    # HyperNEAT predator side (CPPN population, fixed substrate)
    {"algo": "hyperneat", "name": "coev_hyperneat_center", "population_size": 50,
     "substrate_hidden_layers": [XC.EVOLVED_HIDDEN_SIZE]},
    {"algo": "hyperneat", "name": "coev_hyperneat_large_pop", "population_size": 100,
     "substrate_hidden_layers": [XC.EVOLVED_HIDDEN_SIZE]},
    # MATD3 predator side (live RL side)
    {"algo": "matd3", "name": "coev_matd3_center", "expl_noise": 0.2,
     "lr_actor": 1e-4, "lr_critic": 1e-3},
    {"algo": "matd3", "name": "coev_matd3_high_noise", "expl_noise": 0.8,
     "lr_actor": 1e-4, "lr_critic": 1e-3},
    {"algo": "matd3", "name": "coev_matd3_high_lr", "expl_noise": 0.2,
     "lr_actor": 3e-4, "lr_critic": 3e-3},
    {"algo": "matd3", "name": "coev_matd3_more_learning", "expl_noise": 0.2,
     "lr_actor": 1e-4, "lr_critic": 1e-3,
     "learning_steps_per_gen": 256, "rollout_episodes_per_gen": 8},
]


def _make_ea_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    del seed
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="ea",
        optimizer_config={
            "hidden_size": XC.EVOLVED_HIDDEN_SIZE,
            "population_size": cfg["population_size"],
            "crossover_rate": XC.EA_PARAMS["cxpb"],
            "mutation_rate": XC.EA_PARAMS["mutation_rate"],
            "tournament_size": XC.EA_PARAMS["tournament_size"],
            "crossover_alpha": XC.EA_PARAMS["crossover_alpha"],
            "mutation_sigma": cfg["mut_sigma"],
            "mutation_indpb": cfg["mut_indpb"],
            "weight_range": (-1.0, 1.0),
            "activation": XC.EVOLVED_ACTIVATION,
            "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_cma_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    del seed
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="cma",
        optimizer_config={
            "hidden_size": XC.EVOLVED_HIDDEN_SIZE,
            "population_size": cfg["population_size"],
            "sigma0": cfg["sigma0"],
            "activation": XC.EVOLVED_ACTIVATION,
            "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_qdax_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    role = "prey" if role_prefix == "prey" else "predator"
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="qdax_map_elites",
        optimizer_config={
            "hidden_sizes": (XC.QDAX_PARAMS["hidden_size"],),
            "activation": XC.EVOLVED_ACTIVATION,
            "batch_size": cfg["batch_size"],
            "grid_shape": XC.QDAX_PARAMS["grid_shape"],
            "iso_sigma": cfg["iso_sigma"],
            "line_sigma": cfg["line_sigma"],
            "episode_length": int(ENV_CFG["max_time_steps"]),
            "descriptor_fn": XC.make_two_team_descriptor(role, ENV_CFG),
            "position_extractor": XC.make_position_extractor(role, ENV_CFG),
            "seed": seed,
            "pool_mode": XC.QDAX_PARAMS["pool_mode"],
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_pso_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="pso",
        optimizer_config={
            "hidden_size": XC.EVOLVED_HIDDEN_SIZE,
            "population_size": cfg["population_size"],
            "phi1": cfg["phi1"],
            "phi2": cfg["phi2"],
            "weight": cfg["weight"],
            "activation": XC.EVOLVED_ACTIVATION,
            "weight_range": (-1.0, 1.0),
            "seed": seed,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_hyperneat_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="hyperneat",
        optimizer_config={
            "pop_size": cfg["population_size"],
            "substrate_hidden_layers": cfg["substrate_hidden_layers"],
            "eval_episodes_per_genome": 1,
            "seed": seed,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_rl_side(cfg: Dict[str, Any], name: str, role_prefix: str, seed: int) -> Side:
    rl_params = dict(XC.RL_COEV_PARAMS)
    for key in ("learning_steps_per_gen", "rollout_episodes_per_gen"):
        if key in cfg:
            rl_params[key] = cfg[key]
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="rl",
        optimizer_config={
            "algorithm": cfg["algo"],
            **rl_params,
            "device": DEVICE,
            "seed": seed,
            "agent_ctor_kwargs": {
                "expl_noise": cfg["expl_noise"],
                "O_U_noise": False,
                "lr_actor": cfg["lr_actor"],
                "lr_critic": cfg["lr_critic"],
                "gamma": XC.MATD3_PARAMS["gamma"],
            },
        },
    )


SIDE_BUILDERS = {
    "ea": _make_ea_side,
    "cma": _make_cma_side,
    "pso": _make_pso_side,
    "qdax": _make_qdax_side,
    "hyperneat": _make_hyperneat_side,
    "matd3": _make_rl_side,
}


def _canonical_prey_side(seed: int, population_size: int = SWEEP_POP) -> Side:
    """Fixed opponent: canonical EA prey at the scaled sweep population."""
    prey_cfg = {"population_size": population_size,
                "mut_sigma": XC.EA_PARAMS["mut_sigma"],
                "mut_indpb": XC.EA_PARAMS["mut_indpb"]}
    return _make_ea_side(prey_cfg, "prey", "prey", seed)


def _benchmark_vs_apf(
    policy,
    role: str,
    env_cfg: Dict[str, Any],
    n_episodes: int,
    seed: int = 0,
) -> Tuple[List[float], List[int]]:
    """Episode rewards + task metrics of *policy* vs the frozen APF opponent.

    ``role="predator"``: policy drives predators vs APF prey, metric is
    prey caught; ``role="prey"``: policy drives prey vs APF predators,
    metric is prey lost.  Pure env reward — no shaping.
    """
    from marl_aquarium import aquarium_v0

    info_key = "prey_caught_total" if role == "predator" else "prey_lost_total"
    rewards: List[float] = []
    metrics: List[int] = []
    fallback_steps = 0
    total_steps = 0
    for ep in range(n_episodes):
        inner = aquarium_v0.parallel_env(render_mode="rgb_array", **env_cfg)
        _patch_aquarium_wrappers(inner)
        if role == "predator":
            bench_env = AquariumScriptedPreyWrapper(
                inner,
                predator_observe_count=env_cfg["predator_observe_count"],
                prey_observe_count=env_cfg["prey_observe_count"],
                action_count=env_cfg["action_count"],
                predator_reward=env_cfg["predator_reward"],
                apf_config=XC.APF_EVADER_CONFIG,
            )
        else:
            bench_env = AquariumScriptedPredatorWrapper(
                inner,
                predator_observe_count=env_cfg["predator_observe_count"],
                prey_observe_count=env_cfg["prey_observe_count"],
                action_count=env_cfg["action_count"],
                apf_config=XC.APF_PURSUER_CONFIG,
            )
        bench_env = AquariumObservationFeatureWrapper(bench_env, append_agent_id=False)
        obs, infos = bench_env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_reward = 0.0
        ep_metric = 0
        for _ in range(int(env_cfg["max_time_steps"])):
            if not bench_env.agents:
                break
            actions = {}
            for agent in bench_env.agents:
                if agent in obs:
                    total_steps += 1
                    try:
                        actions[agent] = policy.get_action(obs[agent], deterministic=True)
                    except Exception:
                        # A fallback measures the no-op action, not the
                        # policy — count it so corrupted probes are loud.
                        fallback_steps += 1
                        actions[agent] = coev._fallback_action()
            if not actions:
                break
            obs, rews, _, _, infos = bench_env.step(actions)
            ep_reward += sum(float(r) for r in rews.values())
            for info in (infos or {}).values():
                if isinstance(info, dict) and info.get(info_key) is not None:
                    try:
                        ep_metric = max(ep_metric, int(info[info_key]))
                    except (TypeError, ValueError):
                        continue
        try:
            bench_env.close()
        except Exception:
            pass
        rewards.append(ep_reward)
        metrics.append(ep_metric)
    if fallback_steps:
        print(
            f"[Bench] WARNING: {fallback_steps}/{total_steps} agent-steps used "
            f"the fallback action — the probe did NOT measure the policy",
            flush=True,
        )
    return rewards, metrics


def _make_bench_callback(
    pred_name: str, prey_name: str, env_cfg: Dict[str, Any], sink: List[Dict[str, Any]]
):
    """Absolute-progress probe: both champions vs frozen APF, logged to *sink*."""

    def _callback(gen: int, states: Dict[str, Any]) -> None:
        if gen % BENCH_EVERY_N != 0:
            return
        for side_name, role in ((pred_name, "predator"), (prey_name, "prey")):
            state = states.get(side_name)
            if state is None:
                continue
            try:
                policy = state.best_policy(None)
                rewards, metrics = _benchmark_vs_apf(
                    policy, role, env_cfg, BENCH_EPISODES, seed=gen * 100
                )
                entry = {
                    "gen": gen,
                    "side": side_name,
                    "role": role,
                    "bench_reward": float(np.mean(rewards)),
                    "bench_metric": float(np.mean(metrics)),
                }
                sink.append(entry)
                print(
                    f"[Bench] gen {gen:3d}  {side_name:>10} vs APF: "
                    f"reward {entry['bench_reward']:+8.2f}  metric {entry['bench_metric']:.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Bench] gen {gen:3d}  {side_name} failed: {exc}", flush=True)

    return _callback


def run_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    algo = cfg["algo"]
    name = cfg["name"]
    print(f"\n{'=' * 70}\n{algo} :: {name}\n{'=' * 70}", flush=True)

    is_rl = algo == "matd3"
    is_qdax = algo == "qdax"
    budget = QDAX_BUDGET if is_qdax else (RL_BUDGET if is_rl else EVO_BUDGET)

    pred_side = SIDE_BUILDERS[algo](cfg, "predators", "predator", seed)
    prey_side = _canonical_prey_side(seed)
    bench_log: List[Dict[str, Any]] = []

    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[pred_side, prey_side],
        schedule="synchronous",
        k_opponents=K_OPPONENTS,
        aggregation="mean",
        reporting_side="predators",
        max_steps_per_episode=int(ENV_CFG["max_time_steps"]),
        seed=seed,
        num_workers=N_WORKERS,
        parallel_backend="process",
        generation_callback=_make_bench_callback("predators", "prey", ENV_CFG, bench_log),
    )
    train_cfg = TrainingConfig(
        env_name=coev.ENV_NAME_COEV,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=budget,
        # QDax sides participate in the shared consumed pool too (real
        # per-generation ticks; overshoot <= one generation's pooled cost).
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        coevolution_config=coev_cfg,
        max_steps_per_episode=int(ENV_CFG["max_time_steps"]),
        env_config=dict(ENV_CFG),
        output_dir=run_dir,
        recording=RecordingConfig(enabled=False),
    )
    print(
        f"  budget={budget:,} K={K_OPPONENTS} pop={cfg.get('population_size', '—')} "
        f"workers={coev_cfg.num_workers} consumed={train_cfg.budget_on_consumed}",
        flush=True,
    )

    t0 = time.time()
    result = CoEvolutionTrainer().train(train_cfg)
    train_time = time.time() - t0

    rewards, metrics = _benchmark_vs_apf(
        result.policy, "predator", ENV_CFG, FINAL_EVAL_EPISODES, seed=990_000
    )
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    pred_bench = [b for b in bench_log if b["side"] == "predators"]
    prey_bench = [b for b in bench_log if b["side"] == "prey"]
    return {
        "config_name": name,
        "algorithm": algo,
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("algo", "name")},
        "final_bench_reward_mean": float(rewards_arr.mean()),
        "final_bench_reward_std": float(rewards_arr.std()),
        "final_bench_catches_mean": float(np.mean(metrics)),
        "final_bench_catches_std": float(np.std(metrics)),
        "generations": int(result.metrics.get("num_generations", 0)),
        "env_steps_consumed": int(result.metrics.get("env_steps_consumed", 0) or 0),
        "final_best_relative": result.metrics.get("final_best", {}),
        "bench_trajectory": bench_log,
        "pred_bench_last": pred_bench[-1] if pred_bench else None,
        "prey_bench_last": prey_bench[-1] if prey_bench else None,
        "per_side_history": result.metrics.get("per_side_history", {}),
        "training_time_s": float(train_time),
        "budget": budget,
    }


def _write_reports(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coev_sweep_full.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )

    csv_path = output_dir / "coev_sweep_summary.csv"
    rows = []
    for r in all_results:
        row = {
            "config_name": r["config_name"],
            "algorithm": r["algorithm"],
            "final_bench_reward_mean": f"{r['final_bench_reward_mean']:.4f}",
            "final_bench_reward_std": f"{r['final_bench_reward_std']:.4f}",
            "final_bench_catches_mean": f"{r['final_bench_catches_mean']:.4f}",
            "generations": r["generations"],
            "env_steps_consumed": r["env_steps_consumed"],
            "training_time_s": f"{r['training_time_s']:.1f}",
            "budget": r["budget"],
        }
        for k, v in r["hyperparams"].items():
            row[f"hp_{k}"] = str(v)
        rows.append(row)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV -> {csv_path}")

    md_path = output_dir / "coev_sweep_report.md"
    lines = [
        "# Aquarium coevolution hyperparameter sweep",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Regime:** swept algo drives predators vs canonical EA prey "
        f"(pop {SWEEP_POP}, K={K_OPPONENTS})",
        "**Metric:** predator champion vs frozen APF prey "
        f"({FINAL_EVAL_EPISODES} episodes, pure env reward)",
        "",
        "| Rank | Config | Algo | Bench reward (μ ± σ) | Catches/ep | Gens | Time (s) |",
        "|------|--------|------|----------------------|------------|------|----------|",
    ]
    ranked = sorted(all_results, key=lambda x: x["final_bench_reward_mean"], reverse=True)
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | `{r['config_name']}` | {r['algorithm']} | "
            f"{r['final_bench_reward_mean']:+.2f} ± {r['final_bench_reward_std']:.2f} | "
            f"{r['final_bench_catches_mean']:.2f} | {r['generations']} | "
            f"{r['training_time_s']:.0f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD  -> {md_path}")


def main() -> None:
    global N_WORKERS, EVO_BUDGET, QDAX_BUDGET, RL_BUDGET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default="all",
                        help="Comma-separated subset of ea,cma,pso,qdax,hyperneat,matd3.")
    parser.add_argument("--configs", default="",
                        help="Optional comma-separated config-name filter.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help="Evaluation worker processes.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override every per-config budget (smoke tests).")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    N_WORKERS = args.workers
    if args.budget:
        EVO_BUDGET = QDAX_BUDGET = RL_BUDGET = int(args.budget)

    selected = (
        list(SIDE_BUILDERS)
        if args.algos.strip().lower() == "all"
        else [a.strip() for a in args.algos.split(",") if a.strip()]
    )
    unknown = [a for a in selected if a not in SIDE_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown algos {unknown}; available: {sorted(SIDE_BUILDERS)}")

    coev._register_envs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [c for c in CONFIGS if c["algo"] in selected]
    name_filter = {c.strip() for c in args.configs.split(",") if c.strip()}
    if name_filter:
        configs = [c for c in configs if c["name"] in name_filter]

    print(f"Coevolution sweep: {len(configs)} configs, algos={selected}")
    print(f"  env: prey_view={ENV_CFG['prey_view_distance']} "
          f"pred_view={ENV_CFG['predator_view_distance']} (symmetric)")

    all_results: List[Dict[str, Any]] = []
    for cfg in configs:
        run_dir = output_dir / cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            all_results.append(run_config(cfg, args.seed, run_dir))
        except Exception:
            print(f"  config {cfg['name']} FAILED:")
            traceback.print_exc()
        if all_results:
            _write_reports(all_results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
