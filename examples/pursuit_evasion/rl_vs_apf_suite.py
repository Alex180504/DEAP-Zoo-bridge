"""RL-vs-APF head-to-head suite: IPPO vs MATD3 vs MADDPG (no co-evolution).

Goal
----
Answer the user's question empirically: *which AgileRL multi-agent
algorithm learns the aquarium predator pursuit task best when trained
against a FIXED scripted APF prey opponent, and with which
hyperparameters?*

Unlike the co-evolution experiment (where the prey side evolves), here
the opponent is **frozen**: APF-flee prey baked into the env via
:class:`ScriptedOpponentEnvWrapper`.  Only the 3 predators are learnable
agents.  Because the target is stationary-in-distribution, any rise in
the APF benchmark score is unambiguous evidence the algorithm is
learning to catch prey.

deap-bridge API
---------------
All three algorithms are driven through the **same** deap-bridge
``AgileRLTrainer`` surface:

  * ``MATD3Trainer``  — off-policy, twin critics  (off-policy loop)
  * ``MADDPGTrainer`` — off-policy, single critic  (off-policy loop)
  * ``IPPOTrainer``   — on-policy, independent PPO  (on-policy loop)

Each ``trainer.train(TrainingConfig(env_name=ENV_NAME_APF, ...))`` call
trains one agent against the registered APF env.  After training we
score the returned policy with the **noise-free** APF benchmark
(``_run_scripted_benchmark`` reused from the co-evolution experiment),
which counts real catches.

Run
---
    # quick calibration (small budget, 1 config per algo)
    .venv\\Scripts\\python.exe examples\\pursuit_evasion\\rl_vs_apf_suite.py \
        --budget 60000 --bench-eps 8 --device cuda --configs matd3:base,maddpg:base,ippo:base

    # full suite
    .venv\\Scripts\\python.exe examples\\pursuit_evasion\\rl_vs_apf_suite.py \
        --budget 300000 --bench-eps 16 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Headless rendering — must be set before pygame ever calls init().
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Allow running as a plain script *or* as a module (-m).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from deap_pettingzoo.baselines.agilerl_trainer import (
    IPPOTrainer,
    MADDPGTrainer,
    MATD3Trainer,
)
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.environments.registry import register_env

# Reuse the co-evolution experiment's env config, APF wrappers and the
# noise-free APF benchmark so the comparison is apples-to-apples with the
# MATD3 co-evolution results.
import examples.pursuit_evasion.aquarium_coevolution_ea_cma_matd3 as coev
from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    ScriptedOpponentEnvWrapper,
    _patch_aquarium_wrappers,
)


# ===========================================================================
# Registered env: predators learn vs FIXED APF prey (no co-evolution)
# ===========================================================================

ENV_NAME_APF = "aquarium_pred_vs_apf_v0"


def _pred_feature_wrapper(inner):
    """Apply the same observation feature transform used in training."""
    return AquariumObservationFeatureWrapper(inner, append_agent_id=False)


class _PredVsApfFactory:
    """Picklable factory: aquarium + APF prey baked in, predators exposed."""

    def __init__(self, env_config: Dict[str, Any]) -> None:
        self.env_config = dict(env_config)

    def __call__(self, render_mode: Optional[str] = None, **kwargs):
        from marl_aquarium import aquarium_v0

        cfg = {**self.env_config, **kwargs}
        if render_mode is None:
            render_mode = "rgb_array"
        inner = aquarium_v0.parallel_env(render_mode=render_mode, **cfg)
        _patch_aquarium_wrappers(inner)
        return ScriptedOpponentEnvWrapper(
            inner,
            predator_observe_count=int(cfg["predator_observe_count"]),
            prey_observe_count=int(cfg["prey_observe_count"]),
            action_count=int(cfg.get("action_count", 16)),
            predator_reward=int(cfg.get("predator_reward", 20)),
        )


def _register_apf_env() -> None:
    register_env(
        ENV_NAME_APF,
        _PredVsApfFactory(env_config=coev.ENV_CONFIG),
        default_episode_len=int(coev.ENV_CONFIG["max_time_steps"]),
        continuous_default=False,
        notes=(
            "marl_aquarium.aquarium_v0 with APF-flee prey baked in "
            "(ScriptedOpponentEnvWrapper); only the 3 predators are "
            "learnable — used for the RL-vs-APF algorithm comparison."
        ),
        tags={"aquarium", "pursuit_evasion", "scripted_prey", "rl_vs_apf"},
        api="parallel",
        wrappers=(_pred_feature_wrapper,),
        overwrite=True,
    )


# ===========================================================================
# Hyperparameter configurations per algorithm
# ===========================================================================
#
# Each config is a dict of ``algorithm_params`` forwarded to the trainer.
# Off-policy (matd3/maddpg) split (per AgileRLTrainer): loop keys
# {memory_size, learning_delay, learn_step} drive the rollout/learn
# cadence; everything else (expl_noise, O_U_noise, lr_actor, lr_critic,
# gamma, batch_size) goes to the AgileRL constructor.
#
# IPPO (on-policy) uses learn_step as the rollout length and ctor knobs
# lr / ent_coef / clip_coef / gamma / gae_lambda / update_epochs.
#
# References: AgileRL MADDPG/IPPO docs (constructor defaults) + the
# MATD3 co-evolution cuda sweep (expl_noise=0.8 was optimal on discrete).

# Shared off-policy loop knobs (frequent updates ~ proven coev ratio).
_OFF_POLICY_LOOP = {
    "memory_size": 100_000,
    "learning_delay": 256,
    "learn_step": 32,      # call learn() every 32 env steps
    "batch_size": 128,
}

MATD3_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Proven co-evolution winner: expl_noise=0.8, low LRs, O-U off.
    "base": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.8, "O_U_noise": False,
        "lr_actor": 1e-4, "lr_critic": 1e-3, "gamma": 0.95,
    },
    "expl06": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.6, "O_U_noise": False,
        "lr_actor": 1e-4, "lr_critic": 1e-3, "gamma": 0.95,
    },
    "gamma99": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.8, "O_U_noise": False,
        "lr_actor": 1e-4, "lr_critic": 1e-3, "gamma": 0.99,
    },
}

MADDPG_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Mirror the MATD3 winner (same DDPG family, single critic).
    "base": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.8, "O_U_noise": False,
        "lr_actor": 1e-4, "lr_critic": 1e-3, "gamma": 0.95,
    },
    "expl06": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.6, "O_U_noise": False,
        "lr_actor": 1e-4, "lr_critic": 1e-3, "gamma": 0.95,
    },
    # AgileRL library defaults (higher LRs, O-U noise on).
    "agilerl_default": {
        **_OFF_POLICY_LOOP,
        "expl_noise": 0.1, "O_U_noise": True,
        "lr_actor": 1e-3, "lr_critic": 1e-2, "gamma": 0.95,
    },
}

IPPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    # AgileRL IPPO doc defaults, with a moderate rollout length.
    "base": {
        "lr": 1e-4, "ent_coef": 0.01, "clip_coef": 0.2,
        "gamma": 0.99, "gae_lambda": 0.95, "update_epochs": 4,
        "batch_size": 256, "learn_step": 1024,
    },
    # Higher LR + more exploration (entropy) — common for sparse rewards.
    "lr3e4_ent05": {
        "lr": 3e-4, "ent_coef": 0.05, "clip_coef": 0.2,
        "gamma": 0.99, "gae_lambda": 0.95, "update_epochs": 4,
        "batch_size": 256, "learn_step": 1024,
    },
    # Longer rollout (lower-variance advantage estimates).
    "rollout2048": {
        "lr": 3e-4, "ent_coef": 0.01, "clip_coef": 0.2,
        "gamma": 0.99, "gae_lambda": 0.95, "update_epochs": 4,
        "batch_size": 256, "learn_step": 2048,
    },
}

ALGO_REGISTRY: Dict[str, Tuple[Any, Dict[str, Dict[str, Any]]]] = {
    "matd3":  (MATD3Trainer,  MATD3_CONFIGS),
    "maddpg": (MADDPGTrainer, MADDPG_CONFIGS),
    "ippo":   (IPPOTrainer,   IPPO_CONFIGS),
}


# ===========================================================================
# One training + benchmark run
# ===========================================================================


def _train_and_score(
    algo: str,
    config_name: str,
    params: Dict[str, Any],
    *,
    budget: int,
    bench_eps: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    """Train one (algo, config) against APF prey; return benchmark result."""
    trainer_cls, _ = ALGO_REGISTRY[algo]
    trainer = trainer_cls(device=device, verbose=False)

    algorithm_params = dict(params)
    algorithm_params["seed"] = int(seed)

    cfg = TrainingConfig(
        env_name=ENV_NAME_APF,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=int(budget),
        algorithm_params=algorithm_params,
        env_config={},
        output_dir=None,  # NoOpLogger — no disk artefacts
    )

    t0 = time.time()
    result = trainer.train(cfg)
    train_s = time.time() - t0

    # Noise-free APF benchmark (counts real catches) — reuse the exact
    # evaluation harness the co-evolution experiment used so the numbers
    # are directly comparable with the MATD3 coev results.
    t1 = time.time()
    bench = coev._run_scripted_benchmark(
        result.policy, "predator", coev.ENV_CONFIG, n_episodes=bench_eps, seed=10_000 + seed
    )
    bench_s = time.time() - t1

    ep_rewards = list(result.metrics.get("episode_rewards", []) or [])
    train_mean_last = (
        float(np.mean(ep_rewards[-10:])) if ep_rewards else float("nan")
    )

    return {
        "algo": algo,
        "config": config_name,
        "benchmark": float(bench),
        "train_mean_last10": train_mean_last,
        "train_episodes": len(ep_rewards),
        "train_s": train_s,
        "bench_s": bench_s,
    }


# ===========================================================================
# Suite driver
# ===========================================================================


def _parse_configs_arg(arg: Optional[str]) -> List[Tuple[str, str]]:
    """Parse ``--configs matd3:base,ippo:lr3e4_ent05`` into (algo, cfg) pairs.

    When ``arg`` is ``None`` / empty, expand to *all* configs of *all*
    algorithms.
    """
    if not arg:
        pairs: List[Tuple[str, str]] = []
        for algo, (_, configs) in ALGO_REGISTRY.items():
            for cfg_name in configs:
                pairs.append((algo, cfg_name))
        return pairs

    pairs = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"--configs token {token!r} must be 'algo:config' "
                "(e.g. 'matd3:base')."
            )
        algo, cfg_name = token.split(":", 1)
        algo = algo.strip()
        cfg_name = cfg_name.strip()
        if algo not in ALGO_REGISTRY:
            raise ValueError(f"unknown algo {algo!r}; choices: {list(ALGO_REGISTRY)}")
        if cfg_name not in ALGO_REGISTRY[algo][1]:
            raise ValueError(
                f"unknown config {cfg_name!r} for {algo!r}; "
                f"choices: {list(ALGO_REGISTRY[algo][1])}"
            )
        pairs.append((algo, cfg_name))
    return pairs


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=200_000,
                        help="env-step training budget per run")
    parser.add_argument("--bench-eps", type=int, default=12,
                        help="APF benchmark episodes (noise-free eval)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--configs", type=str, default=None,
                        help="comma list of algo:config (default: all)")
    args = parser.parse_args()

    _register_apf_env()

    pairs = _parse_configs_arg(args.configs)

    _section(
        "RL-vs-APF suite (no co-evolution) — IPPO / MATD3 / MADDPG\n"
        f"env={ENV_NAME_APF}  budget={args.budget:,} env-steps  "
        f"bench_eps={args.bench_eps}  seed={args.seed}  device={args.device}\n"
        f"runs: {', '.join(f'{a}:{c}' for a, c in pairs)}"
    )

    results: List[Dict[str, Any]] = []
    for algo, cfg_name in pairs:
        params = ALGO_REGISTRY[algo][1][cfg_name]
        print(f"\n--- training {algo}:{cfg_name} ---", flush=True)
        try:
            res = _train_and_score(
                algo, cfg_name, params,
                budget=args.budget, bench_eps=args.bench_eps,
                seed=args.seed, device=args.device,
            )
        except Exception as exc:  # surface but keep the suite going
            import traceback
            traceback.print_exc()
            res = {
                "algo": algo, "config": cfg_name, "benchmark": float("nan"),
                "train_mean_last10": float("nan"), "train_episodes": 0,
                "train_s": 0.0, "bench_s": 0.0, "error": str(exc),
            }
        results.append(res)
        print(
            f"    -> benchmark={res['benchmark']:.2f}  "
            f"train_last10={res['train_mean_last10']:.2f}  "
            f"train_eps={res['train_episodes']}  "
            f"train={res['train_s']:.0f}s bench={res['bench_s']:.0f}s",
            flush=True,
        )

    # ----- ranking -----
    ranked = sorted(
        results,
        key=lambda r: (np.isnan(r["benchmark"]), -(r["benchmark"] if not np.isnan(r["benchmark"]) else 0.0)),
    )
    _section("RANKING by APF benchmark (mean learner reward; ~higher = more catches)")
    print(f"{'rank':<5}{'algo':<8}{'config':<18}{'benchmark':>10}{'train_last10':>14}{'train_eps':>11}")
    print("-" * 66)
    for i, r in enumerate(ranked, 1):
        bench = "nan" if np.isnan(r["benchmark"]) else f"{r['benchmark']:.2f}"
        tlast = "nan" if np.isnan(r["train_mean_last10"]) else f"{r['train_mean_last10']:.2f}"
        print(f"{i:<5}{r['algo']:<8}{r['config']:<18}{bench:>10}{tlast:>14}{r['train_episodes']:>11}")

    # Best per algorithm.
    _section("BEST hyperparameter config per algorithm")
    best_by_algo: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if np.isnan(r["benchmark"]):
            continue
        cur = best_by_algo.get(r["algo"])
        if cur is None or r["benchmark"] > cur["benchmark"]:
            best_by_algo[r["algo"]] = r
    for algo in ("ippo", "matd3", "maddpg"):
        r = best_by_algo.get(algo)
        if r is None:
            print(f"  {algo:<8}: no successful run")
        else:
            print(f"  {algo:<8}: config={r['config']:<18} benchmark={r['benchmark']:.2f}")


if __name__ == "__main__":
    main()
