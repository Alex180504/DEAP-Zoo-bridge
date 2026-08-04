"""Equal-ability calibration for the aquarium pursuit-evasion env config.

Plays scripted APF pursuers against scripted APF evaders (no learning) on a
staged grid of env-config candidates and reports how one-sided each setup
is.  The balance target is a *catch fraction* (prey caught / prey count at
the episode horizon) of ~0.5: strong pursuit is rewarded, but escape stays
achievable, so neither learner role starts from a hopeless position.

Because both sides use the same APF force-model family, the scripted match
is a cheap, repeatable proxy for learnable balance — far cheaper than
training-based calibration and stable across seeds.

Stages (each keeps the previous stage's winner):

  1. speed ratio        — predator vs prey max velocity
  2. view distances     — symmetric vs one-sided information advantage
  3. topology + catch_radius — rectangle vs torus, catch annulus size

Run directly::

    python examples/pursuit_evasion/aquarium_env_balance.py              # grid
    python examples/pursuit_evasion/aquarium_env_balance.py --reference  # shipped config

``--reference`` skips the grid and measures only the shipped config, in both
action modes, at ``REFERENCE_EPISODES`` episodes — the APF-vs-APF baseline the
campaign quotes its learned catch rates against.

Edit the module constants to change episode counts / the candidate grid.
The chosen config is what ``examples/pursuit_evasion/aquarium/
experiment_config.py`` ships as the canonical balanced setup — re-run this
script after any env upgrade to re-validate it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.pursuit_evasion.aquarium.apf_policy import (
    AquariumAPFEvaderPolicy,
    AquariumAPFPursuerPolicy,
)
from examples.pursuit_evasion.aquarium.experiment_config import (
    APF_EVADER_CONFIG,
    APF_PURSUER_CONFIG,
    BALANCED_ENV_CONFIG,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    _patch_aquarium_wrappers,
)

# ---- Scope ------------------------------------------------------------------

EPISODES_PER_CONFIG: int = 12
SEED_BASE: int = 0
TARGET_CATCH_FRACTION: float = 0.5
VERIFY_CONTINUOUS: bool = True   # re-run the winner in continuous action mode
OUTPUT_DIR: Path = Path("results/aquarium_env_balance")

# Reference measurement (``--reference``): the shipped config only, at enough
# episodes to resolve the imbalance.  Catch fractions swing ~±0.04 run to run,
# so the 12-episode grid pass cannot separate a genuine bias from noise; the
# methodology reports the residual APF-vs-APF imbalance as a caveat rather
# than calibrating it away, and that number needs a real CI.
REFERENCE_EPISODES: int = 128

# ---- Base env config (stage candidates override individual keys) -------------
# Re-validation workflow: the base IS the shipped canonical config, so
# re-running this script after an env upgrade checks whether the canonical
# choice still sits closest to the balance target.

BASE_ENV_CONFIG: Dict[str, Any] = dict(BALANCED_ENV_CONFIG)

PURSUER_APF = APF_PURSUER_CONFIG
EVADER_APF = APF_EVADER_CONFIG

# ---- Staged candidate grid ----------------------------------------------------

STAGE_1_SPEED: List[Tuple[str, Dict[str, Any]]] = [
    ("speed_12v12", {"predator_max_velocity": 12.0, "prey_max_velocity": 12.0}),
    ("speed_13v12", {"predator_max_velocity": 13.0, "prey_max_velocity": 12.0}),
    ("speed_14v12", {"predator_max_velocity": 14.0, "prey_max_velocity": 12.0}),
    ("speed_12v13", {"predator_max_velocity": 12.0, "prey_max_velocity": 13.0}),
]

STAGE_2_VIEW: List[Tuple[str, Dict[str, Any]]] = [
    ("view_400v400", {"predator_view_distance": 400, "prey_view_distance": 400}),
    ("view_500v400", {"predator_view_distance": 500, "prey_view_distance": 400}),
    ("view_400v500", {"predator_view_distance": 400, "prey_view_distance": 500}),
]

STAGE_3_TOPOLOGY: List[Tuple[str, Dict[str, Any]]] = [
    ("rect_catch40", {"observable_walls": True, "catch_radius": 40}),
    ("rect_catch50", {"observable_walls": True, "catch_radius": 50}),
    ("rect_catch60", {"observable_walls": True, "catch_radius": 60}),
    ("torus_catch50", {"observable_walls": False, "catch_radius": 50}),
]


def _play_episode(env_config: Dict[str, Any], seed: int, continuous: bool) -> Dict[str, Any]:
    """Run one fully scripted APF-vs-APF episode; return catch statistics."""
    from marl_aquarium import aquarium_v0

    env_kwargs = {k: v for k, v in env_config.items() if k != "continuous_actions"}
    env = aquarium_v0.parallel_env(
        render_mode="rgb_array", continuous_actions=continuous, **env_kwargs
    )
    _patch_aquarium_wrappers(env)
    try:
        observations, _ = env.reset(seed=seed)
        wall_features = 4 if env_config.get("observable_walls", False) else 0
        pursuer = AquariumAPFPursuerPolicy(
            predator_observe_count=env_config["predator_observe_count"],
            prey_observe_count=env_config["prey_observe_count"],
            action_count=env_config["action_count"],
            config=PURSUER_APF,
            wall_features=wall_features,
            continuous=continuous,
        )
        evader = AquariumAPFEvaderPolicy(
            predator_observe_count=env_config["prey_observe_count"],
            prey_observe_count=env_config["prey_observe_count"],
            action_count=env_config["action_count"],
            config=EVADER_APF,
            wall_features=wall_features,
            continuous=continuous,
        )
        evader.reset()

        prey_total = int(env_config["prey_count"])
        alive = prey_total
        catch_steps: List[int] = []
        for step in range(int(env_config["max_time_steps"])):
            if not env.agents:
                break
            actions = {}
            for agent, obs in observations.items():
                policy = pursuer if agent.startswith("predator") else evader
                actions[agent] = policy.get_action(obs)
            observations, _, _, _, _ = env.step(actions)
            now_alive = sum(1 for a in env.agents if a.startswith("prey"))
            for _ in range(alive - now_alive):
                catch_steps.append(step)
            alive = now_alive
            if alive == 0:
                break

        catches = prey_total - alive
        return {
            "catches": catches,
            "catch_fraction": catches / float(prey_total),
            "median_catch_step": float(np.median(catch_steps)) if catch_steps else None,
        }
    finally:
        try:
            env.close()
        except SystemExit:
            pass


def evaluate_config(
    name: str,
    env_config: Dict[str, Any],
    episodes: int = EPISODES_PER_CONFIG,
    continuous: bool = False,
) -> Dict[str, Any]:
    """Average episode statistics for one env-config candidate."""
    t0 = time.time()
    fractions: List[float] = []
    catch_steps: List[float] = []
    for ep in range(episodes):
        stats = _play_episode(env_config, seed=SEED_BASE + ep, continuous=continuous)
        fractions.append(stats["catch_fraction"])
        if stats["median_catch_step"] is not None:
            catch_steps.append(stats["median_catch_step"])
    mean_fraction = float(np.mean(fractions))
    result = {
        "name": name,
        "continuous": continuous,
        "catch_fraction_mean": mean_fraction,
        "catch_fraction_std": float(np.std(fractions)),
        "balance_error": abs(mean_fraction - TARGET_CATCH_FRACTION),
        "median_catch_step": float(np.median(catch_steps)) if catch_steps else None,
        "episodes": episodes,
        "wall_time_s": time.time() - t0,
        "overrides": {
            k: v for k, v in env_config.items() if BASE_ENV_CONFIG.get(k) != v
        },
    }
    return result


def _print_stage(title: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"\n=== {title} ===")
    print(f"{'config':<16} {'catch_frac':>10} {'±std':>7} {'med_step':>9} {'err':>6}")
    for r in sorted(results, key=lambda r: r["balance_error"]):
        med = f"{r['median_catch_step']:.0f}" if r["median_catch_step"] is not None else "—"
        print(
            f"{r['name']:<16} {r['catch_fraction_mean']:>10.3f} "
            f"{r['catch_fraction_std']:>7.3f} {med:>9} {r['balance_error']:>6.3f}"
        )
    best = min(results, key=lambda r: r["balance_error"])
    print(f"--> winner: {best['name']}")
    return best


def run_reference() -> Dict[str, Any]:
    """Measure the *shipped* config in both action modes, at high precision.

    This is the campaign's APF-vs-APF reference point: how one-sided the env
    is before any learning, for the exact config the suites train on.  It runs
    no grid search — the methodology keeps the env perfectly symmetric and
    reports the residual imbalance as a caveat rather than calibrating it away.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_config(
            f"shipped_{'continuous' if continuous else 'discrete'}",
            BASE_ENV_CONFIG,
            episodes=REFERENCE_EPISODES,
            continuous=continuous,
        )
        for continuous in (False, True)
    ]
    print(f"\n=== APF-vs-APF reference ({REFERENCE_EPISODES} episodes/mode) ===")
    print(f"{'mode':<22} {'catch_frac':>10} {'±sem':>7} {'med_step':>9} {'err':>6}")
    for r in results:
        sem = r["catch_fraction_std"] / (r["episodes"] ** 0.5)
        med = f"{r['median_catch_step']:.0f}" if r["median_catch_step"] is not None else "—"
        print(
            f"{r['name']:<22} {r['catch_fraction_mean']:>10.3f} "
            f"{sem:>7.3f} {med:>9} {r['balance_error']:>6.3f}"
        )
        r["catch_fraction_sem"] = sem

    report = {"config": dict(BASE_ENV_CONFIG), "results": results}
    out = OUTPUT_DIR / "reference_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreference written to {out}")
    return report


def main() -> None:
    if "--reference" in sys.argv[1:]:
        run_reference()
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    config = dict(BASE_ENV_CONFIG)

    stage1 = [
        evaluate_config(name, {**config, **overrides})
        for name, overrides in STAGE_1_SPEED
    ]
    all_results += stage1
    best1 = _print_stage("Stage 1: speed ratio", stage1)
    config.update(next(o for n, o in STAGE_1_SPEED if n == best1["name"]))

    stage2 = [
        evaluate_config(name, {**config, **overrides})
        for name, overrides in STAGE_2_VIEW
    ]
    all_results += stage2
    best2 = _print_stage("Stage 2: view distances", stage2)
    config.update(next(o for n, o in STAGE_2_VIEW if n == best2["name"]))

    stage3 = [
        evaluate_config(name, {**config, **overrides})
        for name, overrides in STAGE_3_TOPOLOGY
    ]
    all_results += stage3
    best3 = _print_stage("Stage 3: topology + catch radius", stage3)
    config.update(next(o for n, o in STAGE_3_TOPOLOGY if n == best3["name"]))

    if VERIFY_CONTINUOUS:
        cont = evaluate_config("winner_continuous", config, continuous=True)
        all_results.append(cont)
        _print_stage("Continuous-mode verification of winner", [cont])

    print("\n=== Calibrated env config ===")
    print(json.dumps(config, indent=2))
    out = OUTPUT_DIR / "balance_report.json"
    out.write_text(
        json.dumps({"chosen_config": config, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
