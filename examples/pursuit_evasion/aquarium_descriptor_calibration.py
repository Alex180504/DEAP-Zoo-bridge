"""Measure achievable QDax descriptor bands on the balanced aquarium config.

The pursuit-evasion descriptors (formation spread, mean distance to
opponents, mean pairwise teammate distance) are normalised to ``[0, 1]``
archive axes.  Dividing by a theoretical maximum (the arena diagonal)
compresses every realistic trajectory into a small corner and the MAP-Elites
grid never fills.  This script measures the band those axes *actually* reach
by rolling out behaviour archetypes that bracket what evolving populations
produce:

  * APF vs APF          — competent pursuit and evasion
  * random vs APF       — incompetent pursuit
  * APF vs random       — incompetent evasion
  * random vs random    — chaos baseline
  * huddle vs APF       — all pursuers hold one shared heading (tightest
                          formations, the classic degenerate attractor)
  * fan-out vs APF      — each pursuer holds a distinct heading (widest
                          formations)

For each axis the 5th–95th percentile across every episode is reported as a
``DESCRIPTOR_RANGES`` entry for
``examples/pursuit_evasion/aquarium/experiment_config.py``.  Positions are
normalised to ``[0, 1]²`` by the arena dimensions — the same coordinate
space both QDax extractors emit.

Run directly::

    python examples/pursuit_evasion/aquarium_descriptor_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
    _AquariumExtractor,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    _patch_aquarium_wrappers,
)

EPISODES_PER_ARCHETYPE: int = 6
SEED_BASE: int = 0
PERCENTILES: Tuple[float, float] = (5.0, 95.0)
OUTPUT_DIR: Path = Path("results/aquarium_env_balance")


class _FixedHeadingPolicy:
    """Constant discrete-heading policy (degenerate-behaviour archetype)."""

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def reset(self) -> None:
        return None

    def get_action(self, observation, deterministic: bool = True) -> int:
        del observation, deterministic
        return self.action


class _RandomPolicy:
    """Uniform random discrete policy."""

    def __init__(self, action_count: int, seed: int) -> None:
        self.action_count = int(action_count)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def get_action(self, observation, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self.rng.integers(self.action_count))


def _make_apf_pursuer(cfg: Dict[str, Any]):
    return AquariumAPFPursuerPolicy(
        predator_observe_count=cfg["predator_observe_count"],
        prey_observe_count=cfg["prey_observe_count"],
        action_count=cfg["action_count"],
        config=APF_PURSUER_CONFIG,
        wall_features=4 if cfg.get("observable_walls") else 0,
    )


def _make_apf_evader(cfg: Dict[str, Any]):
    policy = AquariumAPFEvaderPolicy(
        predator_observe_count=cfg["prey_observe_count"],
        prey_observe_count=cfg["prey_observe_count"],
        action_count=cfg["action_count"],
        config=APF_EVADER_CONFIG,
        wall_features=4 if cfg.get("observable_walls") else 0,
    )
    policy.reset()
    return policy


def _episode_axis_means(
    cfg: Dict[str, Any],
    pred_policies: List[Any],
    prey_policies: List[Any],
    seed: int,
) -> Dict[str, float]:
    """Roll one episode; return the time-mean of each descriptor axis."""
    from marl_aquarium import aquarium_v0

    env_kwargs = {k: v for k, v in cfg.items() if k != "continuous_actions"}
    env = aquarium_v0.parallel_env(render_mode="rgb_array", **env_kwargs)
    _patch_aquarium_wrappers(env)
    try:
        observations, _ = env.reset(seed=seed)
        raw = _AquariumExtractor._walk_to_raw(env)
        w, h = float(cfg["width"]), float(cfg["height"])

        per_step: Dict[str, List[float]] = {
            "pred_spread": [], "prey_spread": [], "cross_dist": [],
            "pred_pair": [], "prey_pair": [],
        }
        pred_ids = [a for a in env.possible_agents if a.startswith("predator")]
        prey_ids = [a for a in env.possible_agents if a.startswith("prey")]

        for _ in range(int(cfg["max_time_steps"])):
            if not env.agents:
                break
            preds = np.asarray(
                [
                    _AquariumExtractor._position_xy(p.position)
                    for p in getattr(raw, "predators", [])
                ],
                dtype=np.float64,
            ) / [w, h]
            prey = np.asarray(
                [
                    _AquariumExtractor._position_xy(p.position)
                    for p in getattr(raw, "prey", [])
                ],
                dtype=np.float64,
            ) / [w, h] if getattr(raw, "prey", []) else np.zeros((0, 2))

            if preds.shape[0] >= 2:
                per_step["pred_spread"].append(
                    float(np.mean(np.linalg.norm(preds - preds.mean(0), axis=1)))
                )
                diffs = preds[:, None, :] - preds[None, :, :]
                d = np.linalg.norm(diffs, axis=-1)
                per_step["pred_pair"].append(
                    float(d.sum() / (preds.shape[0] * (preds.shape[0] - 1)))
                )
            if prey.shape[0] >= 2:
                per_step["prey_spread"].append(
                    float(np.mean(np.linalg.norm(prey - prey.mean(0), axis=1)))
                )
                diffs = prey[:, None, :] - prey[None, :, :]
                d = np.linalg.norm(diffs, axis=-1)
                per_step["prey_pair"].append(
                    float(d.sum() / (prey.shape[0] * (prey.shape[0] - 1)))
                )
            if preds.shape[0] and prey.shape[0]:
                cross = np.linalg.norm(
                    preds[:, None, :] - prey[None, :, :], axis=-1
                )
                per_step["cross_dist"].append(float(cross.mean()))

            actions = {}
            for i, a in enumerate(pred_ids):
                if a in observations:
                    actions[a] = pred_policies[i % len(pred_policies)].get_action(
                        observations[a]
                    )
            for i, a in enumerate(prey_ids):
                if a in observations:
                    actions[a] = prey_policies[i % len(prey_policies)].get_action(
                        observations[a]
                    )
            actions = {a: act for a, act in actions.items() if a in env.agents}
            if not actions:
                break
            observations, _, _, _, _ = env.step(actions)

        return {k: float(np.mean(v)) for k, v in per_step.items() if v}
    finally:
        try:
            env.close()
        except SystemExit:
            pass


def main() -> None:
    cfg = dict(BALANCED_ENV_CONFIG)
    n_pred = int(cfg["predator_count"])
    n_prey = int(cfg["prey_count"])
    n_act = int(cfg["action_count"])

    archetypes: List[Tuple[str, Any, Any]] = [
        (
            "apf_vs_apf",
            lambda s: [_make_apf_pursuer(cfg) for _ in range(n_pred)],
            lambda s: [_make_apf_evader(cfg) for _ in range(n_prey)],
        ),
        (
            "random_vs_apf",
            lambda s: [_RandomPolicy(n_act, s + i) for i in range(n_pred)],
            lambda s: [_make_apf_evader(cfg) for _ in range(n_prey)],
        ),
        (
            "apf_vs_random",
            lambda s: [_make_apf_pursuer(cfg) for _ in range(n_pred)],
            lambda s: [_RandomPolicy(n_act, s + 100 + i) for i in range(n_prey)],
        ),
        (
            "random_vs_random",
            lambda s: [_RandomPolicy(n_act, s + i) for i in range(n_pred)],
            lambda s: [_RandomPolicy(n_act, s + 100 + i) for i in range(n_prey)],
        ),
        (
            "huddle_vs_apf",
            lambda s: [_FixedHeadingPolicy(s % n_act) for _ in range(n_pred)],
            lambda s: [_make_apf_evader(cfg) for _ in range(n_prey)],
        ),
        (
            "fanout_vs_apf",
            lambda s: [
                _FixedHeadingPolicy((s + i * n_act // n_pred) % n_act)
                for i in range(n_pred)
            ],
            lambda s: [_make_apf_evader(cfg) for _ in range(n_prey)],
        ),
    ]

    samples: Dict[str, List[float]] = {
        "pred_spread": [], "prey_spread": [], "cross_dist": [],
        "pred_pair": [], "prey_pair": [],
    }
    for name, make_preds, make_prey in archetypes:
        for ep in range(EPISODES_PER_ARCHETYPE):
            seed = SEED_BASE + ep
            means = _episode_axis_means(cfg, make_preds(seed), make_prey(seed), seed)
            for key, value in means.items():
                samples[key].append(value)
        print(f"  {name:18s} done ({EPISODES_PER_ARCHETYPE} episodes)")

    lo_p, hi_p = PERCENTILES
    bands = {
        key: (
            round(float(np.percentile(vals, lo_p)), 3),
            round(float(np.percentile(vals, hi_p)), 3),
        )
        for key, vals in samples.items()
        if vals
    }

    print("\nPer-axis achievable bands (P5–P95, positions in [0, 1]²):")
    for key, (lo, hi) in bands.items():
        print(f"  {key:12s} ({lo:.3f}, {hi:.3f})")

    ranges = {
        "predator": {"spread": bands["pred_spread"], "dist": bands["cross_dist"]},
        "prey": {"spread": bands["prey_spread"], "dist": bands["cross_dist"]},
        "predator_single": {"spread": bands["pred_spread"], "pair": bands["pred_pair"]},
        "prey_single": {"spread": bands["prey_spread"], "pair": bands["prey_pair"]},
    }
    print("\nDESCRIPTOR_RANGES (paste into experiment_config.py):")
    print(json.dumps(ranges, indent=4))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "descriptor_ranges.json"
    out.write_text(json.dumps(ranges, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
