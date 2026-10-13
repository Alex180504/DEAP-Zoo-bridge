"""Can the evolved-family policy class express the scripted APF pursuer?

The 2026-08-01 pilot left one fork unresolved.  Evolved pursuers reach ~0.74
catches of 5 prey while the scripted APF controller reaches ~3.4 on the same
config — a 5x gap.  Two very different causes produce that:

* **Search failure** — the policy class *can* represent APF-quality pursuit,
  but 27 generations of a population of 24 never find it.  Then the fix is
  budget allocation (K vs generations), population size, or the co-evolution
  dynamics.
* **Representation failure** — the class *cannot* represent it at all, and no
  amount of search will help.  Then the network width or the observation
  encoding has to change before any further cluster time is spent.

This script settles it by behaviour-cloning the APF pursuer *into the evolved
family's own network* (``[obs, hidden, actions]``, tanh hidden, linear output,
decoded by the same action head) and measuring the clone's catch rate through
the same benchmark harness the campaign probes use.

Reading the result:

* clone ~= APF          -> the class is fine; the pilot's gap is search.
* clone >> evolved best -> the class is fine and there is reachable headroom,
  even if cloning is imperfect.
* clone ~= evolved best -> representation (or the encoding) is the binding
  constraint.

Two controls separate capacity from the usual behaviour-cloning artefacts:

* a **wider** hidden layer is cloned alongside the shipped width, so "16 units
  is too few" is distinguishable from "this class cannot do it";
* one **DAgger round** relabels states the clone itself visits, so a low catch
  rate cannot be blamed on covariate shift alone.

Run::

    python examples/pursuit_evasion/aquarium_policy_capacity.py
    python examples/pursuit_evasion/aquarium_policy_capacity.py --mode continuous
"""

from __future__ import annotations

import argparse
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

from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.policies.feedforward_neural import FeedforwardNeuralPolicy
from examples.pursuit_evasion.aquarium import experiment_config as XC

XC.ensure_utf8_stdout()
from examples.pursuit_evasion.aquarium.apf_policy import (
    AquariumAPFEvaderPolicy,
    AquariumAPFPursuerPolicy,
)
from examples.pursuit_evasion.aquarium.observation_features import (
    aquarium_feature_size,
    featurize_aquarium_observation,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    _patch_aquarium_wrappers,
)

OUTPUT_DIR = Path("results/aquarium_policy_capacity")

CLONE_EPISODES = 40          # expert-rollout episodes for the training set
DAGGER_EPISODES = 20         # clone-rollout episodes relabelled by the expert
EVAL_EPISODES = XC.EVAL_EPISODES
HIDDEN_SIZES = (XC.EVOLVED_HIDDEN_SIZE, 64)
EPOCHS = 60
BATCH = 512
LR = 3e-3


def _make_env(env_cfg: Dict[str, Any]):
    from marl_aquarium import aquarium_v0

    kwargs = {k: v for k, v in env_cfg.items() if k != "continuous_actions"}
    env = aquarium_v0.parallel_env(
        render_mode="rgb_array",
        continuous_actions=bool(env_cfg.get("continuous_actions", False)),
        **kwargs,
    )
    _patch_aquarium_wrappers(env)
    return env


def _apf_policies(env_cfg: Dict[str, Any], continuous: bool):
    """The scripted pursuer (the expert) and evader (the fixed opponent)."""
    walls = 4 if env_cfg.get("observable_walls", False) else 0
    pursuer = AquariumAPFPursuerPolicy(
        predator_observe_count=env_cfg["predator_observe_count"],
        prey_observe_count=env_cfg["prey_observe_count"],
        action_count=env_cfg["action_count"],
        config=XC.APF_PURSUER_CONFIG,
        wall_features=walls,
        continuous=continuous,
    )
    evader = AquariumAPFEvaderPolicy(
        predator_observe_count=env_cfg["prey_observe_count"],
        prey_observe_count=env_cfg["prey_observe_count"],
        action_count=env_cfg["action_count"],
        config=XC.APF_EVADER_CONFIG,
        wall_features=walls,
        continuous=continuous,
    )
    return pursuer, evader


def rollout(
    env_cfg: Dict[str, Any],
    n_episodes: int,
    seed_base: int,
    *,
    predator_policy: Optional[Any] = None,
    collect: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Play episodes and optionally record (featurised obs, expert action).

    ``predator_policy`` drives the predators; ``None`` means the APF expert
    drives them.  The expert *labels* every recorded state either way, so
    passing the clone here produces DAgger-style on-policy relabelling.

    Returns:
        ``(features, labels, mean_catches)``.
    """
    continuous = bool(env_cfg.get("continuous_actions", False))
    walls = 4 if env_cfg.get("observable_walls", False) else 0
    expert, evader = _apf_policies(env_cfg, continuous)
    prey_total = int(env_cfg["prey_count"])

    feats: List[np.ndarray] = []
    labels: List[Any] = []
    catches: List[int] = []

    for episode in range(n_episodes):
        env = _make_env(env_cfg)
        try:
            observations, _ = env.reset(seed=seed_base + episode)
            evader.reset()
            if predator_policy is not None and hasattr(predator_policy, "reset"):
                predator_policy.reset()
            alive = prey_total
            for _ in range(int(env_cfg["max_time_steps"])):
                if not env.agents:
                    break
                actions: Dict[str, Any] = {}
                for agent, obs in observations.items():
                    if agent.startswith("predator"):
                        expert_action = expert.get_action(obs)
                        features = featurize_aquarium_observation(
                            obs,
                            env_cfg["predator_observe_count"],
                            env_cfg["prey_observe_count"],
                            wall_features=walls,
                        )
                        if collect:
                            feats.append(features)
                            labels.append(expert_action)
                        if predator_policy is None:
                            actions[agent] = expert_action
                        else:
                            actions[agent] = predator_policy.get_action(
                                features, deterministic=True
                            )
                    else:
                        actions[agent] = evader.get_action(obs)
                observations, _, _, _, _ = env.step(actions)
                alive = sum(1 for a in env.agents if a.startswith("prey"))
                if alive == 0:
                    break
            catches.append(prey_total - alive)
        finally:
            try:
                env.close()
            except SystemExit:
                pass

    x = np.asarray(feats, dtype=np.float32) if feats else np.empty((0, 0), np.float32)
    y = (
        np.asarray(labels, dtype=np.int64 if not continuous else np.float32)
        if labels
        else np.empty((0,), np.int64)
    )
    return x, y, float(np.mean(catches)) if catches else 0.0


def fit_clone(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    out_size: int,
    continuous: bool,
    seed: int = 0,
):
    """Fit the evolved family's exact architecture to the expert's actions."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(x.shape[1], hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_size),
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if continuous else nn.CrossEntropyLoss()

    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y)
    if continuous:
        # The Box head squashes with tanh before rescaling, so regress on the
        # pre-squash target to keep the clone's decode identical to the
        # expert's action.
        yt = torch.atanh(torch.clamp(yt, -0.999, 0.999))

    n = len(xt)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    with torch.no_grad():
        out = net(xt)
        if continuous:
            fit_quality = float(
                1.0 - (out - yt).pow(2).mean() / yt.var().clamp_min(1e-9)
            )
        else:
            fit_quality = float((out.argmax(1) == yt).float().mean())
    return net, fit_quality


def to_genome(net, activation: str, output_activation: str) -> FixedTopologyGenome:
    """Transplant a torch MLP into the flat genome the evolved family uses.

    ``FeedforwardNeuralPolicy.forward`` reads each layer as
    ``[W.ravel() (out x in), b (out)]``, which is exactly torch's
    ``Linear.weight`` row-major layout followed by ``Linear.bias``.
    """
    import torch
    from torch import nn

    layers = [m for m in net if isinstance(m, nn.Linear)]
    sizes = [layers[0].in_features] + [layer.out_features for layer in layers]
    genome = FixedTopologyGenome(
        layer_sizes=sizes,
        activation=activation,
        output_activation=output_activation,
    )
    flat: List[np.ndarray] = []
    with torch.no_grad():
        for layer in layers:
            flat.append(layer.weight.detach().numpy().ravel())
            flat.append(layer.bias.detach().numpy().ravel())
    genome.set_weights(np.concatenate(flat).astype(np.float64))
    return genome


def verify_transplant(net, policy: FeedforwardNeuralPolicy, x: np.ndarray) -> float:
    """Max |torch output - genome policy output| over a sample of states."""
    import torch

    sample = x[:256]
    with torch.no_grad():
        expected = net(torch.as_tensor(sample)).numpy()
    got = np.stack([policy.forward(row.astype(np.float64)) for row in sample])
    return float(np.max(np.abs(expected - got)))


def build_policy(genome: FixedTopologyGenome, env_cfg: Dict[str, Any]):
    """Wrap a genome exactly as the co-evolution evaluator would."""
    from gymnasium import spaces

    if env_cfg.get("continuous_actions", False):
        space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    else:
        space = spaces.Discrete(int(env_cfg["action_count"]))
    return FeedforwardNeuralPolicy(genome, action_space=space)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="discrete", choices=XC.ACTION_MODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    env_cfg = XC.make_env_config(args.mode)
    continuous = bool(env_cfg["continuous_actions"])
    walls = 4 if env_cfg.get("observable_walls", False) else 0
    in_size = aquarium_feature_size(
        env_cfg["predator_observe_count"],
        env_cfg["prey_observe_count"],
        wall_features=walls,
    )
    out_size = 2 if continuous else int(env_cfg["action_count"])

    print(f"\n=== Policy-class capacity probe ({args.mode}) ===")
    print(f"  network      : [{in_size}, <hidden>, {out_size}] "
          f"{XC.EVOLVED_ACTIVATION} hidden / {XC.EVOLVED_OUTPUT_ACTIVATION} out")
    print(f"  shipped width: {XC.EVOLVED_HIDDEN_SIZE}")

    t0 = time.time()
    print(f"\n[1/4] Expert rollouts ({CLONE_EPISODES} episodes)...", flush=True)
    x, y, apf_catches = rollout(env_cfg, CLONE_EPISODES, seed_base=1_000)
    print(f"      {len(x):,} states   APF pursuer catches "
          f"{apf_catches:.2f} of {env_cfg['prey_count']}")

    results: Dict[str, Any] = {
        "mode": args.mode,
        "input_size": in_size,
        "output_size": out_size,
        "apf_expert_catches": apf_catches,
        "clones": [],
    }

    for hidden in HIDDEN_SIZES:
        tag = f"hidden={hidden}"
        print(f"\n[2/4] Cloning {tag} ...", flush=True)
        net, quality = fit_clone(x, y, hidden, out_size, continuous)
        genome = to_genome(net, XC.EVOLVED_ACTIVATION, XC.EVOLVED_OUTPUT_ACTIVATION)
        policy = build_policy(genome, env_cfg)
        drift = verify_transplant(net, policy, x)
        weights = np.abs(genome.get_weights())
        metric = "accuracy" if not continuous else "R^2"
        print(f"      supervised {metric} {quality:.3f}   "
              f"transplant drift {drift:.2e}   "
              f"|w| median {np.median(weights):.2f} max {weights.max():.2f}")

        print(f"[3/4] Evaluating {tag} ({args.eval_episodes} episodes)...", flush=True)
        _, _, catches = rollout(
            env_cfg, args.eval_episodes, seed_base=77_000,
            predator_policy=policy, collect=False,
        )
        print(f"      catches {catches:.2f}")

        print(f"[4/4] DAgger round for {tag} ...", flush=True)
        x_d, y_d, _ = rollout(
            env_cfg, DAGGER_EPISODES, seed_base=5_000, predator_policy=policy
        )
        x2 = np.concatenate([x, x_d])
        y2 = np.concatenate([y, y_d])
        net2, quality2 = fit_clone(x2, y2, hidden, out_size, continuous)
        policy2 = build_policy(
            to_genome(net2, XC.EVOLVED_ACTIVATION, XC.EVOLVED_OUTPUT_ACTIVATION),
            env_cfg,
        )
        _, _, catches2 = rollout(
            env_cfg, args.eval_episodes, seed_base=77_000,
            predator_policy=policy2, collect=False,
        )
        print(f"      supervised {metric} {quality2:.3f}   catches {catches2:.2f}")

        results["clones"].append({
            "hidden": hidden,
            "supervised_quality": quality,
            "catches": catches,
            "supervised_quality_dagger": quality2,
            "catches_dagger": catches2,
            "weight_abs_median": float(np.median(weights)),
            "weight_abs_max": float(weights.max()),
            "transplant_drift": drift,
        })

    print(f"\n=== Verdict ({args.mode}) ===")
    print(f"{'policy':<28}{'catches of 5':>14}")
    print(f"{'scripted APF (expert)':<28}{apf_catches:>14.2f}")
    for entry in results["clones"]:
        print(f"{'clone h=' + str(entry['hidden']):<28}{entry['catches']:>14.2f}")
        print(f"{'  + DAgger':<28}{entry['catches_dagger']:>14.2f}")
    print(f"{'evolved best (2026-08 pilot)':<28}{0.74:>14.2f}")

    best = max(e["catches_dagger"] for e in results["clones"])
    print()
    if best >= 0.7 * apf_catches:
        print("  The class CAN express APF-quality pursuit -> the pilot's gap is")
        print("  a SEARCH problem (budget allocation, population, co-ev dynamics).")
    elif best >= 2 * 0.74:
        print("  The class reaches well past the evolved best but short of APF ->")
        print("  reachable headroom exists; search is the main constraint, and the")
        print("  encoding/width is a secondary one.")
    else:
        print("  The class CANNOT express APF-quality pursuit -> REPRESENTATION is")
        print("  the binding constraint. Fix the width/encoding before more search.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"capacity_{args.mode}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  wall {time.time() - t0:.0f}s   report {out}")


if __name__ == "__main__":
    main()
