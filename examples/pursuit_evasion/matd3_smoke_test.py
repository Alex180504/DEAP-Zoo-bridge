"""Minimal MATD3 learning smoke test via the deap-bridge RL side API.

Goal
----
Answer one question empirically: *can MATD3 learn at all through
deap-bridge's ``RLSideState`` rollout/learn loop?*  The aquarium
co-evolution experiment shows the MATD3 predator collapsing to linear
motion; this script isolates the RL plumbing from the aquarium env,
the co-evolution moving target, and the discrete-vs-continuous action
question.

Design
------
A single MATD3 learner (the ``adversary`` / predator side of PettingZoo
``simple_tag_v3``) is trained against a *fixed random* prey using the
same ``RLSideState`` + ``_RLDriverContext`` harness the co-evolution
trainer uses.  Because the opponent is stationary (random), any rise in
the learner's mean episode return is unambiguous evidence that the
critic + actor are learning to catch the prey.

Two action-space regimes are run back-to-back so we can see whether the
*discrete* (Gumbel-Softmax) path -- which is what the aquarium env uses
-- behaves differently from the *continuous* (MATD3-native) path that
the AgileRL tutorial and the repo's own RL-side tests exercise:

  * ``continuous_actions=True``  -> Box action space  (MATD3 native)
  * ``continuous_actions=False`` -> Discrete(5)        (Gumbel-Softmax)

For each regime we report the mean episode return for the first vs the
last few generations.  A working learner should show a clear upward
trend in the continuous case at minimum.

Run
---
    .venv\\Scripts\\python.exe examples\\pursuit_evasion\\matd3_smoke_test.py
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Fixed random opponent (stationary target for the learner)
# ---------------------------------------------------------------------------


class _RandomActionPolicy:
    """Samples uniformly from the given action space every step."""

    def __init__(self, action_space) -> None:
        self._space = action_space

    def reset(self) -> None:
        pass

    def get_action(self, obs, mask=None, deterministic=False):
        return self._space.sample()


# ---------------------------------------------------------------------------
# One-regime training run
# ---------------------------------------------------------------------------


def run_regime(
    *,
    continuous: bool,
    generations: int,
    rollout_episodes_per_gen: int,
    learning_steps_per_gen: int,
    max_steps: int,
    expl_noise: float,
    o_u_noise: bool,
    lr_actor: float,
    lr_critic: float,
    seed: int,
    device: str,
) -> List[float]:
    """Train one MATD3 learner vs a fixed random prey; return per-gen mean returns."""
    from deap_pettingzoo.baselines.coevolution import (
        FrozenPolicyPool,
        Side,
        make_side_state,
    )
    from deap_pettingzoo.baselines.coevolution.rl_side import _RLDriverContext
    from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper

    env = PettingZooWrapper(
        "simple_tag_v3",
        env_config={"continuous_actions": continuous},
    )

    rl_cfg: Dict[str, Any] = {
        "algorithm": "matd3",
        "rollout_episodes_per_gen": rollout_episodes_per_gen,
        "learning_steps_per_gen": learning_steps_per_gen,
        "batch_size": 64,
        "memory_size": 50_000,
        "learning_delay": 0,
        "device": device,
        "seed": seed,
        "agent_ctor_kwargs": {
            "expl_noise": expl_noise,
            "O_U_noise": o_u_noise,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
        },
    }
    # Learner controls the predators (adversary*), random prey opponent.
    side = Side("predators", ("adversary",), "rl", rl_cfg)
    state = make_side_state(side)
    rng = np.random.default_rng(seed)
    state.initialise(env, rng=rng)

    prey_ids = [a for a in env.possible_agents if a.startswith("agent")]
    opp_policy = _RandomActionPolicy(env.action_space(prey_ids[0]))
    opp_pool = FrozenPolicyPool([opp_policy])

    ctx = _RLDriverContext(
        env_wrapper=env,
        learner_role_prefixes=("adversary",),
        opponent_sides=[("prey", ("agent",), opp_pool)],
        max_steps=max_steps,
        rng=rng,
    )
    state._bootstrap_ctx = ctx
    state.evaluate_population()  # seed the replay buffer

    per_gen_mean: List[float] = []
    try:
        for _ in range(generations):
            stats = state.step_generation(lambda s: ctx)
            per_gen_mean.append(float(stats.mean))
    finally:
        env.close()

    # Report learn-call health for diagnosis.
    print(
        f"    learn_calls={state._total_learn_calls} "
        f"learn_failures={state._total_learn_failures} "
        f"total_transitions={state._total_transitions}"
    )
    return per_gen_mean


def _summarise(label: str, returns: List[float]) -> None:
    if not returns:
        print(f"[{label}] no data")
        return
    n = max(1, len(returns) // 5)
    first = float(np.mean(returns[:n]))
    last = float(np.mean(returns[-n:]))
    delta = last - first
    trend = "UP" if delta > 0 else ("flat" if abs(delta) < 1e-6 else "DOWN")
    print(
        f"[{label}] gens={len(returns)}  "
        f"first{n}={first:.3f}  last{n}={last:.3f}  "
        f"delta={delta:+.3f}  -> {trend}"
    )
    # Compact trajectory print (every gen if short, else ~10 samples).
    step = max(1, len(returns) // 10)
    traj = ", ".join(f"{returns[i]:.2f}" for i in range(0, len(returns), step))
    print(f"    trajectory: {traj}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--rollout-episodes", type=int, default=4)
    parser.add_argument("--learning-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--expl-noise", type=float, default=0.4)
    parser.add_argument("--o-u-noise", action="store_true", default=False)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--regimes",
        type=str,
        default="continuous,discrete",
        help="comma-separated subset of {continuous,discrete}",
    )
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    print("=" * 70)
    print("MATD3 deap-bridge learning smoke test (vs fixed random prey)")
    print(
        f"gens={args.generations} rollout_eps={args.rollout_episodes} "
        f"learn_steps={args.learning_steps} max_steps={args.max_steps}"
    )
    print(
        f"expl_noise={args.expl_noise} O_U_noise={args.o_u_noise} "
        f"lr_actor={args.lr_actor} lr_critic={args.lr_critic} device={args.device}"
    )
    print("=" * 70)

    for regime in regimes:
        continuous = regime == "continuous"
        print(f"\n--- regime: {regime} "
              f"({'Box' if continuous else 'Discrete(5) / Gumbel-Softmax'}) ---")
        returns = run_regime(
            continuous=continuous,
            generations=args.generations,
            rollout_episodes_per_gen=args.rollout_episodes,
            learning_steps_per_gen=args.learning_steps,
            max_steps=args.max_steps,
            expl_noise=args.expl_noise,
            o_u_noise=args.o_u_noise,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            seed=args.seed,
            device=args.device,
        )
        _summarise(regime, returns)


if __name__ == "__main__":
    main()
