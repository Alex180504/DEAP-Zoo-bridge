"""Bidirectional EA-vs-MADDPG co-evolution on simple_tag_v3.

Demonstrates the new ``Side(optimizer="rl", ...)`` path:

* one EA-evolved side (``adversary_*`` pursuers) exchanging fitness
  signal with...
* one AgileRL MADDPG side (``agent_*`` evader) whose neural networks
  learn against the *current* EA pool every generation.

Each side advances one generation at a time under the synchronous
schedule, so the RL side improves *and* mutates the EA side's
fitness-landscape simultaneously.

Run::

    python examples/coevolution_ea_vs_maddpg.py

Requires the ``rl-baselines`` extra::

    pip install 'deap-pettingzoo[rl-baselines]'
"""

from __future__ import annotations

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)


def main() -> None:
    config = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side(
                name="pursuers",
                role_prefixes=("adversary",),
                optimizer="ea",
                optimizer_config={
                    "population_size": 8,
                    "hidden_size": 32,
                    "seed": 0,
                },
            ),
            Side(
                name="evaders",
                role_prefixes=("agent",),
                optimizer="rl",
                optimizer_config={
                    "algorithm": "maddpg",
                    "rollout_episodes_per_gen": 4,
                    "learning_steps_per_gen": 2,
                    "batch_size": 32,
                    "memory_size": 5_000,
                    "learning_delay": 100,
                    # device defaults to "cuda" if torch.cuda.is_available()
                    # else "cpu". Leave unset to follow that default; set
                    # explicitly to override.
                    "seed": 1,
                },
            ),
        ],
        # k_opponents must be 1 when an RL side is in play — the EA
        # side's evaluator samples the same RL policy every episode
        # anyway (RLPolicyPool is degenerate size-1).
        k_opponents=1,
        max_steps_per_episode=25,
        # num_workers is auto-clamped to 1 when an RL side is present
        # (with a one-shot RuntimeWarning); leave at default.
        seed=42,
    )

    tcfg = TrainingConfig(
        env_name="simple_tag_v3",
        env_config={"continuous_actions": True},  # MADDPG needs Box action spaces
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=200,
        algorithm_params={"coevolution": config},
    )

    print("Running EA-vs-MADDPG bidirectional co-evolution...")
    result = CoEvolutionTrainer().train(tcfg)

    print(f"\nGenerations completed: {result.metrics['num_generations']}")
    print(f"Training time: {result.training_time:.1f}s\n")

    for name, hist in result.metrics["per_side_history"].items():
        if not hist:
            continue
        first, last = hist[0], hist[-1]
        print(
            f"{name:>9s}: gen 0 best = {first['best_fitness']:+.3f}  "
            f"-> gen {last['generation']} best = {last['best_fitness']:+.3f}"
        )


if __name__ == "__main__":
    main()
