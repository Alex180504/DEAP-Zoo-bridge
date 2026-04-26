"""Continuous-action CMA-ES demo on ``simple_spread_v3``.

Mirrors the structure of :mod:`examples.compare_ea_baseline` but for a
single CMA-ES run on the *continuous* variant of ``simple_spread_v3``.
Confirms that :class:`~deap_pettingzoo.baselines.CMAESTrainer` wires a
:class:`~deap_pettingzoo.policies.action_heads.BoxHead` into the
returned policy and that its actions stay inside the env's Box bounds.

Run with::

    .\\.venv\\Scripts\\python.exe examples\\continuous\\spread_cma.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deap_pettingzoo.baselines import CMAESTrainer
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.policies.action_heads import BoxHead


ENV_NAME = "simple_spread_v3"
ENV_CONFIG = {"max_cycles": 25, "N": 3}


def main() -> None:
    trainer = CMAESTrainer(hidden_size=16, population_size=8, sigma0=0.3)
    config = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=40,  # ~5 generations at pop=8
        algorithm_params={"seed": 0},
        env_config=ENV_CONFIG,
        continuous=True,
        output_dir=Path("results/continuous_spread_cma"),
    )

    print(f"Training CMA-ES on {ENV_NAME} (continuous actions)...")
    result = trainer.train(config)

    head = result.policy._policy.head
    assert isinstance(head, BoxHead), f"Expected BoxHead, got {type(head).__name__}"

    env = PettingZooWrapper(ENV_NAME, env_config={**ENV_CONFIG, "continuous_actions": True})
    try:
        obs = env.reset(seed=0)
        agent = env.possible_agents[0]
        action = result.policy.get_action(obs[agent])
        low, high = env.action_space(agent).low, env.action_space(agent).high
        print(f"  final fitness     : {result.metrics['final_fitness']:.3f}")
        print(f"  generations       : {result.metrics['num_generations']}")
        print(f"  population size   : {result.metrics['population_size']}")
        print(f"  action head       : {type(head).__name__}(shape={head.shape})")
        print(f"  sample action     : {action}")
        print(f"  action in bounds  : {bool(np.all((action >= low) & (action <= high)))}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
