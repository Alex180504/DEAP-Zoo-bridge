"""CMA-ES pursuers vs APF evaders — CMA-ES counterpart to ea_vs_apf.

Mirrors :mod:`examples.pursuit_evasion_ea_vs_apf` but drives the pursuer
search with :class:`deap.cma.Strategy` via the bridge's
:mod:`~deap_pettingzoo.integration.cma_integration` helpers, keeping all
other components (MPE2 ``simple_tag_v3``, APF evaders,
:class:`MixedPolicyEvaluator`) identical. This lets the two scripts be
used as apples-to-apples evolutionary-search baselines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from deap import creator, tools

from deap_pettingzoo.contrib.pursuit_evasion import PursuitEvasionConfig, PursuitEvasionEnv
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_strategy,
    setup_deap_creators,
)
from deap_pettingzoo.policies import (
    APFConfig,
    APFEvaderPolicy,
    FeedforwardNeuralPolicy,
)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

POPULATION_SIZE = 24        # CMA lambda_
NUM_GENERATIONS = 15
MAX_STEPS_PER_EPISODE = 150
HIDDEN_SIZE = 48
SIGMA0 = 0.3
SEED = 0

ENV_CONFIG = PursuitEvasionConfig(
    num_adversaries=4,
    num_good=4,
    num_obstacles=3,
    max_cycles=MAX_STEPS_PER_EPISODE,
    equal_max_speed=1.0,
    equal_accel=3.0,
    num_agent_neighbors=4,
    num_landmark_neighbors=3,
    boundary_threshold=0.9,
    boundary_scale=1.0,
    boundary_sharpness=2.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


def build_env() -> PursuitEvasionEnv:
    return PursuitEvasionEnv(config=ENV_CONFIG)


def build_scripted_evaders(env: PursuitEvasionEnv) -> Dict[str, APFEvaderPolicy]:
    apf_cfg = APFConfig(
        k_pursuer=1.0,
        k_landmark=0.3,
        k_boundary=2.0,
        boundary_threshold=ENV_CONFIG.boundary_threshold,
    )
    return {
        name: APFEvaderPolicy(layout=env.get_layout(name), config=apf_cfg)
        for name in env.good_agents
    }


def make_learner_factory(env: PursuitEvasionEnv, hidden_size: int):
    def _factory(genome: FixedTopologyGenome, agent_id: str):
        return FeedforwardNeuralPolicy(genome)

    return _factory


# ---------------------------------------------------------------------------
# CMA-ES evolution loop
# ---------------------------------------------------------------------------


def run_evolution() -> Dict[str, Any]:
    env = build_env()
    scripted_evaders = build_scripted_evaders(env)

    pursuer_id = env.adversary_agents[0]
    sample_obs = env.reset(seed=SEED)
    input_size = sample_obs[pursuer_id].shape[0]
    act_space = env.action_space(pursuer_id)
    output_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"  Pursuer NN:  in={input_size}  hidden={HIDDEN_SIZE}  out={output_size}")

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=make_learner_factory(env, HIDDEN_SIZE),
        scripted_policies=scripted_evaders,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # CMA-ES toolbox built from the bridge's integration helpers.
    setup_deap_creators()
    np.random.seed(SEED)

    genome_config = {
        "layer_sizes": [input_size, HIDDEN_SIZE, output_size],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
    }
    strategy, dimension = setup_cma_strategy(
        genome_config=genome_config,
        sigma0=SIGMA0,
        population_size=POPULATION_SIZE,
    )
    ind_init = make_cma_ind_init(genome_config)
    toolbox = register_cma_toolbox(strategy, ind_init, evaluator)

    stats = make_cma_stats()
    hof = tools.HallOfFame(1)
    fitness_history = []

    print(f"  Search dim : {dimension}  |  sigma0={SIGMA0}  |  lambda={strategy.lambda_}")

    # Mirror examples/pursuit_evasion_ea_vs_apf.py, which evaluates gen 0
    # then performs ``NUM_GENERATIONS`` update cycles. CMA's first call to
    # ``toolbox.generate()`` IS gen 0, so we run exactly ``NUM_GENERATIONS``
    # iterations to keep the two scripts apples-to-apples.
    for gen in range(NUM_GENERATIONS):
        population = toolbox.generate()
        for ind, fit in zip(population, map(toolbox.evaluate, population)):
            ind.fitness.values = fit

        toolbox.update(population)
        hof.update(population)

        rec = stats.compile(population)
        fitness_history.append({"gen": gen, **rec})
        print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

    env.close()
    return {
        "best_genome": hof[0].genome,
        "best_fitness": hof[0].fitness.values[0],
        "history": fitness_history,
        "dimension": dimension,
    }


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def evaluate_best(best_genome: FixedTopologyGenome, n_episodes: int = 20) -> Dict[str, float]:
    env = build_env()
    scripted_evaders = build_scripted_evaders(env)

    setup_deap_creators()
    individual = creator.Individual()
    individual.genome = best_genome

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=make_learner_factory(env, HIDDEN_SIZE),
        scripted_policies=scripted_evaders,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    learner_rewards = []
    scripted_rewards = []
    for i in range(n_episodes):
        res = evaluator.run_episode(individual, seed=1000 + i)
        learner_rewards.append(res.learner_reward)
        scripted_rewards.append(res.scripted_reward)

    env.close()
    return {
        "pursuer_mean": float(np.mean(learner_rewards)),
        "pursuer_std":  float(np.std(learner_rewards)),
        "pursuer_min":  float(np.min(learner_rewards)),
        "pursuer_max":  float(np.max(learner_rewards)),
        "evader_mean":  float(np.mean(scripted_rewards)),
        "evader_std":   float(np.std(scripted_rewards)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _section("Pursuit-Evasion Prototype: CMA-ES Pursuers vs APF Evaders")

    print(f"  Scenario   : {ENV_CONFIG.num_adversaries} pursuers vs "
          f"{ENV_CONFIG.num_good} evaders, {ENV_CONFIG.num_obstacles} obstacles")
    print(f"  Visibility : {ENV_CONFIG.num_agent_neighbors} agents / "
          f"{ENV_CONFIG.num_landmark_neighbors} landmarks")
    print(f"  Equal-ability: max_speed={ENV_CONFIG.equal_max_speed}  "
          f"accel={ENV_CONFIG.equal_accel}")
    print(f"  Population : {POPULATION_SIZE} | Generations: {NUM_GENERATIONS} | "
          f"Max steps/ep: {MAX_STEPS_PER_EPISODE}")

    _section("Evolution")
    t0 = time.time()
    result = run_evolution()
    evolution_time = time.time() - t0
    print(f"\n  Evolution complete in {evolution_time:.1f}s")
    print(f"  Best training fitness: {result['best_fitness']:+.2f}")

    _section("Final evaluation (20 fresh seeds)")
    eval_stats = evaluate_best(result["best_genome"], n_episodes=20)
    print(f"  Pursuer team reward : {eval_stats['pursuer_mean']:+.2f} "
          f"± {eval_stats['pursuer_std']:.2f}   "
          f"range=[{eval_stats['pursuer_min']:+.2f}, {eval_stats['pursuer_max']:+.2f}]")
    print(f"  APF evader reward   : {eval_stats['evader_mean']:+.2f} "
          f"± {eval_stats['evader_std']:.2f}")

    out_dir = Path("results/pursuit_evasion_cmaes_vs_apf")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "fitness_history.npz",
        history=np.array([(h["gen"], h["avg"], h["std"], h["min"], h["max"])
                          for h in result["history"]],
                         dtype=[("gen", "i4"), ("avg", "f4"), ("std", "f4"),
                                ("min", "f4"), ("max", "f4")]),
    )
    print(f"\n  Saved fitness history to {out_dir / 'fitness_history.npz'}")
    _section("Done")


if __name__ == "__main__":
    main()
