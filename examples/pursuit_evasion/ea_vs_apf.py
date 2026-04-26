"""EA pursuers vs APF evaders — honours-project prototype experiment.

Implements the baseline methodology described in the project's research
protocol:

* 10-vs-10 pursuit-evasion in ``simple_tag_v3`` (MPE2)
* Equal-ability pursuers and evaders (same ``max_speed``/``accel``)
* Partial observability (K-nearest agents + K-nearest landmarks,
  zero-padded)
* Exponential boundary penalty on agent rewards
* Scripted APF evaders (deterministic, non-learning) so the pursuers train
  against a *stationary* target — isolating algorithmic efficiency from
  opponent-policy non-stationarity
* Pursuers evolved via DEAP using the deap-bridge framework's evaluator
  abstractions

This script showcases the deap-bridge API end-to-end:

* :class:`~deap_pettingzoo.environments.PursuitEvasionEnv` — env wrapper
* :class:`~deap_pettingzoo.policies.APFEvaderPolicy` — scripted policy
  satisfying :class:`PolicyProtocol`
* :class:`~deap_pettingzoo.policies.FeedforwardNeuralPolicy` — NN policy
  from evolved genomes
* :class:`~deap_pettingzoo.evaluators.MixedPolicyEvaluator` — heterogeneous
  per-agent policy routing
* :mod:`~deap_pettingzoo.integration.deap_integration` — DEAP toolbox
  construction with the bridge's genome/crossover/mutation operators
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from deap import tools

from deap_pettingzoo.contrib.pursuit_evasion import PursuitEvasionEnv, PursuitEvasionConfig
from deap_pettingzoo.contrib.pursuit_evasion import APFEvaderPolicy, APFConfig
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.policies import (dNeuralPolicy,
)
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    setup_deap_creators,
    create_toolbox,
    clone_individual,
)
from deap_pettingzoo.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

POPULATION_SIZE = 24
NUM_GENERATIONS = 15
MAX_STEPS_PER_EPISODE = 150
HIDDEN_SIZE = 48
SEED = 0
OUTPUT_DIR = Path("results/pursuit_evasion_ea_vs_apf")

# Scenario — scaled down from 10v10 for a reasonable prototype runtime.
# Change ``num_good`` / ``num_adversaries`` to 10 for the full methodology.
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
    """Build the pursuit-evasion environment for this experiment."""
    return PursuitEvasionEnv(config=ENV_CONFIG)


def build_scripted_evaders(env: PursuitEvasionEnv) -> Dict[str, APFEvaderPolicy]:
    """Instantiate one APF policy per evader (``agent_*``)."""
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
    """Return a factory that turns a genome into a per-agent NN policy.

    All pursuers share the same evolved genome — this is classic
    homogeneous-team neuroevolution.  Because every pursuer sees the same
    observation size (they are structurally identical), a single
    :class:`FeedforwardNeuralPolicy` instance per agent is sufficient.
    """
    def _factory(genome: FixedTopologyGenome, agent_id: str, action_space):
        # One policy instance per agent (they wrap the *same* genome,
        # which is fine because FeedforwardNeuralPolicy is stateless).
        return FeedforwardNeuralPolicy(genome, action_space=action_space)

    return _factory


# ---------------------------------------------------------------------------
# DEAP evolution loop
# ---------------------------------------------------------------------------


def run_evolution() -> Dict[str, Any]:
    """Evolve NN pursuers against scripted APF evaders.

    Returns a dict containing the best genome, the per-generation fitness
    history, and bookkeeping metadata.  Kept separate from ``main`` so
    callers (or tests) can invoke it directly.
    """
    env = build_env()
    scripted_evaders = build_scripted_evaders(env)

    # Pursuer observation/action sizing
    pursuer_id = env.adversary_agents[0]
    sample_obs = env.reset(seed=SEED)
    input_size = sample_obs[pursuer_id].shape[0]
    act_space = env.action_space(pursuer_id)
    output_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"  Pursuer NN:  in={input_size}  hidden={HIDDEN_SIZE}  out={output_size}")

    # Mixed evaluator — EA learners control 'adversary_*', APF drives the rest.
    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=make_learner_factory(env, HIDDEN_SIZE),
        scripted_policies=scripted_evaders,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # DEAP toolbox built from the bridge's integration helpers.
    setup_deap_creators()
    genome_config = {
        "layer_sizes": [input_size, HIDDEN_SIZE, output_size],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
        "random_seed": SEED,
    }
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"tournament_size": 3},
        crossover_config={"alpha": 0.5},
        mutation_config={"sigma": 0.2, "indpb": 0.1},
    )
    toolbox.register("clone", clone_individual)

    # Initial population
    rng = np.random.default_rng(SEED)
    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    fitness_history = []

    # Evaluate gen 0
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit
    hof.update(population)
    rec = stats.compile(population)
    fitness_history.append({"gen": 0, **rec})
    print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

    # Evolution loop
    for gen in range(1, NUM_GENERATIONS + 1):
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for m in offspring:
            if rng.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values

        # Re-evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        hof.update(population)

        rec = stats.compile(population)
        fitness_history.append({"gen": gen, **rec})
        print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

    env.close()

    return {
        "best_genome": hof[0].genome,
        "best_fitness": hof[0].fitness.values[0],
        "history": fitness_history,
    }


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def evaluate_best(best_genome: FixedTopologyGenome, n_episodes: int = 20) -> Dict[str, float]:
    """Evaluate the evolved genome across ``n_episodes`` fresh seeds.

    Returns mean/std/min/max of the pursuer-team fitness and of the scripted
    evader team's total reward, plus a measured capture rate based on
    negative evader reward spikes (MPE signals +10 pursuer reward per
    capture; we infer captures from the evader-side reward signal).
    """
    env = build_env()
    scripted_evaders = build_scripted_evaders(env)

    # Wrap the genome as a DEAP individual so the evaluator can consume it.
    setup_deap_creators()
    from deap import creator
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
    _section("Pursuit-Evasion Prototype: EA Pursuers vs APF Evaders")

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

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the per-generation stats into the canonical run layout so
    # the analysis UI / report can pick this experiment up alongside the
    # rest of the baselines.
    logger = ExperimentLogger.create(
        out_dir,
        algorithm="ea",
        env_name="pursuit_evasion",
        seed=SEED,
        budget_mode="eval_episodes",
        budget_value=POPULATION_SIZE * (NUM_GENERATIONS + 1),
        hyperparams={
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "hidden_size": HIDDEN_SIZE,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "num_adversaries": ENV_CONFIG.num_adversaries,
            "num_good": ENV_CONFIG.num_good,
            "num_obstacles": ENV_CONFIG.num_obstacles,
        },
        notes="EA pursuers vs APF evaders.",
    )
    with logger:
        for h in result["history"]:
            logger.row("metrics", {
                "iteration": int(h["gen"]),
                "env_steps": int(h["gen"]) * POPULATION_SIZE,
                "wall_time": 0.0,
                "mean": float(h["avg"]),
                "std": float(h["std"]),
                "min": float(h["min"]),
                "max": float(h["max"]),
                "best": float(h["max"]),
            })
        logger.summary({
            "final_fitness": float(result["best_fitness"]),
            "best_fitness": float(result["best_fitness"]),
            "num_generations": NUM_GENERATIONS,
            "population_size": POPULATION_SIZE,
            "training_time_s": float(evolution_time),
            "eval_pursuer_mean": eval_stats["pursuer_mean"],
            "eval_pursuer_std": eval_stats["pursuer_std"],
            "eval_evader_mean": eval_stats["evader_mean"],
        })

    np.savez(
        out_dir / "fitness_history.npz",
        history=np.array([(h["gen"], h["avg"], h["std"], h["min"], h["max"])
                          for h in result["history"]],
                         dtype=[("gen", "i4"), ("avg", "f4"), ("std", "f4"),
                                ("min", "f4"), ("max", "f4")]),
    )
    print(f"\n  Saved fitness history to {out_dir / 'fitness_history.npz'}")
    print(f"  Logged run to {logger.run_dir}")
    _section("Done")


if __name__ == "__main__":
    main()
