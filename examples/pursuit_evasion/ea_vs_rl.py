"""Pursuit-evasion EA vs RL — honours-project experiment.

Mirrors :mod:`examples.pursuit_evasion_ea_vs_apf` but swaps the scripted
APF evaders for **MADDPG-trained evaders** (via AgileRL). The pursuit
side is still trained by DEAP-driven neuroevolution; only the target
opponent changes.

Methodology
-----------

* Environment: MPE2 ``simple_tag_v3`` via
  :class:`~deap_pettingzoo.environments.PettingZooWrapper` (3 pursuers +
  1 evader, 2 landmarks). Vanilla (no partial-observability wrapping)
  so that MADDPG's trained observation shapes match those the EA
  pursuers see at evaluation time.
* Stage 1: train MADDPG on ``simple_tag_v3`` for ``MADDPG_STEPS`` env
  steps using the project's :class:`MADDPGTrainer`. This trains actors
  for *all* agents simultaneously; afterwards we re-wrap the trained
  underlying agent with :class:`AgileRLPolicyWrapper` restricted to the
  evader IDs so it only dispatches actions for ``agent_*``.
* Stage 2: evolve EA pursuers (homogeneous ``adversary_*`` agents
  sharing one NN genome) with DEAP. Evaluation uses
  :class:`MixedPolicyEvaluator` with:

  - ``learner_agent_prefixes=['adversary']`` — EA trains only the
    pursuer side.
  - ``scripted_policies={agent_id: AgileRLPolicyWrapper(...)}`` — a
    *frozen* MADDPG opponent driving the evader.

  This matches the research protocol of training one side against a
  stationary benchmark opponent so that learning signal isolates
  algorithmic efficiency rather than co-adaptation dynamics.
* Stage 3: final evaluation across fresh seeds.

Requires the optional ``rl-baselines`` extra::

    pip install 'deap-pettingzoo[rl-baselines]'
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from deap import tools

from deap_pettingzoo.baselines import (
    BudgetMode,
    MADDPGTrainer,
    TrainingConfig,
)
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    clone_individual,
    create_toolbox,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

ENV_NAME = "simple_tag_v3"
ENV_CONFIG = {"max_cycles": 50}
SEED = 0

MADDPG_STEPS = 5_000
MADDPG_BATCH = 64
MADDPG_MEMORY = 5_000
MADDPG_WARMUP = 500

POPULATION_SIZE = 16
NUM_GENERATIONS = 8
HIDDEN_SIZE = 32

EVAL_EPISODES = 20

OUTPUT_DIR = Path("results/pursuit_evasion_ea_vs_rl")


def _section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Stage 1 — Train MADDPG evader opponent
# ---------------------------------------------------------------------------


def train_maddpg_opponent() -> AgileRLPolicyWrapper:
    """Train MADDPG on ``simple_tag_v3`` and return a wrapper restricted
    to the evader agent IDs only.
    """
    trainer = MADDPGTrainer(device="cpu", verbose=False)
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=MADDPG_STEPS,
        algorithm_params={
            "seed": SEED,
            "batch_size": MADDPG_BATCH,
            "memory_size": MADDPG_MEMORY,
            "learning_delay": MADDPG_WARMUP,
            "max_steps_per_episode": ENV_CONFIG["max_cycles"],
        },
    )

    result = trainer.train(cfg)
    full_wrapper: AgileRLPolicyWrapper = result.policy  # type: ignore[assignment]
    evader_ids = [
        a for a in full_wrapper.agent_ids if not a.startswith("adversary")
    ]
    evader_only = AgileRLPolicyWrapper(
        full_wrapper.underlying_agent,
        evader_ids,
        algorithm_name="maddpg-evader",
    )

    print(
        f"  Trained MADDPG in {result.training_time:.1f}s "
        f"(episodes={len(result.metrics.get('episode_rewards', []))}, "
        f"env_steps={MADDPG_STEPS:,})"
    )
    print(f"  Evader IDs under MADDPG control: {evader_ids}")
    return evader_only


# ---------------------------------------------------------------------------
# Stage 2 — Evolve EA pursuers against frozen MADDPG evaders
# ---------------------------------------------------------------------------


def build_env() -> PettingZooWrapper:
    return PettingZooWrapper(env_name=ENV_NAME, env_config=ENV_CONFIG)


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    return FeedforwardNeuralPolicy(genome, action_space=action_space)


def run_evolution(maddpg_evaders: AgileRLPolicyWrapper) -> Dict[str, Any]:
    """Evolve EA pursuers against the frozen MADDPG evaders."""
    env = build_env()

    pursuer_id = next(a for a in env.possible_agents if a.startswith("adversary"))
    sample_obs = env.reset(seed=SEED)
    input_size = int(sample_obs[pursuer_id].shape[0])
    act_space = env.action_space(pursuer_id)
    output_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"  Pursuer NN:  in={input_size}  hidden={HIDDEN_SIZE}  out={output_size}")

    evader_ids = [a for a in env.possible_agents if not a.startswith("adversary")]
    # The same wrapper instance is reused for every evader key; the
    # wrapper rotates internally through its own agent_ids list in the
    # same order that MixedPolicyEvaluator iterates observations.
    scripted_policies = {aid: maddpg_evaders for aid in evader_ids}

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        scripted_policies=scripted_policies,
        max_steps=ENV_CONFIG["max_cycles"],
    )

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

    rng = np.random.default_rng(SEED)
    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    history = []
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit
    hof.update(population)
    rec = stats.compile(population)
    history.append({"gen": 0, **rec})
    print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

    for gen in range(1, NUM_GENERATIONS + 1):
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if rng.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        hof.update(population)
        rec = stats.compile(population)
        history.append({"gen": gen, **rec})
        print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

    return {
        "best_genome": hof[0].genome,
        "best_fitness": hof[0].fitness.values[0],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Final evaluation
# ---------------------------------------------------------------------------


def evaluate_best(
    best_genome: FixedTopologyGenome,
    maddpg_evaders: AgileRLPolicyWrapper,
    n_episodes: int = EVAL_EPISODES,
) -> Dict[str, float]:
    env = build_env()
    evader_ids = [a for a in env.possible_agents if not a.startswith("adversary")]
    scripted_policies = {aid: maddpg_evaders for aid in evader_ids}

    setup_deap_creators()
    from deap import creator

    individual = creator.Individual()
    individual.genome = best_genome

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        scripted_policies=scripted_policies,
        max_steps=ENV_CONFIG["max_cycles"],
    )

    pursuer_rewards, evader_rewards = [], []
    for i in range(n_episodes):
        res = evaluator.run_episode(individual, seed=1000 + i)
        pursuer_rewards.append(res.learner_reward)
        evader_rewards.append(res.scripted_reward)

    return {
        "pursuer_mean": float(np.mean(pursuer_rewards)),
        "pursuer_std": float(np.std(pursuer_rewards)),
        "pursuer_min": float(np.min(pursuer_rewards)),
        "pursuer_max": float(np.max(pursuer_rewards)),
        "evader_mean": float(np.mean(evader_rewards)),
        "evader_std": float(np.std(evader_rewards)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _section("Pursuit-Evasion Experiment: EA Pursuers vs MADDPG (AgileRL) Evaders")
    print(f"  Environment  : {ENV_NAME} (vanilla parallel_env)")
    print(f"  MADDPG budget: {MADDPG_STEPS:,} env steps")
    print(
        f"  EA setup     : pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}, "
        f"hidden={HIDDEN_SIZE}"
    )
    print(f"  Max steps/ep : {ENV_CONFIG['max_cycles']}")

    _section("Stage 1 — Train MADDPG evader opponent")
    t0 = time.time()
    maddpg_evaders = train_maddpg_opponent()
    stage1_time = time.time() - t0

    _section("Stage 2 — Evolve EA pursuers against frozen MADDPG evaders")
    t0 = time.time()
    evo = run_evolution(maddpg_evaders)
    stage2_time = time.time() - t0
    print(f"\n  Evolution complete in {stage2_time:.1f}s")
    print(f"  Best training fitness: {evo['best_fitness']:+.2f}")

    _section(f"Stage 3 — Final evaluation ({EVAL_EPISODES} fresh seeds)")
    t0 = time.time()
    stats = evaluate_best(evo["best_genome"], maddpg_evaders)
    stage3_time = time.time() - t0
    print(
        f"  EA pursuer team : {stats['pursuer_mean']:+.2f} "
        f"± {stats['pursuer_std']:.2f}   "
        f"range=[{stats['pursuer_min']:+.2f}, {stats['pursuer_max']:+.2f}]"
    )
    print(
        f"  MADDPG evader   : {stats['evader_mean']:+.2f} "
        f"± {stats['evader_std']:.2f}"
    )

    _section("Timings")
    print(f"  Stage 1 (MADDPG train): {stage1_time:6.1f}s")
    print(f"  Stage 2 (EA evolve)   : {stage2_time:6.1f}s")
    print(f"  Stage 3 (final eval)  : {stage3_time:6.1f}s")
    print(f"  Total                 : {stage1_time + stage2_time + stage3_time:6.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "summary.npz",
        history=np.array(
            [(h["gen"], h["avg"], h["max"]) for h in evo["history"]],
            dtype=[("gen", "i4"), ("avg", "f4"), ("max", "f4")],
        ),
        final_pursuer_mean=np.float32(stats["pursuer_mean"]),
        final_pursuer_std=np.float32(stats["pursuer_std"]),
        final_evader_mean=np.float32(stats["evader_mean"]),
    )
    _section(f"Saved summary to {OUTPUT_DIR / 'summary.npz'}")


if __name__ == "__main__":
    main()
