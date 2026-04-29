"""Pursuit-evasion in marl-aquarium: EA vs CMA-ES vs MAPPO.

Single-file experiment comparing three pursuer-team training methods
against a fixed scripted APF prey controller in the
:mod:`marl_aquarium.aquarium_v0` environment:

* **EA**: standard DEAP genetic algorithm with CRN parallel evaluation
  (mirrors ``examples/pursuit_evasion/ea_vs_apf.py``).
* **CMA-ES**: covariance matrix adaptation via DEAP's ``cma.Strategy``
  (mirrors ``examples/pursuit_evasion/cmaes_vs_apf.py``).
* **MAPPO**: AgileRL's centralised-critic IPPO/MAPPO baseline
  (:class:`MAPPOTrainer`).

All three trainers see the *same* scripted-prey environment via
:func:`register_scripted_aquarium`, share the same ENV_STEPS budget, and
run for ``NUM_SEEDS`` independent seeds.  After training, every
``(algorithm, seed)`` champion is evaluated for ``EVAL_EPISODES`` fresh
seeds; results are written to ``OUTPUT_DIR`` and aggregated with
:mod:`deap_pettingzoo.analysis`.

Configuration is exposed at the top of the file via plain module-level
constants.  Set ``RUN_EA``, ``RUN_CMAES``, ``RUN_MAPPO`` to toggle the
three sections independently; ``PROBE_ENV=True`` runs a one-shot env
sanity check (random-action episode + obs/reward/info dump) and exits.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Headless rendering — must be set before pygame ever calls init().
# The aquarium env lazily creates its pygame View on the first render()
# call; SDL_VIDEODRIVER=offscreen prevents a visible window from opening
# while still allowing rgb_array frame capture for mp4 recording.
# Use setdefault so an explicit user-level env var is not overridden.
# ---------------------------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from deap import creator, tools

from deap_pettingzoo.environments.registry import build_env
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    clone_individual,
    create_toolbox,
    make_mixed_parallel_evaluate_map,
    setup_deap_creators,
)
from deap_pettingzoo.integration.deap_integration import (  # noqa: E402
    _MixedWorkerSpec,
    _individual_to_payload,
    _mixed_worker_evaluate,
)
from deap_pettingzoo.integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_strategy,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.utils.logging import ExperimentLogger
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.utils.parallel import make_executor

# Allow running as a plain script *or* as a module (-m).
# When invoked as a script the project root is not on sys.path, so the
# `examples.*` package is unreachable.  Adding the repo root here fixes that
# without breaking the `-m` invocation (inserting a duplicate is harmless).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.pursuit_evasion.aquarium.apf_policy import AquariumAPFConfig
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    DictResetEnv,
    register_scripted_aquarium,
)


# ===========================================================================
# Top-level config (edit these — no CLI parsing)
# ===========================================================================

# ---- Toggles --------------------------------------------------------------

RUN_EA: bool = False
RUN_CMAES: bool = False
RUN_MAPPO: bool = True
PROBE_ENV: bool = False        # short env smoke test then exit
RUN_FINAL_ANALYSIS: bool = True

# ---- Reproducibility / hardware ------------------------------------------

SEED_BASE: int = 0
NUM_SEEDS: int = 3             # independent training runs per algorithm
DEVICE: str = "cuda"           # MAPPO; falls back to CPU if cuda unavailable
N_WORKERS: int = 0             # 0 = os.cpu_count(); 1 = sequential

# ---- Shared budget --------------------------------------------------------

# Equal env-steps budget across the three algorithms (Phase 2 of the plan).
TOTAL_ENV_STEPS: int = 1_000_000
EVAL_EPISODES: int = 30

# ---- Aquarium env config (5v5 equal-ability) -----------------------------
#
# Aquarium quirks (see /memories/repo/aquarium_env_quirks.md):
#   - predator_view_distance=200 default fails the assertion in
#     utils.scale() because random spawn distances reach ~535 on the
#     800x800 torus.  Use 2000 for safety.
#   - fov_enabled=True triggers a buggy padding length in
#     prey_nearby_sharks_observations (line 761) when prey see fewer
#     sharks than predator_observe_count.  Disable it.
#   - render_mode=None is rejected; the wrapper maps it to "rgb_array".
#
# Equal-ability: predator and prey share max_velocity, max_acceleration,
# max_steer_force, view_distance, fov, observe_count.

ENV_NAME: str = "aquarium_pursuit_evasion_v0"

ENV_CONFIG: Dict[str, Any] = {
    "predator_count": 5,
    "prey_count": 15,
    "max_time_steps": 200,         # episode length cap
    # Equal-ability physics
    "predator_max_velocity": 7.0,
    "prey_max_velocity": 5.0,
    "predator_max_acceleration": 0.7,
    "prey_max_acceleration": 0.5,
    "predator_max_steer_force": 0.5,
    "prey_max_steer_force": 0.3,
    "predator_view_distance": 2000,
    "prey_view_distance": 2000,
    "predator_fov": 360,
    "prey_fov": 360,
    "predator_observe_count": 5,
    "prey_observe_count": 5,
    "fov_enabled": False,          # MUST be False — see quirks
    "action_count": 16,
    "predator_reward": 10,
    "catch_radius": 100,
    "keep_prey_count_constant": False,   # prey don't respawn; episode ends when all caught
    "width": 2000,
    "height": 2000,
}

APF_CONFIG = AquariumAPFConfig(k_predator=0.3)    # default hyperparameters

# ---- EA config -----------------------------------------------------------

EA_POPULATION_SIZE: int = 32
EA_HIDDEN_SIZE: int = 32
EA_EVAL_EPISODES_TRAIN: int = 1     # CRN, 1 ep per genome per gen
EA_ELITE_SIZE: int = 4
EA_MUT_SIGMA: float = 0.1
EA_MUT_INDPB: float = 0.2
EA_CXPB: float = 0.7

# ---- CMA-ES config -------------------------------------------------------

CMA_POPULATION_SIZE: int = 24
CMA_HIDDEN_SIZE: int = 32
CMA_EVAL_EPISODES_TRAIN: int = 1
CMA_SIGMA0: float = 0.3

# ---- MAPPO config (AgileRL) ----------------------------------------------

MAPPO_LEARN_STEP: int = 2048
MAPPO_BATCH_SIZE: int = 256
MAPPO_LR: float = 3e-4
MAPPO_GAMMA: float = 0.99
# Extra kwargs forwarded to AgileRL's ``IPPO.__init__`` (e.g.
# ``{"net_config": {"encoder_config": {"hidden_size": [128, 128]}}}``).
# Empty dict = use AgileRL's defaults.
MAPPO_EXTRA: Dict[str, Any] = {}

# ---- Recording / logging --------------------------------------------------

RECORD: bool = True
RECORD_FPS: int = 20
RECORD_FMT: str = "mp4"
RECORD_EVERY_N_EA: int = 5          # EA/CMA generations
RECORD_EVERY_N_MAPPO: int = 50      # MAPPO iterations

OUTPUT_DIR: Path = Path("results/aquarium_pursuit_evasion_ea_cma_mappo")


# ===========================================================================
# Helpers
# ===========================================================================


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _episode_steps_per_episode() -> int:
    return int(ENV_CONFIG["max_time_steps"])


def _ea_generations_for_budget(pop_size: int, eval_eps: int) -> int:
    """Convert ENV_STEPS budget → number of EA/CMA generations.

    ``gens = budget / (pop_size * eval_eps * max_time_steps)``.
    Rounds down with a min of 1 so a tiny smoke budget still runs one gen.
    """
    per_gen = pop_size * max(1, eval_eps) * _episode_steps_per_episode()
    return max(1, TOTAL_ENV_STEPS // per_gen)


# ---------------------------------------------------------------------------
# Env factory paths used by all three trainers
# ---------------------------------------------------------------------------

def _registered_env_factory():
    """Zero-arg factory returning a fresh scripted-prey aquarium env.

    Returns the modern PettingZoo parallel-API env (reset -> tuple).
    Used by AgileRL/MAPPO and by direct-stepping helpers (probe,
    recorder, MAPPO eval).
    """
    return build_env(ENV_NAME)


def _evaluator_env_factory():
    """Zero-arg factory returning the env with reset() shimmed to dict-only.

    The deap-bridge :class:`MixedPolicyEvaluator` and CRN pmap helper
    expect ``reset(seed=...) -> dict`` (legacy contract), not the
    modern ``(obs, infos)`` tuple.  :class:`DictResetEnv` strips infos
    on reset for those code paths.
    """
    return DictResetEnv(build_env(ENV_NAME))


def _empty_scripted_factory(env) -> Dict[str, Any]:
    """Return an empty scripted-policies dict.

    The :class:`ScriptedOpponentEnvWrapper` already injects scripted prey
    actions internally; from the public-API perspective there are *no*
    non-learner agents (``possible_agents`` shows only predators).  The
    evaluator's ``_validate_agent_coverage`` then sees an empty
    ``scripted_agents`` list and accepts the empty dict.
    """
    return {}


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    """All predators share the same evolved genome (homogeneous team)."""
    return FeedforwardNeuralPolicy(genome, action_space=action_space)


# ---------------------------------------------------------------------------
# Champion-rollout recording
# ---------------------------------------------------------------------------


def _record_champion_rollout(
    recorder: FrameRecorder,
    episode_idx: int,
    best_genome: FixedTopologyGenome,
    seed: int,
) -> None:
    """Render one champion-vs-APF episode and stream frames to *recorder*."""
    env = build_env(ENV_NAME, render_mode="rgb_array")
    try:
        learner_id = env.possible_agents[0]
        act_space = env.action_space(learner_id)
        policy = FeedforwardNeuralPolicy(best_genome, action_space=act_space)

        obs, _ = env.reset(seed=seed)
        recorder.start_episode(episode_idx)
        try:
            frame = env.render()
            if frame is not None:
                recorder.add_frame(frame)
        except Exception:
            pass

        for _ in range(_episode_steps_per_episode()):
            if not obs:
                break
            actions = {}
            for agent_id, ob in obs.items():
                try:
                    actions[agent_id] = policy.get_action(ob)
                except Exception:
                    actions[agent_id] = env.action_space(agent_id).sample()
            obs, _, terms, truncs, _ = env.step(actions)
            try:
                frame = env.render()
                if frame is not None:
                    recorder.add_frame(frame)
            except Exception:
                pass
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
    finally:
        recorder.end_episode()
        try:
            env.close()
        except SystemExit:
            pass


# ===========================================================================
# Env probe (PROBE_ENV=True)
# ===========================================================================


def _probe_env() -> None:
    _section("ENV PROBE — random-action smoke test")
    env = build_env(ENV_NAME)
    try:
        obs, infos = env.reset(seed=SEED_BASE)
        learner = env.possible_agents
        print(f"  possible_agents : {learner}")
        sample_id = learner[0]
        print(f"  obs space ({sample_id}) : {env.observation_space(sample_id)}")
        print(f"  act space ({sample_id}) : {env.action_space(sample_id)}")
        print(f"  obs[{sample_id}].shape   : {np.asarray(obs[sample_id]).shape}")
        total_r = 0.0
        for _ in range(_episode_steps_per_episode()):
            if not obs:
                break
            actions = {a: env.action_space(a).sample() for a in obs}
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_r += sum(rewards.values())
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
        prey_caught = infos.get(learner[0], {}).get("prey_caught_total", "?")
        print(f"  total team reward: {total_r:+.2f}")
        print(f"  prey_caught     : {prey_caught}")
        print("  OK")
    finally:
        try:
            env.close()
        except SystemExit:
            pass


# ===========================================================================
# EA section (mirrors ea_vs_apf.py with CRN parallel pmap)
# ===========================================================================


def _build_evaluator() -> MixedPolicyEvaluator:
    return MixedPolicyEvaluator(
        env_factory=_evaluator_env_factory,
        scripted_policy_factory=_empty_scripted_factory,
        learner_agent_prefixes=["predator"],
        learner_factory=_learner_factory,
        max_steps=_episode_steps_per_episode(),
    )


def _ea_or_cma_io_sizes(evaluator: MixedPolicyEvaluator, seed: int) -> Tuple[int, int]:
    learner_id = evaluator.learner_agents[0]
    inner = evaluator.env_factory()
    try:
        obs = inner.reset(seed=seed)
        in_size = int(np.asarray(obs[learner_id]).shape[0])
        act = inner.action_space(learner_id)
        out_size = int(act.n) if hasattr(act, "n") else int(act.shape[0])
    finally:
        try:
            inner.close()
        except SystemExit:
            pass
    return in_size, out_size


def run_ea(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"EA  | seed={seed}")
    evaluator = _build_evaluator()
    in_size, out_size = _ea_or_cma_io_sizes(evaluator, seed)
    print(f"  Pursuer NN: in={in_size} hidden={EA_HIDDEN_SIZE} out={out_size}")

    setup_deap_creators()
    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": [in_size, EA_HIDDEN_SIZE, out_size],
        "activation": "tanh",
        "weight_range": (-init_scale, init_scale),
        "random_seed": seed,
    }
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"tournament_size": 3},
        crossover_config={"alpha": 0.5},
        mutation_config={"sigma": EA_MUT_SIGMA, "indpb": EA_MUT_INDPB},
    )
    toolbox.register("clone", clone_individual)

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _ea_generations_for_budget(EA_POPULATION_SIZE, EA_EVAL_EPISODES_TRAIN)
    print(f"  budget→ gens={n_gens}  pop={EA_POPULATION_SIZE}  workers={n_workers}")

    rng = np.random.default_rng(seed)
    population = toolbox.population(n=EA_POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    fitness_history: List[Dict[str, Any]] = []
    gen_seed_holder = {"seed": int(seed)}

    logger = ExperimentLogger.create(
        run_dir,
        algorithm="ea",
        env_name=ENV_NAME,
        seed=seed,
        env_config=ENV_CONFIG,
        budget_mode="env_steps",
        budget_value=TOTAL_ENV_STEPS,
        hyperparams={
            "population_size": EA_POPULATION_SIZE,
            "num_generations": n_gens,
            "hidden_size": EA_HIDDEN_SIZE,
            "eval_episodes_train": EA_EVAL_EPISODES_TRAIN,
            "elite_size": EA_ELITE_SIZE,
            "mut_sigma": EA_MUT_SIGMA,
            "cxpb": EA_CXPB,
        },
    )
    recorder = (
        logger.get_frame_recorder(fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="ea_")
        if RECORD else None
    )

    t0 = time.time()
    with logger, make_executor(n_workers) as executor:
        # Touch upstream pmap once to validate qualnames, then replace.
        _ = make_mixed_parallel_evaluate_map(
            executor, evaluator, eval_episodes=EA_EVAL_EPISODES_TRAIN
        )
        del _

        def _crn_pmap(_evaluate_ignored, individuals):
            inds = list(individuals)
            cur_seed = gen_seed_holder["seed"]
            payloads = []
            for ind in inds:
                spec = _MixedWorkerSpec(
                    env_factory_qualname=(
                        f"{evaluator.env_factory.__module__}."
                        f"{evaluator.env_factory.__qualname__}"
                    ),
                    scripted_factory_qualname=(
                        f"{evaluator.scripted_policy_factory.__module__}."
                        f"{evaluator.scripted_policy_factory.__qualname__}"
                    ),
                    learner_factory_qualname=(
                        f"{evaluator.learner_factory.__module__}."
                        f"{evaluator.learner_factory.__qualname__}"
                    ),
                    learner_prefixes=tuple(evaluator.learner_agent_prefixes),
                    max_steps=evaluator.max_steps,
                    eval_episodes=EA_EVAL_EPISODES_TRAIN,
                    layer_sizes=tuple(ind.genome.layer_sizes),
                    activation=ind.genome.activation,
                )
                payloads.append((_individual_to_payload(ind, seed=cur_seed), spec))
            return list(executor.map(_mixed_worker_evaluate, payloads))

        toolbox.register("map", _crn_pmap)

        # Gen 0
        gen_seed_holder["seed"] = seed * 10_000 + 0
        for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
            ind.fitness.values = fit
        hof.update(population)
        rec = stats.compile(population)
        fitness_history.append({"gen": 0, **rec})
        logger.row("metrics", {
            "iteration": 0,
            "env_steps": EA_POPULATION_SIZE * EA_EVAL_EPISODES_TRAIN
                          * _episode_steps_per_episode(),
            "wall_time": time.time() - t0,
            "mean": float(rec["avg"]),
            "std":  float(rec["std"]),
            "min":  float(rec["min"]),
            "max":  float(rec["max"]),
            "best": float(rec["max"]),
        })
        print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

        if recorder is not None and recorder.enabled:
            try:
                _record_champion_rollout(recorder, 0, hof[0].genome, seed=seed)
            except Exception:
                traceback.print_exc()

        # Evolution loop
        for gen in range(1, n_gens + 1):
            gen_seed_holder["seed"] = seed * 10_000 + gen

            elites = [
                toolbox.clone(ind) for ind in tools.selBest(population, EA_ELITE_SIZE)
            ]
            for e in elites:
                if e.fitness.valid:
                    del e.fitness.values

            n_offspring = len(population) - EA_ELITE_SIZE
            offspring = list(map(
                toolbox.clone, toolbox.select(population, n_offspring)
            ))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < EA_CXPB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values
            for m in offspring:
                if rng.random() < 0.3:
                    toolbox.mutate(m)
                    del m.fitness.values

            invalid = [ind for ind in (elites + offspring) if not ind.fitness.valid]
            for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population[:] = elites + offspring
            hof.update(population)

            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            logger.row("metrics", {
                "iteration": gen,
                "env_steps": (gen + 1) * EA_POPULATION_SIZE * EA_EVAL_EPISODES_TRAIN
                              * _episode_steps_per_episode(),
                "wall_time": time.time() - t0,
                "mean": float(rec["avg"]),
                "std":  float(rec["std"]),
                "min":  float(rec["min"]),
                "max":  float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
            })
            print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

            if (recorder is not None and recorder.enabled
                    and gen % RECORD_EVERY_N_EA == 0):
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=seed)
                except Exception:
                    traceback.print_exc()

        train_time = time.time() - t0
        best = hof[0]
        eval_stats = _evaluate_genome(best.genome, n_episodes=EVAL_EPISODES, seed_base=10_000 * seed)
        logger.summary({
            "final_fitness": float(eval_stats["mean"]),
            "best_fitness": float(best.fitness.values[0]),
            "training_time_s": float(train_time),
            "num_generations": int(n_gens),
            "eval_mean": eval_stats["mean"],
            "eval_std": eval_stats["std"],
            "eval_min": eval_stats["min"],
            "eval_max": eval_stats["max"],
            "eval_episodes": EVAL_EPISODES,
        })
    return {
        "best_genome": best.genome,
        "best_fitness": float(best.fitness.values[0]),
        "history": fitness_history,
        "eval": eval_stats,
        "training_time_s": train_time,
    }


# ===========================================================================
# CMA-ES section
# ===========================================================================


def run_cmaes(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"CMA-ES | seed={seed}")
    evaluator = _build_evaluator()
    in_size, out_size = _ea_or_cma_io_sizes(evaluator, seed)
    print(f"  Pursuer NN: in={in_size} hidden={CMA_HIDDEN_SIZE} out={out_size}")

    setup_deap_creators()
    np.random.seed(seed)

    genome_config = {
        "layer_sizes": [in_size, CMA_HIDDEN_SIZE, out_size],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
    }
    strategy, dimension = setup_cma_strategy(
        genome_config=genome_config,
        sigma0=CMA_SIGMA0,
        population_size=CMA_POPULATION_SIZE,
    )
    ind_init = make_cma_ind_init(genome_config)
    toolbox = register_cma_toolbox(strategy, ind_init, evaluator)
    stats = make_cma_stats()
    hof = tools.HallOfFame(1)

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _ea_generations_for_budget(CMA_POPULATION_SIZE, CMA_EVAL_EPISODES_TRAIN)
    print(f"  Search dim: {dimension}  λ={strategy.lambda_}  σ0={CMA_SIGMA0}  gens={n_gens}  workers={n_workers}")

    fitness_history: List[Dict[str, Any]] = []
    logger = ExperimentLogger.create(
        run_dir,
        algorithm="cmaes",
        env_name=ENV_NAME,
        seed=seed,
        env_config=ENV_CONFIG,
        budget_mode="env_steps",
        budget_value=TOTAL_ENV_STEPS,
        hyperparams={
            "population_size": CMA_POPULATION_SIZE,
            "num_generations": n_gens,
            "hidden_size": CMA_HIDDEN_SIZE,
            "sigma0": CMA_SIGMA0,
            "dimension": dimension,
        },
    )
    recorder = (
        logger.get_frame_recorder(fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="cmaes_")
        if RECORD else None
    )

    cma_gen_seed_holder = {"seed": int(seed)}
    t0 = time.time()
    with logger, make_executor(n_workers) as executor:

        def _cmaes_crn_pmap(_evaluate_ignored, individuals):
            inds = list(individuals)
            cur_seed = cma_gen_seed_holder["seed"]
            payloads = []
            for ind in inds:
                spec = _MixedWorkerSpec(
                    env_factory_qualname=(
                        f"{evaluator.env_factory.__module__}."
                        f"{evaluator.env_factory.__qualname__}"
                    ),
                    scripted_factory_qualname=(
                        f"{evaluator.scripted_policy_factory.__module__}."
                        f"{evaluator.scripted_policy_factory.__qualname__}"
                    ),
                    learner_factory_qualname=(
                        f"{evaluator.learner_factory.__module__}."
                        f"{evaluator.learner_factory.__qualname__}"
                    ),
                    learner_prefixes=tuple(evaluator.learner_agent_prefixes),
                    max_steps=evaluator.max_steps,
                    eval_episodes=CMA_EVAL_EPISODES_TRAIN,
                    layer_sizes=tuple(ind.genome.layer_sizes),
                    activation=ind.genome.activation,
                )
                payloads.append((_individual_to_payload(ind, seed=cur_seed), spec))
            return list(executor.map(_mixed_worker_evaluate, payloads))

        toolbox.register("map", _cmaes_crn_pmap)

        for gen in range(n_gens):
            cma_gen_seed_holder["seed"] = seed * 10_000 + gen
            population = toolbox.generate()
            for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
                ind.fitness.values = fit
            toolbox.update(population)
            hof.update(population)
            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            logger.row("metrics", {
                "iteration": gen,
                "env_steps": (gen + 1) * CMA_POPULATION_SIZE * CMA_EVAL_EPISODES_TRAIN
                              * _episode_steps_per_episode(),
                "wall_time": time.time() - t0,
                "mean": float(rec["avg"]),
                "std":  float(rec.get("std", 0.0)),
                "min":  float(rec["min"]),
                "max":  float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
            })
            print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

            if (recorder is not None and recorder.enabled
                    and gen % RECORD_EVERY_N_EA == 0):
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=seed)
                except Exception:
                    traceback.print_exc()

        train_time = time.time() - t0
        best = hof[0]
        eval_stats = _evaluate_genome(best.genome, n_episodes=EVAL_EPISODES, seed_base=10_000 * seed)
        logger.summary({
            "final_fitness": float(eval_stats["mean"]),
            "best_fitness": float(best.fitness.values[0]),
            "training_time_s": float(train_time),
            "num_generations": int(n_gens),
            "dimension": int(dimension),
            "eval_mean": eval_stats["mean"],
            "eval_std": eval_stats["std"],
            "eval_min": eval_stats["min"],
            "eval_max": eval_stats["max"],
            "eval_episodes": EVAL_EPISODES,
        })
    return {
        "best_genome": best.genome,
        "best_fitness": float(best.fitness.values[0]),
        "history": fitness_history,
        "eval": eval_stats,
        "training_time_s": train_time,
    }


# ===========================================================================
# MAPPO section (AgileRL)
# ===========================================================================


def run_mappo(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"MAPPO | seed={seed}")
    # Lazy import — AgileRL pulls in torch.
    from deap_pettingzoo.baselines.agilerl_trainer import MAPPOTrainer
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        RecordingConfig,
        TrainingConfig,
    )

    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=TOTAL_ENV_STEPS,
        num_seeds=1,
        algorithm_params={
            "seed": int(seed),
            "learn_step": MAPPO_LEARN_STEP,
            "batch_size": MAPPO_BATCH_SIZE,
            "lr": MAPPO_LR,
            "gamma": MAPPO_GAMMA,
            **MAPPO_EXTRA,
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(
            enabled=RECORD,
            every_n_generations=RECORD_EVERY_N_MAPPO,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            episode_prefix="mappo_",
        ),
    )
    trainer = MAPPOTrainer(device=DEVICE)
    t0 = time.time()
    result = trainer.train(cfg)
    train_time = time.time() - t0
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": train_time,
        "seed": seed,
    }


# ===========================================================================
# Final evaluation
# ===========================================================================


def _evaluate_genome(
    genome: FixedTopologyGenome,
    n_episodes: int,
    seed_base: int,
) -> Dict[str, float]:
    """Evaluate an evolved genome against scripted prey for ``n_episodes``."""
    evaluator = _build_evaluator()
    setup_deap_creators()
    individual = creator.Individual()
    individual.genome = genome

    rewards: List[float] = []
    catches: List[int] = []
    try:
        for i in range(n_episodes):
            res = evaluator.run_episode(individual, seed=seed_base + i)
            rewards.append(float(res.learner_reward))
            # Aquarium gives +predator_reward per catch as predator-team
            # reward → catches = round(reward / predator_reward).
            catches.append(int(round(res.learner_reward / float(ENV_CONFIG["predator_reward"]))))
    finally:
        if evaluator.env_wrapper is not None:
            try:
                evaluator.env_wrapper.close()
            except SystemExit:
                pass

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(rewards_arr.mean()),
        "std":  float(rewards_arr.std()),
        "min":  float(rewards_arr.min()),
        "max":  float(rewards_arr.max()),
        "median": float(np.median(rewards_arr)),
        "catches_mean": float(np.mean(catches)),
        "catches_std":  float(np.std(catches)),
        "n_episodes": n_episodes,
    }


def _evaluate_mappo_policy(
    policy,
    n_episodes: int,
    seed_base: int,
) -> Dict[str, float]:
    """Run trained MAPPO policy against scripted prey for ``n_episodes``."""
    rewards: List[float] = []
    catches: List[int] = []
    for i in range(n_episodes):
        env = build_env(ENV_NAME)
        try:
            obs, _ = env.reset(seed=seed_base + i)
            ep_ret = 0.0
            for _ in range(_episode_steps_per_episode()):
                if not obs:
                    break
                actions = {}
                for agent_id, ob in obs.items():
                    try:
                        actions[agent_id] = policy.get_action(ob)
                    except Exception:
                        actions[agent_id] = env.action_space(agent_id).sample()
                obs, rew, terms, truncs, _ = env.step(actions)
                ep_ret += sum(float(r) for r in rew.values())
                if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                    break
            rewards.append(ep_ret)
            catches.append(int(round(ep_ret / float(ENV_CONFIG["predator_reward"]))))
        finally:
            try:
                env.close()
            except SystemExit:
                pass

    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "median": float(np.median(arr)),
        "catches_mean": float(np.mean(catches)),
        "catches_std":  float(np.std(catches)),
        "n_episodes": n_episodes,
    }


# ===========================================================================
# Aggregation / analysis
# ===========================================================================


def _final_analysis(root: Path) -> None:
    _section("Final analysis")
    try:
        from deap_pettingzoo.analysis import compare, loaders, report  # noqa: F401
    except Exception:
        print("  analysis module unavailable — skipping")
        traceback.print_exc()
        return

    try:
        experiment = loaders.load_experiment(root)
    except Exception:
        print("  load_experiment failed")
        traceback.print_exc()
        return
    print(f"  loaded {len(experiment.runs)} runs from {root}")

    try:
        table = compare.pairwise_table(experiment, metric="final_fitness")
        out_csv = root / "pairwise_comparison.csv"
        if hasattr(table, "to_csv"):
            table.to_csv(out_csv, index=False)
            print(f"  wrote {out_csv}")
        else:
            print(f"  pairwise_table: {table!r}")
    except Exception:
        print("  pairwise_table failed")
        traceback.print_exc()

    try:
        report_path = report.build_report(experiment, root, metric="final_fitness")
        print(f"  wrote {report_path}")
    except Exception:
        print("  build_report failed")
        traceback.print_exc()


# ===========================================================================
# Module-level env registration
# ===========================================================================
#
# Done at import so that *all* call sites — main loop, MAPPO worker,
# parallel EA workers — see the same registered name without having
# to re-register.  ``overwrite=True`` makes re-running the script in
# the same Python process safe.

register_scripted_aquarium(
    ENV_NAME,
    ENV_CONFIG,
    apf_config=APF_CONFIG,
    overwrite=True,
)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    _section("Pursuit-evasion in marl-aquarium: EA / CMA-ES / MAPPO")
    print(f"  ENV          : {ENV_NAME}")
    print(f"  predators    : {ENV_CONFIG['predator_count']}")
    print(f"  prey         : {ENV_CONFIG['prey_count']}  (scripted APF)")
    print(f"  episode cap  : {ENV_CONFIG['max_time_steps']} steps")
    print(f"  budget       : {TOTAL_ENV_STEPS:,} env_steps × {NUM_SEEDS} seeds × algos")
    print(f"  toggles      : EA={RUN_EA}  CMA-ES={RUN_CMAES}  MAPPO={RUN_MAPPO}")
    print(f"  output_dir   : {OUTPUT_DIR}")

    if PROBE_ENV:
        _probe_env()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    summary: Dict[str, List[Dict[str, Any]]] = {"ea": [], "cmaes": [], "mappo": []}

    for seed_idx in range(NUM_SEEDS):
        seed = SEED_BASE + seed_idx
        if RUN_EA:
            try:
                res = run_ea(seed, OUTPUT_DIR / "ea")
                summary["ea"].append({"seed": seed, **res["eval"],
                                      "best_fitness": res["best_fitness"]})
            except Exception:
                traceback.print_exc()
        if RUN_CMAES:
            try:
                res = run_cmaes(seed, OUTPUT_DIR / "cmaes")
                summary["cmaes"].append({"seed": seed, **res["eval"],
                                         "best_fitness": res["best_fitness"]})
            except Exception:
                traceback.print_exc()
        if RUN_MAPPO:
            try:
                res = run_mappo(seed, OUTPUT_DIR / "mappo")
                eval_stats = _evaluate_mappo_policy(
                    res["policy"], n_episodes=EVAL_EPISODES, seed_base=10_000 * seed
                )
                summary["mappo"].append({"seed": seed, **eval_stats})
            except Exception:
                traceback.print_exc()

    total_time = time.time() - overall_t0

    _section("Summary")
    for algo, rows in summary.items():
        if not rows:
            continue
        means = [r.get("mean", r.get("eval_mean", float("nan"))) for r in rows]
        catches = [r.get("catches_mean", float("nan")) for r in rows]
        print(f"  {algo:>6} : mean_reward={np.mean(means):+.2f} "
              f"± {np.std(means):.2f}  "
              f"catches/ep={np.nanmean(catches):.2f} ± {np.nanstd(catches):.2f}  "
              f"(n={len(rows)})")
    print(f"  total time   : {total_time:.1f}s")

    if RUN_FINAL_ANALYSIS:
        _final_analysis(OUTPUT_DIR)

    _section("Done")


if __name__ == "__main__":
    main()
