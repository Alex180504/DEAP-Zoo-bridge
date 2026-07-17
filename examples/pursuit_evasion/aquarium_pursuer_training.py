"""Pursuit-evasion in marl-aquarium: pursuer training vs scripted APF prey.

Single-file experiment comparing six campaign algorithms as
pursuer-team trainers against a fixed scripted APF prey controller in
``marl_aquarium.aquarium_v0``:

* **EA**            — DEAP genetic algorithm with CRN parallel evaluation.
* **CMA-ES**        — DEAP ``cma.Strategy`` via deap-bridge's CMA helpers.
* **PSO**           — particle swarm over the same genome family
  (:class:`PSOTrainer`).
* **MAP-Elites**    — QDax repertoire search (numpy scoring path) with the
  calibrated two-team behavioural descriptor.
* **HyperNEAT**     — CPPN evolution over a fixed substrate (``neat`` extra).
* **MATD3**         — AgileRL :class:`MATD3Trainer` (off-policy).

All trainers share one registered scripted-prey env, the same ENV_STEPS
budget, and run ``NUM_SEEDS`` independent seeds.  Champions are evaluated
on ``EVAL_EPISODES`` fresh seeds; results land in ``OUTPUT_DIR`` and are
aggregated with :mod:`deap_pettingzoo.analysis`.

Configuration is exposed as module-level constants below.  ``ACTION_MODE``
selects discrete headings vs continuous desired-velocity control
(marl-aquarium >= 0.1.12).  Env config and per-mode hyperparameters come
from :mod:`examples.pursuit_evasion.aquarium.experiment_config`.
``PROBE_ENV=True`` runs a one-shot env sanity check and exits.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Headless rendering — must be set before pygame ever calls init().
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from deap import tools

from deap_pettingzoo.environments.registry import build_env, get_registry_snapshot
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    _MixedWorkerSpec,
    _individual_to_payload,
    _mixed_worker_evaluate_with_steps,
    clone_individual,
    create_toolbox,
    make_mixed_parallel_evaluate_map,
    setup_deap_creators,
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
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.pursuit_evasion.aquarium import experiment_config as XC

XC.ensure_utf8_stdout()
from deap_pettingzoo.policies.action_heads import required_output_size
from examples.pursuit_evasion.aquarium.heads import (
    make_eval_head,
    make_train_head,
)
from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
    extract_prey_caught_total,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    DictResetEnv,
    register_scripted_aquarium,
)


# ===========================================================================
# Top-level config (edit these — no CLI parsing)
# ===========================================================================

# ---- Toggles ---------------------------------------------------------------

RUN_EA: bool = True
RUN_CMAES: bool = True
RUN_PSO: bool = True
RUN_MAP_ELITES: bool = True
RUN_HYPERNEAT: bool = True
RUN_MATD3: bool = True
PROBE_ENV: bool = False        # short env smoke test, then exit
RUN_FINAL_ANALYSIS: bool = True

# ---- Experiment variant ----------------------------------------------------
# Env-var overrides let the HPC launcher select the variant without editing
# the file; spawned evaluation workers inherit the environment, so their
# import-time env registration matches the driver's.

ACTION_MODE: str = os.environ.get("AQUARIUM_ACTION_MODE", "discrete")   # "discrete" | "continuous"

# ---- Reproducibility / hardware --------------------------------------------

SEED_BASE: int = 0
NUM_SEEDS: int = 3
DEVICE: str = "cuda"           # RL trainer; falls back to CPU if cuda unavailable
N_WORKERS: int = 0             # 0 = os.cpu_count(); 1 = sequential

# ---- Shared budget -----------------------------------------------------------

TOTAL_ENV_STEPS: int = XC.VS_APF_BUDGET_STEPS
EVAL_EPISODES: int = XC.EVAL_EPISODES

# ---- Held-out champion probe ---------------------------------------------------
# Uniform across all six algorithms: every PROBE_EVERY_N_STEPS consumed
# env steps the frozen champion plays PROBE_EPISODES episodes vs the
# scripted APF prey; rows land in each run's probe.csv (off-budget).

PROBE_EPISODES: int = XC.PROBE_EPISODES
PROBE_EVERY_N_STEPS: int = XC.PROBE_EVERY_N_STEPS
PROBE_SEED_OFFSET: int = 77_000        # probe episode seeds, distinct from eval's

# ---- Env / opponent ----------------------------------------------------------

ENV_NAME: str = "aquarium_pursuit_evasion_v0"
ENV_CONFIG: Dict[str, Any] = XC.make_env_config(ACTION_MODE)
APF_CONFIG = XC.APF_EVADER_CONFIG
REWARD_SHAPE_ALPHA: float = XC.VS_APF_PRED_SHAPE_ALPHA

# ---- EA / CMA-ES -------------------------------------------------------------

EA_POPULATION_SIZE: int = XC.EA_PARAMS["population_size"]
EA_HIDDEN_SIZE: int = XC.EA_PARAMS["hidden_size"]
EA_EVAL_EPISODES_TRAIN: int = XC.EVOLUTION_TRAIN_EVAL_EPISODES
EA_ELITE_SIZE: int = XC.EA_PARAMS["elite_size"]
EA_MUT_SIGMA: float = XC.EA_PARAMS["mut_sigma"]
EA_MUT_INDPB: float = XC.EA_PARAMS["mut_indpb"]
EA_CXPB: float = XC.EA_PARAMS["cxpb"]

CMA_POPULATION_SIZE: int = XC.CMA_PARAMS["population_size"]
CMA_HIDDEN_SIZE: int = XC.CMA_PARAMS["hidden_size"]
CMA_EVAL_EPISODES_TRAIN: int = XC.EVOLUTION_TRAIN_EVAL_EPISODES
CMA_SIGMA0: float = XC.params_for("cma", ACTION_MODE)["sigma0"]

SOFTMAX_TEMPERATURE: float = XC.SOFTMAX_TEMPERATURE

# ---- PSO -----------------------------------------------------------------------

PSO_PARAMS: Dict[str, Any] = XC.params_for("pso", ACTION_MODE)

# ---- HyperNEAT -------------------------------------------------------------------

HYPERNEAT_PARAMS: Dict[str, Any] = XC.params_for("hyperneat", ACTION_MODE)

# ---- MAP-Elites (QDax, numpy scoring path) -----------------------------------------

MAP_ELITES_PARAMS: Dict[str, Any] = dict(XC.QDAX_PARAMS)
MAP_ELITES_EVAL_EPISODES_TRAIN: int = XC.EVOLUTION_TRAIN_EVAL_EPISODES
MAP_ELITES_WORKER_IMPORTS: Tuple[str, ...] = (
    "examples.pursuit_evasion.aquarium_pursuer_training",
)

# ---- AgileRL (MATD3) ----------------------------------------------------------------

MATD3_PARAMS: Dict[str, Any] = {
    **XC.MATD3_PARAMS,
    **XC.rl_exploration_kwargs(ACTION_MODE),
}
RL_NUM_ENVS: int = (
    XC.RL_NUM_ENVS_WINDOWS if sys.platform == "win32" else XC.RL_NUM_ENVS
)

# ---- Recording / logging -------------------------------------------------------

RECORD: bool = True
RECORD_FPS: int = 20
RECORD_FMT: str = "mp4"
RECORD_EVERY_N_EA: int = 5          # EA/CMA generations
RECORD_COUNT_RL: int = 10           # trainer-managed videos spread over the run
RECORD_MAX_FRAMES = None            # None = record full episodes
RECORD_MAX_DIM: int = 960

OUTPUT_DIR: Path = Path("results/aquarium_pursuer_training")


# ===========================================================================
# Helpers
# ===========================================================================


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _episode_steps_per_episode() -> int:
    return int(ENV_CONFIG["max_time_steps"])


def _ea_generations_for_budget(pop_size: int, eval_eps: int) -> int:
    """Convert the ENV_STEPS budget into EA/CMA generations (floor, min 1)."""
    per_gen = pop_size * max(1, eval_eps) * _episode_steps_per_episode()
    return max(1, TOTAL_ENV_STEPS // per_gen)


def _evaluator_env_factory():
    """Env with reset() shimmed to the dict-only contract of MixedPolicyEvaluator."""
    return DictResetEnv(build_env(ENV_NAME))


def _empty_scripted_factory(env) -> Dict[str, Any]:
    """No public scripted agents — the wrapper drives prey internally."""
    return {}


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    """All predators share the evolved genome (homogeneous team).

    The head RNG is seeded from a stable digest of ``agent_id`` so identical
    genomes under the same env seed produce identical action streams (CRN).
    """
    head = make_train_head(
        action_space,
        temperature=SOFTMAX_TEMPERATURE,
        rng_seed=zlib.crc32(agent_id.encode("utf-8")),
    )
    return FeedforwardNeuralPolicy(genome, head=head)


def _make_champion_policy(genome: FixedTopologyGenome, action_space, seed: int):
    """Champion-playback policy matching the training head.

    Discrete recording with an argmax head at small weights produces a
    near-constant action and makes learned policies look idle; use the same
    stochastic head as training for fair playback.
    """
    if hasattr(action_space, "n"):
        head = make_train_head(
            action_space, temperature=SOFTMAX_TEMPERATURE, rng_seed=seed
        )
    else:
        head = make_eval_head(action_space)
    return FeedforwardNeuralPolicy(genome, head=head)


def _run_policy_episode(policy, seed: int) -> Tuple[float, int]:
    """Roll one predator-team episode; return ``(team_reward, catches)``."""
    env = build_env(ENV_NAME)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, infos = env.reset(seed=seed)
        catches = extract_prey_caught_total(infos)
        episode_reward = 0.0
        for _ in range(_episode_steps_per_episode()):
            if not obs:
                break
            actions = {}
            for agent_id, ob in obs.items():
                try:
                    actions[agent_id] = policy.get_action(ob)
                except Exception:
                    actions[agent_id] = env.action_space(agent_id).sample()
            obs, rewards, terms, truncs, infos = env.step(actions)
            episode_reward += sum(float(r) for r in rewards.values())
            catches = max(catches, extract_prey_caught_total(infos))
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
        return episode_reward, catches
    finally:
        try:
            env.close()
        except SystemExit:
            pass


def _champion_catch_probe(genome, action_space, seed: int, episodes: int = 3) -> float:
    """Mean catches over a few deterministic (eval-head) episodes."""
    counts = []
    for i in range(episodes):
        policy = FeedforwardNeuralPolicy(genome, head=make_eval_head(action_space))
        _, c = _run_policy_episode(policy, seed=seed + i)
        counts.append(c)
    return float(np.mean(counts))


def _probe_row(iteration: int, env_steps: int, stats: Dict[str, float]) -> Dict[str, Any]:
    """Probe row matching the library ProbeController's probe.csv schema."""
    return {
        "iteration": int(iteration),
        "env_steps": int(env_steps),
        "probe_reward_mean": float(stats["mean"]),
        "probe_reward_std": float(stats["std"]),
        "probe_episodes": int(stats["n_episodes"]),
        "probe_prey_caught_total_mean": float(stats["catches_mean"]),
        "probe_prey_caught_total_std": float(stats["catches_std"]),
    }


def _record_champion_rollout(
    recorder: FrameRecorder,
    episode_idx: int,
    best_genome: FixedTopologyGenome,
    seed: int,
) -> None:
    """Render one champion-vs-APF episode and stream frames to *recorder*."""
    env = build_env(ENV_NAME, render_mode="rgb_array")
    try:
        act_space = env.action_space(env.possible_agents[0])
        policy = _make_champion_policy(best_genome, act_space, seed=seed)

        obs, _ = env.reset(seed=seed)
        recorder.start_episode(episode_idx)
        cap = recorder.max_frames_per_episode
        for _ in range(_episode_steps_per_episode() + 1):
            if cap is None or len(recorder._frames) < cap:
                try:
                    frame = env.render()
                    if frame is not None:
                        recorder.add_frame(frame)
                except Exception:
                    pass
            if not obs:
                break
            actions = {}
            for agent_id, ob in obs.items():
                try:
                    actions[agent_id] = policy.get_action(ob)
                except Exception:
                    actions[agent_id] = env.action_space(agent_id).sample()
            obs, _, terms, truncs, _ = env.step(actions)
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
        sample_id = env.possible_agents[0]
        print(f"  possible_agents : {env.possible_agents}")
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
        print(f"  total team reward: {total_r:+.2f}")
        print(f"  prey_caught     : {extract_prey_caught_total(infos)}")
        print("  OK")
    finally:
        try:
            env.close()
        except SystemExit:
            pass


# ===========================================================================
# EA / CMA shared plumbing
# ===========================================================================


def _build_evaluator() -> MixedPolicyEvaluator:
    return MixedPolicyEvaluator(
        env_factory=_evaluator_env_factory,
        scripted_policy_factory=_empty_scripted_factory,
        learner_agent_prefixes=["predator"],
        learner_factory=_learner_factory,
        max_steps=_episode_steps_per_episode(),
    )


def _learner_io(evaluator: MixedPolicyEvaluator, seed: int):
    """Return ``(in_size, out_size, action_space)`` for the learner role."""
    learner_id = evaluator.learner_agents[0]
    inner = evaluator.env_factory()
    try:
        obs = inner.reset(seed=seed)
        in_size = int(np.asarray(obs[learner_id]).shape[0])
        act_space = inner.action_space(learner_id)
    finally:
        try:
            inner.close()
        except SystemExit:
            pass
    return in_size, required_output_size(act_space), act_space


def _make_crn_pmap(
    executor, evaluator, eval_episodes, gen_seed_holder, registry_snapshot,
    step_sink,
):
    """CRN parallel map: every individual is evaluated on the current
    generation's seed so fitness comparisons share common random numbers.

    Real env-step deltas from the workers accumulate into ``step_sink[0]``
    for consumed-step budgeting.
    """

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
                eval_episodes=eval_episodes,
                layer_sizes=tuple(ind.genome.layer_sizes),
                activation=ind.genome.activation,
                output_activation=getattr(
                    ind.genome, "output_activation", ind.genome.activation
                ),
                registry_snapshot=registry_snapshot,
            )
            payloads.append((_individual_to_payload(ind, seed=cur_seed), spec))
        results = list(executor.map(_mixed_worker_evaluate_with_steps, payloads))
        step_sink[0] += sum(int(steps) for _, steps in results)
        return [fitness for fitness, _ in results]

    return _crn_pmap


# ===========================================================================
# EA section
# ===========================================================================


def run_ea(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"EA  | seed={seed}")
    evaluator = _build_evaluator()
    in_size, out_size, act_space = _learner_io(evaluator, seed)
    layers = [in_size, EA_HIDDEN_SIZE, out_size]
    print(f"  Pursuer NN: in={in_size} hidden={EA_HIDDEN_SIZE} out={out_size} ({ACTION_MODE})")

    setup_deap_creators()
    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": layers,
        "activation": XC.EVOLVED_ACTIVATION,
        "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        "weight_range": (-init_scale, init_scale),
        "random_seed": seed,
    }
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"tournament_size": XC.EA_PARAMS["tournament_size"]},
        crossover_config={"alpha": XC.EA_PARAMS["crossover_alpha"]},
        mutation_config={"sigma": EA_MUT_SIGMA, "indpb": EA_MUT_INDPB},
    )
    toolbox.register("clone", clone_individual)

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _ea_generations_for_budget(EA_POPULATION_SIZE, EA_EVAL_EPISODES_TRAIN)
    consumed = [0]  # Real env-step counter shared with the CRN pmap.
    print(
        f"  budget-> nominal gens={n_gens}  pop={EA_POPULATION_SIZE}  "
        f"workers={n_workers}  consumed_budgeting={XC.BUDGET_ON_CONSUMED}"
    )

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
            "action_mode": ACTION_MODE,
            "budget_on_consumed": XC.BUDGET_ON_CONSUMED,
        },
    )
    recorder = (
        logger.get_frame_recorder(
            fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="ea_",
            max_frames_per_episode=RECORD_MAX_FRAMES, max_dim=RECORD_MAX_DIM,
        )
        if RECORD else None
    )

    t0 = time.time()
    registry_snapshot = get_registry_snapshot()
    with logger, make_executor(n_workers) as executor:
        # Touch upstream pmap once to validate qualnames, then use CRN map.
        make_mixed_parallel_evaluate_map(
            executor, evaluator, eval_episodes=EA_EVAL_EPISODES_TRAIN
        )
        toolbox.register(
            "map",
            _make_crn_pmap(
                executor, evaluator, EA_EVAL_EPISODES_TRAIN,
                gen_seed_holder, registry_snapshot, consumed,
            ),
        )

        probe_state = {"next": 0, "last": -1}

        def _probe_champion(gen: int, force: bool = False) -> None:
            """Held-out champion probe on the consumed-step cadence (off-budget)."""
            if not force and consumed[0] < probe_state["next"]:
                return
            if force and probe_state["last"] == consumed[0]:
                return
            while probe_state["next"] <= consumed[0]:
                probe_state["next"] += PROBE_EVERY_N_STEPS
            probe_stats = _evaluate_genome(
                hof[0].genome, act_space,
                n_episodes=PROBE_EPISODES, seed_base=PROBE_SEED_OFFSET + seed,
            )
            probe_state["last"] = consumed[0]
            logger.row("probe", _probe_row(gen, consumed[0], probe_stats))
            print(
                f"  [probe] gen {gen:>3} | reward={probe_stats['mean']:+.2f}  "
                f"catches/ep={probe_stats['catches_mean']:.2f}"
            )

        def _log_generation(gen: int) -> None:
            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            champ_catches = _champion_catch_probe(
                hof[0].genome, act_space, seed=seed * 10_000 + gen
            )
            logger.row("metrics", {
                "iteration": gen,
                "env_steps": int(consumed[0]),
                "wall_time": time.time() - t0,
                "mean": float(rec["avg"]),
                "std": float(rec["std"]),
                "min": float(rec["min"]),
                "max": float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
                "champion_catches": float(champ_catches),
            })
            print(
                f"  Gen {gen:>3} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}  "
                f"champ_catches={champ_catches:.1f}"
            )
            _probe_champion(gen)
            if recorder is not None and recorder.enabled and gen % RECORD_EVERY_N_EA == 0:
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=seed)
                except Exception:
                    traceback.print_exc()

        # Generation 0
        gen_seed_holder["seed"] = seed * 10_000
        for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
            ind.fitness.values = fit
        hof.update(population)
        _log_generation(0)

        realized_gens = 0
        for gen in XC.budget_generations(
            n_gens, TOTAL_ENV_STEPS, consumed, _episode_steps_per_episode()
        ):
            realized_gens = gen
            gen_seed_holder["seed"] = seed * 10_000 + gen

            # True CRN: elites are re-evaluated on the current generation's
            # seed so all individuals are compared on equal footing (the HOF
            # still preserves the best genome across generations).
            elites = [toolbox.clone(ind) for ind in tools.selBest(population, EA_ELITE_SIZE)]
            for elite in elites:
                del elite.fitness.values

            n_offspring = len(population) - EA_ELITE_SIZE
            offspring = list(map(toolbox.clone, toolbox.select(population, n_offspring)))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < EA_CXPB:
                    toolbox.mate(c1, c2)
            for m in offspring:
                if rng.random() < XC.EA_PARAMS["mutation_rate"]:
                    toolbox.mutate(m)
            for ind in offspring:
                if ind.fitness.valid:
                    del ind.fitness.values

            invalid = [ind for ind in (elites + offspring) if not ind.fitness.valid]
            for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population[:] = elites + offspring
            hof.update(population)
            _log_generation(gen)

        # Guarantee a probe row at the end-of-run consumed count.
        _probe_champion(realized_gens, force=True)

        train_time = time.time() - t0
        best = hof[0]
        eval_stats = _evaluate_genome(
            best.genome, act_space, n_episodes=EVAL_EPISODES, seed_base=10_000 * seed
        )
        logger.summary({
            "final_fitness": float(eval_stats["mean"]),
            "best_fitness": float(best.fitness.values[0]),
            "training_time_s": float(train_time),
            "num_generations": int(realized_gens),
            "env_steps_consumed": int(consumed[0]),
            **{f"eval_{k}": v for k, v in eval_stats.items()},
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
    in_size, out_size, act_space = _learner_io(evaluator, seed)
    layers = [in_size, CMA_HIDDEN_SIZE, out_size]
    print(f"  Pursuer NN: in={in_size} hidden={CMA_HIDDEN_SIZE} out={out_size} ({ACTION_MODE})")

    setup_deap_creators()
    np.random.seed(seed)  # DEAP CMA samples from the global np.random

    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": layers,
        "activation": XC.EVOLVED_ACTIVATION,
        "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        "weight_range": (-init_scale, init_scale),
    }
    strategy, dimension = setup_cma_strategy(
        genome_config=genome_config,
        sigma0=CMA_SIGMA0,
        population_size=CMA_POPULATION_SIZE,
        centroid=None,
    )
    ind_init = make_cma_ind_init(genome_config)
    toolbox = register_cma_toolbox(strategy, ind_init, evaluator)
    stats = make_cma_stats()
    hof = tools.HallOfFame(1)

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _ea_generations_for_budget(CMA_POPULATION_SIZE, CMA_EVAL_EPISODES_TRAIN)
    consumed = [0]  # Real env-step counter shared with the CRN pmap.
    print(
        f"  dim={dimension}  λ={strategy.lambda_}  σ0={CMA_SIGMA0}  "
        f"nominal gens={n_gens}  workers={n_workers}  "
        f"consumed_budgeting={XC.BUDGET_ON_CONSUMED}"
    )

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
            "action_mode": ACTION_MODE,
            "budget_on_consumed": XC.BUDGET_ON_CONSUMED,
        },
    )
    recorder = (
        logger.get_frame_recorder(
            fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="cmaes_",
            max_frames_per_episode=RECORD_MAX_FRAMES, max_dim=RECORD_MAX_DIM,
        )
        if RECORD else None
    )

    gen_seed_holder = {"seed": int(seed)}
    t0 = time.time()
    registry_snapshot = get_registry_snapshot()
    with logger, make_executor(n_workers) as executor:
        toolbox.register(
            "map",
            _make_crn_pmap(
                executor, evaluator, CMA_EVAL_EPISODES_TRAIN,
                gen_seed_holder, registry_snapshot, consumed,
            ),
        )

        probe_state = {"next": 0, "last": -1}

        def _probe_champion(gen: int, force: bool = False) -> None:
            """Held-out champion probe on the consumed-step cadence (off-budget)."""
            if not force and consumed[0] < probe_state["next"]:
                return
            if force and probe_state["last"] == consumed[0]:
                return
            while probe_state["next"] <= consumed[0]:
                probe_state["next"] += PROBE_EVERY_N_STEPS
            probe_stats = _evaluate_genome(
                hof[0].genome, act_space,
                n_episodes=PROBE_EPISODES, seed_base=PROBE_SEED_OFFSET + seed,
            )
            probe_state["last"] = consumed[0]
            logger.row("probe", _probe_row(gen, consumed[0], probe_stats))
            print(
                f"  [probe] gen {gen:>3} | reward={probe_stats['mean']:+.2f}  "
                f"catches/ep={probe_stats['catches_mean']:.2f}"
            )

        realized_gens = 0
        for gen in XC.budget_generations(
            n_gens, TOTAL_ENV_STEPS, consumed, _episode_steps_per_episode()
        ):
            realized_gens = gen
            gen_seed_holder["seed"] = seed * 10_000 + gen
            population = toolbox.generate()
            for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
                ind.fitness.values = fit
            toolbox.update(population)
            hof.update(population)
            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            champ_catches = _champion_catch_probe(
                hof[0].genome, act_space, seed=seed * 10_000 + gen, episodes=1
            )
            logger.row("metrics", {
                "iteration": gen,
                "env_steps": int(consumed[0]),
                "wall_time": time.time() - t0,
                "mean": float(rec["avg"]),
                "std": float(rec.get("std", 0.0)),
                "min": float(rec["min"]),
                "max": float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
                "champion_catches": float(champ_catches),
            })
            print(
                f"  Gen {gen:>3} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}  "
                f"champ_catches={champ_catches:.1f}"
            )
            _probe_champion(gen)
            if (recorder is not None and recorder.enabled
                    and gen % RECORD_EVERY_N_EA == 0):
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=seed)
                except Exception:
                    traceback.print_exc()

        # Guarantee a probe row at the end-of-run consumed count.
        _probe_champion(realized_gens, force=True)

        train_time = time.time() - t0
        best = hof[0]
        eval_stats = _evaluate_genome(
            best.genome, act_space, n_episodes=EVAL_EPISODES, seed_base=10_000 * seed
        )
        logger.summary({
            "final_fitness": float(eval_stats["mean"]),
            "best_fitness": float(best.fitness.values[0]),
            "training_time_s": float(train_time),
            "num_generations": int(realized_gens),
            "env_steps_consumed": int(consumed[0]),
            "dimension": int(dimension),
            **{f"eval_{k}": v for k, v in eval_stats.items()},
        })
    return {
        "best_genome": best.genome,
        "best_fitness": float(best.fitness.values[0]),
        "history": fitness_history,
        "eval": eval_stats,
        "training_time_s": train_time,
    }


# ===========================================================================
# PSO section
# ===========================================================================


def run_pso(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"PSO | seed={seed}")
    from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        ProbeConfig,
        RecordingConfig,
        TrainingConfig,
    )

    trainer = PSOTrainer(
        hidden_size=PSO_PARAMS["hidden_size"],
        population_size=PSO_PARAMS["population_size"],
        phi1=PSO_PARAMS["phi1"],
        phi2=PSO_PARAMS["phi2"],
        weight=PSO_PARAMS["weight"],
        smin=PSO_PARAMS["smin"],
        smax=PSO_PARAMS["smax"],
        activation=XC.EVOLVED_ACTIVATION,
        eval_episodes=PSO_PARAMS["eval_episodes"],
        num_workers=N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1),
        parallel_backend="process",
    )
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=TOTAL_ENV_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        seed=int(seed),
        max_steps_per_episode=_episode_steps_per_episode(),
        algorithm_params={
            "population_size": PSO_PARAMS["population_size"],
            "hidden_size": PSO_PARAMS["hidden_size"],
            "phi1": PSO_PARAMS["phi1"],
            "phi2": PSO_PARAMS["phi2"],
            "weight": PSO_PARAMS["weight"],
            "eval_episodes": PSO_PARAMS["eval_episodes"],
            "action_mode": ACTION_MODE,
            "budget_on_consumed": XC.BUDGET_ON_CONSUMED,
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        probe=ProbeConfig(
            enabled=True,
            every_n_steps=PROBE_EVERY_N_STEPS,
            n_episodes=PROBE_EPISODES,
            info_keys=("prey_caught_total",),
            seed=PROBE_SEED_OFFSET + seed,
        ),
        recording=RecordingConfig(
            enabled=RECORD,
            mode="count",
            max_recordings=RECORD_COUNT_RL,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            episode_prefix="pso_",
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
        ),
    )
    print(
        f"  pso pop={PSO_PARAMS['population_size']} "
        f"phi={PSO_PARAMS['phi1']}/{PSO_PARAMS['phi2']} w={PSO_PARAMS['weight']} "
        f"budget={TOTAL_ENV_STEPS:,}"
    )
    t0 = time.time()
    result = trainer.train(cfg)
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": time.time() - t0,
        "seed": seed,
    }


# ===========================================================================
# MAP-Elites section (QDax, numpy scoring path)
# ===========================================================================


def run_map_elites(seed: int, run_dir: Path) -> Dict[str, Any]:
    """MAP-Elites repertoire search for the predator team vs scripted prey.

    Uses the calibrated two-team descriptor (predator formation spread ×
    mean distance to prey) driven by :class:`AquariumStateExtractor`.
    """
    _section(f"MAP-Elites | seed={seed}")
    from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        ProbeConfig,
        RecordingConfig,
        TrainingConfig,
    )

    # QDax honours budget_on_consumed on the numpy/PettingZoo scoring path
    # (the one this suite uses); overshoot is at most one batch's real cost.
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=TOTAL_ENV_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        continuous=ACTION_MODE == "continuous",
        algorithm_params={
            "seed": seed,
            "hidden_sizes": (MAP_ELITES_PARAMS["hidden_size"],),
            "activation": XC.EVOLVED_ACTIVATION,
            "init_batch_size": MAP_ELITES_PARAMS["batch_size"],
            "init_mode": "gaussian",
            "init_sigma": MAP_ELITES_PARAMS["init_sigma"],
            "batch_size": MAP_ELITES_PARAMS["batch_size"],
            "eval_episodes": MAP_ELITES_EVAL_EPISODES_TRAIN,
            "grid_shape": MAP_ELITES_PARAMS["grid_shape"],
            "iso_sigma": MAP_ELITES_PARAMS["iso_sigma"],
            "line_sigma": MAP_ELITES_PARAMS["line_sigma"],
            "episode_length": _episode_steps_per_episode(),
            "descriptor_fn": XC.make_two_team_descriptor("predator", ENV_CONFIG),
            "state_extractor": XC.make_state_extractor("predator", ENV_CONFIG),
            "save_repertoire": True,
            "num_workers": N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1),
            "parallel_backend": "process",
            "worker_imports": MAP_ELITES_WORKER_IMPORTS,
            "layer_sizes": None,   # auto-infer from env spaces
            "reeval_top_k": MAP_ELITES_PARAMS["reeval_top_k"],
            "reeval_rounds": MAP_ELITES_PARAMS["reeval_rounds"],
            "action_mode": ACTION_MODE,
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        probe=ProbeConfig(
            enabled=True,
            every_n_steps=PROBE_EVERY_N_STEPS,
            n_episodes=PROBE_EPISODES,
            info_keys=("prey_caught_total",),
            seed=PROBE_SEED_OFFSET + seed,
        ),
        recording=RecordingConfig(
            enabled=RECORD,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
            episode_prefix="map_elites_",
        ),
    )
    print(
        f"  MAP-Elites: hidden={MAP_ELITES_PARAMS['hidden_size']}  "
        f"batch={MAP_ELITES_PARAMS['batch_size']}  "
        f"grid={MAP_ELITES_PARAMS['grid_shape']}  "
        f"iso={MAP_ELITES_PARAMS['iso_sigma']}  budget={TOTAL_ENV_STEPS:,}"
    )
    trainer = MAPElitesTrainer()
    t0 = time.time()
    result = trainer.train(cfg)
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": time.time() - t0,
        "seed": seed,
    }


# ===========================================================================
# HyperNEAT section
# ===========================================================================


def run_hyperneat(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"HyperNEAT | seed={seed}")
    try:
        from deap_pettingzoo.contrib.neat import HyperNEATTrainer
    except ImportError as exc:
        raise ImportError(
            "HyperNEAT requires the 'neat' extra. Install with: pip install deap-pettingzoo[neat]"
        ) from exc
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        ProbeConfig,
        RecordingConfig,
        TrainingConfig,
    )

    # NEAT-family trainers honour budget_on_consumed by stepping the
    # population n=1 generations until the real env-tick budget is exhausted.
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=TOTAL_ENV_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        num_workers=N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1),
        algorithm_params={
            "seed": int(seed),
            "pop_size": HYPERNEAT_PARAMS["pop_size"],
            "eval_episodes_per_genome": HYPERNEAT_PARAMS["eval_episodes_per_genome"],
            "substrate_hidden_layers": HYPERNEAT_PARAMS["substrate_hidden_layers"],
            "phenotype_activation": HYPERNEAT_PARAMS["phenotype_activation"],
            "stochastic_head": HYPERNEAT_PARAMS["stochastic_head"],
            "head_temperature": HYPERNEAT_PARAMS["head_temperature"],
            "action_mode": ACTION_MODE,
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        probe=ProbeConfig(
            enabled=True,
            every_n_steps=PROBE_EVERY_N_STEPS,
            n_episodes=PROBE_EPISODES,
            info_keys=("prey_caught_total",),
            seed=PROBE_SEED_OFFSET + seed,
        ),
        recording=RecordingConfig(
            enabled=RECORD,
            mode="count",
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            episode_prefix="hyperneat_",
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
        ),
    )
    print(
        f"  hyperneat pop={HYPERNEAT_PARAMS['pop_size']} "
        f"substrate={HYPERNEAT_PARAMS['substrate_hidden_layers']} "
        f"eval_eps={HYPERNEAT_PARAMS['eval_episodes_per_genome']} budget={TOTAL_ENV_STEPS:,}"
    )
    t0 = time.time()
    result = HyperNEATTrainer().train(cfg)
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": time.time() - t0,
        "seed": seed,
    }


# ===========================================================================
# MATD3 section (AgileRL)
# ===========================================================================


def run_matd3(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"MATD3 | seed={seed}")
    from deap_pettingzoo.baselines.agilerl_trainer import MATD3Trainer
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        ProbeConfig,
        RecordingConfig,
        TrainingConfig,
    )

    # RL trainers budget on real consumed env steps by construction, so no
    # budget_on_consumed flag is needed for parity with the EA/CMA loops.
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=TOTAL_ENV_STEPS,
        num_seeds=1,
        seed=int(seed),
        max_steps_per_episode=_episode_steps_per_episode(),
        algorithm_params={
            **MATD3_PARAMS,
            "num_envs": RL_NUM_ENVS,
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        probe=ProbeConfig(
            enabled=True,
            every_n_steps=PROBE_EVERY_N_STEPS,
            n_episodes=PROBE_EPISODES,
            info_keys=("prey_caught_total",),
            seed=PROBE_SEED_OFFSET + seed,
        ),
        recording=RecordingConfig(
            enabled=RECORD,
            mode="interval",
            every_n_steps=20_000,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
            episode_prefix="matd3_",
        ),
    )
    trainer = MATD3Trainer(device=DEVICE, verbose=True)
    t0 = time.time()
    result = trainer.train(cfg)
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": time.time() - t0,
        "seed": seed,
    }


# ===========================================================================
# Final evaluation / analysis
# ===========================================================================


def _summarize_episode_stats(rewards: List[float], catches: List[int]) -> Dict[str, float]:
    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "catches_mean": float(np.mean(catches)),
        "catches_std": float(np.std(catches)),
        "n_episodes": len(rewards),
    }


def _evaluate_genome(genome, act_space, n_episodes: int, seed_base: int) -> Dict[str, float]:
    """Deterministic champion evaluation against scripted prey."""
    rewards, catches = [], []
    for i in range(n_episodes):
        policy = FeedforwardNeuralPolicy(genome, head=make_eval_head(act_space))
        r, c = _run_policy_episode(policy, seed=seed_base + i)
        rewards.append(float(r))
        catches.append(int(c))
    return _summarize_episode_stats(rewards, catches)


def _evaluate_rl_policy(policy, n_episodes: int, seed_base: int) -> Dict[str, float]:
    rewards, catches = [], []
    for i in range(n_episodes):
        r, c = _run_policy_episode(policy, seed=seed_base + i)
        rewards.append(float(r))
        catches.append(int(c))
    return _summarize_episode_stats(rewards, catches)


def _final_analysis(root: Path) -> None:
    _section("Final analysis")
    try:
        from deap_pettingzoo.analysis import compare, loaders, report
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
# Done at import so every call site — main loop, library trainers, parallel EA
# workers (which re-import this module via factory qualnames) — sees the
# same registered name.

register_scripted_aquarium(
    ENV_NAME,
    ENV_CONFIG,
    apf_config=APF_CONFIG,
    reward_shape_alpha=REWARD_SHAPE_ALPHA,
    wrappers=(AquariumObservationFeatureWrapper,),
    overwrite=True,
)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    _section("Pursuit-evasion in marl-aquarium — PURSUER training")
    print(f"  ENV          : {ENV_NAME}  ({ACTION_MODE})")
    print(f"  predators    : {ENV_CONFIG['predator_count']}  (learners)")
    print(f"  prey         : {ENV_CONFIG['prey_count']}  (scripted APF)")
    print(f"  episode cap  : {ENV_CONFIG['max_time_steps']} steps")
    print(f"  budget       : {TOTAL_ENV_STEPS:,} env_steps × {NUM_SEEDS} seeds × algos")
    print(
        f"  toggles      : EA={RUN_EA}  CMA-ES={RUN_CMAES}  PSO={RUN_PSO}  "
        f"MAP-Elites={RUN_MAP_ELITES}  HyperNEAT={RUN_HYPERNEAT}  MATD3={RUN_MATD3}"
    )
    print(f"  output_dir   : {OUTPUT_DIR}")

    if PROBE_ENV:
        _probe_env()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()
    summary: Dict[str, List[Dict[str, Any]]] = {
        "ea": [], "cmaes": [], "pso": [], "map_elites": [],
        "hyperneat": [], "matd3": [],
    }

    def _try(label: str, fn, seed: int) -> None:
        """Run one algorithm and record its champion evaluation."""
        try:
            res = fn(seed, OUTPUT_DIR / label)
            if "eval" in res:
                summary[label].append(
                    {"seed": seed, **res["eval"], "best_fitness": res["best_fitness"]}
                )
            else:
                eval_stats = _evaluate_rl_policy(
                    res["policy"], n_episodes=EVAL_EPISODES, seed_base=10_000 * seed
                )
                summary[label].append({"seed": seed, **eval_stats})
        except Exception:
            traceback.print_exc()

    for seed_idx in range(NUM_SEEDS):
        seed = SEED_BASE + seed_idx
        if RUN_EA:
            _try("ea", run_ea, seed)
        if RUN_CMAES:
            _try("cmaes", run_cmaes, seed)
        if RUN_PSO:
            _try("pso", run_pso, seed)
        if RUN_MAP_ELITES:
            _try("map_elites", run_map_elites, seed)
        if RUN_HYPERNEAT:
            _try("hyperneat", run_hyperneat, seed)
        if RUN_MATD3:
            _try("matd3", run_matd3, seed)

    _section("Summary")
    for algo, rows in summary.items():
        if not rows:
            continue
        means = [r.get("mean", float("nan")) for r in rows]
        catches = [r.get("catches_mean", float("nan")) for r in rows]
        print(
            f"  {algo:>6} : mean_reward={np.mean(means):+.2f} ± {np.std(means):.2f}  "
            f"catches/ep={np.nanmean(catches):.2f} ± {np.nanstd(catches):.2f}  (n={len(rows)})"
        )
    print(f"  total time   : {time.time() - overall_t0:.1f}s")

    if RUN_FINAL_ANALYSIS:
        _final_analysis(OUTPUT_DIR)

    _section("Done")


if __name__ == "__main__":
    main()
