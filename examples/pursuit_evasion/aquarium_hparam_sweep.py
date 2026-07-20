"""Hyperparameter sweep for the aquarium pursuit-evasion suite (both roles).

Runs a small grid per campaign algorithm (EA, CMA-ES, PSO, MAP-Elites,
HyperNEAT, MATD3) against the fixed scripted APF opponent on the symmetric
balanced env.  Each grid spans the shipped per-mode values in
:mod:`examples.pursuit_evasion.aquarium.experiment_config` plus variants
either side, so a sweep run directly validates (or refutes) the shipped
choice for the active action mode.

``ROLE`` selects which side learns (``"pursuer"`` -> predators vs scripted
APF prey; ``"evader"`` -> prey vs scripted APF pursuers); ``ACTION_MODE``
selects discrete headings vs continuous desired-velocity control.  Each
configuration runs ``SWEEP_BUDGET_STEPS`` env steps for one seed; a ranked
CSV + Markdown report is written to ``OUTPUT_DIR``.

Usage::

    python examples/pursuit_evasion/aquarium_hparam_sweep.py

Toggles at the top control roles, variants, and which algorithms run.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from deap import tools
from gymnasium import spaces as gym_spaces

from deap_pettingzoo.environments.registry import build_env, get_registry_snapshot
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    _MixedWorkerSpec,
    _individual_to_payload,
    _mixed_worker_evaluate_with_steps,
    clone_individual,
    create_toolbox,
    setup_deap_creators,
)
from deap_pettingzoo.integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_strategy,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.utils.parallel import make_executor

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
    register_scripted_predator_aquarium,
)


# ===========================================================================
# Top-level toggles
# ===========================================================================

# Env-var overrides let the HPC launcher select the variant without editing
# the file; spawned evaluation workers inherit the environment.
ROLE: str = os.environ.get("AQUARIUM_ROLE", "pursuer")                  # "pursuer" | "evader"
ACTION_MODE: str = os.environ.get("AQUARIUM_ACTION_MODE", "discrete")   # "discrete" | "continuous"

RUN_EA: bool = True
RUN_CMAES: bool = True
RUN_PSO: bool = True
RUN_MAP_ELITES: bool = True
RUN_HYPERNEAT: bool = True
RUN_MATD3: bool = True

# ---- Hardware / scope -------------------------------------------------------

SEED: int = 0
DEVICE: str = "cuda"
N_WORKERS: int = 0                 # 0 = os.cpu_count()
SWEEP_BUDGET_STEPS: int = XC.SWEEP_BUDGET_STEPS
EVAL_EPISODES: int = 20            # post-training champion evaluation
EVOLUTION_TRAIN_EVAL_EPISODES: int = XC.EVOLUTION_TRAIN_EVAL_EPISODES

# ---- Env (calibrated, shared) --------------------------------------------------

LEARNER_PREFIX: str = "predator" if ROLE == "pursuer" else "prey"
METRIC_NAME: str = "catches" if ROLE == "pursuer" else "prey_lost"
ENV_NAME: str = f"aquarium_{ROLE}_hparam_sweep_v0"
ENV_CONFIG: Dict[str, Any] = XC.make_env_config(ACTION_MODE)

SOFTMAX_TEMPERATURE: float = XC.SOFTMAX_TEMPERATURE
RL_NUM_ENVS: int = (
    XC.RL_NUM_ENVS_WINDOWS if sys.platform == "win32" else XC.RL_NUM_ENVS
)

OUTPUT_DIR: Path = Path(f"results/aquarium_{ROLE}_hparam_sweep")


class ClipContinuousActionsWrapper:
    """Clip continuous actions to the Box bounds before stepping.

    Unbounded Gaussian action heads crash the pettingzoo assert wrapper
    inside marl-aquarium.  Bounded-actor learners are unaffected — clipping
    is then the identity.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def observation_space(self, agent):
        return self.inner.observation_space(agent)

    def action_space(self, agent):
        return self.inner.action_space(agent)

    def reset(self, seed=None, options=None):
        return self.inner.reset(seed=seed, options=options)

    def step(self, actions):
        clipped = {}
        for agent, action in actions.items():
            space = self.inner.action_space(agent)
            if isinstance(space, gym_spaces.Box):
                arr = np.asarray(action, dtype=space.dtype).reshape(space.shape)
                clipped[agent] = np.clip(arr, space.low, space.high)
            else:
                clipped[agent] = action
        return self.inner.step(clipped)

    def close(self):
        return self.inner.close()


if ROLE == "pursuer":
    register_scripted_aquarium(
        ENV_NAME,
        ENV_CONFIG,
        apf_config=XC.APF_EVADER_CONFIG,
        reward_shape_alpha=XC.VS_APF_PRED_SHAPE_ALPHA,
        wrappers=(AquariumObservationFeatureWrapper, ClipContinuousActionsWrapper),
        overwrite=True,
    )
else:
    register_scripted_predator_aquarium(
        ENV_NAME,
        ENV_CONFIG,
        apf_config=XC.APF_PURSUER_CONFIG,
        reward_shape_alpha=XC.VS_APF_PREY_SHAPE_ALPHA,
        close_range_penalty=XC.PREY_CLOSE_RANGE_PENALTY,
        close_range_sigma2=XC.PREY_CLOSE_RANGE_SIGMA2,
        wrappers=(AquariumObservationFeatureWrapper, ClipContinuousActionsWrapper),
        overwrite=True,
    )


# ===========================================================================
# Hyperparameter grids (absolute configs spanning the shipped per-mode
# values; the shipped choice for the active mode is always one of the rows)
# ===========================================================================

_H = XC.EVOLVED_HIDDEN_SIZE

# Shipped canonical: ea_center 
EA_CONFIGS: List[Dict[str, Any]] = [
    {"name": "ea_center", "population_size": 48, "hidden_size": _H,
     "mut_sigma": 0.2, "mut_indpb": 0.3, "cxpb": 0.7, "elite_size": 3},
    {"name": "ea_small_pop", "population_size": 32, "hidden_size": _H,
     "mut_sigma": 0.2, "mut_indpb": 0.3, "cxpb": 0.7, "elite_size": 3},
    {"name": "ea_large_pop", "population_size": 64, "hidden_size": _H,
     "mut_sigma": 0.2, "mut_indpb": 0.3, "cxpb": 0.7, "elite_size": 3},
    {"name": "ea_low_mutation", "population_size": 48, "hidden_size": _H,
     "mut_sigma": 0.1, "mut_indpb": 0.2, "cxpb": 0.7, "elite_size": 3},
    {"name": "ea_high_mutation", "population_size": 48, "hidden_size": _H,
     "mut_sigma": 0.4, "mut_indpb": 0.4, "cxpb": 0.7, "elite_size": 3},
    {"name": "ea_low_cx", "population_size": 48, "hidden_size": _H,
     "mut_sigma": 0.2, "mut_indpb": 0.3, "cxpb": 0.3, "elite_size": 3},
]

# Shipped canonical: cmaes_center 
CMAES_CONFIGS: List[Dict[str, Any]] = [
    {"name": "cmaes_center", "population_size": 48, "hidden_size": _H, "sigma0": 0.2},
    {"name": "cmaes_low_sigma", "population_size": 48, "hidden_size": _H, "sigma0": 0.1},
    {"name": "cmaes_high_sigma", "population_size": 48, "hidden_size": _H, "sigma0": 0.4},
    {"name": "cmaes_small_pop", "population_size": 24, "hidden_size": _H, "sigma0": 0.2},
]

# PSO grid: shipped canonical is per-mode — pso_constriction for discrete, pso_high_inertia (w 0.9) for
# continuous.  The grid keeps the trainer-default inertia and the
# generations-vs-samples population trade as challengers.
PSO_CONFIGS: List[Dict[str, Any]] = [
    {"name": "pso_center", "population_size": 48, "hidden_size": _H,
     "phi1": 2.0, "phi2": 2.0, "weight": 0.7298},
    {"name": "pso_constriction", "population_size": 48, "hidden_size": _H,
     "phi1": 1.49618, "phi2": 1.49618, "weight": 0.7298},
    {"name": "pso_high_inertia", "population_size": 48, "hidden_size": _H,
     "phi1": 2.0, "phi2": 2.0, "weight": 0.9},
    {"name": "pso_small_pop", "population_size": 24, "hidden_size": _H,
     "phi1": 2.0, "phi2": 2.0, "weight": 0.7298},
]

# Shipped canonical: me_center 
MAP_ELITES_CONFIGS: List[Dict[str, Any]] = [
    {"name": "me_center", "hidden_size": _H, "iso_sigma": 0.05, "line_sigma": 0.1,
     "batch_size": 32, "init_sigma": 0.2},
    {"name": "me_low_iso", "hidden_size": _H, "iso_sigma": 0.005, "line_sigma": 0.05,
     "batch_size": 32, "init_sigma": 0.2},
    {"name": "me_high_iso", "hidden_size": _H, "iso_sigma": 0.15, "line_sigma": 0.3,
     "batch_size": 32, "init_sigma": 0.2},
    {"name": "me_large_batch", "hidden_size": _H, "iso_sigma": 0.05, "line_sigma": 0.1,
     "batch_size": 64, "init_sigma": 0.2},
]

# HyperNEAT grid: substrate width matched to the evolved family (16) vs the
# trainer-default small substrate, plus the population trade.  Shipped
# canonical: hyperneat_large_pop
HYPERNEAT_CONFIGS: List[Dict[str, Any]] = [
    {"name": "hyperneat_center", "population_size": 50,
     "substrate_hidden_layers": [_H],
     "eval_episodes_per_genome": EVOLUTION_TRAIN_EVAL_EPISODES},
    {"name": "hyperneat_large_pop", "population_size": 100,
     "substrate_hidden_layers": [_H],
     "eval_episodes_per_genome": EVOLUTION_TRAIN_EVAL_EPISODES},
    {"name": "hyperneat_small_substrate", "population_size": 50,
     "substrate_hidden_layers": [4, 4],
     "eval_episodes_per_genome": EVOLUTION_TRAIN_EVAL_EPISODES},
]

# Off-policy exploration noise is the primary MATD3 sweep axis.  The
# high-noise variant keeps re-testing the rejected "discrete argmax needs
# big noise" hypothesis (0.2 beat 0.5 at 1M steps and 0.8 at 100k; see
# docs/aquarium_hyperparameters.md).
_center_noise = XC.rl_exploration_kwargs(ACTION_MODE)["expl_noise"]

MATD3_CONFIGS: List[Dict[str, Any]] = [
    {"name": "matd3_center", "lr_actor": 1e-4, "lr_critic": 1e-3,
     "expl_noise": _center_noise},
    {"name": "matd3_high_noise", "lr_actor": 1e-4, "lr_critic": 1e-3,
     "expl_noise": min(1.0, _center_noise * 4)},
    {"name": "matd3_high_lr", "lr_actor": 3e-4, "lr_critic": 3e-3,
     "expl_noise": _center_noise},
    {"name": "matd3_small_batch", "lr_actor": 1e-4, "lr_critic": 1e-3,
     "expl_noise": _center_noise, "batch_size": 256},
]


# ===========================================================================
# Helpers
# ===========================================================================


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _subsection(title: str) -> None:
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


def _episode_steps() -> int:
    return int(ENV_CONFIG["max_time_steps"])


def _generations_for_budget(pop_size: int, eval_eps: int) -> int:
    per_gen = pop_size * max(1, eval_eps) * _episode_steps()
    return max(1, SWEEP_BUDGET_STEPS // per_gen)


def _evaluator_env_factory():
    return DictResetEnv(build_env(ENV_NAME))


def _empty_scripted_factory(env) -> Dict[str, Any]:
    return {}


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    head = make_train_head(
        action_space,
        temperature=SOFTMAX_TEMPERATURE,
        rng_seed=zlib.crc32(agent_id.encode("utf-8")),
    )
    return FeedforwardNeuralPolicy(genome, head=head)


def _build_evaluator() -> MixedPolicyEvaluator:
    return MixedPolicyEvaluator(
        env_factory=_evaluator_env_factory,
        scripted_policy_factory=_empty_scripted_factory,
        learner_agent_prefixes=[LEARNER_PREFIX],
        learner_factory=_learner_factory,
        max_steps=_episode_steps(),
    )


def _learner_io(evaluator: MixedPolicyEvaluator, seed: int):
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


def _extract_task_metric(infos) -> int:
    """Role-specific progress counter (catches or prey lost)."""
    if ROLE == "pursuer":
        return extract_prey_caught_total(infos)
    best = 0
    for info in (infos or {}).values():
        if isinstance(info, dict) and info.get("prey_lost_total") is not None:
            try:
                best = max(best, int(info["prey_lost_total"]))
            except (TypeError, ValueError):
                continue
    return best


def _run_policy_episode(policy, seed: int) -> Tuple[float, int]:
    env = build_env(ENV_NAME)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, infos = env.reset(seed=seed)
        metric = _extract_task_metric(infos)
        episode_reward = 0.0
        for _ in range(_episode_steps()):
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
            metric = max(metric, _extract_task_metric(infos))
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
        return episode_reward, metric
    finally:
        try:
            env.close()
        except SystemExit:
            pass


def _final_eval(policy_factory, seed: int) -> Tuple[np.ndarray, List[int]]:
    """Evaluate a fresh policy per episode; return (rewards, metrics)."""
    rewards: List[float] = []
    metrics: List[int] = []
    for i in range(EVAL_EPISODES):
        r, m = _run_policy_episode(policy_factory(), seed=10_000 * seed + i)
        rewards.append(r)
        metrics.append(m)
    return np.asarray(rewards, dtype=np.float64), metrics


def _result_row(
    name: str,
    algorithm: str,
    hyperparams: Dict[str, Any],
    rewards: np.ndarray,
    metrics: List[int],
    best_fitness: float,
    train_time: float,
    per_gen: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "config_name": name,
        "algorithm": algorithm,
        "hyperparams": hyperparams,
        "final_fitness_mean": float(rewards.mean()),
        "final_fitness_std": float(rewards.std()),
        "final_fitness_min": float(rewards.min()),
        "final_fitness_max": float(rewards.max()),
        "final_metric_mean": float(np.mean(metrics)),
        "final_metric_std": float(np.std(metrics)),
        "best_fitness": float(best_fitness),
        "training_time_s": float(train_time),
        "total_env_steps": SWEEP_BUDGET_STEPS,
        "per_gen": per_gen,
    }


def _make_crn_pmap(
    executor, evaluator, eval_episodes, gen_seed_holder, registry_snapshot,
    step_sink,
):
    """CRN parallel map (all individuals share the generation's seed).

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
# EA runner
# ===========================================================================


def run_ea_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    name = cfg["name"]
    _subsection(f"EA :: {name}")
    evaluator = _build_evaluator()
    in_size, out_size, act_space = _learner_io(evaluator, seed)
    layers = [in_size, cfg["hidden_size"], out_size]
    print(f"  NN: in={in_size} hidden={cfg['hidden_size']} out={out_size} ({ACTION_MODE})")

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
        mutation_config={"sigma": cfg["mut_sigma"], "indpb": cfg["mut_indpb"]},
    )
    toolbox.register("clone", clone_individual)

    pop_size = cfg["population_size"]
    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _generations_for_budget(pop_size, EVOLUTION_TRAIN_EVAL_EPISODES)
    consumed = [0]  # Real env-step counter shared with the CRN pmap.
    print(
        f"  pop={pop_size} nominal gens={n_gens} workers={n_workers} "
        f"consumed_budgeting={XC.BUDGET_ON_CONSUMED}"
    )

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    rng = np.random.default_rng(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    per_gen_log: List[Dict[str, Any]] = []
    gen_seed_holder = {"seed": int(seed)}
    t0 = time.time()
    registry_snapshot = get_registry_snapshot()
    with make_executor(n_workers) as executor:
        toolbox.register(
            "map",
            _make_crn_pmap(
                executor, evaluator, EVOLUTION_TRAIN_EVAL_EPISODES,
                gen_seed_holder, registry_snapshot, consumed,
            ),
        )

        def _log_gen(gen: int) -> None:
            rec = stats.compile(population)
            _, champ_metric = _run_policy_episode(
                FeedforwardNeuralPolicy(hof[0].genome, head=make_eval_head(act_space)),
                seed=seed * 10_000 + gen,
            )
            per_gen_log.append({
                "gen": gen, "mean": float(rec["avg"]), "std": float(rec["std"]),
                "min": float(rec["min"]), "max": float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
                "champion_metric": int(champ_metric),
            })
            print(
                f"  Gen {gen:>3} | mean={rec['avg']:+.2f} max={rec['max']:+.2f} "
                f"{METRIC_NAME}={champ_metric}"
            )

        gen_seed_holder["seed"] = seed * 10_000
        for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
            ind.fitness.values = fit
        hof.update(population)
        _log_gen(0)

        for gen in XC.budget_generations(
            n_gens, SWEEP_BUDGET_STEPS, consumed, _episode_steps()
        ):
            gen_seed_holder["seed"] = seed * 10_000 + gen
            elites = [toolbox.clone(ind) for ind in tools.selBest(population, cfg["elite_size"])]
            for elite in elites:
                del elite.fitness.values
            n_offspring = len(population) - cfg["elite_size"]
            offspring = list(map(toolbox.clone, toolbox.select(population, n_offspring)))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < cfg["cxpb"]:
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
            _log_gen(gen)

    train_time = time.time() - t0
    rewards, metrics = _final_eval(
        lambda: FeedforwardNeuralPolicy(hof[0].genome, head=make_eval_head(act_space)),
        seed,
    )
    return _result_row(
        name, "EA",
        {k: cfg[k] for k in ("population_size", "hidden_size", "mut_sigma",
                              "mut_indpb", "cxpb", "elite_size")},
        rewards, metrics, float(hof[0].fitness.values[0]), train_time, per_gen_log,
    )


# ===========================================================================
# CMA-ES runner
# ===========================================================================


def run_cmaes_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    name = cfg["name"]
    _subsection(f"CMA-ES :: {name}")
    evaluator = _build_evaluator()
    in_size, out_size, act_space = _learner_io(evaluator, seed)
    layers = [in_size, cfg["hidden_size"], out_size]
    print(f"  NN: in={in_size} hidden={cfg['hidden_size']} out={out_size} ({ACTION_MODE})")

    setup_deap_creators()
    np.random.seed(seed)
    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": layers,
        "activation": XC.EVOLVED_ACTIVATION,
        "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        "weight_range": (-init_scale, init_scale),
    }
    strategy, dimension = setup_cma_strategy(
        genome_config=genome_config,
        sigma0=cfg["sigma0"],
        population_size=cfg["population_size"],
        centroid=None,
    )
    ind_init = make_cma_ind_init(genome_config)
    toolbox = register_cma_toolbox(strategy, ind_init, evaluator)
    stats_cma = make_cma_stats()
    hof = tools.HallOfFame(1)

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    n_gens = _generations_for_budget(cfg["population_size"], EVOLUTION_TRAIN_EVAL_EPISODES)
    consumed = [0]  # Real env-step counter shared with the CRN pmap.
    print(
        f"  pop={cfg['population_size']} σ0={cfg['sigma0']} dim={dimension} "
        f"nominal gens={n_gens} consumed_budgeting={XC.BUDGET_ON_CONSUMED}"
    )

    per_gen_log: List[Dict[str, Any]] = []
    gen_seed_holder = {"seed": int(seed)}
    t0 = time.time()
    registry_snapshot = get_registry_snapshot()
    with make_executor(n_workers) as executor:
        toolbox.register(
            "map",
            _make_crn_pmap(
                executor, evaluator, EVOLUTION_TRAIN_EVAL_EPISODES,
                gen_seed_holder, registry_snapshot, consumed,
            ),
        )

        for gen in XC.budget_generations(
            n_gens, SWEEP_BUDGET_STEPS, consumed, _episode_steps()
        ):
            gen_seed_holder["seed"] = seed * 10_000 + gen
            population = toolbox.generate()
            for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
                ind.fitness.values = fit
            toolbox.update(population)
            hof.update(population)
            rec = stats_cma.compile(population)
            _, champ_metric = _run_policy_episode(
                FeedforwardNeuralPolicy(hof[0].genome, head=make_eval_head(act_space)),
                seed=seed * 10_000 + gen,
            )
            per_gen_log.append({
                "gen": gen, "mean": float(rec["avg"]), "std": float(rec.get("std", 0.0)),
                "min": float(rec["min"]), "max": float(rec["max"]),
                "best": float(hof[0].fitness.values[0]),
                "champion_metric": int(champ_metric),
            })
            print(
                f"  Gen {gen:>3} | mean={rec['avg']:+.2f} max={rec['max']:+.2f} "
                f"{METRIC_NAME}={champ_metric}"
            )

    train_time = time.time() - t0
    rewards, metrics = _final_eval(
        lambda: FeedforwardNeuralPolicy(hof[0].genome, head=make_eval_head(act_space)),
        seed,
    )
    return _result_row(
        name, "CMA-ES",
        {"population_size": cfg["population_size"], "hidden_size": cfg["hidden_size"],
         "sigma0": cfg["sigma0"], "dimension": dimension},
        rewards, metrics, float(hof[0].fitness.values[0]), train_time, per_gen_log,
    )


# ===========================================================================
# MAP-Elites runner
# ===========================================================================


def run_map_elites_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    name = cfg["name"]
    _subsection(f"MAP-Elites :: {name}")
    from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer
    from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig

    role = "predator" if ROLE == "pursuer" else "prey"
    # QDax honours budget_on_consumed on the numpy/PettingZoo scoring path
    # (the one this sweep uses); overshoot is at most one batch's real cost.
    train_cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=SWEEP_BUDGET_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        continuous=ACTION_MODE == "continuous",
        algorithm_params={
            "seed": seed,
            "hidden_sizes": (cfg["hidden_size"],),
            "activation": XC.EVOLVED_ACTIVATION,
            "init_batch_size": cfg["batch_size"],
            "init_mode": "gaussian",
            "init_sigma": cfg["init_sigma"],
            "batch_size": cfg["batch_size"],
            "eval_episodes": EVOLUTION_TRAIN_EVAL_EPISODES,
            "grid_shape": XC.QDAX_PARAMS["grid_shape"],
            "iso_sigma": cfg["iso_sigma"],
            "line_sigma": cfg["line_sigma"],
            "episode_length": _episode_steps(),
            "descriptor_fn": XC.make_two_team_descriptor(role, ENV_CONFIG),
            "state_extractor": XC.make_state_extractor(role, ENV_CONFIG),
            "save_repertoire": False,
            "num_workers": N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1),
            "parallel_backend": "process",
            "worker_imports": ("examples.pursuit_evasion.aquarium_hparam_sweep",),
            "layer_sizes": None,
            # Re-select the champion on fresh CRN batches so the config is
            # ranked on genuine champion quality..
            "reeval_top_k": XC.QDAX_PARAMS["reeval_top_k"],
            "reeval_rounds": XC.QDAX_PARAMS["reeval_rounds"],
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(enabled=False),
    )
    print(f"  hidden={cfg['hidden_size']} iso={cfg['iso_sigma']} batch={cfg['batch_size']}")

    trainer = MAPElitesTrainer()
    t0 = time.time()
    result = trainer.train(train_cfg)
    train_time = time.time() - t0

    rewards, metrics = _final_eval(lambda: result.policy, seed)
    per_gen_log = [
        {
            "gen": it.get("iteration", 0),
            "coverage": it.get("coverage", 0),
            "qd_score": it.get("qd_score", 0.0),
            "max_fitness": it.get("max_fitness", 0.0),
        }
        for it in result.metrics.get("iterations", [])
        if isinstance(it, dict)
    ]
    return _result_row(
        name, "MAP-Elites",
        {"hidden_size": cfg["hidden_size"], "iso_sigma": cfg["iso_sigma"],
         "line_sigma": cfg["line_sigma"], "batch_size": cfg["batch_size"],
         "init_sigma": cfg["init_sigma"],
         "coverage": result.metrics.get("coverage")},
        rewards, metrics,
        float(result.metrics.get("best_fitness", float("nan"))),
        train_time, per_gen_log,
    )


# ===========================================================================
# PSO runner
# ===========================================================================


def run_pso_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    """PSO over the shared evolved-family genome via :class:`PSOTrainer`.

    Library-trainer path: ``eval_episodes`` common-random-number episodes per
    particle per iteration (de-noised fitness, matching the vs-APF campaign),
    parallel process evaluation, and native consumed-step budgeting.
    """
    name = cfg["name"]
    _subsection(f"PSO :: {name}")
    from deap_pettingzoo.baselines.pso_trainer import PSOTrainer
    from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    eval_episodes = int(cfg.get("eval_episodes", EVOLUTION_TRAIN_EVAL_EPISODES))
    trainer = PSOTrainer(
        hidden_size=cfg["hidden_size"],
        population_size=cfg["population_size"],
        phi1=cfg["phi1"],
        phi2=cfg["phi2"],
        weight=cfg["weight"],
        smin=cfg.get("smin"),
        smax=cfg.get("smax"),
        activation=XC.EVOLVED_ACTIVATION,
        eval_episodes=eval_episodes,
        num_workers=n_workers,
        parallel_backend="process",
    )
    train_cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=SWEEP_BUDGET_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        seed=int(seed),
        max_steps_per_episode=_episode_steps(),
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(enabled=False),
    )
    print(
        f"  pop={cfg['population_size']} phi={cfg['phi1']}/{cfg['phi2']} "
        f"w={cfg['weight']} workers={n_workers}"
    )
    t0 = time.time()
    result = trainer.train(train_cfg)
    train_time = time.time() - t0

    rewards, metrics = _final_eval(lambda: result.policy, seed)
    return _result_row(
        name, "PSO",
        {"population_size": cfg["population_size"], "hidden_size": cfg["hidden_size"],
         "phi1": cfg["phi1"], "phi2": cfg["phi2"], "weight": cfg["weight"]},
        rewards, metrics,
        float(result.metrics.get("final_fitness", float("nan"))),
        train_time, [],
    )


# ===========================================================================
# HyperNEAT runner
# ===========================================================================


def run_hyperneat_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    name = cfg["name"]
    _subsection(f"HyperNEAT :: {name}")
    try:
        from deap_pettingzoo.contrib.neat import HyperNEATTrainer
    except ImportError as exc:
        raise ImportError(
            "HyperNEAT requires the 'neat' extra. Install with: pip install deap-pettingzoo[neat]"
        ) from exc
    from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig

    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    train_cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=SWEEP_BUDGET_STEPS,
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        num_workers=n_workers,
        algorithm_params={
            "seed": int(seed),
            "pop_size": cfg["population_size"],
            "eval_episodes_per_genome": cfg["eval_episodes_per_genome"],
            "substrate_hidden_layers": cfg["substrate_hidden_layers"],
            # Match the campaign head convention: tanh phenotype + softmax
            # search head (argmax over a saturating sigmoid phenotype produced
            # a constant, non-steering action and a step-function landscape).
            "phenotype_activation": XC.HYPERNEAT_PARAMS["phenotype_activation"],
            "stochastic_head": XC.HYPERNEAT_PARAMS["stochastic_head"],
            "head_temperature": XC.HYPERNEAT_PARAMS["head_temperature"],
        },
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(enabled=False),
    )
    print(
        f"  pop={cfg['population_size']} substrate={cfg['substrate_hidden_layers']} "
        f"eval_eps={cfg['eval_episodes_per_genome']} workers={n_workers}"
    )
    t0 = time.time()
    result = HyperNEATTrainer().train(train_cfg)
    train_time = time.time() - t0

    rewards, metrics = _final_eval(lambda: result.policy, seed)
    return _result_row(
        name, "HyperNEAT",
        {"population_size": cfg["population_size"],
         "substrate_hidden_layers": str(cfg["substrate_hidden_layers"]),
         "eval_episodes_per_genome": cfg["eval_episodes_per_genome"]},
        rewards, metrics,
        float(result.metrics.get("best_fitness", float("nan"))),
        train_time, [],
    )


# ===========================================================================
# AgileRL runner (MATD3)
# ===========================================================================


def run_matd3_config(cfg: Dict[str, Any], seed: int, run_dir: Path) -> Dict[str, Any]:
    name = cfg["name"]
    _subsection(f"MATD3 :: {name}")
    from deap_pettingzoo.baselines.agilerl_trainer import MATD3Trainer
    from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig

    # Every non-name config key overrides the canonical base params, so
    # grids can sweep any trainer knob (batch_size, learn_step, ...).
    cfg_overrides = {k: v for k, v in cfg.items() if k != "name"}
    params = {
        **XC.MATD3_PARAMS,
        "num_envs": RL_NUM_ENVS,
        **cfg_overrides,
        "O_U_noise": False,
    }
    train_cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=SWEEP_BUDGET_STEPS,
        num_seeds=1,
        seed=int(seed),
        max_steps_per_episode=_episode_steps(),
        algorithm_params=params,
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(enabled=False),
    )
    print(f"  lr_actor={cfg['lr_actor']} expl_noise={cfg['expl_noise']}")

    trainer = MATD3Trainer(device=DEVICE, verbose=True)
    t0 = time.time()
    result = trainer.train(train_cfg)
    train_time = time.time() - t0

    rewards, metrics = _final_eval(lambda: result.policy, seed)
    return _result_row(
        name, "MATD3",
        {"lr_actor": cfg["lr_actor"], "lr_critic": cfg["lr_critic"],
         "expl_noise": cfg["expl_noise"]},
        rewards, metrics, float(rewards.max()), train_time, [],
    )


# ===========================================================================
# Report generation
# ===========================================================================


def _generate_report(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Write CSV summary, per-generation CSV, Markdown report, console table."""
    _section("Generating Report")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sweep_summary.csv"
    csv_rows: List[Dict[str, Any]] = []
    all_fields: List[str] = []
    seen_fields: set = set()
    for r in all_results:
        row = {
            "config_name": r["config_name"],
            "algorithm": r["algorithm"],
            "final_fitness_mean": f"{r['final_fitness_mean']:.4f}",
            "final_fitness_std": f"{r['final_fitness_std']:.4f}",
            f"final_{METRIC_NAME}_mean": f"{r['final_metric_mean']:.4f}",
            f"final_{METRIC_NAME}_std": f"{r['final_metric_std']:.4f}",
            "best_fitness": f"{r['best_fitness']:.4f}",
            "training_time_s": f"{r['training_time_s']:.1f}",
            "total_env_steps": r["total_env_steps"],
        }
        for k, v in r["hyperparams"].items():
            row[f"hp_{k}"] = str(v)
        csv_rows.append(row)
        for key in row:
            if key not in seen_fields:
                seen_fields.add(key)
                all_fields.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV summary  -> {csv_path}")

    per_gen_path = output_dir / "sweep_per_gen.csv"
    gen_rows: List[Dict[str, Any]] = []
    for r in all_results:
        for g in r["per_gen"]:
            gen_rows.append({
                "config_name": r["config_name"],
                "algorithm": r["algorithm"],
                **g,
            })
    if gen_rows:
        fields: List[str] = []
        for row in gen_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(per_gen_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(gen_rows)
        print(f"  Per-gen CSV -> {per_gen_path}")

    md_path = output_dir / "sweep_report.md"
    lines: List[str] = [
        f"# Aquarium {ROLE.capitalize()} Hyperparameter Sweep Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Variant:** {ACTION_MODE} actions",
        f"**Budget per config:** {SWEEP_BUDGET_STEPS:,} env steps",
        f"**Evaluation episodes:** {EVAL_EPISODES}",
        f"**Total configs tested:** {len(all_results)}",
        "",
        "---",
        "",
        "## Final Results (ranked by mean evaluation reward)",
        "",
        f"| Rank | Config | Algorithm | Reward (μ ± σ) | {METRIC_NAME} (μ ± σ) | Best Fit | Time (s) |",
        "|------|--------|-----------|----------------|------------------|----------|----------|",
    ]
    ranked = sorted(all_results, key=lambda x: x["final_fitness_mean"], reverse=True)
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | `{r['config_name']}` | {r['algorithm']} | "
            f"{r['final_fitness_mean']:+.2f} ± {r['final_fitness_std']:.2f} | "
            f"{r['final_metric_mean']:.1f} ± {r['final_metric_std']:.1f} | "
            f"{r['best_fitness']:+.2f} | {r['training_time_s']:.0f} |"
        )
    lines += ["", "---", "", "## Per-Algorithm Best Configuration", ""]
    for algo in sorted(set(r["algorithm"] for r in all_results)):
        algo_results = [r for r in all_results if r["algorithm"] == algo]
        best = max(algo_results, key=lambda x: x["final_fitness_mean"])
        lines.append(f"### {algo}")
        lines.append(f"- **Best config:** `{best['config_name']}`")
        lines.append(
            f"- **Reward:** {best['final_fitness_mean']:+.2f} ± {best['final_fitness_std']:.2f}"
        )
        lines.append(
            f"- **{METRIC_NAME}:** {best['final_metric_mean']:.1f} ± {best['final_metric_std']:.1f}"
        )
        lines.append("- **Hyperparams:**")
        for k, v in best["hyperparams"].items():
            lines.append(f"  - `{k}`: {v}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown    -> {md_path}")

    _section("Sweep Results (ranked)")
    print(f"{'Rank':<5} {'Config':<24} {'Algo':<12} {'Reward μ±σ':<20} {METRIC_NAME + ' μ±σ':<16} {'Time':<8}")
    print("-" * 92)
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5} {r['config_name']:<24} {r['algorithm']:<12} "
            f"{r['final_fitness_mean']:>+7.2f} ± {r['final_fitness_std']:>5.2f}   "
            f"{r['final_metric_mean']:>5.1f} ± {r['final_metric_std']:>4.1f}   "
            f"{r['training_time_s']:>5.0f}s"
        )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    _section(f"Aquarium {ROLE} hyperparameter sweep")
    print(f"  ENV          : {ENV_NAME}  ({ACTION_MODE})")
    print(f"  learners     : {LEARNER_PREFIX}")
    print(f"  budget/config: {SWEEP_BUDGET_STEPS:,} env steps")
    print(f"  eval episodes: {EVAL_EPISODES}")
    print(
        f"  toggles      : EA={RUN_EA} CMA-ES={RUN_CMAES} PSO={RUN_PSO} "
        f"MAP-Elites={RUN_MAP_ELITES} HyperNEAT={RUN_HYPERNEAT} MATD3={RUN_MATD3}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    total_t0 = time.time()

    # Optional single-config selection so a launcher can fan configs out
    # as parallel processes (AQUARIUM_SWEEP_CONFIGS="matd3_center,...").
    cfg_filter = {
        c.strip()
        for c in os.environ.get("AQUARIUM_SWEEP_CONFIGS", "").split(",")
        if c.strip()
    }

    batches = [
        (RUN_EA, EA_CONFIGS, run_ea_config, "ea"),
        (RUN_CMAES, CMAES_CONFIGS, run_cmaes_config, "cmaes"),
        (RUN_PSO, PSO_CONFIGS, run_pso_config, "pso"),
        (RUN_MAP_ELITES, MAP_ELITES_CONFIGS, run_map_elites_config, "map_elites"),
        (RUN_HYPERNEAT, HYPERNEAT_CONFIGS, run_hyperneat_config, "hyperneat"),
        (RUN_MATD3, MATD3_CONFIGS, run_matd3_config, "matd3"),
    ]
    for enabled, configs, runner, subdir in batches:
        if not enabled:
            continue
        for cfg in configs:
            if cfg_filter and cfg["name"] not in cfg_filter:
                continue
            try:
                all_results.append(runner(cfg, SEED, OUTPUT_DIR / subdir))
            except Exception:
                print(f"  config {cfg['name']} FAILED:")
                traceback.print_exc()

    total_time = time.time() - total_t0
    print(f"\n  Total wall time: {total_time:.1f}s  ({total_time / 60:.1f} min)")

    if all_results:
        _generate_report(all_results, OUTPUT_DIR)
    else:
        print("\n  No results to report.")

    _section("Done")


if __name__ == "__main__":
    main()
