"""Canonical configuration for the aquarium pursuit-evasion experiment suite.

Single source of truth consumed by the three experiment drivers
(``aquarium_pursuer_training``, ``aquarium_evader_training``,
``aquarium_coevolution``), the hyperparameter sweeps, and the
HPC launcher.  Nothing here is env- or trainer-registry magic — just plain
constants and small pure helpers, so every entry point stays reproducible.

Where the numbers come from
---------------------------
* **Env config** — perfectly symmetric agent abilities (equal speed, view,
  FOV, acceleration/steer for both teams) on the bounded rectangle, per the
  2026-07-14 methodology revision: no calibration handicap, so any later
  performance gap is attributable to the learner, not to an ability edge.
  The residual APF-vs-APF imbalance this accepts is documented in
  ``docs/aquarium_experiment_methodology.md``.  The torus topology was
  dropped from the campaign entirely.
* **Algorithm hyperparameters** — per-action-mode best values from the
  hyperparameter sweeps.
* **QDax descriptor ranges** — measured by
  ``examples/pursuit_evasion/aquarium_descriptor_calibration.py`` on the
  balanced env config (5th–95th percentile of the observed axis values
  across scripted + random behaviours), so grid archives stretch over the
  *achievable* band instead of a theoretical maximum.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

from .apf_policy import AquariumAPFConfig


def ensure_utf8_stdout() -> None:
    """Best-effort switch of stdout/stderr to UTF-8.

    Redirected Windows streams default to cp1252, which cannot encode the
    progress glyphs (``->``, ``σ``, ``λ``) the experiment scripts print —
    the run then dies with ``UnicodeEncodeError`` inside a log pipe.  Every
    suite entry point calls this at import.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# ===========================================================================
# Environment
# ===========================================================================

# Ability-symmetric 3-pursuer / 6-evader setup on a bounded 1500x1500
# rectangle.  APF-vs-APF catch fraction: 0.432 (discrete) / 0.495
# (continuous) at 128 episodes/mode; see
# results/aquarium_env_balance/balance_report.json.
BALANCED_ENV_CONFIG: Dict[str, Any] = {
    "predator_count": 3,
    "prey_count": 6,
    "max_time_steps": 250,
    "predator_max_velocity": 12.0,
    "prey_max_velocity": 12.0,
    "predator_max_acceleration": 0.8,
    "prey_max_acceleration": 0.8,
    "predator_max_steer_force": 2.0,
    "prey_max_steer_force": 2.0,
    "predator_view_distance": 400,
    "prey_view_distance": 400,
    "predator_observe_count": 6,
    "prey_observe_count": 6,
    "fov_enabled": True,
    "predator_fov": 360,
    "prey_fov": 360,
    "action_count": 16,
    "continuous_actions": False,
    "observable_walls": True,
    "predator_reward": 20,
    "prey_reward": 0.5,
    "prey_punishment": 20.0,
    "catch_radius": 50,
    "keep_prey_count_constant": False,
    "width": 1500,
    "height": 1500,
    "draw_view_cones": False,
    "draw_action_vectors": False,
}

ACTION_MODES: Tuple[str, str] = ("discrete", "continuous")


def make_env_config(action_mode: str = "discrete", **overrides: Any) -> Dict[str, Any]:
    """Return a copy of the symmetric env config for one action mode.

    The ``AQUARIUM_ENV_OVERRIDES`` environment variable (a JSON object) is
    applied between the base config and the caller's explicit overrides.
    Launchers use it to run a whole experiment family on an env variant
    without editing this module — spawned evaluation workers inherit the
    variable, so worker-side env registration matches the driver's.

    Args:
        action_mode: ``"discrete"`` (``Discrete(action_count)`` headings)
            or ``"continuous"`` (``Box(-1, 1, (2,))`` desired velocity;
            requires marl-aquarium >= 0.1.12).
        **overrides: Final per-experiment key overrides.
    """
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
    cfg = dict(BALANCED_ENV_CONFIG)
    cfg["continuous_actions"] = action_mode == "continuous"
    env_var_overrides = os.environ.get("AQUARIUM_ENV_OVERRIDES")
    if env_var_overrides:
        cfg.update(json.loads(env_var_overrides))
    cfg.update(overrides)
    return cfg


# ===========================================================================
# Scripted APF opponents (shared by benchmarks, balance calibration, envs)
# ===========================================================================

# Same base gain for both roles so neither side gets a hand-tuning edge;
# the evader keeps its tangential dodge + flock separation (without them it
# freezes between symmetric threats and the match degenerates).
APF_PURSUER_CONFIG = AquariumAPFConfig(
    k_predator=5.0, k_tangential=0.0, k_prey_separation=0.0
)
APF_EVADER_CONFIG = AquariumAPFConfig(
    k_predator=5.0, k_tangential=3.0, k_prey_separation=0.5
)

# ===========================================================================
# Reward shaping
# ===========================================================================

# Dense progress shaping, ``alpha * (Δ nearest-opponent distance) / 200`` per
# step.  A catch (±20) must stay the dominant signal; too-large alpha
# creates an approach-without-catch local optimum.
#
# Coevolution uses 1.0: its shaping is clipped to one step of plausible
# motion (robust to target-switch spikes) and K_OPPONENTS averaging keeps
# fitness variance low enough for the gradient to matter.
COEV_PRED_SHAPE_ALPHA: float = 1.0
COEV_PREY_SHAPE_ALPHA: float = 1.0
# Vs-scripted-APF training uses the sweep-validated 0.3 (unclipped wrapper).
VS_APF_PRED_SHAPE_ALPHA: float = 0.3
VS_APF_PREY_SHAPE_ALPHA: float = 0.3

# Prey close-range danger penalty (evader training only): Gaussian bump when
# the nearest predator is inside the catch annulus.
PREY_CLOSE_RANGE_PENALTY: float = 5.0
PREY_CLOSE_RANGE_SIGMA2: float = 0.005

# ===========================================================================
# Shared training scope
# ===========================================================================

# Equal env-steps budgets across algorithms make the comparisons fair.
SMOKE_BUDGET_STEPS: int = 60_000        # local plumbing check
SWEEP_BUDGET_STEPS: int = 1_000_000     # per sweep config
VS_APF_BUDGET_STEPS: int = 2_500_000    # per (algorithm, seed) final run
COEV_BUDGET_STEPS: int = 5_000_000      # per (pairing, seed) final run

EVAL_EPISODES: int = 30                 # post-training champion evaluation
EVOLUTION_TRAIN_EVAL_EPISODES: int = 5  # per-candidate fitness episodes (EA/CMA/QDax)

# Held-out champion probe: every PROBE_EVERY_N_STEPS consumed env steps the
# current champion plays PROBE_EPISODES episodes against the scripted APF
# opponent and the scores land in probe.csv (per-run).
PROBE_EPISODES: int = 32
PROBE_EVERY_N_STEPS: int = 125_000      # ~20 checkpoints per 2.5M vs-APF run

# Consumed-step budgeting: stop when the *real* env-tick count hits the
# budget instead of a nominal generation count that assumes full-length
# episodes.  Aquarium episodes end early once every prey is caught
# (keep_prey_count_constant=False), so nominal budgeting under-funds the
# evolutionary algorithms relative to the RL learners, which always
# consume exactly budget_value real steps.
BUDGET_ON_CONSUMED: bool = True


def budget_generations(nominal_gens: int, budget_steps: int, step_counter, max_steps: int):
    """Yield 1-based generation indices under the active budgeting mode.

    Under nominal budgeting this is simply ``range(1, nominal_gens + 1)``.
    Under consumed-step budgeting (:data:`BUDGET_ON_CONSUMED`) generations
    are yielded until ``step_counter[0]`` — a 1-element list the caller
    shares with its parallel-evaluation step sink — reaches
    ``budget_steps``.  The safety cap mirrors the library trainers: each
    generation consumes at least one tick per evaluated individual and
    episodes run at most ``max_steps`` ticks, so no more than
    ``nominal_gens * max_steps`` generations are ever needed.

    Args:
        nominal_gens: Generation count implied by full-length episodes.
        budget_steps: Total real env-step budget.
        step_counter: 1-element mutable list accumulating consumed steps.
        max_steps: Per-episode step cap of the environment.
    """
    if not BUDGET_ON_CONSUMED:
        yield from range(1, nominal_gens + 1)
        return
    safety_cap = nominal_gens * max(1, max_steps) + 10
    gen = 0
    while gen < safety_cap and step_counter[0] < budget_steps:
        gen += 1
        yield gen

# ===========================================================================
# Evolved family (EA / CMA-ES / PSO / QDax share one genome shape)
# ===========================================================================

# One network family for EA / CMA-ES / PSO / QDax across every experiment:
# a single 16-unit tanh hidden layer with a linear output.  Bounded hidden
# activations keep logits in the range the softmax-temperature calibration
# assumes and are the coevolution-validated setting.
EVOLVED_HIDDEN_SIZE: int = 16
EVOLVED_ACTIVATION: str = "tanh"
EVOLVED_OUTPUT_ACTIVATION: str = "linear"

EA_PARAMS: Dict[str, Any] = {
    "population_size": 48,
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "mut_sigma": 0.2,
    "mut_indpb": 0.3,
    "cxpb": 0.3,
    "elite_size": 3,
    "tournament_size": 3,
    "crossover_alpha": 0.5,
    "mutation_rate": 0.3,
}

CMA_PARAMS: Dict[str, Any] = {
    "population_size": 48,
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "sigma0": 0.2,
}

# PSO searches the same genome family.  Population matched to the EA for
# budget parity.  ``eval_episodes`` matches the EA/CMA per-candidate episode
# count so PSO fitness is de-noised by the same common-random-number averaging.
PSO_PARAMS: Dict[str, Any] = {
    "population_size": 48,
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "phi1": 2.0,
    "phi2": 2.0,
    "weight": 0.9,
    "smin": None,
    "smax": None,
    "eval_episodes": EVOLUTION_TRAIN_EVAL_EPISODES,
}

# Softmax sampling temperature used by EA/CMA discrete heads during
# training (smooth fitness landscape; argmax is a step function).
SOFTMAX_TEMPERATURE: float = 0.3
MODE_BEST_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "cma": {
        "discrete": {"sigma0": 0.4},
        "continuous": {"sigma0": 0.2},
    },
}


def params_for(algo: str, action_mode: str) -> Dict[str, Any]:
    """Per-action-mode hyperparameters for *algo*.

    Returns a copy of the algorithm's base dict with the sweep-validated
    per-mode overrides from :data:`MODE_BEST_OVERRIDES` applied.
    """
    base_dicts = {
        "ea": EA_PARAMS,
        "cma": CMA_PARAMS,
        "pso": PSO_PARAMS,
        "map_elites": QDAX_PARAMS,
        "hyperneat": HYPERNEAT_PARAMS,
        "matd3": MATD3_PARAMS,
    }
    if algo not in base_dicts:
        raise ValueError(f"Unknown algo {algo!r}; available: {sorted(base_dicts)}")
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
    params = dict(base_dicts[algo])
    params.update(MODE_BEST_OVERRIDES.get(algo, {}).get(action_mode, {}))
    return params

# ===========================================================================
# QDax MAP-Elites
# ===========================================================================

QDAX_PARAMS: Dict[str, Any] = {
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "batch_size": 32,
    "grid_shape": (8, 8),
    "iso_sigma": 0.15,
    "line_sigma": 0.3,
    "init_sigma": 0.2,
    "pool_mode": "uniform",
    # Re-score the top cells on fresh CRN batches and keep the genuinely-best
    # one so best_fitness is comparable to the EA/CMA CRN champions.
    "reeval_top_k": 16,
    "reeval_rounds": 4,
}

DESCRIPTOR_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Two-team descriptor (formation spread, mean distance to opponents).
    "predator": {"spread": (0.167, 0.509), "dist": (0.394, 0.728)},
    "prey": {"spread": (0.187, 0.357), "dist": (0.394, 0.728)},
    # Single-team descriptor (centroid spread, mean pairwise distance).
    "predator_single": {"spread": (0.167, 0.509), "pair": (0.292, 0.886)},
    "prey_single": {"spread": (0.187, 0.357), "pair": (0.306, 0.595)},
}


def make_two_team_descriptor(role: str, env_config: Dict[str, Any]):
    """Two-axis (spread, mean-dist-to-opponents) descriptor for *role*.

    Expects positions ordered learner-role first — pair with
    :func:`make_position_extractor` / :func:`make_state_extractor` built
    for the same role.
    """
    from examples.qdax.descriptors import formation_spread_and_mean_distance

    n_learner = int(env_config["prey_count" if role == "prey" else "predator_count"])
    n_opponent = int(env_config["predator_count" if role == "prey" else "prey_count"])
    ranges = DESCRIPTOR_RANGES[role]
    return formation_spread_and_mean_distance(
        num_pursuers=n_learner,
        num_evaders=n_opponent,
        spread_range=ranges["spread"],
        dist_range=ranges["dist"],
    )


def make_single_team_descriptor(role: str, env_config: Dict[str, Any]):
    """Single-team (spread, pairwise) descriptor for envs exposing one role."""
    from examples.qdax.descriptors import single_team_formation

    n = int(env_config["prey_count" if role == "prey" else "predator_count"])
    ranges = DESCRIPTOR_RANGES[f"{role}_single"]
    return single_team_formation(
        num_agents=n,
        spread_range=ranges["spread"],
        pair_range=ranges["pair"],
    )


def make_position_extractor(role: str, env_config: Dict[str, Any]):
    """[0, 1]²-normalised position extractor for coevolution QDax sides."""
    from examples.pursuit_evasion.aquarium.qdax_position_extractor import (
        make_aquarium_extractor,
    )

    return make_aquarium_extractor(
        arena_width=float(env_config["width"]),
        arena_height=float(env_config["height"]),
        role_order=("prey", "predator") if role == "prey" else ("predator", "prey"),
    )


def make_state_extractor(role: str, env_config: Dict[str, Any]):
    """[0, 1]²-normalised state extractor for standalone MAP-Elites scoring."""
    from examples.pursuit_evasion.aquarium.qdax_state_extractor import (
        AquariumStateExtractor,
    )

    return AquariumStateExtractor(
        arena_width=float(env_config["width"]),
        arena_height=float(env_config["height"]),
        predator_count=int(env_config["predator_count"]),
        prey_count=int(env_config["prey_count"]),
        role_order=("prey", "predator") if role == "prey" else ("predator", "prey"),
    )


# ===========================================================================
# NEAT-family (HyperNEAT)
# ===========================================================================

# Fixed-substrate HyperNEAT: the CPPN paints weights onto a substrate whose
# hidden layer matches the evolved family's 16-unit width, so the phenotype
# capacity is comparable to the EA/CMA/PSO/QDax networks.  Per-genome
# evaluation cost matches the EA's (same episode count).
#
# ``phenotype_activation="tanh"`` and ``stochastic_head=True`` bring HyperNEAT
# to parity with the EA/CMA/PSO head convention: a sigmoid phenotype saturates
# to (0,1) and, decoded by argmax, collapses to a single constant heading (the
# agent swims straight and the fitness landscape is a step function). ``tanh``
# restores signed dynamic range and the softmax head (temperature matched to
# the EA's) gives gradient-free search a usable signal; the probe and returned
# champion still score the deterministic argmax policy.
HYPERNEAT_PARAMS: Dict[str, Any] = {
    "pop_size": 50,
    "substrate_hidden_layers": [EVOLVED_HIDDEN_SIZE],
    "eval_episodes_per_genome": EVOLUTION_TRAIN_EVAL_EPISODES,
    "phenotype_activation": EVOLVED_ACTIVATION,
    "stochastic_head": True,
    "head_temperature": SOFTMAX_TEMPERATURE,
}

RL_NUM_ENVS: int = 4
RL_NUM_ENVS_WINDOWS: int = 1

# ===========================================================================
# AgileRL off-policy (MATD3)
# ===========================================================================

# Standalone (vs scripted APF) trainer knobs.
MATD3_PARAMS: Dict[str, Any] = {
    "batch_size": 1_024,
    "learn_step": 25,
    "memory_size": 500_000,
    "learning_delay": 10_000,
    "lr_actor": 1e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "tau": 0.01,
    "policy_freq": 2,
    "net_config": {
        "encoder_config": {"hidden_size": [128, 128]},
        "head_config": {"hidden_size": [128]},
    },
}


def rl_exploration_kwargs(action_mode: str) -> Dict[str, Any]:
    """Exploration-noise constructor kwargs for MATD3 per action mode.
    """
    del action_mode  # Same exploration setting in both modes.
    return {"expl_noise": 0.2, "O_U_noise": False}


RL_COEV_PARAMS: Dict[str, Any] = {
    "batch_size": 64,
    "memory_size": 50_000,
    "learning_delay": 256,
    "rollout_episodes_per_gen": 4,
    "learning_steps_per_gen": 64,
}

# ===========================================================================
# Coevolution loop
# ===========================================================================

# Episodes per candidate-vs-opponent fitness estimate: the dominant
# variance-reduction knob (each catch swings fitness by ±20).
K_OPPONENTS: int = 8
# Past-champion snapshots kept per side for archive-based opponent sampling.
HOF_CAPACITY: int = 10
ARCHIVE_MIX: float = 0.5
# Scripted-APF benchmark cadence: an absolute progress signal decoupled from
# the coevolutionary moving-target effect.  Episode count matches
# PROBE_EPISODES so coevolution probe rows carry the same measurement
# precision as the vs-APF probe curves they are plotted against.
BENCHMARK_EVERY_N: int = 5
BENCHMARK_N_EPISODES: int = PROBE_EPISODES
