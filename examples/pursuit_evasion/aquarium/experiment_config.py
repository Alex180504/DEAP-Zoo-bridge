"""Canonical configuration for the aquarium pursuit-evasion experiment suite.

Single source of truth consumed by the three experiment drivers
(``aquarium_pursuer_training``, ``aquarium_evader_training``,
``aquarium_coevolution``), the hyperparameter sweeps, and the
HPC launcher.  Nothing here is env- or trainer-registry magic — just plain
constants and small pure helpers, so every entry point stays reproducible.

Where the numbers come from
---------------------------
* **Env config** — perfectly symmetric agent abilities (equal speed, view,
  FOV, acceleration/steer for both teams) on the bounded rectangle.
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
from typing import Any, Dict, List, Tuple

import numpy as np

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

# Ability-symmetric, equal-size 5-pursuer / 5-evader setup on a bounded
# 1500x1500 rectangle.  The env is kept perfectly symmetric (no ability
# handicap); with equal team sizes any residual APF-vs-APF imbalance is
# reported as a caveat rather than calibrated away.
#
# APF-vs-APF reference, measured on THIS config with
# ``aquarium_env_balance.py --reference`` (128 episodes/mode, 2026-07-31):
#
#   discrete    catch fraction 0.684 +/- 0.019 (SEM), median catch step 111
#   continuous  catch fraction 0.720 +/- 0.019 (SEM), median catch step 115
#
# So the 5v5 setup is materially predator-favoured under scripted play — ~10
# SEM above the 0.5 balance target, i.e. a real bias, not sampling noise.  It
# is the reference every learned catch rate is quoted against: 3.4-3.6 of 5
# prey caught by a scripted pursuer team.  (The 0.432 / 0.495 figures quoted
# before this measurement were from the earlier 3-pursuer / 6-evader config
# and do not apply.)
BALANCED_ENV_CONFIG: Dict[str, Any] = {
    "predator_count": 5,
    "prey_count": 5,
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
    "draw_view_cones": True,
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

# Gains perturbed for the anchor set. The shipped configs above stay held
# out for scoring, so a run cannot post a good probe number by fitting the
# one opponent it trained against.
ANCHOR_GAIN_SPREAD: float = 0.35
ANCHOR_VARIANTS: int = 4


def anchor_apf_configs(
    base: AquariumAPFConfig,
    n: int = ANCHOR_VARIANTS,
    *,
    seed: int = 0,
    spread: float = ANCHOR_GAIN_SPREAD,
) -> List[AquariumAPFConfig]:
    """Return ``n`` perturbed APF configs for use as fitness anchors.

    An anchor's job is to be a *fixed* reference, not a strong opponent, so
    what matters is that the set stays constant for the whole run and is
    disjoint from the config used to score it.  Training on the exact
    scoring opponent would let a run overfit the one controller it will be
    measured against and report progress that does not generalise; a small
    family of perturbed variants keeps the reference stable while leaving
    the shipped config genuinely held out.

    The gains are perturbed log-uniformly so a 35% spread means the same
    proportional change up as down.  ``wall_threshold`` is a normalised
    distance rather than a gain, so it is clipped to a sane band.

    Args:
        base: Config to perturb — :data:`APF_EVADER_CONFIG` or
            :data:`APF_PURSUER_CONFIG`.
        n: How many variants to produce.
        seed: Fixes the perturbations; the same seed always yields the same
            anchor set, which is what makes runs comparable.
        spread: Fractional gain spread.

    Returns:
        ``n`` configs, none equal to ``base``.
    """
    if n < 1:
        raise ValueError(f"anchor_apf_configs: n must be >= 1, got {n}.")
    rng = np.random.default_rng(seed)

    def jitter(value: float) -> float:
        if abs(value) < 1e-12:
            return 0.0   # A disabled term stays disabled.
        return float(value * np.exp(rng.uniform(-1.0, 1.0) * np.log1p(spread)))

    out: List[AquariumAPFConfig] = []
    for _ in range(n):
        out.append(
            AquariumAPFConfig(
                k_predator=jitter(base.k_predator),
                k_tangential=jitter(base.k_tangential),
                k_prey_separation=jitter(base.k_prey_separation),
                k_wall=jitter(base.k_wall),
                wall_threshold=float(
                    np.clip(jitter(base.wall_threshold), 0.05, 0.6)
                ),
                separation_cap_ratio=base.separation_cap_ratio,
                softening=base.softening,
                no_op_threshold=base.no_op_threshold,
            )
        )
    return out


# ===========================================================================
# Reward shaping
# ===========================================================================

# Dense progress shaping, ``alpha * (Δ nearest-opponent distance) / 200`` per
# step.  A catch (±20) must stay the dominant signal; too-large alpha
# creates an approach-without-catch local optimum.
#
# Coevolution shaping is potential-based (see
# :mod:`examples.pursuit_evasion.aquarium.reward_shaping`): an episode's total
# telescopes to ``Φ(s_T) − Φ(s_0)``, so at alpha=1.0 it is bounded by the arena
# diagonal / 200 ≈ ±10.6 regardless of episode length, against ±20 per catch.
# The ENUCC campaign measured the catch term at only 12–19% of median predator
# fitness under the *previous* non-telescoping form, which is what put alpha
# under suspicion.  Measured since on the 40-run K sweep, whose archives record
# the unshaped and shaping halves of each probe separately: the catch term is
# a median 0.686 of predator fitness, against the 0.50 the gate asks for.  So
# alpha=1.0 stands on measurement rather than inheritance, and shaping
# dominance is ruled out as an explanation for the co-evolution plateau —
# selection is already looking at a fitness that is mostly catches.
COEV_PRED_SHAPE_ALPHA: float = 1.0
COEV_PREY_SHAPE_ALPHA: float = 1.0
# Vs-scripted-APF training now shapes on the same potential as coevolution,
# so the two arms are scored on one reward definition and a co-evolved
# champion can be compared with a fixed-opponent one directly.  0.3 is carried
# over from the sweep that validated the old per-step form: the divisor and
# the sign are unchanged, and dropping the FOV gate can only *reduce* an
# episode's shaping total (the gate was what stopped it telescoping), so 0.3
# stays a safe upper bound until the re-run measures its catch share.
VS_APF_PRED_SHAPE_ALPHA: float = 0.3
VS_APF_PREY_SHAPE_ALPHA: float = 0.3


def run_provenance(
    alpha: float,
    *,
    probe_env_name: str,
    curriculum: bool,
    action_mode: str,
) -> Dict[str, Any]:
    """Manifest keys describing the conditions a run trained under.

    Every backend must record the same set, because these are exactly the
    facts that separate two archives whose numbers look equally plausible.
    ``shaping_form`` in particular is the only reliable way to tell the
    potential rewrite from what came before — the run timestamp does not
    survive a re-extract, and a grid missing these keys cannot be told from
    one that simply ran older code.

    Pass the result as ``TrainingConfig.run_metadata``, never as
    ``algorithm_params``: AgileRL's trainers splat that dict into their
    algorithm constructor, where an unknown key is a ``TypeError``.

    Args:
        alpha: The shaping weight the run actually used.
        probe_env_name: Env the held-out probe scores on, which differs from
            the training env whenever the curriculum is active.
        curriculum: Whether the opponent-speed curriculum was enabled.
        action_mode: ``"discrete"`` or ``"continuous"``.

    Returns:
        Dict to assign to a run's ``run_metadata``.
    """
    return {
        "shaping_form": "potential",
        "shape_alpha": float(alpha),
        "probe_env_name": probe_env_name,
        "curriculum": bool(curriculum),
        "action_mode": action_mode,
    }


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

# Post-training champion evaluation.  128 episodes, not the earlier 30:
# aquarium catch counts swing enough run-to-run that 30 episodes give SEM
# ~0.04 catches, the same order as the effects being compared, while 128 give
# ~0.019 (methodology §5).
EVAL_EPISODES: int = 128

# Per-candidate fitness episodes for every population-based trainer.
EVOLUTION_TRAIN_EVAL_EPISODES: int = 10

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

# ===========================================================================
# Optional opponent-speed curriculum (vs-APF suites, AQUARIUM_CURRICULUM=1)
# ===========================================================================

# Anneal the scripted opponent's max velocity toward the symmetric 12.0 at
# these consumed-budget fractions.
CURRICULUM_PHASE_FRACS: Tuple[float, ...] = (0.0, 0.4, 0.7)
CURRICULUM_OPPONENT_SPEEDS: Tuple[float, ...] = (9.0, 10.5, 12.0)


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
    "population_size": 24,
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "mut_sigma": 0.2,
    "mut_indpb": 0.3,
    "cxpb": 0.7,
    "elite_size": 3,
    "tournament_size": 3,
    "crossover_alpha": 0.5,
    "mutation_rate": 0.3,
}

CMA_PARAMS: Dict[str, Any] = {
    "population_size": 24,
    "hidden_size": EVOLVED_HIDDEN_SIZE,
    "sigma0": 0.2,
}

# PSO searches the same genome family.  Population matched to the EA for
# budget parity.  ``eval_episodes`` matches the EA/CMA per-candidate episode
# count so PSO fitness is de-noised by the same common-random-number averaging.
# The velocity constants are per-action-mode (MODE_BEST_OVERRIDES): the base
# dict carries the continuous-mode values.
PSO_PARAMS: Dict[str, Any] = {
    "population_size": 24,
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
    "pso": {
        "discrete": {"phi1": 1.49618, "phi2": 1.49618, "weight": 0.7298},
        "continuous": {"phi1": 2.0, "phi2": 2.0, "weight": 0.9},
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
    "batch_size": 16,
    "grid_shape": (8, 8),
    "iso_sigma": 0.05,
    "line_sigma": 0.1,
    "init_sigma": 0.2,
    "pool_mode": "uniform",
    # Re-score the top cells on fresh CRN batches and keep the genuinely-best
    # one so best_fitness is comparable to the EA/CMA CRN champions.
    "reeval_top_k": 16,
    "reeval_rounds": 4,
}

# Measured on the symmetric 5v5 config by aquarium_descriptor_calibration.py
# (P5-P95 across scripted + random behaviours).
DESCRIPTOR_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Two-team descriptor (formation spread, mean distance to opponents).
    "predator": {"spread": (0.191, 0.565), "dist": (0.393, 0.709)},
    "prey": {"spread": (0.185, 0.423), "dist": (0.393, 0.709)},
    # Single-team descriptor (centroid spread, mean pairwise distance).
    "predator_single": {"spread": (0.191, 0.565), "pair": (0.3, 0.85)},
    "prey_single": {"spread": (0.185, 0.423), "pair": (0.321, 0.641)},
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


# The campaign's MATD3 slot runs the evolutionary-HPO variant
# (:class:`~deap_pettingzoo.baselines.agilerl_hpo_trainer.MATD3HPOTrainer`):
# a population of MATD3 agents evolved by tournament selection + mutation
# (weights, architecture, RL hyperparameters), with a persistent per-slot
# replay buffer and a best-ever hall-of-fame re-scored under a clean final
# eval budget.  These knobs sit on top of the base MATD3_PARAMS RL settings.
MATD3_HPO_PARAMS: Dict[str, Any] = {
    "population_size": 6,
    # Env steps each agent trains between generations.  At the 2.5M vs-APF
    # budget this is ~8 generations (budget / (population * evo_every)).
    "evo_every_n_steps": 50_000,
    # Best-ever hall-of-fame kept + re-scored; hof[0] is the returned champion.
    "hof_size": 5,
    "eval_episodes": 8,          # per-generation tournament scoring
    "final_eval_episodes": 32,   # clean HOF re-score for the returned champion
    "tournament_size": 2,
    "mutation_sd": 0.1,
    # RL-hyperparameter mutation search ranges.
    "hp_config": {
        "lr_actor": [1e-5, 1e-3],
        "lr_critic": [1e-4, 1e-2],
        "batch_size": [256, 2048],
    },
}


def matd3_hpo_algorithm_params(action_mode: str) -> Dict[str, Any]:
    """``algorithm_params`` for the MATD3 evolutionary-HPO trainer.

    Combines the base MATD3 RL knobs (network / lr / batch / replay buffer)
    with the population-level HPO knobs.  Consumed selectively by
    :meth:`~deap_pettingzoo.baselines.agilerl_hpo_trainer.EvolveHPOTrainer.train`
    (population / evo cadence / HOF / eval-episode counts are read by name;
    only the base RL keys reach the AgileRL agent constructor), so — unlike
    the plain trainer — extra keys here are not forwarded to the algorithm
    ``__init__``.  ``num_envs`` is intentionally omitted: the HPO rollout
    loop always uses a single (non-vectorised) env.
    """
    params = dict(MATD3_PARAMS)
    params.update(rl_exploration_kwargs(action_mode))
    params.update(MATD3_HPO_PARAMS)
    return params


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

# Episodes per candidate-vs-opponent fitness estimate.
K_OPPONENTS: int = 16
# Past-champion snapshots kept per side for archive-based opponent sampling.
HOF_CAPACITY: int = 10
ARCHIVE_MIX: float = 0.5
# Fraction of each individual's K_OPPONENTS episodes played against a fixed
# scripted anchor instead of the co-evolving opponent.  Off: the 4-arm grid of
# 2026-08-04 (10 seeds/arm) found no benefit at any dose — 4/16 scored
# -0.152 +- 0.164 against the unanchored control and 8/16 -0.109 +- 0.162,
# with the control the best arm.  The mechanism did fire (within-generation
# spread fell 22% at the heavier dose), so this is the technique failing on
# this task rather than the implementation failing; the machinery is kept and
# reachable via AQUARIUM_ANCHOR_MIX.
ANCHOR_MIX: float = 0.0
# Give every individual in a generation the same K episode seeds. Off for the
# same grid: CRN halved between-run spread (sd 0.44 -> 0.24) exactly as
# intended, but carried an unexplained -0.200 +- 0.159 on the mean, the
# largest single contrast in the grid.  Leaving it on would make every future
# run non-comparable with the archive for no demonstrated benefit.
COMMON_RANDOM_SEEDS: bool = False
# Scripted-APF benchmark cadence: an absolute progress signal decoupled from
# the coevolutionary moving-target effect.  Episode count matches
# PROBE_EPISODES so coevolution probe rows carry the same measurement
# precision as the vs-APF probe curves they are plotted against.
#
# The cadence is in consumed env steps, not generations: under
# side_pacing="equal_steps" a generation no longer costs a fixed amount (a
# cheap side takes many more of them), so a generation-indexed probe would
# fire at a pairing-dependent rate and cost more wall clock than the training
# it measures.  BENCHMARK_EVERY_N is the fallback for runs with no consumed
# counter, and 250k steps of the 5M pooled budget gives the same ~20
# checkpoints per run as the vs-APF PROBE_EVERY_N_STEPS.
BENCHMARK_EVERY_N: int = 5
BENCHMARK_EVERY_N_STEPS: int = 250_000
BENCHMARK_N_EPISODES: int = PROBE_EPISODES
