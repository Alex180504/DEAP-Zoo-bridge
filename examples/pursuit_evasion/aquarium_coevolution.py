"""Competitive coevolution in marl-aquarium — EA / CMA-ES / PSO / QDax / HyperNEAT / MATD3.

Every enabled pairing "X_vs_Y" trains predators with X against prey trained
by Y via true bidirectional coevolution
(:class:`CoEvolutionTrainer` / ``competitive_2pop``).  Six side types are
available for either role:

* ``ea`` / ``cma`` / ``pso`` — DEAP populations / particle swarm over the
  shared evolved-family genome (parallel opponent evaluation).
* ``qdax``           — MAP-Elites archive side (:class:`QDaxSideState`) with
  the calibrated behavioural descriptor, so the QD grid genuinely fills.
* ``hyperneat``      — CPPN population painting weights onto a fixed
  substrate matched to the evolved-family width.
* ``matd3``          — AgileRL MATD3 evolutionary-HPO population
  (:class:`RLEvolveHPOSideState`, off-policy): a population evolved by
  tournament + mutation against the current opponent pool, with persistent
  per-slot replay buffers.

Enable/disable pairings in the ``PAIRINGS`` dict; any ``"<a>_vs_<b>"`` key
with side types from ``SIDE_BUILDERS`` works.  ``ACTION_MODE`` selects the
experiment variant.  Env config, side hyperparameters, coevolution knobs,
and QDax descriptor ranges come from
:mod:`examples.pursuit_evasion.aquarium.experiment_config`.

Every ``BENCHMARK_EVERY_N_STEPS`` consumed env steps each side's champion
also plays a fixed scripted APF opponent — an absolute progress signal that
is immune to the coevolutionary moving-target effect — and one final
``FINAL_EVAL_EPISODES`` evaluation closes every run with a score that is
comparable across pairings.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Headless rendering — must be set before pygame ever calls init().
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from deap_pettingzoo.baselines.base import RecordingConfig
from deap_pettingzoo.baselines.coevolution import CoEvolutionConfig, Side
from deap_pettingzoo.baselines.coevolution.archive import RingHallOfFame
from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer
from deap_pettingzoo.environments.registry import register_env

# Allow running as a plain script *or* as a module (-m).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.pursuit_evasion.aquarium import experiment_config as XC

XC.ensure_utf8_stdout()
from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
)
from examples.pursuit_evasion.aquarium.reward_shaping import (
    entities_by_id,
    find_raw_env,
    nearest_opponent_distances,
    potential_shaping,
)
from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
    AquariumScriptedPredatorWrapper,
    AquariumScriptedPreyWrapper,
    _patch_aquarium_wrappers,
)


# ===========================================================================
# Top-level config (edit these — no CLI parsing)
# ===========================================================================

# ---- Pairings ("<predator_algo>_vs_<prey_algo>") -----------------------------

# The campaign grid is a full round-robin: every ordered pair of distinct
# campaign algorithms, so each algorithm meets every other both as
# predators and as prey. Self-pairings (X_vs_X) are excluded: they measure an
# intra-algorithm arms race, not the cross-algorithm comparison. Edit
# COEV_ALGOS or flip individual entries to False to run a subset.
COEV_ALGOS = ("ea", "cma", "pso", "qdax", "hyperneat", "matd3")
PAIRINGS: Dict[str, bool] = {
    f"{pred}_vs_{prey}": True
    for pred in COEV_ALGOS
    for prey in COEV_ALGOS
    if pred != prey
}

# ---- Experiment variant --------------------------------------------------------
# Env-var overrides let the HPC launcher select the variant without editing
# the file; spawned evaluation workers inherit the environment, so their
# import-time env registration matches the driver's.

ACTION_MODE: str = os.environ.get("AQUARIUM_ACTION_MODE", "discrete")   # "discrete" | "continuous"

# ---- Reproducibility / hardware ---------------------------------------------------

SEED_BASE: int = 0
NUM_SEEDS: int = 1
DEVICE: str = "cuda"
N_WORKERS: int = os.cpu_count() or 1   # coevolution evaluator worker processes

# ---- Budget -------------------------------------------------------------------------

COEV_TOTAL_STEPS: int = XC.COEV_BUDGET_STEPS

# ---- Env / shaping -------------------------------------------------------------------

ENV_NAME_COEV = "aquarium_coevolution_v0"
ENV_CONFIG: Dict[str, Any] = XC.make_env_config(ACTION_MODE)
PRED_REWARD_SHAPE_ALPHA: float = XC.COEV_PRED_SHAPE_ALPHA
PREY_REWARD_SHAPE_ALPHA: float = XC.COEV_PREY_SHAPE_ALPHA

# ---- Coevolution loop ------------------------------------------------------------------

K_OPPONENTS: int = XC.K_OPPONENTS
HOF_CAPACITY: int = XC.HOF_CAPACITY
ARCHIVE_MIX: float = XC.ARCHIVE_MIX
BENCHMARK_EVERY_N: int = XC.BENCHMARK_EVERY_N
BENCHMARK_EVERY_N_STEPS: int = XC.BENCHMARK_EVERY_N_STEPS
BENCHMARK_N_EPISODES: int = XC.BENCHMARK_N_EPISODES
# End-of-run champion evaluation — more episodes than the periodic probe
# because it is the number the cross-run comparison rests on.
FINAL_EVAL_EPISODES: int = XC.EVAL_EPISODES

# ---- Recording ------------------------------------------------------------------------

RECORD_VIDEOS: bool = True
RECORD_FPS: int = 30
RECORD_FMT: str = "mp4"
RECORD_MAX_FRAMES: Optional[int] = None
RECORD_MAX_DIM: int = 512

OUTPUT_DIR = Path("results/aquarium_coevolution")


# ===========================================================================
# Utility helpers
# ===========================================================================


def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def _coev_feature_wrapper(inner):
    """Featurize observations without the agent-id one-hot (both sides learn)."""
    return AquariumObservationFeatureWrapper(inner, append_agent_id=False)


def _fallback_action():
    """Mode-appropriate no-op action."""
    if ENV_CONFIG.get("continuous_actions", False):
        return np.zeros(2, dtype=np.float32)
    return 0


# ===========================================================================
# _CoevBothSidesAdapter — expose both predators and prey as learners
# ===========================================================================


class _CoevBothSidesAdapter:
    """Thin adapter over a raw marl-aquarium parallel env.

    Adds the ``predator_observe_count`` / ``prey_observe_count`` attributes
    that :class:`AquariumObservationFeatureWrapper` reads, normalises
    ``reset()`` to the modern ``(obs, infos)`` tuple, and exposes *all*
    agents (predators and prey) as learners.

    Optional dense *potential-based* progress shaping: predators earn
    ``alpha * (Φ(s') − Φ(s))`` with ``Φ(s) = −d(nearest prey) / 200``, prey the
    symmetric evasion potential.  The episode total therefore telescopes to
    ``Φ(s_T) − Φ(s_0)`` — bounded by the arena diagonal and impossible to farm
    by circling — instead of accumulating unrelated per-step deltas.  See
    :mod:`examples.pursuit_evasion.aquarium.reward_shaping`; the raw distance
    delta keeps a symmetric clip so a caught prey being removed cannot inject a
    large spurious reward at the exact step a catch happens.

    Attributes:
        shaping_totals: Per-side shaping reward injected during the current
            episode (reset by :meth:`reset`).  Diagnostic only — it lets a
            caller split an episode return into its env and shaping parts.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array"], "name": "coev_aquarium"}

    def __init__(
        self,
        inner,
        predator_observe_count: int,
        prey_observe_count: int,
        pred_shape_alpha: float = 0.0,
        prey_shape_alpha: float = 0.0,
    ) -> None:
        self.inner = inner
        self.predator_observe_count = int(predator_observe_count)
        self.prey_observe_count = int(prey_observe_count)
        self.possible_agents: List[str] = list(inner.possible_agents)
        self.agents: List[str] = list(getattr(inner, "agents", self.possible_agents))
        self._pred_shape_alpha = float(pred_shape_alpha)
        self._prey_shape_alpha = float(prey_shape_alpha)
        self._predator_agents = [a for a in self.possible_agents if a.startswith("predator")]
        self._prey_agents = [a for a in self.possible_agents if a.startswith("prey")]
        self._raw_env = find_raw_env(inner)
        if self._raw_env is None and (self._pred_shape_alpha > 0.0 or self._prey_shape_alpha > 0.0):
            # Shaping needs entity positions; failing loudly beats a run that
            # silently trains on the unshaped reward.
            raise RuntimeError(
                "_CoevBothSidesAdapter: reward shaping requested but the raw "
                "aquarium env was not found in the wrapper chain"
            )
        self._prev_min_prey_dists: Dict[str, float] = {}
        self._prev_min_pred_dists: Dict[str, float] = {}
        self.shaping_totals: Dict[str, float] = {"predator": 0.0, "prey": 0.0}
        pred_v = float(getattr(self._raw_env, "predator_max_velocity", 18.0) or 18.0)
        prey_v = float(getattr(self._raw_env, "prey_max_velocity", 12.0) or 12.0)
        self._shape_progress_cap = 1.5 * (pred_v + prey_v)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(self, seed=None, options=None):
        result = self.inner.reset(seed=seed, options=options)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))
        self.shaping_totals = {"predator": 0.0, "prey": 0.0}
        if self._pred_shape_alpha > 0.0:
            self._prev_min_prey_dists = self._nearest_prey_distances()
        if self._prey_shape_alpha > 0.0:
            self._prev_min_pred_dists = self._nearest_pred_distances()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.inner.step(actions)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))

        # Shaping errors are *not* swallowed: a silently unshaped run is
        # indistinguishable from a shaped one that failed to learn.
        if self._pred_shape_alpha > 0.0:
            curr = self._nearest_prey_distances()
            self._apply_potential_shaping(
                rewards, curr, self._prev_min_prey_dists,
                side="predator", alpha=self._pred_shape_alpha, closing=True,
            )
            self._prev_min_prey_dists = curr

        if self._prey_shape_alpha > 0.0:
            curr = self._nearest_pred_distances()
            self._apply_potential_shaping(
                rewards, curr, self._prev_min_pred_dists,
                side="prey", alpha=self._prey_shape_alpha, closing=False,
            )
            self._prev_min_pred_dists = curr

        return obs, rewards, terminations, truncations, infos

    def _apply_potential_shaping(
        self,
        rewards: Dict[str, float],
        curr: Dict[str, float],
        prev: Dict[str, float],
        *,
        side: str,
        alpha: float,
        closing: bool,
    ) -> None:
        """Add ``alpha * (Φ(s') − Φ(s))`` to one side's rewards, in place."""
        shaped = potential_shaping(
            curr, prev, alpha=alpha, closing=closing, cap=self._shape_progress_cap
        )
        for agent_id, bonus in shaped.items():
            rewards[agent_id] = rewards.get(agent_id, 0.0) + bonus
            self.shaping_totals[side] += bonus

    def render(self, *args, **kwargs):
        return self.inner.render(*args, **kwargs)

    def close(self) -> None:
        self.inner.close()

    # Internal helpers

    def _nearest_prey_distances(self) -> Dict[str, float]:
        """Nearest-prey torus distance per live predator, keyed by identity."""
        raw = self._raw_env
        if raw is None:
            return {}
        return nearest_opponent_distances(
            self._predator_agents,
            entities_by_id(getattr(raw, "predators", [])),
            getattr(raw, "prey", []),
            getattr(raw, "torus", None),
        )

    def _nearest_pred_distances(self) -> Dict[str, float]:
        """Nearest-predator torus distance per live prey, keyed by identity."""
        raw = self._raw_env
        if raw is None:
            return {}
        return nearest_opponent_distances(
            self._prey_agents,
            entities_by_id(getattr(raw, "prey", [])),
            getattr(raw, "predators", []),
            getattr(raw, "torus", None),
        )


# ===========================================================================
# Env factory + registration
# ===========================================================================


class _CoevAquariumFactory:
    """Picklable factory building the raw both-sides coevolution env."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        pred_shape_alpha: float = 0.0,
        prey_shape_alpha: float = 0.0,
    ) -> None:
        self.env_config = dict(env_config)
        self.pred_shape_alpha = float(pred_shape_alpha)
        self.prey_shape_alpha = float(prey_shape_alpha)

    def __call__(self, render_mode: Optional[str] = None, **kwargs) -> _CoevBothSidesAdapter:
        from marl_aquarium import aquarium_v0  # optional dep — import locally

        cfg = {**self.env_config, **kwargs}
        if render_mode is None:
            render_mode = "rgb_array"
        inner = aquarium_v0.parallel_env(render_mode=render_mode, **cfg)
        _patch_aquarium_wrappers(inner)
        return _CoevBothSidesAdapter(
            inner,
            predator_observe_count=int(cfg["predator_observe_count"]),
            prey_observe_count=int(cfg["prey_observe_count"]),
            pred_shape_alpha=self.pred_shape_alpha,
            prey_shape_alpha=self.prey_shape_alpha,
        )


def _register_envs() -> None:
    register_env(
        ENV_NAME_COEV,
        _CoevAquariumFactory(
            env_config=ENV_CONFIG,
            pred_shape_alpha=PRED_REWARD_SHAPE_ALPHA,
            prey_shape_alpha=PREY_REWARD_SHAPE_ALPHA,
        ),
        default_episode_len=int(ENV_CONFIG["max_time_steps"]),
        continuous_default=bool(ENV_CONFIG.get("continuous_actions", False)),
        notes="Both predators and prey exposed as learnable agents for CoEvolutionTrainer.",
        tags={"aquarium", "coevolution", "competitive"},
        api="parallel",
        wrappers=(_coev_feature_wrapper,),
        overwrite=True,
    )


# ===========================================================================
# Side builders
# ===========================================================================


def _make_ea_side(name: str, role_prefix: str, seed: int) -> Side:
    del seed
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="ea",
        optimizer_config={
            "hidden_size": XC.EA_PARAMS["hidden_size"],
            "population_size": XC.EA_PARAMS["population_size"],
            "crossover_rate": XC.EA_PARAMS["cxpb"],
            "mutation_rate": XC.EA_PARAMS["mutation_rate"],
            "tournament_size": XC.EA_PARAMS["tournament_size"],
            "crossover_alpha": XC.EA_PARAMS["crossover_alpha"],
            "mutation_sigma": XC.EA_PARAMS["mut_sigma"],
            "mutation_indpb": XC.EA_PARAMS["mut_indpb"],
            "weight_range": (-1.0, 1.0),
            "activation": XC.EVOLVED_ACTIVATION,
            "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_cma_side(name: str, role_prefix: str, seed: int) -> Side:
    del seed
    cma_params = XC.params_for("cma", ACTION_MODE)
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="cma",
        optimizer_config={
            "hidden_size": cma_params["hidden_size"],
            "population_size": cma_params["population_size"],
            "sigma0": cma_params["sigma0"],
            "activation": XC.EVOLVED_ACTIVATION,
            "output_activation": XC.EVOLVED_OUTPUT_ACTIVATION,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_pso_side(name: str, role_prefix: str, seed: int) -> Side:
    pso_params = XC.params_for("pso", ACTION_MODE)
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="pso",
        optimizer_config={
            "hidden_size": pso_params["hidden_size"],
            "population_size": pso_params["population_size"],
            "phi1": pso_params["phi1"],
            "phi2": pso_params["phi2"],
            "weight": pso_params["weight"],
            "smin": pso_params["smin"],
            "smax": pso_params["smax"],
            "activation": XC.EVOLVED_ACTIVATION,
            "weight_range": (-1.0, 1.0),
            "seed": seed,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_hyperneat_side(name: str, role_prefix: str, seed: int) -> Side:
    hn_params = XC.params_for("hyperneat", ACTION_MODE)
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="hyperneat",
        optimizer_config={
            "pop_size": hn_params["pop_size"],
            "substrate_hidden_layers": hn_params["substrate_hidden_layers"],
            # Threaded through so HyperNEAT decodes like the rest of the
            # evolved family (tanh phenotype + softmax head); leaving them
            # unset silently gave it a sigmoid phenotype and an argmax head.
            "phenotype_activation": hn_params["phenotype_activation"],
            "stochastic_head": hn_params["stochastic_head"],
            "head_temperature": hn_params["head_temperature"],
            "seed": seed,
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_qdax_side(name: str, role_prefix: str, seed: int) -> Side:
    """QDax MAP-Elites side with the calibrated behavioural descriptor.

    The position extractor reads true entity positions each step
    (normalised to [0, 1]²) and the descriptor stretches the measured
    achievable band over the archive axes — without both, every genotype
    lands in the same cell and MAP-Elites degenerates to a (1+batch) ES.
    """
    role = "prey" if role_prefix == "prey" else "predator"
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="qdax_map_elites",
        optimizer_config={
            "hidden_sizes": (XC.QDAX_PARAMS["hidden_size"],),
            "activation": XC.EVOLVED_ACTIVATION,
            "batch_size": XC.QDAX_PARAMS["batch_size"],
            "grid_shape": XC.QDAX_PARAMS["grid_shape"],
            "iso_sigma": XC.QDAX_PARAMS["iso_sigma"],
            "line_sigma": XC.QDAX_PARAMS["line_sigma"],
            "episode_length": int(ENV_CONFIG["max_time_steps"]),
            "descriptor_fn": XC.make_two_team_descriptor(role, ENV_CONFIG),
            "position_extractor": XC.make_position_extractor(role, ENV_CONFIG),
            "seed": seed,
            "pool_mode": XC.QDAX_PARAMS["pool_mode"],
        },
        hof=RingHallOfFame(capacity=HOF_CAPACITY),
        use_archive_in_opponent_pool=True,
        archive_mix=ARCHIVE_MIX,
    )


def _make_matd3_side(name: str, role_prefix: str, seed: int) -> Side:
    """MATD3 evolutionary-HPO side (:class:`RLEvolveHPOSideState`).

    The campaign's MATD3 slot runs the population-HPO variant in every
    regime; here it evolves an AgileRL MATD3 population against the live
    coevolutionary opponents, with persistent per-slot replay buffers.
    """
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="rl_hpo",
        optimizer_config={
            "algorithm": "matd3_hpo",
            "population_size": XC.MATD3_HPO_PARAMS["population_size"],
            # Coevolution generations are shorter than standalone HPO.
            "evo_every_n_steps": 20_000,
            "device": DEVICE,
            "seed": seed,
            # Only constructor knobs may go here (RLEvolveHPOSideState forwards
            # agent_ctor_kwargs verbatim; memory_size/learning_delay are not
            # accepted by the AgileRL ctor).
            "agent_ctor_kwargs": {
                **XC.rl_exploration_kwargs(ACTION_MODE),
                "batch_size": XC.RL_COEV_PARAMS["batch_size"],
                "lr_actor": XC.MATD3_PARAMS["lr_actor"],
                "lr_critic": XC.MATD3_PARAMS["lr_critic"],
                "gamma": XC.MATD3_PARAMS["gamma"],
            },
            "net_config": XC.MATD3_PARAMS["net_config"],
            "hp_config": {
                "lr_actor": [1e-5, 1e-3],
                "lr_critic": [1e-4, 1e-2],
            },
        },
    )


SIDE_BUILDERS = {
    "ea": _make_ea_side,
    "cma": _make_cma_side,
    "pso": _make_pso_side,
    "qdax": _make_qdax_side,
    "hyperneat": _make_hyperneat_side,
    "matd3": _make_matd3_side,
}


# ===========================================================================
# Scripted-APF benchmark
# ===========================================================================


class _ShapingAccountant:
    """Replays the training shaping term over an *unshaped* probe episode.

    The benchmark scores pure env reward, but choosing alpha needs the shaped
    and unshaped halves of the same trajectory — the previous schema recorded
    only their sum, so reward misalignment and plain disengagement could not be
    told apart.  This recomputes exactly what :class:`_CoevBothSidesAdapter`
    would have injected, without perturbing the reward the probe reports.
    """

    def __init__(self, raw_env, role: str, alpha: float, cap: float) -> None:
        self._raw = raw_env
        self._alpha = float(alpha)
        self._cap = float(cap)
        self._closing = role == "predator"
        self._observer_attr = "predators" if self._closing else "prey"
        self._target_attr = "prey" if self._closing else "predators"
        self._prev: Dict[str, float] = {}
        self.total: float = 0.0

    def _distances(self) -> Dict[str, float]:
        if self._raw is None or self._alpha <= 0.0:
            return {}
        observers = entities_by_id(getattr(self._raw, self._observer_attr, []))
        return nearest_opponent_distances(
            list(observers),
            observers,
            getattr(self._raw, self._target_attr, []),
            getattr(self._raw, "torus", None),
        )

    def reset(self) -> None:
        """Latch the episode-start potential and zero the running total."""
        self._prev = self._distances()
        self.total = 0.0

    def step(self) -> None:
        """Accumulate one step of shaping after the env has advanced."""
        if self._alpha <= 0.0:
            return
        curr = self._distances()
        self.total += sum(
            potential_shaping(
                curr, self._prev,
                alpha=self._alpha, closing=self._closing, cap=self._cap,
            ).values()
        )
        self._prev = curr


def _run_scripted_benchmark(
    policy,
    role: str,
    env_cfg: Dict[str, Any],
    n_episodes: int,
    seed: int = 0,
) -> Dict[str, float]:
    """Episode stats of *policy* against a fixed scripted APF opponent.

    ``role="predator"``: policy drives predators vs APF prey;
    ``role="prey"``: policy drives prey vs APF predators.  The reported
    ``reward`` is pure env reward — no shaping — while ``shaping`` records, on
    the same trajectory, what the configured alpha *would* have added, so the
    two halves of training fitness can be compared without conflating them.
    ``catches`` is the episode-max of the wrapper's cumulative catch counter
    (``prey_caught_total`` for the predator role, ``prey_lost_total`` for the
    prey role) — one physical quantity, so the two sides share a column: high
    is good for predators, bad for prey.
    """
    from marl_aquarium import aquarium_v0

    info_key = "prey_caught_total" if role == "predator" else "prey_lost_total"
    shape_alpha = (
        PRED_REWARD_SHAPE_ALPHA if role == "predator" else PREY_REWARD_SHAPE_ALPHA
    )
    # Same clip the training adapter applies to the raw distance delta.
    shape_cap = 1.5 * (
        float(env_cfg.get("predator_max_velocity", 18.0))
        + float(env_cfg.get("prey_max_velocity", 12.0))
    )
    rewards: List[float] = []
    catches: List[int] = []
    shapings: List[float] = []
    for ep in range(n_episodes):
        inner = aquarium_v0.parallel_env(render_mode="rgb_array", **env_cfg)
        _patch_aquarium_wrappers(inner)

        if role == "predator":
            bench_env = AquariumScriptedPreyWrapper(
                inner,
                predator_observe_count=env_cfg["predator_observe_count"],
                prey_observe_count=env_cfg["prey_observe_count"],
                action_count=env_cfg["action_count"],
                predator_reward=env_cfg["predator_reward"],
                apf_config=XC.APF_EVADER_CONFIG,
            )
        else:
            bench_env = AquariumScriptedPredatorWrapper(
                inner,
                predator_observe_count=env_cfg["predator_observe_count"],
                prey_observe_count=env_cfg["prey_observe_count"],
                action_count=env_cfg["action_count"],
                apf_config=XC.APF_PURSUER_CONFIG,
            )

        # Same observation transform as coevolution training.
        bench_env = AquariumObservationFeatureWrapper(bench_env, append_agent_id=False)
        accountant = _ShapingAccountant(
            find_raw_env(inner), role, shape_alpha, shape_cap
        )
        obs, _ = bench_env.reset(seed=seed + ep)
        accountant.reset()
        # RL policy wrappers dispatch actions round-robin through an agent
        # roster; reset the rotation at every episode start.
        if hasattr(policy, "reset"):
            policy.reset()
        ep_reward = 0.0
        ep_catches = 0
        while bench_env.agents:
            actions = {}
            for agent in bench_env.agents:
                if agent in obs:
                    try:
                        actions[agent] = policy.get_action(
                            obs[agent], deterministic=True
                        )
                    except Exception:
                        actions[agent] = _fallback_action()
            if not actions:
                break
            obs, rews, _, _, infos = bench_env.step(actions)
            ep_reward += sum(rews.values())
            accountant.step()
            for info in (infos or {}).values():
                if isinstance(info, dict) and info.get(info_key) is not None:
                    try:
                        ep_catches = max(ep_catches, int(info[info_key]))
                    except (TypeError, ValueError):
                        pass
        # Deliberately NOT closed: marl-aquarium's raw close() calls the
        # process-global pygame.quit(), which permanently kills rendering in
        # the trainer's cached recording env (videos silently stop after the
        # first benchmark).  The bench env never renders, so it holds no
        # pygame resources and plain GC is enough.
        rewards.append(ep_reward)
        catches.append(ep_catches)
        shapings.append(accountant.total)

    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "catches_mean": float(np.mean(catches)) if catches else 0.0,
        "catches_std": float(np.std(catches)) if catches else 0.0,
        "shaping_mean": float(np.mean(shapings)) if shapings else 0.0,
        "shaping_std": float(np.std(shapings)) if shapings else 0.0,
        "n_episodes": len(rewards),
    }


class _ProbeCadence:
    """Decides when the scripted-APF benchmark fires.

    Generations are not a fixed amount of experience under
    ``side_pacing="equal_steps"`` — a cheap side runs many more of them — so a
    generation-indexed probe fires at a pairing-dependent rate and can cost
    more wall clock than the training it measures.  Cadence is therefore in
    consumed env steps whenever the trainer reports them, with the generation
    stride as the fallback for runs that do not.

    Args:
        every_n: Generation stride; ``<= 0`` disables the fallback.
        every_n_steps: Consumed-step stride; ``<= 0`` disables step cadence.
    """

    def __init__(self, every_n: int, every_n_steps: int) -> None:
        self.every_n = int(every_n)
        self.every_n_steps = int(every_n_steps)
        self._last_bucket = -1

    @property
    def enabled(self) -> bool:
        return self.every_n > 0 or self.every_n_steps > 0

    def due(self, gen: int, consumed: Optional[int]) -> bool:
        """Whether a probe should run now, advancing the cadence state."""
        if self.every_n_steps > 0 and consumed is not None:
            bucket = int(consumed) // self.every_n_steps
            if bucket <= self._last_bucket:
                return False
            self._last_bucket = bucket
            return True
        return self.every_n > 0 and gen % self.every_n == 0


class _ScriptedBenchmark:
    """Each side's champion vs a fixed scripted APF opponent.

    The only absolute progress signal in a coevolution run: training fitness is
    measured against co-evolved opponents, so it moves with the opponent and
    lives on a different scale per algorithm and pairing.  This benchmark plays
    the noise-free champion against a frozen scripted opponent on pure env
    reward, which is comparable across every run in the campaign.

    It serves two cadences from one place:

    * :meth:`callback` — the periodic probe during training, on consumed env
      steps (falling back to a generation stride).  Generations are not a fixed
      amount of experience under ``side_pacing="equal_steps"``, so a
      generation-indexed probe would fire at a pairing-dependent rate.
    * :meth:`final_eval` — one higher-precision evaluation at the end of the
      run, wired as ``CoEvolutionConfig.final_eval_fn``.  It writes the
      end-of-run probe row (so every run has a probe at its final consumed
      count, which the generation gate alone never guaranteed) *and* returns
      the summary scalars, which are the run's only comparable final score.

    Probe rows carry a superset of the vs-APF probe schema, so the UI/rliable
    curves work across regimes; the ``probe_shaping_*`` /
    ``probe_shaped_reward_mean`` columns split what training fitness would have
    been on those same episodes into its env and shaping halves.  Episode seeds
    are fixed per run (common random numbers across checkpoints).
    """

    def __init__(
        self,
        pred_side_name: str,
        prey_side_name: str,
        env_cfg: Dict[str, Any],
        n_episodes: int,
        every_n: int,
        probe_seed: int = 77_000,
        every_n_steps: int = 0,
        final_episodes: int = 0,
    ) -> None:
        self.roles = ((pred_side_name, "predator"), (prey_side_name, "prey"))
        self.env_cfg = env_cfg
        self.n_episodes = int(n_episodes)
        self.final_episodes = int(final_episodes or n_episodes)
        self.probe_seed = int(probe_seed)
        self.cadence = _ProbeCadence(every_n, every_n_steps)

    def callback(
        self,
        gen: int,
        states: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generation callback: probe both champions when the cadence is due."""
        ctx = context or {}
        consumed = ctx.get("env_steps_consumed")
        if not self.cadence.due(gen, consumed):
            return
        self._benchmark_all(
            states,
            logger=ctx.get("logger"),
            gen=gen,
            consumed=consumed,
            n_episodes=self.n_episodes,
            tag=f"gen {gen:4d}",
        )

    def final_eval(
        self, states: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """End-of-run champion evaluation; also lands as the final probe row."""
        results = self._benchmark_all(
            states,
            logger=context.get("logger"),
            gen=int(context.get("num_generations") or 0),
            consumed=context.get("env_steps_consumed"),
            n_episodes=self.final_episodes,
            tag="final",
        )
        summary: Dict[str, Any] = {"episodes": self.final_episodes}
        for side_name, bench in results.items():
            for key in (
                "reward_mean", "reward_std",
                "catches_mean", "catches_std",
                "shaping_mean",
            ):
                summary[f"{side_name}_{key}"] = bench[key]
            summary[f"{side_name}_shaped_reward_mean"] = (
                bench["reward_mean"] + bench["shaping_mean"]
            )
        return summary

    def _benchmark_all(
        self,
        states: Dict[str, Any],
        *,
        logger: Any,
        gen: int,
        consumed: Optional[int],
        n_episodes: int,
        tag: str,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark every side, print, and write one probe row each."""
        from deap_pettingzoo.baselines.coevolution.rl_side import RLSideState

        results: Dict[str, Dict[str, float]] = {}
        for side_name, role in self.roles:
            state = states.get(side_name)
            if state is None:
                continue
            try:
                bench = _run_scripted_benchmark(
                    state.best_policy(None), role, self.env_cfg,
                    n_episodes, seed=self.probe_seed,
                )
            except Exception as exc:
                print(f"[Bench] {tag}  {side_name} benchmark failed: {exc}", flush=True)
                continue
            results[side_name] = bench
            scripted_role = "prey" if role == "predator" else "predator"
            kind = "RL" if isinstance(state, RLSideState) else "EA"
            print(
                f"[Bench] {tag}  {side_name} ({kind}) vs "
                f"APF-scripted {scripted_role}: {bench['reward_mean']:.2f}  "
                f"catches {bench['catches_mean']:.2f}  "
                f"shaping {bench['shaping_mean']:+.2f}  "
                f"({bench['n_episodes']} eps)",
                flush=True,
            )
            if logger is not None:
                self._write_probe_row(logger, side_name, gen, consumed, bench)
        return results

    @staticmethod
    def _write_probe_row(
        logger: Any,
        side_name: str,
        gen: int,
        consumed: Optional[int],
        bench: Dict[str, float],
    ) -> None:
        try:
            logger.row("probe", {
                "side": side_name,
                "iteration": int(gen),
                "env_steps": int(consumed) if consumed is not None else 0,
                "probe_reward_mean": bench["reward_mean"],
                "probe_reward_std": bench["reward_std"],
                "probe_episodes": bench["n_episodes"],
                "probe_catches_mean": bench["catches_mean"],
                "probe_catches_std": bench["catches_std"],
                # Unshaped (env) and shaping halves of what training fitness
                # would have been on these same episodes — separate columns so
                # the realised catch share is measurable rather than inferred.
                "probe_shaping_mean": bench["shaping_mean"],
                "probe_shaping_std": bench["shaping_std"],
                "probe_shaped_reward_mean": (
                    bench["reward_mean"] + bench["shaping_mean"]
                ),
            })
        except Exception as exc:
            print(f"[Bench] probe row for {side_name} failed: {exc}", flush=True)


# ===========================================================================
# Core coevolution runner
# ===========================================================================


def _run_coev(
    pred_side: Side,
    prey_side: Side,
    seed: int,
    run_dir: Path,
    label: str,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one coevolution training run and return a result dict."""
    from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig

    benchmark = _ScriptedBenchmark(
        pred_side.name,
        prey_side.name,
        ENV_CONFIG,
        BENCHMARK_N_EPISODES,
        BENCHMARK_EVERY_N,
        probe_seed=77_000 + seed,
        every_n_steps=BENCHMARK_EVERY_N_STEPS,
        final_episodes=FINAL_EVAL_EPISODES,
    )

    # Consumed-step budgeting keeps the aggregate real env-tick pool equal
    # across pairings (early-terminating episodes no longer under-fund the
    # run).  QDax sides participate too: they report real per-generation
    # env ticks into the shared consumed pool, and the overshoot bound is
    # the same finish-generation-then-stop bound as every other side type
    # (one generation's pooled cost, including the QDax batch — negligible
    # at the suite budgets).

    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[pred_side, prey_side],
        schedule="synchronous",
        k_opponents=K_OPPONENTS,
        aggregation="mean",
        reporting_side=pred_side.name,
        max_steps_per_episode=int(ENV_CONFIG["max_time_steps"]),
        seed=seed,
        # num_workers applies to RL pairings too: EA/CMA sides evaluate
        # against frozen CPU snapshots of the RL policy in parallel; only
        # on-policy rollout collection itself stays serial.
        num_workers=N_WORKERS,
        parallel_backend="process",
        # Per-side budget with the "freeze" policy: each side gets an equal
        # slice of the consumed-step pool, and the side that exhausts its
        # slice first stops evolving (keeps serving its final pool as the
        # opponent) while the other keeps training to its own slice. This
        # equalises real per-side experience across pairings whose sides have
        # very different per-generation costs (e.g. an EA population vs a
        # single MATD3 learner), instead of letting the cheaper-to-evaluate
        # side drain a shared pool. Requires budget_on_consumed=True below.
        budget_split="per_side",
        on_side_exhausted="freeze",
        # Equal-env-step pacing: without it a side steps once per generation
        # regardless of what a generation costs it, so the expensive side
        # drained its slice at ~0.66 of the run and the last third was not
        # co-evolution at all (84% of the ENUCC runs froze a side). Cheap
        # sides now simply take more generations to spend the same slice.
        side_pacing="equal_steps",
        generation_callback=benchmark.callback,
        # Absolute end-of-run score: final_best is a training fitness against
        # co-evolved opponents (84 -> 661 across backends) and cannot be
        # compared across runs. This also guarantees a probe row at the run's
        # final consumed count, which the periodic cadence alone does not.
        final_eval_fn=benchmark.final_eval,
    )
    train_cfg = TrainingConfig(
        env_name=ENV_NAME_COEV,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=int(COEV_TOTAL_STEPS if budget is None else budget),
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        seed=seed,
        # Descriptive only — the logger merges these into the manifest's
        # hyperparams so Browse/Compare can identify the cell, and no trainer
        # ever sees them.
        run_metadata={
            "pairing": label,
            "predator_algo": label.partition("_vs_")[0],
            "prey_algo": label.partition("_vs_")[2],
            "action_mode": ACTION_MODE,
            "k_opponents": K_OPPONENTS,
            "hof_capacity": HOF_CAPACITY,
            "archive_mix": ARCHIVE_MIX,
            "pred_shape_alpha": PRED_REWARD_SHAPE_ALPHA,
            "prey_shape_alpha": PREY_REWARD_SHAPE_ALPHA,
            # Distinguishes these archives from the ENUCC campaign's, whose
            # shaping summed unrelated per-step deltas at the same alphas.
            "shaping_form": "potential",
            "budget_on_consumed": XC.BUDGET_ON_CONSUMED,
            "budget_split": "per_side",
            "on_side_exhausted": "freeze",
            "side_pacing": "equal_steps",
        },
        coevolution_config=coev_cfg,
        max_steps_per_episode=int(ENV_CONFIG["max_time_steps"]),
        env_config=dict(ENV_CONFIG),
        output_dir=run_dir,
        recording=RecordingConfig(
            enabled=RECORD_VIDEOS,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            mode="count",
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
        ),
    )
    t0 = time.time()
    result = CoEvolutionTrainer().train(train_cfg)
    elapsed = time.time() - t0
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": elapsed,
        "seed": seed,
        "label": label,
    }


def run_pairing(pairing: str, seed: int, run_dir: Path) -> Dict[str, Any]:
    """Run one ``"<pred_algo>_vs_<prey_algo>"`` coevolution pairing."""
    pred_algo, _, prey_algo = pairing.partition("_vs_")
    for algo in (pred_algo, prey_algo):
        if algo not in SIDE_BUILDERS:
            raise ValueError(
                f"Unknown side type {algo!r} in pairing {pairing!r}; "
                f"available: {sorted(SIDE_BUILDERS)}"
            )
    _section(f"{pred_algo} vs {prey_algo} | seed={seed}")
    return _run_coev(
        pred_side=SIDE_BUILDERS[pred_algo]("predators", "predator", seed),
        prey_side=SIDE_BUILDERS[prey_algo]("prey", "prey", seed),
        seed=seed,
        run_dir=run_dir,
        label=pairing,
    )


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    _register_envs()
    _section("Aquarium competitive coevolution")
    print(f"  ENV       : {ENV_NAME_COEV}  ({ACTION_MODE})")
    print(f"  budget    : {COEV_TOTAL_STEPS:,} env_steps × {NUM_SEEDS} seeds × pairings")
    print(f"  enabled   : {[p for p, on in PAIRINGS.items() if on]}")
    print(f"  output    : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for pairing, enabled in PAIRINGS.items():
        if not enabled:
            print(f"[skip] {pairing}")
            continue
        for seed_offset in range(NUM_SEEDS):
            seed = SEED_BASE + seed_offset
            run_dir = OUTPUT_DIR / pairing / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = run_pairing(pairing, seed, run_dir)
                final_eval = result.get("metrics", {}).get("final_eval") or {}
                summary.append({
                    "pairing": pairing,
                    "seed": seed,
                    "training_time_s": result.get("training_time_s", 0.0),
                    "pred_catches": final_eval.get("predators_catches_mean"),
                    "status": "ok",
                })
            except Exception as exc:
                print(f"[ERROR] {pairing} seed={seed}: {exc}")
                traceback.print_exc()
                summary.append({
                    "pairing": pairing,
                    "seed": seed,
                    "training_time_s": 0.0,
                    "pred_catches": None,
                    "status": f"error: {exc}",
                })

    _section("Summary")
    header = (
        f"{'Pairing':<20}  {'Seed':>4}  {'Time (s)':>9}  "
        f"{'Catches':>8}  Status"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        catches = row.get("pred_catches")
        catches_txt = f"{catches:.2f}" if catches is not None else "—"
        print(
            f"{row['pairing']:<20}  {row['seed']:>4}  "
            f"{row['training_time_s']:>9.1f}  {catches_txt:>8}  {row['status']}"
        )


if __name__ == "__main__":
    main()
