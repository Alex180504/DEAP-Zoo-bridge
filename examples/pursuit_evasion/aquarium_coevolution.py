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

Every ``BENCHMARK_EVERY_N`` generations each side's champion also plays a
fixed scripted APF opponent — an absolute progress signal that is immune to
the coevolutionary moving-target effect.
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
BENCHMARK_N_EPISODES: int = XC.BENCHMARK_N_EPISODES

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

    Optional dense progress shaping: predators earn
    ``+alpha * Δ(dist to nearest visible prey) / 200`` per step, prey the
    symmetric flee bonus.  The per-step delta is clipped to one step of
    plausible motion so discontinuities (a caught prey being removed, a
    target leaving the FOV cone) cannot inject a large spurious reward at
    the exact step a catch happens.
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
        self._raw_env = self._find_raw_env(inner)
        self._prev_min_prey_dists: Dict[str, Optional[float]] = {}
        self._prev_min_pred_dists: Dict[str, Optional[float]] = {}
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
        if self._pred_shape_alpha > 0.0:
            self._prev_min_prey_dists = self._compute_min_prey_dists_for_predators()
        if self._prey_shape_alpha > 0.0:
            self._prev_min_pred_dists = self._compute_min_pred_dists_for_prey()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.inner.step(actions)
        self.agents = list(getattr(self.inner, "agents", self.possible_agents))

        if self._pred_shape_alpha > 0.0:
            try:
                curr = self._compute_min_prey_dists_for_predators()
                self._apply_progress_shaping(
                    rewards, curr, self._prev_min_prey_dists,
                    prefix="predator", alpha=self._pred_shape_alpha, closing=True,
                )
                self._prev_min_prey_dists = curr
            except Exception:
                pass

        if self._prey_shape_alpha > 0.0:
            try:
                curr = self._compute_min_pred_dists_for_prey()
                self._apply_progress_shaping(
                    rewards, curr, self._prev_min_pred_dists,
                    prefix="prey", alpha=self._prey_shape_alpha, closing=False,
                )
                self._prev_min_pred_dists = curr
            except Exception:
                pass

        return obs, rewards, terminations, truncations, infos

    def _apply_progress_shaping(
        self, rewards, curr, prev, *, prefix: str, alpha: float, closing: bool
    ) -> None:
        cap = self._shape_progress_cap
        for agent_id in self.possible_agents:
            if not agent_id.startswith(prefix):
                continue
            cur_d = curr.get(agent_id)
            prev_d = prev.get(agent_id)
            if cur_d is None or prev_d is None:
                continue
            progress = (prev_d - cur_d) if closing else (cur_d - prev_d)
            progress = max(-cap, min(cap, progress))
            rewards[agent_id] = rewards.get(agent_id, 0.0) + alpha * (progress / 200.0)

    def render(self, *args, **kwargs):
        return self.inner.render(*args, **kwargs)

    def close(self) -> None:
        self.inner.close()

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _find_raw_env(env):
        """Walk the PettingZoo wrapper chain to find the raw aquarium env."""
        inner = env
        for _ in range(12):
            if hasattr(inner, "predators") and hasattr(inner, "prey"):
                return inner
            advanced = False
            for attr in ("aec_env", "env"):
                if hasattr(inner, attr):
                    inner = getattr(inner, attr)
                    advanced = True
                    break
            if not advanced:
                break
        return None

    def _min_visible_distances(
        self, observers, targets, *, observer_prefix: str,
        view_dist, fov,
    ) -> Dict[str, Optional[float]]:
        """Nearest FOV-visible target torus distance per observer agent."""
        raw = self._raw_env
        torus = getattr(raw, "torus", None)
        if not observers or not targets or torus is None:
            return {}
        fov_enabled = bool(getattr(raw, "fov_enabled", False))
        check = getattr(torus, "check_if_entity_is_in_view_in_torus", None)
        result: Dict[str, Optional[float]] = {}
        observer_agents = [
            a for a in self.possible_agents if a.startswith(observer_prefix)
        ]
        for idx, agent_id in enumerate(observer_agents):
            if idx >= len(observers):
                continue
            entity = observers[idx]
            if fov_enabled and check is not None:
                visible = [t for t in targets if check(entity, t, view_dist, fov)]
            else:
                visible = targets
            if not visible:
                result[agent_id] = None
            else:
                result[agent_id] = min(
                    float(torus.get_distance_in_torus(entity.position, t.position))
                    for t in visible
                )
        return result

    def _compute_min_prey_dists_for_predators(self) -> Dict[str, Optional[float]]:
        raw = self._raw_env
        if raw is None:
            return {}
        return self._min_visible_distances(
            list(getattr(raw, "predators", [])),
            list(getattr(raw, "prey", [])),
            observer_prefix="predator",
            view_dist=getattr(raw, "predator_view_distance", 2000),
            fov=getattr(raw, "predator_fov", 360),
        )

    def _compute_min_pred_dists_for_prey(self) -> Dict[str, Optional[float]]:
        raw = self._raw_env
        if raw is None:
            return {}
        return self._min_visible_distances(
            list(getattr(raw, "prey", [])),
            list(getattr(raw, "predators", [])),
            observer_prefix="prey",
            view_dist=getattr(raw, "prey_view_distance", 2000),
            fov=getattr(raw, "prey_fov", 360),
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


def _run_scripted_benchmark(
    policy,
    role: str,
    env_cfg: Dict[str, Any],
    n_episodes: int,
    seed: int = 0,
) -> Dict[str, float]:
    """Episode stats of *policy* against a fixed scripted APF opponent.

    ``role="predator"``: policy drives predators vs APF prey;
    ``role="prey"``: policy drives prey vs APF predators.  Pure env reward —
    no shaping.  ``catches`` is the episode-max of the wrapper's cumulative
    catch counter (``prey_caught_total`` for the predator role,
    ``prey_lost_total`` for the prey role) — one physical quantity, so the
    two sides share a column: high is good for predators, bad for prey.
    """
    from marl_aquarium import aquarium_v0

    info_key = "prey_caught_total" if role == "predator" else "prey_lost_total"
    rewards: List[float] = []
    catches: List[int] = []
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
        obs, _ = bench_env.reset(seed=seed + ep)
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

    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "catches_mean": float(np.mean(catches)) if catches else 0.0,
        "catches_std": float(np.std(catches)) if catches else 0.0,
        "n_episodes": len(rewards),
    }


def _make_benchmark_callback(
    pred_side_name: str,
    prey_side_name: str,
    env_cfg: Dict[str, Any],
    n_episodes: int,
    every_n: int,
    probe_seed: int = 77_000,
):
    """Generation callback benchmarking each champion vs a scripted APF opponent.

    The benchmark evaluates each side's noise-free champion on pure env
    reward (no shaping) — the only absolute progress signal in a coevolution
    run.  Returns ``None`` when *every_n* <= 0 (disabled).

    The callback takes the trainer's three-argument context form: besides
    printing, each benchmark writes one row per side into the run's
    ``probe.csv`` via ``context["logger"]`` with ``env_steps`` set to the
    pooled consumed-step count — the same probe schema as the vs-APF suites,
    so the UI/rliable curves work across regimes.  Episode seeds are fixed
    per run (common random numbers across checkpoints).
    """
    if every_n <= 0:
        return None

    from deap_pettingzoo.baselines.coevolution.rl_side import RLSideState

    def _callback(
        gen: int,
        states: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if gen % every_n != 0:
            return
        ctx = context or {}
        logger = ctx.get("logger")
        consumed = ctx.get("env_steps_consumed")
        for side_name, role in [
            (pred_side_name, "predator"),
            (prey_side_name, "prey"),
        ]:
            state = states.get(side_name)
            if state is None:
                continue
            try:
                policy = state.best_policy(None)
                bench = _run_scripted_benchmark(
                    policy, role, env_cfg, n_episodes, seed=probe_seed
                )
                scripted_role = "prey" if role == "predator" else "predator"
                kind = "RL" if isinstance(state, RLSideState) else "EA"
                print(
                    f"[Bench] gen {gen:4d}  {side_name} ({kind}) vs "
                    f"APF-scripted {scripted_role}: {bench['reward_mean']:.2f}  "
                    f"catches {bench['catches_mean']:.2f}",
                    flush=True,
                )
                if logger is not None:
                    logger.row("probe", {
                        "side": side_name,
                        "iteration": int(gen),
                        "env_steps": int(consumed) if consumed is not None else 0,
                        "probe_reward_mean": bench["reward_mean"],
                        "probe_reward_std": bench["reward_std"],
                        "probe_episodes": bench["n_episodes"],
                        "probe_catches_mean": bench["catches_mean"],
                        "probe_catches_std": bench["catches_std"],
                    })
            except Exception as exc:
                print(
                    f"[Bench] gen {gen:4d}  {side_name} benchmark failed: {exc}",
                    flush=True,
                )

    return _callback


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

    generation_callback = _make_benchmark_callback(
        pred_side.name,
        prey_side.name,
        ENV_CONFIG,
        BENCHMARK_N_EPISODES,
        BENCHMARK_EVERY_N,
        probe_seed=77_000 + seed,
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
        generation_callback=generation_callback,
    )
    train_cfg = TrainingConfig(
        env_name=ENV_NAME_COEV,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=int(COEV_TOTAL_STEPS if budget is None else budget),
        budget_on_consumed=XC.BUDGET_ON_CONSUMED,
        num_seeds=1,
        seed=seed,
        # Run-manifest hyperparams (the trainer ignores these; the logger's
        # from_config picks them up so Browse/Compare can identify the cell).
        algorithm_params={
            "pairing": label,
            "predator_algo": label.partition("_vs_")[0],
            "prey_algo": label.partition("_vs_")[2],
            "action_mode": ACTION_MODE,
            "k_opponents": K_OPPONENTS,
            "hof_capacity": HOF_CAPACITY,
            "archive_mix": ARCHIVE_MIX,
            "pred_shape_alpha": PRED_REWARD_SHAPE_ALPHA,
            "prey_shape_alpha": PREY_REWARD_SHAPE_ALPHA,
            "budget_on_consumed": XC.BUDGET_ON_CONSUMED,
            "budget_split": "per_side",
            "on_side_exhausted": "freeze",
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
                summary.append({
                    "pairing": pairing,
                    "seed": seed,
                    "training_time_s": result.get("training_time_s", 0.0),
                    "status": "ok",
                })
            except Exception as exc:
                print(f"[ERROR] {pairing} seed={seed}: {exc}")
                traceback.print_exc()
                summary.append({
                    "pairing": pairing,
                    "seed": seed,
                    "training_time_s": 0.0,
                    "status": f"error: {exc}",
                })

    _section("Summary")
    header = f"{'Pairing':<20}  {'Seed':>4}  {'Time (s)':>9}  Status"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['pairing']:<20}  {row['seed']:>4}  "
            f"{row['training_time_s']:>9.1f}  {row['status']}"
        )


if __name__ == "__main__":
    main()
