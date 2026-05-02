"""Pursuit-evasion in mpe2 simple_tag: EA vs CMA-ES vs MAPPO.

Mirrors the structure of ``aquarium_ea_cma_mappo.py`` but targets the
:mod:`mpe2.simple_tag_v3` environment via
:class:`~deap_pettingzoo.contrib.pursuit_evasion.PursuitEvasionEnv`,
without any aquarium-specific workarounds (no SDL/pygame setup, no
torus-arena drift attractor, no aquarium scripted-prey wrapper).

Three pursuer-team training methods are compared against fixed scripted
APF prey:

* **EA**     — DEAP genetic algorithm with CRN parallel evaluation.
* **CMA-ES** — DEAP ``cma.Strategy`` via deap-bridge's CMA helpers.
* **MAPPO**  — AgileRL :class:`GroupedIPPOTrainer` (prefix-shared IPPO).

All three trainers share the same registered env name so the prey-control
contract is identical across algorithms.  The env is a thin
:class:`ScriptedPreyWrapper` over :class:`PursuitEvasionEnv` that hides
good agents from the public API and steps them internally with
:class:`APFEvaderPolicy`.

Configuration is exposed at the top of the file via plain module-level
constants — set ``RUN_EA``, ``RUN_CMAES``, ``RUN_MAPPO`` to toggle the
three sections independently; ``PROBE_ENV=True`` runs a one-shot env
sanity check (random-action episode + obs/reward/info dump) and exits.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from deap import tools

from deap_pettingzoo.contrib.pursuit_evasion import (
    APFConfig,
    APFEvaderPolicy,
    PursuitEvasionConfig,
    PursuitEvasionEnv,
)
from deap_pettingzoo.environments.registry import build_env, is_registered, register_env
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
from deap_pettingzoo.policies.action_heads import ActionHead, DiscreteHead
from deap_pettingzoo.utils.logging import ExperimentLogger
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.utils.parallel import make_executor

# Allow running as a plain script *or* as a module (-m).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ===========================================================================
# Top-level config (edit these — no CLI parsing)
# ===========================================================================

# ---- Toggles --------------------------------------------------------------

RUN_EA: bool = True
RUN_CMAES: bool = True
RUN_MAPPO: bool = True
PROBE_ENV: bool = False
RUN_FINAL_ANALYSIS: bool = True

# ---- Reproducibility / hardware ------------------------------------------

SEED_BASE: int = 0
NUM_SEEDS: int = 3
DEVICE: str = "cuda"
N_WORKERS: int = 0   # 0 = os.cpu_count(); 1 = sequential

# ---- Shared budget --------------------------------------------------------

TOTAL_ENV_STEPS: int = 2_000_000
EVAL_EPISODES: int = 30

# ---- mpe2 simple_tag env config (4v4 with 3 obstacles) -------------------

ENV_NAME: str = "simple_tag_pursuit_evasion_scripted_v0"

PE_CONFIG = PursuitEvasionConfig(
    num_adversaries=4,
    num_good=4,
    num_obstacles=3,
    # 300 (was 200) so the team has time to corner prey against walls.
    # At max_speed=1.1 vs 1.0 the relative-velocity gap is only 0.1 u/s;
    # cornering through soft-boundary deflection takes ~150 steps from
    # mid-arena spawns.  200 was below the cornering horizon, which is
    # one reason champ_catches plateaued at 0-2.
    max_cycles=300,
    # Equal-ability is ON; pursuers are then bumped to max_speed=1.1 in
    # ScriptedPreyWrapper.reset (POST_RESET_PURSUER_SPEED_ADVANTAGE).
    # Rationale (the dominant cause of the gyration we observed): with
    # the simple_tag default asymmetry (prey 1.3 vs pursuer 1.0) a lone
    # pursuer is *physically incapable* of catching a fleeing APF prey
    # on open ground.  All three trainers (EA, CMA-ES, MAPPO) converged
    # the distance-shaping reward but champ_catches stayed at 0-2 — the
    # policy plateaued at the equilibrium follow-distance where its
    # speed component along delta matches prey speed.  Inverting the
    # asymmetry (pursuer 1.1 > prey 1.0) makes the chase tractable from
    # scratch; once consistent catches are achieved the user can lower
    # the advantage to study harder regimes.
    equal_ability=True,
    equal_max_speed=1.0,
    equal_accel=3.0,
    # Full observability — see explanatory comment in ea_vs_apf.py: under
    # K-nearest PO, agent type is not recoverable from obs-vector
    # position, which kills the EA's ability to learn pursuit.
    num_agent_neighbors=None,
    num_landmark_neighbors=None,
    # Soft boundary (also from ea_vs_apf.py — the default sharpness
    # produces 1e8+ outliers that drown the +10 collision signal).
    boundary_threshold=1.0,
    boundary_scale=0.05,
    boundary_sharpness=0.5,
    terminate_on_success=True,
)

APF_CONFIG = APFConfig(
    k_pursuer=0.3,         # weakened evader to leave room for early EA learning
    k_landmark=0.3,
    k_boundary=2.0,
    boundary_threshold=0.9,
)

# Per-step distance-based reward shaping for adversaries on top of the
# sparse catch bonus.  Set False for the pure-sparse baseline.
REWARD_SHAPING: bool = True

# Pursuer max-speed advantage applied post-reset (overrides simple_tag's
# defaults).  See PE_CONFIG comment above for rationale.  Set to 1.0 to
# disable the advantage.
POST_RESET_PURSUER_SPEED: float = 1.1
POST_RESET_PREY_SPEED: float = 1.0

# Close-range catch attractor (added to shaping).  Standard -1.0*dist
# shaping has a flat slope: gradient at dist=2.0 equals gradient at
# dist=0.2 (just outside catch radius ~0.15), so the policy has no
# pressure to push through the *last* 0.5 units against fast-reacting
# APF prey.  A Gaussian bump localised at small dist provides the
# missing pressure.  Magnitudes: peak +CLOSE_RANGE_BONUS at dist=0,
# half-max around dist=sqrt(CLOSE_RANGE_SIGMA2 * ln 2) ≈ 0.12 — i.e.
# it kicks in only inside the catch annulus.
CLOSE_RANGE_BONUS: float = 5.0
CLOSE_RANGE_SIGMA2: float = 0.02

# Mild anti-clustering term: small repulsion between adversaries when
# all-too-close, to discourage the homogeneous policy from collapsing
# every pursuer onto the same prey (the EA "gyration in centre" mode
# and the CMA "cluster on the right wall" mode are both expressions of
# this collapse — every adversary's egocentric obs leads to the same
# action).  Active only when teammate dist < ANTICLUSTER_RADIUS.
ANTICLUSTER_WEIGHT: float = 0.3
ANTICLUSTER_RADIUS: float = 0.4

# Prey-absorption mechanics (the fix for the centre-gyration attractor).
#
# When an adversary collides with a prey, the prey is *consumed*: it is
# frozen (movable=False, collide=False), teleported out of the arena so
# it no longer appears in observations or distance-shaping gradients,
# and the adversary team receives ``CATCH_BONUS``.  The episode
# terminates with an additional ``TERMINAL_CLEAR_BONUS`` once all prey
# are consumed.  CATCH_BONUS is set high (200) relative to the
# *integrated* dist-shaping difference between "stand still in centre"
# (avg dist ~ 1.5) and "actively chase" (avg dist ~ 0.7) over an
# episode — ~ 4 agents × (1.5-0.7) × 0.1 × 300 steps = 96 reward.  Without
# this margin, the EA's local optimum is to sit near the prey-spawn
# centroid and accept zero catches (the gyration mode).
CATCH_BONUS: float = 200.0
TERMINAL_CLEAR_BONUS: float = 800.0

# Distance-shaping coefficient.  Was 1.0; reduced to 0.1 so the catch
# bonus is the dominant signal rather than "stand slightly closer".
DIST_SHAPING_WEIGHT: float = 0.1
VTOWARD_SHAPING_WEIGHT: float = 0.05

# Hard arena radius (terminates the episode if any adversary flies
# beyond this in either axis).  Without this, runaway policies
# accumulate astronomical soft-boundary penalties (the env's exp-grow
# penalty reaches ~ 10^9/step at |x|=50), poisoning the population mean
# and making EA selection signal noisy.  3.0 is well outside the
# nominal [-1, +1] arena but bounded enough that exp(0.5*(3-1))*0.05 =
# ~0.18 per axis per step — a few hundred reward at most.
HARD_ARENA_RADIUS: float = 3.0

# ---- EA config -----------------------------------------------------------

EA_POPULATION_SIZE: int = 32
EA_HIDDEN_SIZE: int = 16
EA_EVAL_EPISODES_TRAIN: int = 8
EA_ELITE_SIZE: int = 3
EA_MUT_SIGMA: float = 0.3
EA_MUT_INDPB: float = 0.3
EA_CXPB: float = 0.7

# ---- Oracle warm-start config (ported from ea_vs_apf.py) ------------------
# Without an oracle warm-start, neither EA (32 pop x 100 gens) nor CMA-ES
# (24 lambda x N gens) can find pursuit behaviour from random init in a
# 600+ weight network: the basin of attraction is too small.  Half the
# initial EA population is seeded as small perturbations of a hand-coded
# "chase one randomly chosen prey" controller; CMA-ES' centroid is set
# to the same controller (un-perturbed).  The EA/CMA then refines.
ORACLE_FRACTION: float = 0.5     # fraction of EA pop seeded with oracle
ORACLE_SEED_SIGMA: float = 0.05  # Gaussian noise added per oracle copy
ORACLE_OUTPUT_GAIN: float = 4.0  # logit gain for cardinal-direction output

# ---- CMA-ES config -------------------------------------------------------

CMA_POPULATION_SIZE: int = 24
CMA_HIDDEN_SIZE: int = 16
CMA_EVAL_EPISODES_TRAIN: int = 8
CMA_SIGMA0: float = 0.3

# ---- MAPPO config (AgileRL) ----------------------------------------------

MAPPO_LEARN_STEP: int = 10_240
MAPPO_BATCH_SIZE: int = 1024
MAPPO_LR: float = 3e-4
MAPPO_GAMMA: float = 0.99
MAPPO_EXTRA: Dict[str, Any] = {
    "ent_coef": 0.005,
    "net_config": {
        "encoder_config": {"hidden_size": [64, 64]},
        "head_config": {"hidden_size": [64]},
    },
}

# ---- Recording / logging --------------------------------------------------

RECORD: bool = True
RECORD_FPS: int = 20
RECORD_FMT: str = "mp4"
RECORD_EVERY_N_EA: int = 5
RECORD_EVERY_N_MAPPO: int = 20
RECORD_MAX_FRAMES: Optional[int] = None
RECORD_MAX_DIM: int = 960
RECORD_SEED: int = 9000

OUTPUT_DIR: Path = Path("results/simple_tag_pursuit_evasion_ea_cma_mappo")


# ===========================================================================
# Reward shaping (optional)
# ===========================================================================


def _maybe_apply_reward_shaping(env: PursuitEvasionEnv) -> None:
    """Patch simple_tag's adversary_reward to add a -1.0 * distance term.

    Mirrors the shaping in ``ea_vs_apf.py``: replaces the scenario's
    ``adversary_reward`` on the *instance* (so other envs are unaffected)
    with a version that gives each adversary a continuous penalty
    proportional to its distance to the nearest prey, plus a small
    velocity-toward-prey bonus to break iso-distance drift orbits.
    """
    if not REWARD_SHAPING:
        return
    import types

    scenario = None
    try:
        scenario = env.env.aec_env.env.env.scenario
    except AttributeError:
        pass
    if scenario is None:
        try:
            scenario = env.env.unwrapped.scenario
        except AttributeError:
            return

    def _shaped_adversary_reward(self, agent, world):
        # NOTE: only *active* (movable) prey contribute to shaping.  When
        # the wrapper consumes a prey it sets movable=False, so the
        # nearest-prey gradient automatically retargets to surviving
        # prey.  The +10 collision term is intentionally absent: the
        # wrapper applies a much larger CATCH_BONUS per consumption to
        # the *team*, which is the only catch signal in this regime.
        rew = 0.0
        good_agents = [a for a in self.good_agents(world) if a.movable]
        if good_agents:
            self_pos = agent.state.p_pos
            self_vel = agent.state.p_vel
            nearest = min(
                good_agents,
                key=lambda a: float(np.sum((self_pos - a.state.p_pos) ** 2)),
            )
            delta = nearest.state.p_pos - self_pos
            dist = float(np.linalg.norm(delta))
            rew -= DIST_SHAPING_WEIGHT * dist
            if dist > 1e-6:
                v_toward = float(np.dot(self_vel, delta)) / dist
                rew += VTOWARD_SHAPING_WEIGHT * v_toward
            # Close-range Gaussian attractor: gives the policy a strong
            # gradient inside the last ~0.3 units, breaking the
            # equilibrium-follow-distance plateau.
            rew += CLOSE_RANGE_BONUS * float(
                np.exp(-(dist * dist) / max(CLOSE_RANGE_SIGMA2, 1e-9))
            )
        # Anti-clustering: penalise being too close to teammate adversaries.
        # Encourages the homogeneous policy to spread across multiple
        # prey rather than collapsing onto one, which is the dominant
        # failure mode under egocentric observations + shared genome.
        adversaries = [a for a in world.agents if a.adversary and a is not agent]
        if adversaries and ANTICLUSTER_WEIGHT > 0.0:
            for other in adversaries:
                d = float(np.linalg.norm(agent.state.p_pos - other.state.p_pos))
                if d < ANTICLUSTER_RADIUS:
                    rew -= ANTICLUSTER_WEIGHT * (ANTICLUSTER_RADIUS - d)
        return rew

    scenario.adversary_reward = types.MethodType(_shaped_adversary_reward, scenario)


# ===========================================================================
# ScriptedPreyWrapper — hides good agents, drives them with APF internally
# ===========================================================================


class ScriptedPreyWrapper:
    """Parallel-API facade over :class:`PursuitEvasionEnv` exposing only adversaries.

    On every :meth:`step`, prey actions are generated internally by per-
    agent :class:`APFEvaderPolicy` instances using the prey's last
    observation; the public API's ``actions`` dict therefore only needs
    adversary entries.  ``possible_agents`` is filtered to adversaries.

    Catches per episode are counted directly via the underlying scenario's
    ``is_collision`` and exposed in ``infos[<agent>]['prey_caught_total']``
    so callers can use a sparse fitness signal alongside team reward.

    The :meth:`reset` returns ``(obs, infos)`` per the modern PettingZoo
    parallel API (required by AgileRL's MAPPO trainer).  The
    :class:`DictResetEnv` shim below adapts this for legacy callers
    (deap-bridge's :class:`MixedPolicyEvaluator` and CRN pmap helper).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "simple_tag_scripted"}

    def __init__(
        self,
        inner: PursuitEvasionEnv,
        apf_config: Optional[APFConfig] = None,
    ) -> None:
        self.inner = inner
        self._apf_config = apf_config or APFConfig()
        self.render_mode = inner.render_mode
        self.possible_agents: List[str] = list(inner.adversary_agents)
        self.agents: List[str] = list(self.possible_agents)
        # Per-prey APF policy + last-obs cache.
        self._scripted_policies: Dict[str, APFEvaderPolicy] = {
            name: APFEvaderPolicy(
                layout=inner.get_layout(name), config=self._apf_config
            )
            for name in inner.good_agents
        }
        self._scripted_last_obs: Dict[str, np.ndarray] = {}
        self.prey_caught: int = 0
        # Names of prey that have been consumed this episode (frozen +
        # hidden); cleared on reset.  Used to skip APF stepping for
        # consumed prey and to detect terminal-clear.
        self._consumed_prey: Set[str] = set()
        # Cached path to the underlying scenario for catch detection.
        self._scenario = self._find_scenario()
        self._world = inner._get_world() if hasattr(inner, "_get_world") else None
        self._world_agents_by_name: Dict[str, Any] = (
            {a.name: a for a in self._world.agents} if self._world is not None else {}
        )

    # ParallelEnv API ------------------------------------------------------

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        observations = self.inner.reset(seed=seed, options=options)
        # The underlying mpe2 world is rebuilt on each reset; refresh
        # our cached references so consume-on-catch mutates the live
        # world rather than a stale one.
        if hasattr(self.inner, "_get_world"):
            try:
                self._world = self.inner._get_world()
                if self._world is not None:
                    self._world_agents_by_name = {
                        a.name: a for a in self._world.agents
                    }
                    # Override per-agent max_speed to give pursuers a
                    # small advantage (see PE_CONFIG comment).  This
                    # is applied AFTER PursuitEvasionEnv's equal_ability
                    # post-reset hook so it takes precedence.  Without
                    # this override the chase is physically unwinnable
                    # for a homogeneous policy and all three trainers
                    # plateau at ~0 catches/episode despite shaping
                    # convergence.
                    for a in self._world.agents:
                        if a.adversary:
                            a.max_speed = POST_RESET_PURSUER_SPEED
                        else:
                            a.max_speed = POST_RESET_PREY_SPEED
            except Exception:
                pass
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if not a.startswith("adversary")
        }
        for pol in self._scripted_policies.values():
            pol.reset()
        self.agents = [a for a in self.inner.adversary_agents if a in observations]
        learner_obs = {a: observations[a] for a in self.agents}
        self.prey_caught = 0
        self._consumed_prey = set()
        learner_infos = {
            a: {"prey_caught_total": 0,
                "prey_remaining": len(self.inner.good_agents)}
            for a in self.agents
        }
        return learner_obs, learner_infos

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        # Build full action dict: learner actions for adversaries, APF
        # for *active* prey, no-op for consumed prey.
        full_actions: Dict[str, Any] = {}
        for agent_id in self.inner.possible_agents:
            if agent_id.startswith("adversary"):
                if agent_id not in actions:
                    # Adversary may have terminated; simple_tag still expects
                    # an entry — emit no-op (action 0).
                    full_actions[agent_id] = 0
                else:
                    full_actions[agent_id] = actions[agent_id]
            elif agent_id in self._consumed_prey:
                # Consumed prey: no-op so it stays frozen.
                full_actions[agent_id] = 0
            else:
                pol = self._scripted_policies.get(agent_id)
                last_obs = self._scripted_last_obs.get(agent_id)
                if pol is None or last_obs is None:
                    full_actions[agent_id] = 0
                else:
                    try:
                        full_actions[agent_id] = int(pol.get_action(last_obs))
                    except Exception:
                        full_actions[agent_id] = 0

        observations, rewards, terminations, truncations, infos = self.inner.step(
            full_actions
        )

        # Update prey-obs cache for the next step (active prey only).
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if not a.startswith("adversary") and a not in self._consumed_prey
        }

        # Detect new catches and consume prey via the live mpe2 world.
        new_catches = 0
        if self._scenario is not None and self._world is not None:
            try:
                advs = [a for a in self._world.agents if a.adversary]
                # Only iterate over *active* prey (skip already-consumed).
                active_preys = [
                    a for a in self._world.agents
                    if (not a.adversary)
                    and a.movable
                    and a.name not in self._consumed_prey
                ]
                for prey in active_preys:
                    for adv in advs:
                        if self._scenario.is_collision(prey, adv):
                            self._consumed_prey.add(prey.name)
                            new_catches += 1
                            # Freeze + deactivate the prey in place.
                            # We deliberately do *not* teleport: simple_tag
                            # observations include raw relative positions
                            # to all other agents, so a teleport to a far
                            # position would inject huge ReLU-saturating
                            # outliers into adversary obs.  Instead we
                            # leave the carcass at its last position with
                            # movable=False (no motion), collide=False
                            # (no re-detection, no further catch bonus,
                            # no physical collision response), and zero
                            # velocity.  The shaping reward filters
                            # consumed prey via ``a.movable``, so the
                            # gradient retargets to surviving prey.
                            prey.movable = False
                            prey.collide = False
                            prey.state.p_vel = np.zeros_like(prey.state.p_vel)
                            break
                self.prey_caught += new_catches
            except Exception:
                pass

        # Team-wide catch bonus.
        catch_bonus = CATCH_BONUS * float(new_catches)
        all_consumed = (
            len(self._consumed_prey) >= len(self.inner.good_agents)
            and len(self.inner.good_agents) > 0
        )
        if all_consumed and new_catches > 0:
            catch_bonus += TERMINAL_CLEAR_BONUS

        # Hard arena termination.  If any adversary leaves the hard
        # radius, terminate the episode immediately.  This caps the
        # damage from runaway policies (which would otherwise accumulate
        # astronomical soft-boundary penalties via exp-grow) and keeps
        # the population fitness mean bounded so EA selection signal
        # is not drowned by noise.
        out_of_bounds = False
        if self._world is not None and HARD_ARENA_RADIUS > 0.0:
            try:
                for a in self._world.agents:
                    if a.adversary and float(np.max(np.abs(a.state.p_pos))) > HARD_ARENA_RADIUS:
                        out_of_bounds = True
                        break
            except Exception:
                pass

        # Filter outputs to adversary side only.
        self.agents = [a for a in self.inner.adversary_agents if a in observations]
        learner_obs = {a: observations[a] for a in self.agents}
        learner_rewards = {
            a: float(rewards.get(a, 0.0)) + catch_bonus for a in self.agents
        }
        learner_terms = {
            a: bool(terminations.get(a, False)) or all_consumed or out_of_bounds
            for a in self.agents
        }
        learner_truncs = {a: bool(truncations.get(a, False)) for a in self.agents}
        prey_remaining = len(self.inner.good_agents) - len(self._consumed_prey)
        learner_infos = {
            a: {
                **infos.get(a, {}),
                "prey_caught_total": int(self.prey_caught),
                "prey_remaining": int(prey_remaining),
            }
            for a in self.agents
        }
        return learner_obs, learner_rewards, learner_terms, learner_truncs, learner_infos

    def render(self):
        return self.inner.env.render() if self.render_mode is not None else None

    def close(self) -> None:
        try:
            self.inner.close()
        except (SystemExit, AttributeError):
            pass

    # Internals ------------------------------------------------------------

    def _find_scenario(self):
        try:
            return self.inner.env.aec_env.env.env.scenario
        except AttributeError:
            pass
        try:
            return self.inner.env.unwrapped.scenario
        except AttributeError:
            return None


class DictResetEnv:
    """Adapter exposing legacy ``reset() -> dict`` (drops infos) for an inner env.

    deap-bridge's :class:`MixedPolicyEvaluator` and CRN pmap helper expect
    ``reset(seed=...) -> dict``; the modern parallel API returns a
    ``(obs, infos)`` tuple.  This shim delegates to ``inner.reset`` and
    discards the infos so both call paths can share the same env class.
    """

    def __init__(self, inner: ScriptedPreyWrapper) -> None:
        self.inner = inner
        self.possible_agents = list(inner.possible_agents)
        self.metadata = inner.metadata
        self.render_mode = inner.render_mode

    @property
    def agents(self) -> List[str]:
        return self.inner.agents

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        obs, _infos = self.inner.reset(seed=seed, options=options)
        return obs

    def step(self, actions: Dict[str, Any]):
        return self.inner.step(actions)

    def render(self):
        return self.inner.render()

    def close(self) -> None:
        self.inner.close()


# ===========================================================================
# Module-level env registration
# ===========================================================================


def _scripted_simple_tag_factory(render_mode: Optional[str] = None, **_kwargs):
    inner = PursuitEvasionEnv(config=PE_CONFIG, render_mode=render_mode)
    _maybe_apply_reward_shaping(inner)
    return ScriptedPreyWrapper(inner, apf_config=APF_CONFIG)


if not is_registered(ENV_NAME):
    register_env(
        ENV_NAME,
        _scripted_simple_tag_factory,
        default_episode_len=PE_CONFIG.max_cycles,
        continuous_default=False,
        notes=(
            "mpe2.simple_tag_v3 wrapped by PursuitEvasionEnv and "
            "ScriptedPreyWrapper (APF prey hidden from public API)."
        ),
        tags={"mpe2", "pursuit_evasion", "scripted_prey"},
        api="parallel",
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _episode_steps_per_episode() -> int:
    return int(PE_CONFIG.max_cycles)


def _ea_generations_for_budget(pop_size: int, eval_eps: int) -> int:
    per_gen = pop_size * max(1, eval_eps) * _episode_steps_per_episode()
    return max(1, TOTAL_ENV_STEPS // per_gen)


# ---------------------------------------------------------------------------
# Env factory paths
# ---------------------------------------------------------------------------


def _registered_env_factory():
    """Modern parallel-API env (used by MAPPO + helper rollouts)."""
    return build_env(ENV_NAME)


def _evaluator_env_factory():
    """Legacy ``reset()->dict`` shim (used by EA/CMA evaluator)."""
    return DictResetEnv(build_env(ENV_NAME))


def _empty_scripted_factory(env) -> Dict[str, Any]:
    """No external scripted policies — prey are stepped inside the wrapper."""
    return {}


# ---------------------------------------------------------------------------
# Action head + learner factory
# ---------------------------------------------------------------------------


SOFTMAX_TEMPERATURE: float = 0.2
"""Softmax temperature for ``SoftmaxSamplingHead`` during EA/CMA training.

T=0.2 (vs argmax) gives a smooth fitness landscape: small weight
perturbations continuously shift action probabilities, which is the
standard ES-on-discrete fix for the staircase-fitness problem of argmax.
"""


class SoftmaxSamplingHead(ActionHead):
    """Stochastic softmax-sampling head for Discrete spaces (EA/CMA only)."""

    def __init__(
        self,
        n: int,
        temperature: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"SoftmaxSamplingHead requires n >= 1, got {n}")
        if temperature <= 0:
            raise ValueError(
                f"SoftmaxSamplingHead requires temperature > 0, got {temperature}"
            )
        self.n = int(n)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def output_size(self) -> int:
        return self.n

    def decode(self, raw_output: np.ndarray) -> int:
        arr = np.asarray(raw_output, dtype=np.float64).ravel()
        if arr.size != self.n:
            raise ValueError(
                f"SoftmaxSamplingHead expected output of size {self.n}, "
                f"got size {arr.size}"
            )
        scaled = arr / self.temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return int(self.rng.choice(self.n, p=probs))


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    """All adversaries share one evolved genome (homogeneous team).

    The head's RNG is seeded *deterministically from agent_id* so two
    identical genomes evaluated under the same env seed produce the same
    action stream — this preserves CRN at the policy layer.
    """
    head = SoftmaxSamplingHead(
        n=int(action_space.n),
        temperature=SOFTMAX_TEMPERATURE,
        rng=np.random.default_rng(abs(hash(agent_id)) & 0xFFFF_FFFF),
    )
    return FeedforwardNeuralPolicy(genome, head=head)


# ---------------------------------------------------------------------------
# Episode rollout helpers
# ---------------------------------------------------------------------------


def _build_oracle_weights(
    layer_sizes: List[int],
    good_rel_slice: slice,
    init_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a flat weight vector encoding a single-prey-pursuit oracle.

    Hidden[0] is wired to ``sign(dx_target)`` and hidden[1] to
    ``sign(dy_target)`` of one randomly chosen prey (saturated tanh-like
    via ReLU-friendly large weights).  Output logits are wired so the
    cardinal-direction action whose force is most aligned with the prey
    direction wins the argmax / dominates the softmax.  The remaining
    hidden units stay as small noise so the EA/CMA can grow specialised
    features without disrupting the baseline pursuit behaviour.

    Ported from ``ea_vs_apf.py`` — there it is what makes the EA actually
    learn pursuit; without it gradient-free search cannot find the
    pursuit basin from random init in a budget of ~3,000 evaluations.
    """
    if len(layer_sizes) != 3:
        raise ValueError(
            "oracle warm start currently supports a single hidden layer; "
            f"got {layer_sizes}"
        )
    input_size, hidden_size, output_size = layer_sizes
    if hidden_size < 2:
        raise ValueError(f"hidden_size must be >= 2, got {hidden_size}")
    if output_size != 5:
        raise ValueError(f"oracle expects 5 discrete actions, got {output_size}")

    W1 = rng.uniform(-init_scale, init_scale, size=(hidden_size, input_size))
    b1 = rng.uniform(-init_scale, init_scale, size=(hidden_size,))

    rel_indices = list(range(good_rel_slice.start, good_rel_slice.stop))
    if len(rel_indices) % 2 != 0:
        raise ValueError(
            f"good_rel slice length {len(rel_indices)} is not a multiple of 2"
        )
    dx_indices = rel_indices[0::2]
    dy_indices = rel_indices[1::2]

    W1[0, :] = 0.0
    W1[1, :] = 0.0
    n_good = len(dx_indices)
    target_prey = int(rng.integers(0, n_good))
    SATURATION_GAIN = 5.0
    W1[0, dx_indices[target_prey]] = SATURATION_GAIN
    W1[1, dy_indices[target_prey]] = SATURATION_GAIN
    b1[0] = 0.0
    b1[1] = 0.0

    W2 = rng.uniform(-init_scale, init_scale, size=(output_size, hidden_size))
    b2 = rng.uniform(-init_scale, init_scale, size=(output_size,))

    W2[:, 0] = 0.0
    W2[:, 1] = 0.0
    W2[0, 0] = 0.0  # no-op
    W2[0, 1] = 0.0
    W2[1, 0] = -ORACLE_OUTPUT_GAIN  # -x action
    W2[2, 0] = +ORACLE_OUTPUT_GAIN  # +x action
    W2[3, 1] = -ORACLE_OUTPUT_GAIN  # -y action
    W2[4, 1] = +ORACLE_OUTPUT_GAIN  # +y action
    b2[:] = 0.0

    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])


def _make_oracle_seeded_population(
    toolbox,
    population_size: int,
    oracle_fraction: float,
    layer_sizes: List[int],
    good_rel_slice: slice,
    init_scale: float,
    seed_sigma: float,
    rng: np.random.Generator,
) -> List[Any]:
    """Build a population mixing oracle-seeded and random individuals.

    The first ``round(population_size * oracle_fraction)`` individuals are
    *perturbed* copies of the oracle weights (oracle + N(0, seed_sigma));
    the remainder are random per ``toolbox.individual``.  Each oracle
    copy independently picks a target prey, so ~pop/N_good copies chase
    each prey, naturally giving team coverage.
    """
    population = toolbox.population(n=population_size)
    n_oracle = int(round(population_size * oracle_fraction))
    if n_oracle <= 0:
        return population
    for i in range(n_oracle):
        oracle = _build_oracle_weights(layer_sizes, good_rel_slice, init_scale, rng)
        perturbed = oracle + rng.normal(0.0, seed_sigma, size=oracle.size)
        population[i].genome.set_weights(perturbed.astype(np.float64))
        if hasattr(population[i], "fitness") and population[i].fitness.valid:
            del population[i].fitness.values
    return population


def _make_champion_policy(genome: FixedTopologyGenome, out_size: int, seed: int):
    """Build a champion-rollout policy that *matches the training head*.

    Earlier versions used :class:`DiscreteHead` (argmax) for champion
    eval, but training uses :class:`SoftmaxSamplingHead` (T=0.2,
    stochastic).  At small init weights argmax is essentially a constant
    action, so the deterministic champion can never catch anything \u2014
    not because the policy is bad, but because the eval head doesn't
    match the training head.  Use the same head for fair measurement.
    """
    return FeedforwardNeuralPolicy(
        genome,
        head=SoftmaxSamplingHead(
            n=out_size,
            temperature=SOFTMAX_TEMPERATURE,
            rng=np.random.default_rng(seed),
        ),
    )


def _run_policy_episode(policy, seed: int) -> Tuple[float, int]:
    """Roll one team episode and return ``(team_reward, catches)``."""
    env = build_env(ENV_NAME)
    try:
        if hasattr(policy, "reset"):
            policy.reset()

        obs, infos = env.reset(seed=seed)
        catches = 0
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
            if infos:
                first_info = next(iter(infos.values()))
                catches = max(catches, int(first_info.get("prey_caught_total", 0)))
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
        return episode_reward, catches
    finally:
        try:
            env.close()
        except (SystemExit, AttributeError):
            pass


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
        # Use the same head as training (SoftmaxSamplingHead at
        # SOFTMAX_TEMPERATURE).  Recording with DiscreteHead (argmax)
        # produces near-constant action at small init weights and gives
        # the false impression that learned policies "do nothing".
        out_size = int(act_space.n) if hasattr(act_space, "n") else int(act_space.shape[0])
        policy = _make_champion_policy(best_genome, out_size=out_size, seed=seed)

        obs, _ = env.reset(seed=seed)
        recorder.start_episode(episode_idx)
        try:
            frame = env.render()
            if frame is not None:
                recorder.add_frame(frame)
        except Exception:
            pass

        _cap = recorder.max_frames_per_episode
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
            _frames_buffered = len(recorder._frames)
            if _cap is None or _frames_buffered < _cap:
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
        except (SystemExit, AttributeError):
            pass


# ===========================================================================
# Env probe (PROBE_ENV=True)
# ===========================================================================


def _probe_env() -> None:
    _section("ENV PROBE — random-action smoke test")
    env = build_env(ENV_NAME)
    try:
        obs, infos = env.reset(seed=SEED_BASE)
        learners = env.possible_agents
        print(f"  possible_agents : {learners}")
        sample_id = learners[0]
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
        prey_caught = next(iter(infos.values()), {}).get("prey_caught_total", "?") if infos else "?"
        print(f"  total team reward: {total_r:+.2f}")
        print(f"  prey_caught     : {prey_caught}")
        print("  OK")
    finally:
        try:
            env.close()
        except (SystemExit, AttributeError):
            pass


# ===========================================================================
# EA section
# ===========================================================================


def _build_evaluator() -> MixedPolicyEvaluator:
    return MixedPolicyEvaluator(
        env_factory=_evaluator_env_factory,
        scripted_policy_factory=_empty_scripted_factory,
        learner_agent_prefixes=["adversary"],
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
        except (SystemExit, AttributeError):
            pass
    return in_size, out_size


def run_ea(seed: int, run_dir: Path) -> Dict[str, Any]:
    _section(f"EA  | seed={seed}")
    evaluator = _build_evaluator()
    in_size, out_size = _ea_or_cma_io_sizes(evaluator, seed)
    _ea_layers = [in_size, out_size] if EA_HIDDEN_SIZE == 0 else [in_size, EA_HIDDEN_SIZE, out_size]
    _ea_arch = "linear" if EA_HIDDEN_SIZE == 0 else f"hidden={EA_HIDDEN_SIZE}"
    print(f"  Pursuer NN: in={in_size} {_ea_arch} out={out_size}")

    setup_deap_creators()
    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": _ea_layers,
        "activation": "relu",
        "output_activation": "linear",
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
    # Oracle warm-start: half the initial population starts as small
    # perturbations of a hand-coded single-prey-pursuit controller so
    # the EA has actual pursuit behaviour in the gene pool from gen 0.
    # Without this the search budget is far too small to discover
    # pursuit from random init in a 600+ weight network.
    pursuer_layout = None
    if EA_HIDDEN_SIZE > 0:
        try:
            _probe = evaluator.env_factory()
            try:
                if hasattr(_probe, "inner") and hasattr(_probe.inner, "get_layout"):
                    pursuer_layout = _probe.inner.get_layout(
                        _probe.inner.adversary_agents[0]
                    )
            finally:
                try:
                    _probe.close()
                except (SystemExit, AttributeError):
                    pass
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            print(f"  oracle warm-start unavailable: {exc!r}; using random init")

    if pursuer_layout is not None:
        population = _make_oracle_seeded_population(
            toolbox=toolbox,
            population_size=EA_POPULATION_SIZE,
            oracle_fraction=ORACLE_FRACTION,
            layer_sizes=_ea_layers,
            good_rel_slice=pursuer_layout.good_rel,
            init_scale=init_scale,
            seed_sigma=ORACLE_SEED_SIGMA,
            rng=rng,
        )
        n_oracle = int(round(EA_POPULATION_SIZE * ORACLE_FRACTION))
        print(f"  oracle warm-start: {n_oracle}/{EA_POPULATION_SIZE} individuals")
    else:
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
        env_config={"pe_config": PE_CONFIG.__dict__, "reward_shaping": REWARD_SHAPING},
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
        logger.get_frame_recorder(
            fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="ea_",
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
        )
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
                    output_activation=getattr(ind.genome, "output_activation", ind.genome.activation),
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
        _, champ_catches = _run_policy_episode(
            _make_champion_policy(hof[0].genome, out_size, seed=seed * 10_000 + 0),
            seed=seed * 10_000 + 0,
        )
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
            "champion_catches": int(champ_catches),
        })
        print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}  champ_catches={champ_catches}")

        if recorder is not None and recorder.enabled:
            try:
                _record_champion_rollout(recorder, 0, hof[0].genome, seed=RECORD_SEED)
            except Exception:
                traceback.print_exc()

        # Evolution loop
        for gen in range(1, n_gens + 1):
            gen_seed_holder["seed"] = seed * 10_000 + gen

            elites = [
                toolbox.clone(ind) for ind in tools.selBest(population, EA_ELITE_SIZE)
            ]
            for elite in elites:
                del elite.fitness.values

            n_offspring = len(population) - EA_ELITE_SIZE
            offspring = list(map(toolbox.clone, toolbox.select(population, n_offspring)))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < EA_CXPB:
                    toolbox.mate(c1, c2)
            for m in offspring:
                if rng.random() < 0.3:
                    toolbox.mutate(m)

            for ind in offspring:
                if ind.fitness.valid:
                    del ind.fitness.values

            invalid = [ind for ind in (elites + offspring) if not ind.fitness.valid]
            for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population[:] = elites + offspring
            hof.update(population)

            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            _, champ_catches = _run_policy_episode(
                _make_champion_policy(hof[0].genome, out_size, seed=seed * 10_000 + gen),
                seed=seed * 10_000 + gen,
            )
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
                "champion_catches": int(champ_catches),
            })
            print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}  champ_catches={champ_catches}")

            if (recorder is not None and recorder.enabled
                    and gen % RECORD_EVERY_N_EA == 0):
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=RECORD_SEED)
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
    _cma_layers = [in_size, out_size] if CMA_HIDDEN_SIZE == 0 else [in_size, CMA_HIDDEN_SIZE, out_size]
    _cma_arch = "linear" if CMA_HIDDEN_SIZE == 0 else f"hidden={CMA_HIDDEN_SIZE}"
    print(f"  Pursuer NN: in={in_size} {_cma_arch} out={out_size}")

    setup_deap_creators()
    np.random.seed(seed)

    init_scale = float(np.sqrt(1.0 / max(in_size, 1)))
    genome_config = {
        "layer_sizes": _cma_layers,
        "activation": "relu",
        "output_activation": "linear",
        "weight_range": (-init_scale, init_scale),
    }
    # Oracle centroid warm-start (mirrors the EA path).  CMA-ES from a
    # zero-mean centroid in 600+ dimensions has effectively zero chance
    # of finding pursuit in our budget; seeding the centroid with the
    # hand-coded chase-one-prey controller gives the strategy a sane
    # starting basin to refine.
    cma_centroid = None
    if CMA_HIDDEN_SIZE > 0:
        try:
            _probe = build_env(ENV_NAME)
            try:
                if hasattr(_probe, "inner") and hasattr(_probe.inner, "get_layout"):
                    _layout = _probe.inner.get_layout(_probe.inner.adversary_agents[0])
                    cma_centroid = _build_oracle_weights(
                        _cma_layers,
                        _layout.good_rel,
                        init_scale,
                        np.random.default_rng(seed),
                    )
            finally:
                try:
                    _probe.close()
                except (SystemExit, AttributeError):
                    pass
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            print(f"  oracle centroid unavailable: {exc!r}; using zero centroid")
    strategy, dimension = setup_cma_strategy(
        genome_config=genome_config,
        sigma0=CMA_SIGMA0,
        population_size=CMA_POPULATION_SIZE,
        centroid=cma_centroid,
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
        env_config={"pe_config": PE_CONFIG.__dict__, "reward_shaping": REWARD_SHAPING},
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
        logger.get_frame_recorder(
            fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="cmaes_",
            max_frames_per_episode=RECORD_MAX_FRAMES,
            max_dim=RECORD_MAX_DIM,
        )
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
                    output_activation=getattr(ind.genome, "output_activation", ind.genome.activation),
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
            _, champ_catches = _run_policy_episode(
                _make_champion_policy(hof[0].genome, out_size, seed=seed * 10_000 + gen),
                seed=seed * 10_000 + gen,
            )
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
                "champion_catches": int(champ_catches),
            })
            print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}  champ_catches={champ_catches}")

            if (recorder is not None and recorder.enabled
                    and gen % RECORD_EVERY_N_EA == 0):
                try:
                    _record_champion_rollout(recorder, gen, hof[0].genome, seed=RECORD_SEED)
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
    from deap_pettingzoo.baselines.agilerl_trainer import GroupedIPPOTrainer
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
        env_config={},  # env is fully configured at registration time
        output_dir=run_dir,
        recording=RecordingConfig(
            enabled=RECORD,
            every_n_generations=RECORD_EVERY_N_MAPPO,
            fps=RECORD_FPS,
            fmt=RECORD_FMT,
            episode_prefix="mappo_",
        ),
    )
    trainer = GroupedIPPOTrainer(device=DEVICE)
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
    rewards: List[float] = []
    catches: List[int] = []
    for i in range(n_episodes):
        env = build_env(ENV_NAME)
        try:
            learner_id = env.possible_agents[0]
            policy = FeedforwardNeuralPolicy(genome, action_space=env.action_space(learner_id))
        finally:
            try:
                env.close()
            except (SystemExit, AttributeError):
                pass

        reward, catch_total = _run_policy_episode(policy, seed=seed_base + i)
        rewards.append(float(reward))
        catches.append(int(catch_total))

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
    rewards: List[float] = []
    catches: List[int] = []
    for i in range(n_episodes):
        ep_ret, catch_total = _run_policy_episode(policy, seed=seed_base + i)
        rewards.append(ep_ret)
        catches.append(catch_total)

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
# Entry point
# ===========================================================================


def main() -> None:
    _section("Pursuit-evasion in mpe2 simple_tag: EA / CMA-ES / MAPPO")
    print(f"  ENV          : {ENV_NAME}")
    print(f"  adversaries  : {PE_CONFIG.num_adversaries}")
    print(f"  prey         : {PE_CONFIG.num_good}  (scripted APF)")
    print(f"  obstacles    : {PE_CONFIG.num_obstacles}")
    print(f"  episode cap  : {PE_CONFIG.max_cycles} steps")
    print(f"  reward shape : {REWARD_SHAPING}")
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
