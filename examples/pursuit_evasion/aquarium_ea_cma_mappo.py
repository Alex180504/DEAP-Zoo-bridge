"""Pursuit-evasion in marl-aquarium: EA vs CMA-ES vs MAPPO.

Single-file experiment comparing three pursuer-team training methods
against a fixed scripted APF prey controller in the
:mod:`marl_aquarium.aquarium_v0` environment:

* **EA**: standard DEAP genetic algorithm with CRN parallel evaluation
  (mirrors ``examples/pursuit_evasion/ea_vs_apf.py``).
* **CMA-ES**: covariance matrix adaptation via DEAP's ``cma.Strategy``
  (mirrors ``examples/pursuit_evasion/cmaes_vs_apf.py``).
* **MAPPO**: AgileRL's prefix-grouped IPPO baseline
  (:class:`GroupedIPPOTrainer`).

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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
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
from deap_pettingzoo.policies.action_heads import ActionHead, DiscreteHead
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
from examples.pursuit_evasion.aquarium.observation_features import (
    AquariumObservationFeatureWrapper,
    LazyActionPenaltyWrapper,
    extract_prey_caught_total,
)
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

# Equal env-steps budget across the three algorithms
TOTAL_ENV_STEPS: int = 2_000_000
EVAL_EPISODES: int = 30

# ---- Aquarium env config (5v5, prey speed advantage) ---------------------
#
# Aquarium quirks (see /memories/repo/aquarium_env_quirks.md):
#   - predator_view_distance=200 default fails the assertion in
#     utils.scale() because random spawn distances reach ~535 on the
#     800x800 torus.  Use 2000 for safety.
#   - fov_enabled=True padding bug in prey_nearby_sharks_observations
#     is fixed by the monkey-patch in builtin._patch_aquarium(); safe
#     to enable here.
#   - render_mode=None is rejected; the wrapper maps it to "rgb_array".
#

ENV_NAME: str = "aquarium_pursuit_evasion_v0"

ENV_CONFIG: Dict[str, Any] = {
    "predator_count": 5,
    "prey_count": 10,
    "max_time_steps": 400,
    "predator_max_velocity": 12.0,
    "prey_max_velocity": 13.0,
    "predator_max_acceleration": 0.8,
    "prey_max_acceleration": 0.8,
    "predator_max_steer_force": 2.5,
    "prey_max_steer_force": 2.5,
    "predator_view_distance": 500,
    "prey_view_distance": 600,
    "predator_observe_count": 5,
    "prey_observe_count": 5,
    # Limited predator FOV (120°) forces the policy to actively steer toward
    # prey — a pursuer moving straight north cannot see prey behind it.
    # Prey retain 360° awareness so they always know where to flee.
    "fov_enabled": True,
    "predator_fov": 120,
    "prey_fov": 360,
    # Show FOV cones and steering vectors in recorded frames.
    "draw_view_cones": False,
    "draw_action_vectors": True,
    "action_count": 16,
    # Reduced from 200 to 20 to keep value-function regression stable: at
    # 200 a single catch dwarfs the dense shaping signal by 4 orders of
    # magnitude, so the MSE value loss fits "did a catch happen?" instead
    # of state quality.  20 is still ~5x a typical episode's shaping mass.
    "predator_reward": 100,
    "catch_radius": 50,
    # Disable respawn — prevents the "highway" effect where a catch spawns
    # new prey directly into a linear predator's path.  The episode runs
    # until all prey are caught or max_time_steps is reached.
    "keep_prey_count_constant": False,
    "width": 2000,
    "height": 2000,
}

APF_CONFIG = AquariumAPFConfig(
    k_predator=5.0,
    k_tangential=3.0,
    k_prey_separation=0.5,
)  # tangential dodge breaks force-cancellation; k_prey_separation separates flock

# Progress-based reward shaping: dense guidance for closing distance to prey
# (0 = off). In the wrapper this is applied as ``alpha * (prev_dist -
# curr_dist) / 200`` per predator per step. With alpha=1.0 the shaping
# mass over 5 predators * 400 steps dominates the sparse catch bonus, so
# the optimiser converges on rigid drift headings instead of true pursuit
# (EA/CMA ran +28..+40 with no visible steering). alpha=0.1 cuts shaping
# 10x while keeping a faint gradient signal so catches dominate fitness.
REWARD_SHAPE_ALPHA: float = 0.03
FOV_SHAPE_ALPHA: float = 0.01  # +0.01 per visible prey per step (3 prey * 400 steps * 0.01 = 12 max; catches still dominate)

# ---- Lazy-action entropy penalty -----------------------------------------
#
# Punishes predators whose action distribution collapses (e.g. constant
# heading drift).  Applied once at episode end as a terminal-step reward
# delta keyed by per-agent *top-1 action frequency* (the share of steps
# spent on the single most-used action).  Top-1 frequency is robust to
# the artificial action diversity injected by the softmax sampling head
# at low temperature, where a policy whose argmax never changes still
# samples minority actions a few percent of the time and inflates
# Shannon entropy enough to slip past any reasonable entropy threshold.
# A genuine pursuit policy alternates headings and rarely sits above
# ~40–50% top-1; a herding/drifting policy spends 60–100% of steps on
# the same heading.  Penalty -50 is large vs typical sparse catch
# rewards (~+20 each) but small enough that two real catches still net
# positive fitness.  Set LAZY_PENALTY = 0.0 to disable.
LAZY_PENALTY: float = -10.0
LAZY_TOP1_THRESHOLD: float = 0.7
LAZY_MIN_ACTIONS: int = 50

# ---- EA config -----------------------------------------------------------

EA_POPULATION_SIZE: int = 48
EA_HIDDEN_SIZE: int = 16
EA_EVAL_EPISODES_TRAIN: int = 5    
EA_ELITE_SIZE: int = 3
EA_MUT_SIGMA: float = 0.2           # increased from 0.1 — helps escape fitness plateaus
EA_MUT_INDPB: float = 0.3          # increased from 0.2 — more genes perturbed per mutation
EA_CXPB: float = 0.7

# ---- Oracle warm-start config (ported from simple_tag_ea_cma_mappo.py) ----
# Without an oracle warm-start, gradient-free search at pop=32 cannot find
# pursuit behaviour from random init in a ~640-weight network: the basin
# of attraction is far too small.  Half the EA pop is seeded as small
# perturbations of a hand-coded "chase one randomly chosen prey"
# controller; CMA-ES' centroid is set to the same controller (un-
# perturbed).  The aquarium oracle uses 4 hidden units (one per signed
# axis half) so it produces correct heading logits in *all four
# quadrants* of the prey-relative position — unlike the 2-hidden-unit
# simple_tag oracle which loses x or y signal under ReLU when the prey
# is in the -x and/or -y quadrant.
ORACLE_FRACTION: float = 0.5      # fraction of EA pop seeded with oracle
ORACLE_SEED_SIGMA: float = 0.05   # Gaussian noise added per oracle copy
ORACLE_OUTPUT_GAIN: float = 4.0   # logit gain for cardinal-direction output

# ---- CMA-ES config -------------------------------------------------------

CMA_POPULATION_SIZE: int = 48
CMA_HIDDEN_SIZE: int = 16   
CMA_EVAL_EPISODES_TRAIN: int = 5  
CMA_SIGMA0: float = 0.1

# ---- MAPPO config (AgileRL) ----------------------------------------------

# learn_step lowered from 10_240 to 2_048 (canonical PPO single-env value):
# with num_envs=1 and a 400-step episode horizon, 10_240 contiguous steps
# from one trajectory are extremely correlated and degrade SGD updates.
# 2_048 = ~5 episodes per learn-call which is what AgileRL's IPPO default
# assumes.
MAPPO_LEARN_STEP: int = 2_048
# batch_size lowered from 1024 to 256 (PPO canonical): with the new
# learn_step=2048 this gives 8 minibatches × 4 update_epochs = 32 SGD steps
# per rollout, restoring proper minibatch noise for the value-function
# regression.
MAPPO_BATCH_SIZE: int = 256
MAPPO_LR: float = 3e-4  # canonical PPO LR; 1e-3 was collapsing the policy
# gamma raised from 0.95 to 0.99 (AgileRL default): the catch reward must
# propagate ~50–150 steps back to the steering decisions that set it up.
# At 0.95 the effective horizon is only ~20 steps, so the catch signal is
# discounted to nothing before it can shape early-episode actions.
MAPPO_GAMMA: float = 0.99
# Extra kwargs forwarded to AgileRL's ``IPPO.__init__`` (e.g.
# ``{"net_config": {"encoder_config": {"hidden_size": [128, 128]}}}``).
# Empty dict = use AgileRL's defaults.
MAPPO_EXTRA: Dict[str, Any] = {
    # ent_coef=0.05 was overwhelming the policy gradient: with 16 actions,
    # the entropy bonus over a 400-step / 5-predator episode is ~277 "reward"
    # for staying near-uniform random — comparable to the entire catch signal.
    # 0.01 is canonical PPO and lets the actor actually commit to headings.
    "ent_coef": 0.01,
    # Early-stop PPO update epochs once the policy has moved this far in KL
    # space.  Standard PPO defensive measure; prevents catastrophic policy
    # drift on the few rollouts where advantages happen to be skewed.
    #"target_kl": 0.02,
    "net_config": {
        "encoder_config": {"hidden_size": [64, 64]},
        "head_config": {"hidden_size": [64]},
    },
}

# ---- Recording / logging --------------------------------------------------

RECORD: bool = True
RECORD_FPS: int = 20
RECORD_FMT: str = "mp4"
RECORD_EVERY_N_EA: int = 5          # EA/CMA generations
RECORD_EVERY_N_MAPPO: int = 10      # MAPPO iterations
RECORD_MAX_FRAMES: Optional[int] = None  # None = record full episode; set e.g. 150 to cap at 7.5 s
RECORD_MAX_DIM: int = 960           # scale largest dimension to this (4000→960 avoids H.264 level limit)

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


class SoftmaxSamplingHead(ActionHead):
    """Stochastic softmax-sampling head for Discrete spaces (EA/CMA only).

    The default :class:`DiscreteHead` returns ``argmax(logits)``, which
    creates a step-function fitness landscape: small weight perturbations
    almost never flip the argmax, so gradient-free search crawls. This
    head samples from ``softmax(logits / temperature)`` for a smooth
    landscape. Mirrors the working precedent in ``ea_vs_apf.py``.
    """

    def __init__(self, n: int, temperature: float = 0.2,
                 rng: Optional[np.random.Generator] = None) -> None:
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


SOFTMAX_TEMPERATURE: float = 0.3
"""Softmax temperature for ``SoftmaxSamplingHead`` used during EA/CMA training.

With weights initialised at ``\u00b11/\u221ain_size`` and ReLU activations of
magnitude \u22480.5, output-layer logits are typically O(\u00b10.3).  At
temperature=0.5 the resulting softmax differs by only ~1.7\u00d7 between the
strongest and weakest action \u2014 sampling noise dominates and EA finds it
easier to bias the *distribution* by inflating one logit (constant-heading
drift) than to learn an obs\u2192action mapping.  At temperature=0.1 the same
\u00b10.3 logits give a ~20\u00d7 ratio (near-argmax behaviour), restoring the
policy as the dominant fitness driver while keeping the landscape smooth
at decision boundaries (small perturbations still shift the distribution
continuously)."""


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    """All predators share the same evolved genome (homogeneous team).

    Uses :class:`SoftmaxSamplingHead` instead of argmax so EA/CMA see a
    smooth fitness landscape (precedent: ``ea_vs_apf.py``).  The head's
    RNG is seeded *deterministically from agent_id* so two identical
    genomes evaluated under the same env seed produce the same action
    stream \u2014 this preserves CRN at the policy layer.  Without the seed,
    fitness comparisons include sampling noise that hides genome signal
    and biases EA toward the linear-drift attractor.
    """
    head = SoftmaxSamplingHead(
        n=int(action_space.n),
        temperature=SOFTMAX_TEMPERATURE,
        rng=np.random.default_rng(abs(hash(agent_id)) & 0xFFFF_FFFF),
    )
    return FeedforwardNeuralPolicy(genome, head=head)


# ---------------------------------------------------------------------------
# Oracle warm-start (chase one prey, picked at random per individual)
# ---------------------------------------------------------------------------

# 4 hidden units encode the four signed half-axes of the prey-relative
# direction: |dx|+, |dx|-, |dy|+, |dy|-.  ReLU passes only one of each
# pair.  Output logits are then the dot product of the corresponding
# action heading with the target prey direction, so the action whose
# heading best aligns with the prey wins the softmax.
_ORACLE_HIDDEN_UNITS: int = 4


def _aquarium_action_headings(action_count: int) -> np.ndarray:
    """Return the per-action heading vectors used by the env.

    The aquarium env decodes a discrete action ``k`` to the 2-D unit
    heading ``(sin(2π k / N), -cos(2π k / N))`` (see ``apf_policy.py``).
    Returns an array of shape ``(action_count, 2)``.
    """
    thetas = 2.0 * np.pi * np.arange(int(action_count)) / float(action_count)
    return np.stack([np.sin(thetas), -np.cos(thetas)], axis=1)


def _build_oracle_weights(
    layer_sizes: List[int],
    *,
    feature_offset_prey: int,
    prey_observe_count: int,
    action_count: int,
    init_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a flat weight vector encoding a single-prey-pursuit oracle.

    The featurized predator observation packs each prey slot as
    ``[present, dx, dy, distance, speed]`` (5 dims) starting at index
    ``feature_offset_prey``.  The oracle picks one prey slot at random
    per individual, wires the first 4 hidden units to encode the four
    signed half-axes ``ReLU(±G·dx)`` / ``ReLU(±G·dy)`` of that prey, and
    sets the output W2 so each of the ``action_count`` action logits is
    the dot product of that action's heading with the target's direction.
    The remaining hidden units stay as small noise so the EA/CMA can
    grow specialised features without disrupting baseline pursuit.
    """
    if len(layer_sizes) != 3:
        raise ValueError(
            "oracle warm start currently supports a single hidden layer; "
            f"got {layer_sizes}"
        )
    input_size, hidden_size, output_size = layer_sizes
    if hidden_size < _ORACLE_HIDDEN_UNITS:
        raise ValueError(
            f"hidden_size must be >= {_ORACLE_HIDDEN_UNITS}, got {hidden_size}"
        )
    if output_size != int(action_count):
        raise ValueError(
            f"oracle expects output_size == action_count={action_count}, "
            f"got {output_size}"
        )
    if prey_observe_count < 1:
        raise ValueError(f"prey_observe_count must be >= 1, got {prey_observe_count}")

    W1 = rng.uniform(-init_scale, init_scale, size=(hidden_size, input_size))
    b1 = rng.uniform(-init_scale, init_scale, size=(hidden_size,))

    base = int(feature_offset_prey) + 0 * 5  # pick the first prey slot (closest prey)
    dx_idx = base + 1
    dy_idx = base + 2
    SATURATION_GAIN = 5.0

    # Zero out the four oracle hidden rows, then wire each to one signed
    # half-axis.  ReLU yields a positive activation only when that half-
    # axis is active (e.g. h[0] fires iff dx > 0).
    for h in range(_ORACLE_HIDDEN_UNITS):
        W1[h, :] = 0.0
        b1[h] = 0.0
    W1[0, dx_idx] = +SATURATION_GAIN  # h[0] = ReLU(+G·dx)  -> "right"
    W1[1, dx_idx] = -SATURATION_GAIN  # h[1] = ReLU(-G·dx)  -> "left"
    W1[2, dy_idx] = +SATURATION_GAIN  # h[2] = ReLU(+G·dy)  -> "down"
    W1[3, dy_idx] = -SATURATION_GAIN  # h[3] = ReLU(-G·dy)  -> "up"

    W2 = rng.uniform(-init_scale, init_scale, size=(output_size, hidden_size))
    b2 = rng.uniform(-init_scale, init_scale, size=(output_size,))

    headings = _aquarium_action_headings(action_count)  # (N, 2)
    # logit_k = GAIN * (sx_k * (h0 - h1) + sy_k * (h2 - h3))
    for k in range(output_size):
        sx, sy = float(headings[k, 0]), float(headings[k, 1])
        W2[k, 0] = +ORACLE_OUTPUT_GAIN * sx
        W2[k, 1] = -ORACLE_OUTPUT_GAIN * sx
        W2[k, 2] = +ORACLE_OUTPUT_GAIN * sy
        W2[k, 3] = -ORACLE_OUTPUT_GAIN * sy
    b2[:] = 0.0

    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])


def _make_oracle_seeded_population(
    toolbox,
    *,
    population_size: int,
    oracle_fraction: float,
    layer_sizes: List[int],
    feature_offset_prey: int,
    prey_observe_count: int,
    action_count: int,
    init_scale: float,
    seed_sigma: float,
    rng: np.random.Generator,
) -> List[Any]:
    """Build a population mixing oracle-seeded and random individuals.

    The first ``round(population_size * oracle_fraction)`` individuals are
    *perturbed* copies of the oracle weights (oracle + N(0, seed_sigma));
    the remainder are random per ``toolbox.individual``.  Each oracle
    copy independently picks a target prey slot, so several copies chase
    each prey, naturally giving team coverage.
    """
    population = toolbox.population(n=population_size)
    n_oracle = int(round(population_size * oracle_fraction))
    if n_oracle <= 0:
        return population
    for i in range(n_oracle):
        oracle = _build_oracle_weights(
            layer_sizes,
            feature_offset_prey=feature_offset_prey,
            prey_observe_count=prey_observe_count,
            action_count=action_count,
            init_scale=init_scale,
            rng=rng,
        )
        perturbed = oracle + rng.normal(0.0, seed_sigma, size=oracle.size)
        population[i].genome.set_weights(perturbed.astype(np.float64))
        if hasattr(population[i], "fitness") and population[i].fitness.valid:
            del population[i].fitness.values
    return population


def _make_champion_policy(genome: FixedTopologyGenome, out_size: int, seed: int):
    """Build a champion-rollout policy that *matches the training head*.

    Training uses :class:`SoftmaxSamplingHead` (T=SOFTMAX_TEMPERATURE,
    stochastic).  Recording with the default :class:`DiscreteHead`
    (argmax) at small init weights produces near-constant action and
    gives the false impression that learned policies do nothing.  Use
    the same head for fair playback.
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
    """Roll one predator-team episode and return ``(team_reward, catches)``.

    The lazy-action entropy penalty is *not* applied here — it lives in
    :class:`LazyActionPenaltyWrapper` so it reaches the worker training
    path automatically.  This helper just sums whatever rewards the env
    emits.
    """
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
    import os
    import sys

    # On Windows the default SDL2 video drivers (DirectX/GDI) try to create
    # a real on-screen window, which hangs headlessly and is impossibly slow
    # at 6000×6000.  Switch to the pure-software "dummy" driver for this
    # recording env so surfarray.array3d() still returns real pixel data
    # without any display dependency.
    _sdl_orig = os.environ.get("SDL_VIDEODRIVER")
    if sys.platform == "win32" and _sdl_orig != "dummy":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        try:
            import pygame as _pg
            if _pg.get_init():
                _pg.display.quit()
        except Exception:
            pass

    env = build_env(ENV_NAME, render_mode="rgb_array")
    try:
        learner_id = env.possible_agents[0]
        act_space = env.action_space(learner_id)
        # Use the same head as training (SoftmaxSamplingHead at
        # SOFTMAX_TEMPERATURE).  Recording with DiscreteHead (argmax)
        # gives a constant action at small init weights and makes
        # learned policies look idle in playback.
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
            # Skip render once the frame cap is reached — avoids rendering
            # the full episode (400 steps) when only 150 frames are wanted.
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
        except SystemExit:
            pass
        # Restore SDL_VIDEODRIVER so the rest of the process is unaffected.
        if _sdl_orig is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = _sdl_orig


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
    # perturbations of a hand-coded "chase one randomly chosen prey"
    # controller so EA actually has pursuit behaviour in the gene pool
    # from gen 0.  Without this the search budget is far too small to
    # discover pursuit from random init in a ~640-weight network.
    pred_obs_n = int(ENV_CONFIG.get("predator_observe_count", 5))
    prey_obs_n = int(ENV_CONFIG.get("prey_observe_count", 5))
    action_n = int(ENV_CONFIG.get("action_count", 16))
    feature_offset_prey = 3 + 5 * pred_obs_n  # see featurize_aquarium_observation
    if EA_HIDDEN_SIZE >= _ORACLE_HIDDEN_UNITS and out_size == action_n:
        population = _make_oracle_seeded_population(
            toolbox,
            population_size=EA_POPULATION_SIZE,
            oracle_fraction=ORACLE_FRACTION,
            layer_sizes=_ea_layers,
            feature_offset_prey=feature_offset_prey,
            prey_observe_count=prey_obs_n,
            action_count=action_n,
            init_scale=init_scale,
            seed_sigma=ORACLE_SEED_SIGMA,
            rng=rng,
        )
        n_oracle = int(round(EA_POPULATION_SIZE * ORACLE_FRACTION))
        print(f"  oracle warm-start: {n_oracle}/{EA_POPULATION_SIZE} individuals")
    else:
        print(f"  oracle warm-start skipped (hidden={EA_HIDDEN_SIZE}, out={out_size}, N={action_n})")
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
        logger.get_frame_recorder(fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="ea_",
                                   max_frames_per_episode=RECORD_MAX_FRAMES,
                                   max_dim=RECORD_MAX_DIM)
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
        # Champion catch probe (1 episode) — true task-success metric.
        # Uses argmax (DiscreteHead) so the printed catches reflect the
        # genome's actual capability, not the stochastic head's sampling
        # noise.  Training fitness still uses SoftmaxSamplingHead for the
        # smooth landscape (see _learner_factory).
        _champ_catches_list = []
        for _probe_i in range(3):
            _, c = _run_policy_episode(
                FeedforwardNeuralPolicy(hof[0].genome, head=DiscreteHead(out_size)),
                seed=seed * 10_000 + 0 + _probe_i, 
            )
            _champ_catches_list.append(c)
        champ_catches = float(np.mean(_champ_catches_list))
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
                _record_champion_rollout(recorder, 0, hof[0].genome, seed=seed)
            except Exception:
                traceback.print_exc()

        # Evolution loop
        for gen in range(1, n_gens + 1):
            gen_seed_holder["seed"] = seed * 10_000 + gen

            elites = [
                toolbox.clone(ind) for ind in tools.selBest(population, EA_ELITE_SIZE)
            ]
            # True CRN: re-evaluate elites on the current gen's seeds so all
            # individuals are compared on equal footing.  Without this, a
            # champion that happened to get easy spawn positions in gen-7
            # persists in the HOF for 20+ gens while offspring are scored on
            # different seeds — a systematic bias that masks genuine progress.
            # The HOF preserves the best genome regardless of score fluctuation.
            for elite in elites:
                del elite.fitness.values

            n_offspring = len(population) - EA_ELITE_SIZE
            offspring = list(map(
                toolbox.clone, toolbox.select(population, n_offspring)
            ))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < EA_CXPB:
                    toolbox.mate(c1, c2)
            for m in offspring:
                if rng.random() < 0.3:
                    toolbox.mutate(m)

            # True CRN: invalidate fitnesses for ALL offspring so unmodified
            # clones are forcefully re-evaluated on the current generation's seed.
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
                FeedforwardNeuralPolicy(hof[0].genome, head=DiscreteHead(out_size)),
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
    _cma_layers = [in_size, out_size] if CMA_HIDDEN_SIZE == 0 else [in_size, CMA_HIDDEN_SIZE, out_size]
    _cma_arch = "linear" if CMA_HIDDEN_SIZE == 0 else f"hidden={CMA_HIDDEN_SIZE}"
    print(f"  Pursuer NN: in={in_size} {_cma_arch} out={out_size}")

    setup_deap_creators()
    np.random.seed(seed)

    genome_config = {
        "layer_sizes": _cma_layers,
        "activation": "relu",
        "output_activation": "linear",
        "weight_range": (-float(np.sqrt(1.0 / max(in_size, 1))), float(np.sqrt(1.0 / max(in_size, 1)))),
    }
    # Oracle centroid warm-start (mirrors the EA path).  CMA-ES from a
    # zero-mean centroid in ~640 dimensions has effectively zero chance
    # of finding pursuit in our budget; seeding the centroid with a
    # single chase-one-prey oracle gives the strategy a sane starting
    # basin to refine.
    pred_obs_n = int(ENV_CONFIG.get("predator_observe_count", 5))
    prey_obs_n = int(ENV_CONFIG.get("prey_observe_count", 5))
    action_n = int(ENV_CONFIG.get("action_count", 16))
    feature_offset_prey = 3 + 5 * pred_obs_n
    init_scale_cma = float(np.sqrt(1.0 / max(in_size, 1)))
    cma_centroid = None
    if CMA_HIDDEN_SIZE >= _ORACLE_HIDDEN_UNITS and out_size == action_n:
        try:
            cma_centroid = _build_oracle_weights(
                _cma_layers,
                feature_offset_prey=feature_offset_prey,
                prey_observe_count=prey_obs_n,
                action_count=action_n,
                init_scale=init_scale_cma,
                rng=np.random.default_rng(seed),
            )
            print(f"  oracle centroid warm-start enabled (chase prey slot, action_count={action_n})")
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
        logger.get_frame_recorder(fps=RECORD_FPS, fmt=RECORD_FMT, episode_prefix="cmaes_",
                                   max_frames_per_episode=RECORD_MAX_FRAMES,
                                   max_dim=RECORD_MAX_DIM)
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
                FeedforwardNeuralPolicy(hof[0].genome, head=DiscreteHead(out_size)),
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
    # Use GroupedIPPOTrainer: shares one actor+critic across the homogeneous
    # predator_* team (~5x sample efficiency vs per-agent IPPO/MAPPO).
    from deap_pettingzoo.baselines.agilerl_trainer import GroupedIPPOTrainer
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        RecordingConfig,
        TrainingConfig,
    )
    
    register_scripted_aquarium(
    ENV_NAME + "_mappo",   # different name → different registration
    ENV_CONFIG,
    apf_config=APF_CONFIG,
    reward_shape_alpha=REWARD_SHAPE_ALPHA,
    fov_shape_alpha=FOV_SHAPE_ALPHA,
    wrappers=(AquariumObservationFeatureWrapper,),  # no lazy penalty
    overwrite=True,
    )
    
    from deap_pettingzoo.environments.registry import get_env_spec
    spec = get_env_spec(ENV_NAME + "_mappo")
    print(f"  MAPPO env wrappers: {[w.__name__ if hasattr(w,'__name__') else w for w in spec.wrappers]}")

    cfg = TrainingConfig(
        env_name=ENV_NAME + "_mappo",
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
    # Diagnostic: confirm whether AgileRL's IPPO accepts shared_critic in this version.
    import inspect
    from agilerl.algorithms.ippo import IPPO as _IPPO
    _sig = inspect.signature(_IPPO.__init__)
    print(f"  IPPO accepts shared_critic kwarg: {'shared_critic' in _sig.parameters}")

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
            except SystemExit:
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
    """Run trained MAPPO policy against scripted prey for ``n_episodes``."""
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
# Module-level env registration
# ===========================================================================
#
# Done at import so that *all* call sites — main loop, MAPPO worker,
# parallel EA workers — see the same registered name without having
# to re-register.  ``overwrite=True`` makes re-running the script in
# the same Python process safe.

def _lazy_penalty_wrapper(inner):
    """Factory that wires module-level config into ``LazyActionPenaltyWrapper``.

    Defined at module scope (not a lambda) so it pickles cleanly into
    the parallel worker processes used by EA/CMA.
    """
    return LazyActionPenaltyWrapper(
        inner,
        top1_threshold=LAZY_TOP1_THRESHOLD,
        penalty=LAZY_PENALTY,
        min_actions=LAZY_MIN_ACTIONS,
        horizon=int(ENV_CONFIG["max_time_steps"]),
    )


register_scripted_aquarium(
    ENV_NAME,
    ENV_CONFIG,
    apf_config=APF_CONFIG,
    reward_shape_alpha=REWARD_SHAPE_ALPHA,
    fov_shape_alpha=FOV_SHAPE_ALPHA,
    # Order matters: feature wrapper first (transforms obs/space),
    # lazy-action penalty second (only touches rewards/infos).
    wrappers=(AquariumObservationFeatureWrapper, _lazy_penalty_wrapper),
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
