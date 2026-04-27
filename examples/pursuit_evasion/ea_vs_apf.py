"""EA pursuers vs APF evaders — honours-project prototype experiment.

Implements the baseline methodology described in the project's research
protocol:

* Equal-ability pursuit-evasion in ``simple_tag_v3`` (mpe2) via
  :class:`~deap_pettingzoo.contrib.pursuit_evasion.PursuitEvasionEnv`
  (K-nearest partial observability, boundary penalty, optional curriculum)
* Scripted APF evaders (deterministic, non-learning) so the pursuers train
  against a *stationary* target — isolating algorithmic efficiency from
  opponent-policy non-stationarity
* Per-step distance-based reward shaping on the adversary side to convert
  the sparse +10 collision bonus into a continuous fitness landscape
* Pursuers evolved via DEAP using the deap-bridge framework's evaluator
  abstractions, with multi-episode fitness averaging to reduce ranking noise
* Champion rollouts recorded to mp4 every ``RECORD_EVERY_N_GENS`` generations

This script showcases the deap-bridge API end-to-end:

* :class:`~deap_pettingzoo.contrib.pursuit_evasion.PursuitEvasionEnv`
* :class:`~deap_pettingzoo.contrib.pursuit_evasion.APFEvaderPolicy`
* :class:`~deap_pettingzoo.policies.FeedforwardNeuralPolicy`
* :class:`~deap_pettingzoo.evaluators.MixedPolicyEvaluator`
* :mod:`~deap_pettingzoo.integration.deap_integration`
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from deap import tools

from deap_pettingzoo.contrib.pursuit_evasion import (
    APFConfig,
    APFEvaderPolicy,
    PursuitEvasionConfig,
    PursuitEvasionEnv,
)
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    clone_individual,
    create_toolbox,
    make_mixed_parallel_evaluate_map,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.policies.action_heads import ActionHead
from deap_pettingzoo.utils.logging import ExperimentLogger
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.utils.parallel import make_executor


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

SEED = 0

# Enable per-step distance-based reward shaping for adversaries.
# mpe2 simple_tag's default adversary_reward is purely sparse (+10 on
# collision).  With REWARD_SHAPING = True each adversary additionally receives
# -0.2 × (its Euclidean distance to the nearest prey) per step, giving the
# EA a dense fitness landscape to follow.
REWARD_SHAPING = True

POPULATION_SIZE = 32
NUM_GENERATIONS = 100
HIDDEN_SIZE = 16      # 357 params — tractable for gradient-free search at pop=32

# Episodes averaged per genome per evaluation.  Single-episode evaluation is
# too noisy for reliable selection (lucky start ≠ better genome).  3 episodes
# adds 3× cost but roughly halves ranking noise.
EVAL_EPISODES_TRAIN = 3

EVAL_EPISODES = 20

OUTPUT_DIR = Path("results/pursuit_evasion_ea_vs_apf")

# Scenario — scaled down from 10-vs-10 for a reasonable prototype runtime.
# Change num_adversaries / num_good to 10 for the full methodology.
#
# Boundary penalty is *exponential* (`scale * expm1(sharpness * excess)`).
# At default `sharpness=2.0` even a few units of excess produce 1e8+ values
# that drown the distance-shaping signal and create catastrophic outliers
# (observed -6e16 fitness in early generations).  We tame it heavily:
#   - sharpness 2.0 -> 0.5  (slower exponential growth)
#   - scale     1.0 -> 0.2  (smaller absolute penalty)
# This keeps boundary avoidance meaningful while leaving headroom for
# the -0.2 * distance pursuit signal to drive selection.
#
# Visibility: K-nearest PO was producing observation vectors in which
# the network could NOT distinguish prey from teammates (under PO the
# `other_rel` block is mixed-by-distance with no type label).  We disable
# K-nearest by setting both neighbor counts to None — each pursuer now
# observes ALL other agents, with adversaries appearing first in
# `other_rel` and good agents second, so type is recoverable from
# obs-vector position alone.  Obs dim grows from 26 to ~32, still tiny.
#
# terminate_on_success: end the episode as soon as every prey is caught.
# This raises the relative value of finishing fast (avoids paying the
# remaining steps of distance penalty), giving the EA a stronger
# gradient toward complete-capture policies.
# Boundary penalty further softened (scale 0.2 -> 0.05, threshold 0.9 -> 1.0):
# at the previous setting the APF evaders were fleeing to the edges and the
# pursuers could not follow without paying a fitness penalty larger than the
# +10 collision bonus, so the population converged to a center-camping
# local optimum (max plateaued at -867 from gen 22 onward).  With
# threshold=1.0 the penalty only fires when an agent leaves the [-1,1]
# arena box; with scale=0.05 a brief excursion costs O(0.05) rather than
# O(0.2) per step, freeing pursuers to chase prey across the whole map.
ENV_CONFIG = PursuitEvasionConfig(
    num_adversaries=4,
    num_good=4,
    num_obstacles=3,
    max_cycles=200,
    equal_max_speed=1.0,
    equal_accel=3.0,
    num_agent_neighbors=None,
    num_landmark_neighbors=None,
    boundary_threshold=1.0,
    boundary_scale=0.05,
    boundary_sharpness=0.5,
    terminate_on_success=True,
)

# Number of best individuals carried over unchanged each generation
# (elitism).  Without this the +19.12 outlier observed at gen 67 in our
# logs was destroyed the next generation.  ELITE_SIZE=2 preserves the top
# two genomes verbatim while keeping ~94%% of the population mutable.
ELITE_SIZE: int = 2

# ---------------------------------------------------------------------------
# Recording config
# ---------------------------------------------------------------------------

RECORD = True
RECORD_FPS = 20
RECORD_EVERY_N_GENS = 1   # record champion every N generations (1 = every gen)
# Fixed seed for champion rollout recordings so obstacles always start in the
# same positions across generations — makes visual comparison easy.
RECORD_SEED = 9000

# Number of parallel worker processes for fitness evaluation.
# 0 = use os.cpu_count(); 1 = sequential (useful for debugging / profiling).
# Each worker rebuilds the env and evaluators independently — safe for mpe2.
N_WORKERS: int = 0


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def _enable_reward_shaping(wrapper: PursuitEvasionEnv) -> None:
    """Patch mpe2 simple_tag to enable per-step distance shaping for adversaries.

    Replaces the Scenario's ``adversary_reward`` method on the *instance* (so
    no other envs are affected) with a version that gives each adversary a
    continuous penalty of ``-0.2 × distance_to_nearest_prey`` per step on top
    of the existing sparse ``+10`` collision bonus.  This produces a dense
    fitness landscape so DEAP's EA can learn pursuit behaviour.

    Silently no-ops if the expected internal attribute chain is absent.
    """
    # Try pettingzoo-style chain first, then mpe2's unwrapped shortcut.
    scenario = None
    try:
        scenario = wrapper.env.aec_env.env.env.scenario
    except AttributeError:
        pass
    if scenario is None:
        try:
            scenario = wrapper.env.unwrapped.scenario
        except AttributeError:
            pass
    if scenario is None:
        return

    def _shaped_adversary_reward(self, agent, world):
        # Stronger shaping than the original -0.2*d:
        #   - distance term coefficient 0.2 -> 1.0 (firmer pull toward prey)
        #   - NEW velocity-toward-prey term: rewards moving in the
        #     direction of the nearest prey, even when the absolute
        #     distance change per step is tiny.  This breaks the
        #     gradient-flat iso-distance arc that lets pursuers drift
        #     sideways without penalty.
        rew = 0.0
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if good_agents:
            # Find nearest prey by squared distance (avoid sqrt cost twice).
            self_pos = agent.state.p_pos
            self_vel = agent.state.p_vel
            nearest = min(
                good_agents,
                key=lambda a: float(np.sum((self_pos - a.state.p_pos) ** 2)),
            )
            delta = nearest.state.p_pos - self_pos
            dist = float(np.linalg.norm(delta))
            rew -= 1.0 * dist
            if dist > 1e-6:
                # Project agent's velocity onto the unit vector toward prey.
                # Positive when closing, negative when retreating.
                v_toward = float(np.dot(self_vel, delta)) / dist
                rew += 0.5 * v_toward
        if agent.collide:
            for ag in good_agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10.0
        return rew

    scenario.adversary_reward = types.MethodType(_shaped_adversary_reward, scenario)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Recording helper
# ---------------------------------------------------------------------------


def _record_champion_rollout(
    recorder: FrameRecorder,
    episode_idx: int,
    best_genome: FixedTopologyGenome,
    scripted_evaders: Dict[str, APFEvaderPolicy],
) -> None:
    """Render one champion-vs-APF episode and stream frames into *recorder*.

    Uses a dedicated ``rgb_array`` env so the training env's RNG/state is
    not perturbed.  Best-effort: the caller wraps this in ``try/except``.
    """
    record_env = PursuitEvasionEnv(config=ENV_CONFIG, render_mode="rgb_array")
    try:
        pursuer_id = record_env.adversary_agents[0]
        act_space = record_env.action_space(pursuer_id)
        pursuer_policy = FeedforwardNeuralPolicy(best_genome, action_space=act_space)

        obs_dict = record_env.reset(seed=RECORD_SEED)
        recorder.start_episode(episode_idx)
        try:
            frame = record_env.render()
            if frame is not None:
                recorder.add_frame(frame)
        except Exception:
            pass

        for _ in range(ENV_CONFIG.max_cycles):
            if not obs_dict:
                break
            actions: Dict[str, Any] = {}
            for agent_id, obs in obs_dict.items():
                if agent_id.startswith("adversary"):
                    try:
                        actions[agent_id] = pursuer_policy.get_action(obs)
                    except Exception:
                        actions[agent_id] = record_env.action_space(agent_id).sample()
                else:
                    pol = scripted_evaders.get(agent_id)
                    try:
                        actions[agent_id] = pol.get_action(obs) if pol else record_env.action_space(agent_id).sample()
                    except Exception:
                        actions[agent_id] = record_env.action_space(agent_id).sample()
            obs_dict, _, terms, truncs, _ = record_env.step(actions)
            try:
                frame = record_env.render()
                if frame is not None:
                    recorder.add_frame(frame)
            except Exception:
                pass
            if (terms and all(terms.values())) or (truncs and all(truncs.values())):
                break
    finally:
        recorder.end_episode()
        record_env.close()


# ---------------------------------------------------------------------------
# Environment / policy factories
# ---------------------------------------------------------------------------


def build_env() -> PursuitEvasionEnv:
    """Build the pursuit-evasion environment for this experiment."""
    env = PursuitEvasionEnv(config=ENV_CONFIG)
    if REWARD_SHAPING:
        _enable_reward_shaping(env)
    return env


def build_scripted_evaders(env: PursuitEvasionEnv) -> Dict[str, APFEvaderPolicy]:
    """Instantiate one APF policy per evader (``agent_*``).

    APF evader strength was reduced from k_pursuer=1.0 to 0.3: at full
    strength the scripted evaders were unbeatable for the EA budget
    (pop=32, gens=100), denying any +10 collision events and producing
    a plateaued max-fitness curve.  At 0.3 the evaders still maintain
    distance and exhibit recognisable evasive motion, but with closeable
    gaps that let the EA discover catching policies in early generations.
    Once the EA reliably catches at 0.3, k_pursuer can be ramped up as
    a curriculum.
    """
    apf_cfg = APFConfig(
        k_pursuer=0.3,
        k_landmark=0.3,
        k_boundary=2.0,
        boundary_threshold=0.9,
    )
    return {
        name: APFEvaderPolicy(layout=env.get_layout(name), config=apf_cfg)
        for name in env.good_agents
    }


class SoftmaxSamplingHead(ActionHead):
    """Stochastic softmax-sampling action head for Discrete spaces.

    The default :class:`DiscreteHead` returns ``argmax(logits)``, which
    creates a step-function fitness landscape: small weight perturbations
    almost never flip the argmax, so gradient-free search crawls.  This
    head instead samples from ``softmax(logits / temperature)``, giving a
    smooth fitness landscape where every weight change continuously
    shifts the action distribution.

    Lower ``temperature`` -> closer to argmax (sharper).  Higher
    ``temperature`` -> closer to uniform (more exploratory).
    ``temperature=1.0`` is the standard evolution-strategies default.

    Args:
        n: Number of discrete actions.
        temperature: Softmax temperature (must be > 0).
        rng: Optional ``numpy.random.Generator`` for reproducibility.
            Defaults to a fresh default_rng().
    """

    def __init__(
        self,
        n: int,
        temperature: float = 1.0,
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
        # Numerically stable softmax with temperature scaling.
        scaled = arr / self.temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return int(self.rng.choice(self.n, p=probs))


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    """All pursuers share the same evolved genome — homogeneous team NE.

    Uses :class:`SoftmaxSamplingHead` instead of the default argmax
    :class:`DiscreteHead` so that small weight changes produce continuous
    changes in the action *distribution* (and therefore in expected
    fitness).  With argmax, the fitness landscape was a step function:
    tiny weight perturbations almost never flipped the argmax, so the
    EA crawled at staircase pace (max -892 -> -703 over 100 gens with
    no catches).  With softmax sampling at temperature=1.0, every weight
    perturbation shifts the action probabilities smoothly — standard
    technique for evolution strategies on discrete-action environments.
    """
    # NOTE: Temperature 0.3 (was 1.0).  The genome's output layer applies
    # tanh, clamping logits to [-1, 1].  At T=1.0, softmax of ([+1,-1,0,0,0])
    # only places ~45% on the correct cardinal direction — too noisy for
    # any pursuit signal to survive.  At T=0.3, the same clamped logits give
    # ~90% on the correct direction while preserving a smooth (non-step)
    # fitness landscape (small weight changes still shift probabilities
    # continuously, unlike argmax).
    head = SoftmaxSamplingHead(
        n=int(action_space.n),
        temperature=0.3,
        rng=np.random.default_rng(),
    )
    return FeedforwardNeuralPolicy(genome, head=head)


# ---------------------------------------------------------------------------
# Oracle-aware warm start
# ---------------------------------------------------------------------------
#
# Sanity-check finding (oracle test, 5 episodes): a hand-coded "go toward
# nearest prey" policy averages 22.4 catches and +13.8 reward; our EA from
# random init averages 0 catches and -734.  The env is fully winnable; the
# EA simply cannot find pursuit from random init within 32 * 100 = 3,200
# evaluations over 357 weights.  Standard ES fix: warm-start the population
# near a known-decent genome (here, hand-crafted) and let mutation/selection
# refine from there.
#
# The oracle genome encodes:
#   hidden[0] = sum of dx components in `good_rel` slice (centroid pull X)
#   hidden[1] = sum of dy components in `good_rel` slice (centroid pull Y)
#   hidden[2:] = small random noise (allows future specialisation)
#   output[0] (no-op) = 0
#   output[1] (-x)    = -K * tanh(hidden[0])
#   output[2] (+x)    = +K * tanh(hidden[0])
#   output[3] (-y)    = -K * tanh(hidden[1])
#   output[4] (+y)    = +K * tanh(hidden[1])
# With softmax temperature=1.0, K=4 produces a sharp distribution favouring
# the cardinal direction toward the prey centroid.
#
# Pursuing the centroid (rather than the *nearest*) prey is a small
# approximation but the EA can refine "centroid -> nearest" given a working
# pursuit baseline.

ORACLE_OUTPUT_GAIN: float = 4.0


def _build_oracle_weights(
    layer_sizes: List[int],
    good_rel_slice: slice,
    init_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a flat weight vector encoding the centroid-pursuit oracle.

    Args:
        layer_sizes: ``[input, hidden, output]`` (must have one hidden layer).
        good_rel_slice: Slice into the obs vector covering the flattened
            relative-position block for good agents (length 2*N_good).
        init_scale: Scale used for the small-noise residual weights.
        rng: RNG for the residual noise so each seeded individual is unique.

    Returns:
        Flat weight vector compatible with :class:`FixedTopologyGenome`.
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

    # ---- Layer 1: obs -> hidden  (W1 shape (hidden, input), b1 shape (hidden,)) ----
    W1 = rng.uniform(-init_scale, init_scale, size=(hidden_size, input_size))
    b1 = rng.uniform(-init_scale, init_scale, size=(hidden_size,))

    # Override hidden[0] and hidden[1] with the centroid-pursuit projection.
    # good_rel block is [dx_0, dy_0, dx_1, dy_1, ...] of length 2*N_good.
    rel_indices = list(range(good_rel_slice.start, good_rel_slice.stop))
    if len(rel_indices) % 2 != 0:
        raise ValueError(
            f"good_rel slice length {len(rel_indices)} is not a multiple of 2"
        )
    dx_indices = rel_indices[0::2]
    dy_indices = rel_indices[1::2]

    W1[0, :] = 0.0
    W1[1, :] = 0.0
    # Single-prey targeting: each oracle individual chases ONE randomly
    # chosen prey index, not the centroid.  Centroid pursuit fails when
    # prey scatter around the pursuer (sum of dx cancels to ~0, killing
    # hidden[0]).  Single-prey gives a strong, non-cancelling signal.
    # With ~16 oracle individuals and 4 prey, ~4 individuals chase each
    # prey naturally producing team coverage.
    n_good = len(dx_indices)
    target_prey = int(rng.integers(0, n_good))
    # Saturate tanh on hidden[0,1] so they read as sign(dx_target),
    # sign(dy_target) regardless of absolute distance.  W=5.0 means
    # |dx|>0.2 already saturates tanh near ±1.
    SATURATION_GAIN = 5.0
    W1[0, dx_indices[target_prey]] = SATURATION_GAIN
    W1[1, dy_indices[target_prey]] = SATURATION_GAIN
    b1[0] = 0.0
    b1[1] = 0.0

    # ---- Layer 2: hidden -> logits  (W2 shape (5, hidden), b2 shape (5,)) ----
    W2 = rng.uniform(-init_scale, init_scale, size=(output_size, hidden_size))
    b2 = rng.uniform(-init_scale, init_scale, size=(output_size,))

    # Wire only hidden[0] and hidden[1] to the cardinal-direction logits;
    # leave the rest of the hidden->logit matrix as small noise so the EA
    # can grow specialised features in hidden[2:] without disrupting the
    # baseline pursuit behaviour.
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
    the remainder are random per ``toolbox.individual``.  Perturbation is
    necessary so DEAP's selection sees variance in initial fitness and
    crossover has different parents.
    """
    population = toolbox.population(n=population_size)
    n_oracle = int(round(population_size * oracle_fraction))
    if n_oracle <= 0:
        return population
    oracle = _build_oracle_weights(layer_sizes, good_rel_slice, init_scale, rng)
    n_weights = oracle.size
    for i in range(n_oracle):
        perturbed = oracle + rng.normal(0.0, seed_sigma, size=n_weights)
        # In this project Individual is NOT a list — it holds a
        # FixedTopologyGenome at .genome.  Write directly to the genome.
        population[i].genome.set_weights(perturbed.astype(np.float64))
        # Reset fitness so it gets evaluated fresh.
        if hasattr(population[i], "fitness") and population[i].fitness.valid:
            del population[i].fitness.values
    return population


# ---------------------------------------------------------------------------
# DEAP evolution loop
# ---------------------------------------------------------------------------


def run_evolution(
    recorder: Optional[FrameRecorder] = None,
    record_every_n: int = 1,
) -> Dict[str, Any]:
    """Evolve NN pursuers against scripted APF evaders.

    Returns a dict containing the best genome, the per-generation fitness
    history, and bookkeeping metadata.  Kept separate from ``main`` so
    callers (or tests) can invoke it directly.
    """
    # -----------------------------------------------------------------------
    # Mixed evaluator — factory path (picklable for parallel workers)
    # -----------------------------------------------------------------------
    # Using env_factory + scripted_policy_factory so MixedPolicyEvaluator
    # can dispatch genome eval to subprocess workers via
    # make_mixed_parallel_evaluate_map without pickling live env objects.
    evaluator = MixedPolicyEvaluator(
        env_factory=build_env,
        scripted_policy_factory=build_scripted_evaders,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        max_steps=ENV_CONFIG.max_cycles,
    )

    # Pursuer observation/action sizing (derived from evaluator's validated env).
    pursuer_id = evaluator.learner_agents[0]
    input_size = int(evaluator.env_wrapper.reset(seed=SEED)[pursuer_id].shape[0])
    act_space = evaluator.env_wrapper.action_space(pursuer_id)
    output_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"  Pursuer NN:  in={input_size}  hidden={HIDDEN_SIZE}  out={output_size}")

    # DEAP toolbox built from the bridge's integration helpers.
    #
    # Initial weight range was (-1, 1) but with input dim ~26 + tanh that
    # saturates the hidden layer immediately (pre-act std ~ sqrt(26/3)),
    # producing observation-blind "fixed-direction" policies — the source
    # of the erratic non-pursuit behavior in early generations.  We use
    # a Glorot-style range scaled to fan-in so random initial networks
    # actually respond to their input.
    setup_deap_creators()
    init_scale = float(np.sqrt(1.0 / max(input_size, 1)))
    genome_config = {
        "layer_sizes": [input_size, HIDDEN_SIZE, output_size],
        "activation": "tanh",
        "weight_range": (-init_scale, init_scale),
        "random_seed": SEED,
    }
    # Mutation sigma 0.1 + indpb 0.2: previous σ=0.05 / indpb=0.1 made the
    # population converge to a center-camping local optimum by gen 22 with
    # no further improvement.  Doubled both to inject more exploration while
    # elitism (ELITE_SIZE=2) protects the current champion from disruption.
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"tournament_size": 3},
        crossover_config={"alpha": 0.5},
        mutation_config={"sigma": 0.1, "indpb": 0.2},
    )
    toolbox.register("clone", clone_individual)

    # -----------------------------------------------------------------------
    # Parallel evaluation via make_mixed_parallel_evaluate_map
    # -----------------------------------------------------------------------
    n_workers = N_WORKERS if N_WORKERS > 0 else (os.cpu_count() or 1)
    print(f"  Workers      : {n_workers} ({'parallel' if n_workers > 1 else 'sequential'})")

    # Scripted evaders already built in evaluator's validation env — reuse
    # them for champion rollout recording (policies are stateless at episode
    # start; any env reference they hold is for layout config only).
    scripted_evaders_for_recording = evaluator.scripted_policies

    # Initial population — oracle-seeded warm start.
    # Half the population starts as small perturbations of a hand-crafted
    # centroid-pursuit oracle (see _build_oracle_weights); the other half
    # is random for diversity.  This injects pursuit behaviour into the
    # gene pool from gen 0, which the EA then refines.  Required because
    # 32 * 100 = 3,200 evaluations is far too small a budget to discover
    # pursuit from random init in a 357-weight network.
    rng = np.random.default_rng(SEED)
    pursuer_layout = evaluator.env_wrapper.get_layout(pursuer_id)
    population = _make_oracle_seeded_population(
        toolbox=toolbox,
        population_size=POPULATION_SIZE,
        oracle_fraction=0.5,
        layer_sizes=[input_size, HIDDEN_SIZE, output_size],
        good_rel_slice=pursuer_layout.good_rel,
        init_scale=init_scale,
        seed_sigma=0.05,
        rng=rng,
    )
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    fitness_history = []

    with make_executor(n_workers) as executor:
        pmap = make_mixed_parallel_evaluate_map(
            executor, evaluator, eval_episodes=EVAL_EPISODES_TRAIN
        )
        toolbox.register("map", pmap)

        # Evaluate gen 0
        for ind, fit in zip(population, toolbox.map(toolbox.evaluate, population)):
            ind.fitness.values = fit
        hof.update(population)
        rec = stats.compile(population)
        fitness_history.append({"gen": 0, **rec})
        print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")
        if recorder is not None and recorder.enabled:
            try:
                _record_champion_rollout(
                    recorder, 0, hof[0].genome, scripted_evaders_for_recording
                )
            except Exception:
                pass

        # Evolution loop
        for gen in range(1, NUM_GENERATIONS + 1):
            # Elitism: copy the top ELITE_SIZE individuals verbatim into
            # next generation.  Without this, +19-fitness champions seen
            # in our earlier logs were destroyed by mutation/crossover.
            elites = [
                toolbox.clone(ind)
                for ind in tools.selBest(population, ELITE_SIZE)
            ]

            # Variation: select POP_SIZE - ELITE_SIZE parents, breed.
            n_offspring = len(population) - ELITE_SIZE
            offspring = list(map(
                toolbox.clone,
                toolbox.select(population, n_offspring),
            ))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < 0.7:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation
            for m in offspring:
                if rng.random() < 0.3:
                    toolbox.mutate(m)
                    del m.fitness.values

            # Re-evaluate invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            # New population = elites + offspring (mu + lambda style).
            population[:] = elites + offspring
            hof.update(population)

            rec = stats.compile(population)
            fitness_history.append({"gen": gen, **rec})
            print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")

            if recorder is not None and recorder.enabled and gen % record_every_n == 0:
                try:
                    _record_champion_rollout(
                        recorder, gen, hof[0].genome, scripted_evaders_for_recording
                    )
                except Exception:
                    pass

    # Close the evaluator's validation env (built during factory-path __init__).
    evaluator.env_wrapper.close()

    return {
        "best_genome": hof[0].genome,
        "best_fitness": hof[0].fitness.values[0],
        "history": fitness_history,
    }


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def evaluate_best(
    best_genome: FixedTopologyGenome,
    n_episodes: int = EVAL_EPISODES,
) -> Dict[str, float]:
    """Evaluate the evolved genome across ``n_episodes`` fresh seeds."""
    setup_deap_creators()
    from deap import creator
    individual = creator.Individual()
    individual.genome = best_genome

    evaluator = MixedPolicyEvaluator(
        env_factory=build_env,
        scripted_policy_factory=build_scripted_evaders,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        max_steps=ENV_CONFIG.max_cycles,
    )

    learner_rewards = []
    scripted_rewards = []
    try:
        for i in range(n_episodes):
            res = evaluator.run_episode(individual, seed=1000 + i)
            learner_rewards.append(res.learner_reward)
            scripted_rewards.append(res.scripted_reward)
    finally:
        evaluator.env_wrapper.close()

    return {
        "pursuer_mean": float(np.mean(learner_rewards)),
        "pursuer_std":  float(np.std(learner_rewards)),
        "pursuer_min":  float(np.min(learner_rewards)),
        "pursuer_max":  float(np.max(learner_rewards)),
        "evader_mean":  float(np.mean(scripted_rewards)),
        "evader_std":   float(np.std(scripted_rewards)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _section("Pursuit-Evasion Prototype: EA Pursuers vs APF Evaders")
    print(f"  Scenario     : {ENV_CONFIG.num_adversaries} pursuers vs "
          f"{ENV_CONFIG.num_good} evaders, {ENV_CONFIG.num_obstacles} obstacles")
    visibility = (
        f"{ENV_CONFIG.num_agent_neighbors} agents / "
        f"{ENV_CONFIG.num_landmark_neighbors} landmarks (K-nearest PO)"
        if ENV_CONFIG.num_agent_neighbors is not None
        else "full observability (all agents/landmarks)"
    )
    print(f"  Visibility   : {visibility}")
    print(f"  Equal-ability: max_speed={ENV_CONFIG.equal_max_speed}  "
          f"accel={ENV_CONFIG.equal_accel}")
    print(f"  EA setup     : pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}, "
          f"hidden={HIDDEN_SIZE}, eval_eps_train={EVAL_EPISODES_TRAIN}")
    print(f"  Max steps/ep : {ENV_CONFIG.max_cycles}")
    print(f"  Reward shape : {'on (dist=-1.0, v_toward=+0.5)' if REWARD_SHAPING else 'off (sparse)'}")
    print(f"  Recording    : {'enabled (every %d gen)' % RECORD_EVERY_N_GENS if RECORD else 'disabled'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ea_logger = ExperimentLogger.create(
        OUTPUT_DIR / "ea_pursuers",
        algorithm="ea",
        env_name="pursuit_evasion",
        seed=SEED,
        budget_mode="eval_episodes",
        budget_value=POPULATION_SIZE * (NUM_GENERATIONS + 1),
        hyperparams={
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "hidden_size": HIDDEN_SIZE,
            "max_steps_per_episode": ENV_CONFIG.max_cycles,
            "eval_episodes_train": EVAL_EPISODES_TRAIN,
            "opponent": "apf_scripted",
            "num_adversaries": ENV_CONFIG.num_adversaries,
            "num_good": ENV_CONFIG.num_good,
            "num_obstacles": ENV_CONFIG.num_obstacles,
        },
        notes="EA pursuers vs APF evaders.",
    )
    recorder: Optional[FrameRecorder] = None
    if RECORD:
        recorder = ea_logger.get_frame_recorder(
            fps=RECORD_FPS, fmt="mp4", episode_prefix="ea_"
        )

    _section("Evolution")
    t0 = time.time()
    result = run_evolution(recorder=recorder, record_every_n=RECORD_EVERY_N_GENS)
    evolution_time = time.time() - t0
    print(f"\n  Evolution complete in {evolution_time:.1f}s")
    print(f"  Best training fitness: {result['best_fitness']:+.2f}")

    _section(f"Final evaluation ({EVAL_EPISODES} fresh seeds)")
    t1 = time.time()
    # Record one final champion rollout tagged beyond the generation range.
    if recorder is not None and recorder.enabled:
        try:
            _evaders_tmp = build_scripted_evaders(build_env())
            _record_champion_rollout(
                recorder, NUM_GENERATIONS + 1, result["best_genome"], _evaders_tmp
            )
        except Exception:
            pass

    eval_stats = evaluate_best(result["best_genome"], n_episodes=EVAL_EPISODES)
    eval_time = time.time() - t1
    print(f"  Pursuer team reward : {eval_stats['pursuer_mean']:+.2f} "
          f"± {eval_stats['pursuer_std']:.2f}   "
          f"range=[{eval_stats['pursuer_min']:+.2f}, {eval_stats['pursuer_max']:+.2f}]")
    print(f"  APF evader reward   : {eval_stats['evader_mean']:+.2f} "
          f"± {eval_stats['evader_std']:.2f}")

    _section("Timings")
    print(f"  Evolution : {evolution_time:6.1f}s")
    print(f"  Eval      : {eval_time:6.1f}s")
    print(f"  Total     : {evolution_time + eval_time:6.1f}s")

    with ea_logger:
        for h in result["history"]:
            ea_logger.row("metrics", {
                "iteration": int(h["gen"]),
                "env_steps": int(h["gen"]) * POPULATION_SIZE,
                "wall_time": 0.0,
                "mean": float(h["avg"]),
                "std": float(h["std"]),
                "min": float(h["min"]),
                "max": float(h["max"]),
                "best": float(h["max"]),
            })
        ea_logger.summary({
            "final_fitness": float(result["best_fitness"]),
            "best_fitness": float(result["best_fitness"]),
            "num_generations": NUM_GENERATIONS,
            "population_size": POPULATION_SIZE,
            "training_time_s": float(evolution_time),
            "eval_pursuer_mean": eval_stats["pursuer_mean"],
            "eval_pursuer_std": eval_stats["pursuer_std"],
            "eval_evader_mean": eval_stats["evader_mean"],
        })

    np.savez(
        OUTPUT_DIR / "fitness_history.npz",
        history=np.array(
            [(h["gen"], h["avg"], h["std"], h["min"], h["max"]) for h in result["history"]],
            dtype=[("gen", "i4"), ("avg", "f4"), ("std", "f4"), ("min", "f4"), ("max", "f4")],
        ),
    )
    print(f"\n  Saved fitness history to {OUTPUT_DIR / 'fitness_history.npz'}")
    print(f"  Logged run to {ea_logger.run_dir}")
    _section("Done")


if __name__ == "__main__":
    main()

