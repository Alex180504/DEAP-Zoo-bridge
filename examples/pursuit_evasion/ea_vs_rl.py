"""Pursuit-evasion EA vs RL — honours-project experiment.

Mirrors :mod:`examples.pursuit_evasion_ea_vs_apf` but swaps the scripted
APF evaders for **MADDPG-trained evaders** (via AgileRL). The pursuit
side is still trained by DEAP-driven neuroevolution; only the target
opponent changes.

Methodology
-----------

* Environment: MPE2 ``simple_tag_v3`` via
  :class:`~deap_pettingzoo.environments.PettingZooWrapper` (3 pursuers +
  1 evader, 2 landmarks). Vanilla (no partial-observability wrapping)
  so that MADDPG's trained observation shapes match those the EA
  pursuers see at evaluation time.
* Stage 1: train MADDPG on ``simple_tag_v3`` for ``MADDPG_STEPS`` env
  steps using the project's :class:`MADDPGTrainer`. This trains actors
  for *all* agents simultaneously; afterwards we re-wrap the trained
  underlying agent with :class:`AgileRLPolicyWrapper` restricted to the
  evader IDs so it only dispatches actions for ``agent_*``.
* Stage 2: evolve EA pursuers (homogeneous ``adversary_*`` agents
  sharing one NN genome) with DEAP. Evaluation uses
  :class:`MixedPolicyEvaluator` with:

  - ``learner_agent_prefixes=['adversary']`` — EA trains only the
    pursuer side.
  - ``scripted_policies={agent_id: AgileRLPolicyWrapper(...)}`` — a
    *frozen* MADDPG opponent driving the evader.

  This matches the research protocol of training one side against a
  stationary benchmark opponent so that learning signal isolates
  algorithmic efficiency rather than co-adaptation dynamics.
* Stage 3: final evaluation across fresh seeds.

Requires the optional ``rl-baselines`` extra::

    pip install 'deap-pettingzoo[rl-baselines]'
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import types

import numpy as np
from deap import tools

from deap_pettingzoo.baselines import (
    BudgetMode,
    MADDPGTrainer,
    RecordingConfig,
    TrainingConfig,
)
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import MixedPolicyEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    clone_individual,
    create_toolbox,
    setup_deap_creators,
)
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

ENV_NAME = "simple_tag_v3"
ENV_CONFIG = {"max_cycles": 200}
SEED = 0

# Enable per-step distance-based reward shaping for adversaries.
# simple_tag's default `adversary_reward` is purely sparse (+10 on collision).
# With `REWARD_SHAPING = True`, each adversary additionally receives
# -0.1 × (its own Euclidean distance to the nearest prey) per step, giving
# the EA a dense gradient to follow rather than an all-zero fitness landscape.
REWARD_SHAPING = True

MADDPG_STEPS = 300_000
MADDPG_BATCH = 128
MADDPG_MEMORY = 100_000
MADDPG_WARMUP = 1_000

POPULATION_SIZE = 32
NUM_GENERATIONS = 100
HIDDEN_SIZE = 16      # 357 params vs 1413 — tractable for gradient-free search at pop=32

# Episodes averaged per genome per generation.  Single-episode evaluation is
# too noisy for reliable selection (lucky start ≠ better genome).  3 episodes
# adds 3× cost but roughly halves the ranking noise.
EVAL_EPISODES_TRAIN = 3

EVAL_EPISODES = 50

OUTPUT_DIR = Path("results/pursuit_evasion_ea_vs_rl")

# ---------------------------------------------------------------------------
# Recording config
# ---------------------------------------------------------------------------

RECORD = True            # set False to disable all recording
RECORD_FPS = 20
RECORD_EVERY_N_GENS = 1  # record champion every N generations (1 = every gen)
# Fixed seed for champion rollout recordings so obstacles always start in the
# same positions across generations — makes visual comparison easy.
RECORD_SEED = 9000


def _enable_reward_shaping(wrapper: "PettingZooWrapper") -> None:
    """Patch MPE simple_tag to enable per-step distance shaping for adversaries.

    Replaces the Scenario's ``adversary_reward`` method on the *instance* (so
    no other envs are affected) with a version that gives each adversary a
    continuous penalty of ``-0.1 × distance_to_nearest_prey`` at every step
    on top of the existing sparse ``+10`` collision bonus.  This produces a
    dense fitness landscape so DEAP's EA can learn pursuit behaviour.

    Silently no-ops if the env is not an MPE environment or if the expected
    internal attribute chain is absent.
    """
    try:
        # PettingZooWrapper.env  → parallel_env wrapper
        # .aec_env               → OrderEnforcingWrapper
        # .env                   → AssertOutOfBoundsWrapper
        # .env                   → raw_env (SimpleEnv) — owns .scenario
        base = wrapper.env.aec_env.env.env
        scenario = base.scenario
    except AttributeError:
        return

    def _shaped_adversary_reward(self, agent, world):
        rew = 0.0
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # Dense shaping: penalise this adversary's Euclidean distance to the
        # nearest good agent so the EA receives a non-zero gradient every step.
        if good_agents:
            rew -= 0.2 * min(
                float(np.linalg.norm(agent.state.p_pos - a.state.p_pos))
                for a in good_agents
            )
        # Sparse collision bonus (unchanged from original).
        if agent.collide:
            for ag in good_agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10.0
        return rew

    scenario.adversary_reward = types.MethodType(_shaped_adversary_reward, scenario)


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
    maddpg_evaders: AgileRLPolicyWrapper,
) -> None:
    """Render one champion-vs-MADDPG episode and stream frames into *recorder*.

    Uses a dedicated ``rgb_array`` env so the training env's RNG/state is
    not perturbed. Best-effort: the caller should wrap this in a
    ``try/except`` so recording failures never break the experiment.
    """
    record_env = PettingZooWrapper(
        env_name=ENV_NAME, env_config=ENV_CONFIG, render_mode="rgb_array"
    )  # _patch_fixed_cam_range applied automatically by PettingZooWrapper
    try:
        pursuer_id = next(a for a in record_env.possible_agents if a.startswith("adversary"))
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

        for _ in range(ENV_CONFIG["max_cycles"]):
            if not obs_dict:
                break
            actions: Dict[str, Any] = {}
            for agent_id, obs in obs_dict.items():
                if agent_id.startswith("adversary"):
                    try:
                        actions[agent_id] = pursuer_policy.get_action(
                            obs, action_space=act_space
                        )
                    except Exception:
                        actions[agent_id] = record_env.action_space(agent_id).sample()
                else:
                    try:
                        actions[agent_id] = maddpg_evaders.get_action(obs)
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
# Stage 1 — Train MADDPG evader opponent
# ---------------------------------------------------------------------------


def train_maddpg_opponent() -> AgileRLPolicyWrapper:
    """Train MADDPG on ``simple_tag_v3`` and return a wrapper restricted
    to the evader agent IDs only.
    """
    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    print(f"  MADDPG device: {_device}")
    trainer = MADDPGTrainer(device=_device, verbose=False)
    cfg = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=MADDPG_STEPS,
        algorithm_params={
            "seed": SEED,
            "batch_size": MADDPG_BATCH,
            "memory_size": MADDPG_MEMORY,
            "learning_delay": MADDPG_WARMUP,
            "max_steps_per_episode": ENV_CONFIG["max_cycles"],
        },
        # MADDPGTrainer auto-logs through build_logger when output_dir
        # is set; emits its own manifest/episodes/summary under this dir.
        output_dir=OUTPUT_DIR / "maddpg_opponent",
        recording=RecordingConfig(
            enabled=RECORD,
            fps=RECORD_FPS,
            episode_prefix="maddpg_",
        ),
    )

    result = trainer.train(cfg)
    full_wrapper: AgileRLPolicyWrapper = result.policy  # type: ignore[assignment]
    evader_ids = [
        a for a in full_wrapper.agent_ids if not a.startswith("adversary")
    ]
    evader_only = AgileRLPolicyWrapper(
        full_wrapper.underlying_agent,
        evader_ids,
        algorithm_name="maddpg-evader",
    )

    print(
        f"  Trained MADDPG in {result.training_time:.1f}s "
        f"(episodes={len(result.metrics.get('episode_rewards', []))}, "
        f"env_steps={MADDPG_STEPS:,})"
    )
    print(f"  Evader IDs under MADDPG control: {evader_ids}")
    return evader_only


# ---------------------------------------------------------------------------
# Stage 2 — Evolve EA pursuers against frozen MADDPG evaders
# ---------------------------------------------------------------------------


def build_env() -> PettingZooWrapper:
    env = PettingZooWrapper(env_name=ENV_NAME, env_config=ENV_CONFIG)
    if REWARD_SHAPING:
        _enable_reward_shaping(env)
    return env


def _learner_factory(genome: FixedTopologyGenome, agent_id: str, action_space):
    return FeedforwardNeuralPolicy(genome, action_space=action_space)


def run_evolution(
    maddpg_evaders: AgileRLPolicyWrapper,
    recorder: Optional[FrameRecorder] = None,
    record_every_n: int = 1,
) -> Dict[str, Any]:
    """Evolve EA pursuers against the frozen MADDPG evaders."""
    env = build_env()

    pursuer_id = next(a for a in env.possible_agents if a.startswith("adversary"))
    sample_obs = env.reset(seed=SEED)
    input_size = int(sample_obs[pursuer_id].shape[0])
    act_space = env.action_space(pursuer_id)
    output_size = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"  Pursuer NN:  in={input_size}  hidden={HIDDEN_SIZE}  out={output_size}")

    evader_ids = [a for a in env.possible_agents if not a.startswith("adversary")]
    # The same wrapper instance is reused for every evader key; the
    # wrapper rotates internally through its own agent_ids list in the
    # same order that MixedPolicyEvaluator iterates observations.
    scripted_policies = {aid: maddpg_evaders for aid in evader_ids}

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        scripted_policies=scripted_policies,
        max_steps=ENV_CONFIG["max_cycles"],
    )

    setup_deap_creators()
    genome_config = {
        "layer_sizes": [input_size, HIDDEN_SIZE, output_size],
        "activation": "tanh",
        "weight_range": (-1.0, 1.0),
        "random_seed": SEED,
    }
    toolbox = create_toolbox(
        genome_config=genome_config,
        evaluator=evaluator,
        selection_config={"tournament_size": 3},
        crossover_config={"alpha": 0.5},
        mutation_config={"sigma": 0.2, "indpb": 0.1},
    )
    toolbox.register("clone", clone_individual)

    # Override evaluate to average over EVAL_EPISODES_TRAIN independent episodes.
    # This reduces selection noise from lucky/unlucky starting positions without
    # changing the per-episode reward structure.
    if EVAL_EPISODES_TRAIN > 1:
        _ev = evaluator
        _n_eval = EVAL_EPISODES_TRAIN
        toolbox.register(
            "evaluate",
            lambda ind: (
                sum(_ev.run_episode(ind).fitness for _ in range(_n_eval)) / _n_eval,
            ),
        )

    rng = np.random.default_rng(SEED)
    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    history = []
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit
    hof.update(population)
    rec = stats.compile(population)
    history.append({"gen": 0, **rec})
    print(f"  Gen  0 | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")
    if recorder is not None and recorder.enabled:
        try:
            _record_champion_rollout(recorder, 0, hof[0].genome, maddpg_evaders)
        except Exception:
            pass

    for gen in range(1, NUM_GENERATIONS + 1):
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if rng.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        hof.update(population)
        rec = stats.compile(population)
        history.append({"gen": gen, **rec})
        print(f"  Gen {gen:>2} | mean={rec['avg']:+.2f}  max={rec['max']:+.2f}")
        if recorder is not None and recorder.enabled and gen % record_every_n == 0:
            try:
                _record_champion_rollout(recorder, gen, hof[0].genome, maddpg_evaders)
            except Exception:
                pass

    return {
        "best_genome": hof[0].genome,
        "best_fitness": hof[0].fitness.values[0],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Final evaluation
# ---------------------------------------------------------------------------


def evaluate_best(
    best_genome: FixedTopologyGenome,
    maddpg_evaders: AgileRLPolicyWrapper,
    n_episodes: int = EVAL_EPISODES,
) -> Dict[str, float]:
    env = build_env()
    evader_ids = [a for a in env.possible_agents if not a.startswith("adversary")]
    scripted_policies = {aid: maddpg_evaders for aid in evader_ids}

    setup_deap_creators()
    from deap import creator

    individual = creator.Individual()
    individual.genome = best_genome

    evaluator = MixedPolicyEvaluator(
        env_wrapper=env,
        learner_agent_prefixes=["adversary"],
        learner_factory=_learner_factory,
        scripted_policies=scripted_policies,
        max_steps=ENV_CONFIG["max_cycles"],
    )

    pursuer_rewards, evader_rewards = [], []
    for i in range(n_episodes):
        res = evaluator.run_episode(individual, seed=1000 + i)
        pursuer_rewards.append(res.learner_reward)
        evader_rewards.append(res.scripted_reward)

    return {
        "pursuer_mean": float(np.mean(pursuer_rewards)),
        "pursuer_std": float(np.std(pursuer_rewards)),
        "pursuer_min": float(np.min(pursuer_rewards)),
        "pursuer_max": float(np.max(pursuer_rewards)),
        "evader_mean": float(np.mean(evader_rewards)),
        "evader_std": float(np.std(evader_rewards)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _section("Pursuit-Evasion Experiment: EA Pursuers vs MADDPG (AgileRL) Evaders")
    print(f"  Environment  : {ENV_NAME} (vanilla parallel_env)")
    print(f"  MADDPG budget: {MADDPG_STEPS:,} env steps")
    print(
        f"  EA setup     : pop={POPULATION_SIZE}, gens={NUM_GENERATIONS}, "
        f"hidden={HIDDEN_SIZE}"
    )
    print(f"  Max steps/ep : {ENV_CONFIG['max_cycles']}")
    print(f"  Recording    : {'enabled (every %d gen)' % RECORD_EVERY_N_GENS if RECORD else 'disabled'}")

    # Create the EA experiment logger early so we can attach a FrameRecorder
    # before evolution starts and record champion rollouts per generation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ea_logger = ExperimentLogger.create(
        OUTPUT_DIR / "ea_pursuers",
        algorithm="ea",
        env_name=ENV_NAME,
        seed=SEED,
        budget_mode="eval_episodes",
        budget_value=POPULATION_SIZE * (NUM_GENERATIONS + 1),
        hyperparams={
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "hidden_size": HIDDEN_SIZE,
            "max_steps_per_episode": ENV_CONFIG["max_cycles"],
            "opponent": "maddpg_frozen",
            "maddpg_steps": MADDPG_STEPS,
        },
        notes="EA pursuers vs frozen MADDPG evaders.",
    )
    recorder: Optional[FrameRecorder] = None
    if RECORD:
        recorder = ea_logger.get_frame_recorder(
            fps=RECORD_FPS, fmt="mp4", episode_prefix="ea_"
        )

    _section("Stage 1 — Train MADDPG evader opponent")
    t0 = time.time()
    maddpg_evaders = train_maddpg_opponent()
    stage1_time = time.time() - t0

    _section("Stage 2 — Evolve EA pursuers against frozen MADDPG evaders")
    t0 = time.time()
    evo = run_evolution(maddpg_evaders, recorder=recorder, record_every_n=RECORD_EVERY_N_GENS)
    stage2_time = time.time() - t0
    print(f"\n  Evolution complete in {stage2_time:.1f}s")
    print(f"  Best training fitness: {evo['best_fitness']:+.2f}")

    _section(f"Stage 3 — Final evaluation ({EVAL_EPISODES} fresh seeds)")
    t0 = time.time()
    stats = evaluate_best(evo["best_genome"], maddpg_evaders)
    stage3_time = time.time() - t0
    # Record one final champion rollout tagged beyond the generation range.
    if recorder is not None and recorder.enabled:
        try:
            _record_champion_rollout(
                recorder, NUM_GENERATIONS + 1, evo["best_genome"], maddpg_evaders
            )
        except Exception:
            pass
    print(
        f"  EA pursuer team : {stats['pursuer_mean']:+.2f} "
        f"± {stats['pursuer_std']:.2f}   "
        f"range=[{stats['pursuer_min']:+.2f}, {stats['pursuer_max']:+.2f}]"
    )
    print(
        f"  MADDPG evader   : {stats['evader_mean']:+.2f} "
        f"± {stats['evader_std']:.2f}"
    )

    _section("Timings")
    print(f"  Stage 1 (MADDPG train): {stage1_time:6.1f}s")
    print(f"  Stage 2 (EA evolve)   : {stage2_time:6.1f}s")
    print(f"  Stage 3 (final eval)  : {stage3_time:6.1f}s")
    print(f"  Total                 : {stage1_time + stage2_time + stage3_time:6.1f}s")

    # Flush metrics and summary into the logger that was created at the
    # start of main() (and whose run_dir already holds the recorded frames).
    with ea_logger:
        for h in evo["history"]:
            ea_logger.row("metrics", {
                "iteration": int(h["gen"]),
                "env_steps": int(h["gen"]) * POPULATION_SIZE,
                "wall_time": 0.0,
                "mean": float(h["avg"]),
                "max": float(h["max"]),
                "best": float(h["max"]),
            })
        ea_logger.summary({
            "final_fitness": float(evo["best_fitness"]),
            "best_fitness": float(evo["best_fitness"]),
            "num_generations": NUM_GENERATIONS,
            "population_size": POPULATION_SIZE,
            "training_time_s": float(stage2_time),
            "eval_pursuer_mean": stats["pursuer_mean"],
            "eval_pursuer_std": stats["pursuer_std"],
            "eval_evader_mean": stats["evader_mean"],
            "stage1_maddpg_train_time_s": float(stage1_time),
        })

    np.savez(
        OUTPUT_DIR / "summary.npz",
        history=np.array(
            [(h["gen"], h["avg"], h["max"]) for h in evo["history"]],
            dtype=[("gen", "i4"), ("avg", "f4"), ("max", "f4")],
        ),
        final_pursuer_mean=np.float32(stats["pursuer_mean"]),
        final_pursuer_std=np.float32(stats["pursuer_std"]),
        final_evader_mean=np.float32(stats["evader_mean"]),
    )
    _section(f"Saved summary to {OUTPUT_DIR / 'summary.npz'}")
    print(f"  Logged EA run to {ea_logger.run_dir}")


if __name__ == "__main__":
    main()
