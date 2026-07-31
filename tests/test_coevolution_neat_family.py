"""Cross-backend co-evolution contracts, exercised on every optimizer kind.

These pin the behaviours whose absence let the NEAT-family sides ship a
silent self-play bypass: opponents must come from the *opponent pool*,
fitness must be the learner side's reward only, episodes must respect
``max_steps_per_episode``, and the reported per-generation statistics must
cover the whole population rather than the handful of carried-over elites.

Style follows ``test_budget_fairness.py``: a real env, tiny populations, no
fixtures and no monkeypatching, so the whole module runs in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from deap_pettingzoo.baselines import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution.evaluator import (
    CoEvolutionEvaluator,
    GenomeIndividual,
)
from deap_pettingzoo.baselines.coevolution.pools import FrozenPolicyPool
from deap_pettingzoo.baselines.coevolution.sides import Side, make_side_state
from deap_pettingzoo.baselines.coevolution.trainer import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
)
from deap_pettingzoo.environments import PettingZooWrapper

neat = pytest.importorskip("neat")

ENV_NAME = "simple_tag_v3"
ENV_CONFIG = {"max_cycles": 25, "continuous_actions": False}
LEARNER_PREFIX = "adversary"
OPPONENT_PREFIX = "agent_"
MAX_STEPS = 10
POP_SIZE = 4

#: One entry per registered co-evolution optimizer that owns a population of
#: genomes and drives the shared evaluator.
OPTIMIZER_CONFIGS = {
    "ea": {"population_size": POP_SIZE, "hidden_size": 4},
    "cma": {"population_size": POP_SIZE, "hidden_size": 4},
    "pso": {"population_size": POP_SIZE, "hidden_size": 4},
    "neat": {"pop_size": POP_SIZE, "seed": 7},
    "hyperneat": {"pop_size": POP_SIZE, "seed": 7, "substrate_hidden_layers": [3]},
    "es_hyperneat": {"pop_size": POP_SIZE, "seed": 7},
}


#: ``simple_tag`` discrete action that pushes the evader steadily right; held
#: for a full episode it takes the evader past the arena bound, so its side
#: reward is reliably non-zero and distinguishable from the pursuers'.
DRIFT_ACTION = 2


class _TagPolicy:
    """Opponent double that records every observation it was asked about."""

    def __init__(self, tag: str, action: int = 0) -> None:
        self.tag = tag
        self.action = action
        self.calls = 0

    def get_action(self, observation, deterministic: bool = True, *, action_mask=None):
        del observation, deterministic, action_mask
        self.calls += 1
        return self.action

    def reset(self) -> None:
        pass


def _make_env() -> PettingZooWrapper:
    return PettingZooWrapper(ENV_NAME, env_config=dict(ENV_CONFIG))


def _make_side(optimizer: str) -> Side:
    return Side(
        name="learner",
        role_prefixes=(LEARNER_PREFIX,),
        optimizer=optimizer,
        optimizer_config=dict(OPTIMIZER_CONFIGS[optimizer]),
    )


def _initialised_state(optimizer: str, env: PettingZooWrapper):
    state = make_side_state(_make_side(optimizer))
    state.initialise(env, rng=np.random.default_rng(0))
    return state


def _first_individual(state):
    """Return one evaluatable member of *state*'s population.

    DEAP-backed sides carry ``.genome`` on their individuals already; NEAT
    populations hold bare genomes, which the evaluator consumes through
    :class:`GenomeIndividual`.
    """
    population = getattr(state, "population", None)
    if population:
        return population[0]
    return GenomeIndividual(
        genome=next(iter(state._population_obj.population.values()))
    )


def _make_evaluator(state, env, opponent, *, k_opponents=1, max_steps=MAX_STEPS):
    return CoEvolutionEvaluator(
        env_wrapper=env,
        learner_role_prefixes=(LEARNER_PREFIX,),
        learner_factory=state.learner_policy_factory(),
        opponent_sides=[("opponent", (OPPONENT_PREFIX,), FrozenPolicyPool([opponent]))],
        k_opponents=k_opponents,
        max_steps=max_steps,
    )


def _learner_policy(state, env):
    """Build one learner policy the way the evaluator would."""
    agent = next(a for a in env.possible_agents if a.startswith(LEARNER_PREFIX))
    individual = _first_individual(state)
    factory = state.learner_policy_factory()
    return factory(individual.genome, agent, env.action_space(agent))


def _replay_episode(env, learner_policy, opponent_policy, *, seed, max_steps):
    """Roll one episode independently, summing rewards per side.

    Deliberately does not reuse the evaluator's own accounting — the point
    is to check that partition from the outside.
    """
    observations = env.reset(seed=seed)
    learner_sum = 0.0
    opponent_sum = 0.0
    for _ in range(max_steps):
        actions = {
            agent: (
                learner_policy.get_action(obs, deterministic=True)
                if agent.startswith(LEARNER_PREFIX)
                else opponent_policy.get_action(obs)
            )
            for agent, obs in observations.items()
        }
        observations, rewards, terms, truncs, _ = env.step(actions)
        for agent, reward in rewards.items():
            if agent.startswith(LEARNER_PREFIX):
                learner_sum += float(reward)
            else:
                opponent_sum += float(reward)
        if (terms and all(terms.values())) or (truncs and all(truncs.values())):
            break
    return learner_sum, opponent_sum


@pytest.mark.parametrize("optimizer", sorted(OPTIMIZER_CONFIGS))
class TestOpponentsComeFromThePool:
    """Every backend must drive opponents from the pool, not from itself."""

    def test_opponent_policy_drives_the_opposing_side(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        opponent = _TagPolicy("pool")
        evaluator = _make_evaluator(state, env, opponent)

        evaluator.evaluate(_first_individual(state))

        # One decision per opposing agent per step. Zero would mean the
        # learner's own policy drove the whole roster (the self-play bypass).
        assert opponent.calls > 0
        assert opponent.calls == MAX_STEPS * len(
            [a for a in env.possible_agents if a.startswith(OPPONENT_PREFIX)]
        )

    def test_learner_policy_is_not_the_opponent_policy(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        individual = _first_individual(state)
        learner_agent = next(
            a for a in env.possible_agents if a.startswith(LEARNER_PREFIX)
        )
        policy = state.learner_policy_factory()(
            individual.genome, learner_agent, env.action_space(learner_agent)
        )
        assert not isinstance(policy, _TagPolicy)

    def test_fitness_is_the_learner_sides_reward_only(self, optimizer):
        episode_steps = int(ENV_CONFIG["max_cycles"])
        env = _make_env()
        state = _initialised_state(optimizer, env)
        evaluator = _make_evaluator(
            state,
            env,
            _TagPolicy("pool", action=DRIFT_ACTION),
            max_steps=episode_steps,
        )

        (fitness,) = evaluator.evaluate(_first_individual(state), seed=0)

        learner_sum, opponent_sum = _replay_episode(
            env,
            _learner_policy(state, env),
            _TagPolicy("replay", action=DRIFT_ACTION),
            seed=0,
            max_steps=episode_steps,
        )
        assert fitness == pytest.approx(learner_sum)
        # The evader is driven past the arena bound, so its own reward is
        # non-zero and a sum over *all* agents is a different number — that
        # sum is precisely what the bypassed loop reported as fitness.
        assert opponent_sum < 0.0
        assert fitness != pytest.approx(learner_sum + opponent_sum)

    def test_episode_respects_max_steps_per_episode(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        evaluator = _make_evaluator(state, env, _TagPolicy("pool"), k_opponents=2)

        evaluator.evaluate(_first_individual(state))

        assert evaluator.last_details.steps == 2 * MAX_STEPS
        assert evaluator.env_steps_consumed == 2 * MAX_STEPS


class _CountingEvaluator:
    """Evaluator proxy counting ``evaluate`` calls and the scores returned."""

    def __init__(self, inner: CoEvolutionEvaluator) -> None:
        self._inner = inner
        self.calls = 0
        self.fitnesses: list[float] = []

    def evaluate(self, individual, seed=None, rng=None):
        self.calls += 1
        result = self._inner.evaluate(individual, seed=seed, rng=rng)
        self.fitnesses.append(float(result[0]))
        return result

    @property
    def env_steps_consumed(self) -> int:
        return self._inner.env_steps_consumed


@pytest.mark.parametrize("optimizer", ["neat", "hyperneat", "es_hyperneat"])
class TestNeatFamilyGenerationStats:
    """The NEAT sides must report over the whole population, not the elites."""

    def test_stats_cover_every_genome(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        proxy = _CountingEvaluator(_make_evaluator(state, env, _TagPolicy("pool")))

        stats = state.step_generation(lambda _side: proxy)

        # ``pop_size`` evaluations per NEAT generation — reading fitnesses back
        # off the post-reproduction population would leave 2-6 of them.
        assert proxy.calls == POP_SIZE
        assert stats.mean == pytest.approx(float(np.mean(proxy.fitnesses)))
        assert stats.std == pytest.approx(float(np.std(proxy.fitnesses)))
        assert stats.min == pytest.approx(float(np.min(proxy.fitnesses)))
        assert stats.max == pytest.approx(float(np.max(proxy.fitnesses)))

    def test_env_steps_match_the_evaluator_counter(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        proxy = _CountingEvaluator(_make_evaluator(state, env, _TagPolicy("pool")))

        stats = state.step_generation(lambda _side: proxy)

        assert stats.env_steps > 0
        assert stats.env_steps == proxy.env_steps_consumed
        assert stats.env_steps == POP_SIZE * MAX_STEPS
        assert stats.wall_time_s > 0.0

    def test_bootstrap_populates_every_fitness(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        # The trainer's bootstrap gate is a hasattr check on these two names.
        assert hasattr(state, "rebind_evaluator")
        assert hasattr(state, "evaluate_population")

        state.rebind_evaluator(_make_evaluator(state, env, _TagPolicy("pool")))
        state.evaluate_population()

        genomes = list(state._population_obj.population.values())
        assert len(genomes) == POP_SIZE
        assert all(g.fitness is not None for g in genomes)
        # The trainer snapshots the bootstrap champion straight after this, so
        # best_fitness must already be real — not the -inf sentinel.
        assert np.isfinite(state.best_fitness)
        assert state.best_fitness == max(float(g.fitness) for g in genomes)

    def test_history_and_generation_advance(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        proxy = _CountingEvaluator(_make_evaluator(state, env, _TagPolicy("pool")))

        state.step_generation(lambda _side: proxy)
        state.step_generation(lambda _side: proxy)

        assert state.generation == 2
        assert [s.generation for s in state.history] == [1, 2]

    def test_best_snapshot_is_a_copy(self, optimizer):
        env = _make_env()
        state = _initialised_state(optimizer, env)
        proxy = _CountingEvaluator(_make_evaluator(state, env, _TagPolicy("pool")))
        state.step_generation(lambda _side: proxy)

        snapshot = state.best_snapshot()
        assert snapshot is not state._best_genome
        assert snapshot.key == state._best_genome.key

    def test_learner_snapshot_survives_a_pickle_round_trip(self, optimizer):
        import pickle

        from deap_pettingzoo.baselines.coevolution.pools import (
            rebuild_opponent_policy,
        )

        env = _make_env()
        state = _initialised_state(optimizer, env)
        individual = _first_individual(state)

        snapshot = state.learner_snapshot(individual.genome)
        revived = pickle.loads(pickle.dumps(snapshot))
        learner_agent = next(
            a for a in env.possible_agents if a.startswith(LEARNER_PREFIX)
        )
        policy = rebuild_opponent_policy(
            revived, action_space=env.action_space(learner_agent)
        )
        obs = env.reset(seed=0)[learner_agent]
        assert isinstance(policy.get_action(obs), (int, np.integer))


@pytest.mark.parametrize("optimizer", ["neat", "hyperneat", "es_hyperneat"])
def test_neat_family_trains_end_to_end(optimizer):
    """A NEAT-family side must survive a real ``CoEvolutionTrainer.train()``."""
    per_gen: list = []

    def _callback(gen, states):
        del gen
        state = states["learner"]
        if state.history:
            per_gen.append(state.history[-1])

    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            _make_side(optimizer),
            Side(
                name="opponent",
                role_prefixes=(OPPONENT_PREFIX,),
                optimizer="ea",
                optimizer_config={"population_size": POP_SIZE, "hidden_size": 4},
            ),
        ],
        k_opponents=1,
        max_steps_per_episode=MAX_STEPS,
        seed=0,
        reporting_side="learner",
        generation_callback=_callback,
    )
    config = TrainingConfig(
        env_name=ENV_NAME,
        budget_mode=BudgetMode.GENERATIONS,
        budget_value=2,
        env_config=dict(ENV_CONFIG),
        coevolution_config=coev_cfg,
    )
    result = CoEvolutionTrainer().train(config)

    assert result.metrics["num_generations"] == 2
    assert len(per_gen) == 2
    assert all(s.env_steps > 0 for s in per_gen)
    # Real per-generation ticks, capped by the configured episode length.
    assert all(s.env_steps <= POP_SIZE * MAX_STEPS for s in per_gen)
    assert np.isfinite(result.metrics["final_best"]["learner"])
