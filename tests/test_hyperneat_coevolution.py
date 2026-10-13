"""Tests for HyperNEAT / ES-HyperNEAT co-evolution side states."""

import numpy as np
import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.baselines.coevolution.evaluator import (  # noqa: E402
    CoEvolutionEvaluator,
    GenomeIndividual,
)
from deap_pettingzoo.baselines.coevolution.pools import (  # noqa: E402
    FrozenPolicyPool,
    rebuild_opponent_policy,
)
from deap_pettingzoo.baselines.coevolution.sides import (  # noqa: E402
    Side,
    make_side_state,
)
from deap_pettingzoo.contrib.neat.hyperneat_coevolution import (  # noqa: E402
    HyperNeatSideState,
    ESHyperNeatSideState,
)

SUBSTRATE_HIDDEN_LAYERS = [4]


class _ZeroPolicy:
    """Opponent double — always picks action 0."""

    def get_action(self, observation, deterministic=True, *, action_mask=None):
        del observation, deterministic, action_mask
        return 0

    def reset(self):
        pass


def _make_evaluator(state, env):
    """Build the real co-evolution evaluator this side is meant to consume.

    ``agent_0`` is the learner side; the remaining agents are driven by a
    pool-supplied opponent, which is exactly the partition the side's own
    ``step_generation`` used to ignore.
    """
    return CoEvolutionEvaluator(
        env_wrapper=env,
        learner_role_prefixes=("agent_0",),
        learner_factory=state.learner_policy_factory(),
        opponent_sides=[
            ("opponent", ("agent_1", "agent_2"), FrozenPolicyPool([_ZeroPolicy()]))
        ],
        k_opponents=1,
        max_steps=10,
    )


class TestHyperNeatSideState:
    @pytest.fixture
    def side(self):
        return Side(
            name="hyperneat_agents",
            role_prefixes=("agent_0",),
            optimizer="hyperneat",
            optimizer_config={
                "pop_size": 6,
                "seed": 42,
                "n_generations_per_step": 1,
                "substrate_hidden_layers": SUBSTRATE_HIDDEN_LAYERS,
            },
        )

    @pytest.fixture
    def env(self):
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )
        return PettingZooWrapper(
            "simple_spread_v3",
            env_config={"max_cycles": 25, "continuous_actions": False},
        )

    def test_factory_dispatches(self, side):
        state = make_side_state(side)
        assert isinstance(state, HyperNeatSideState)

    def test_initialise(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        assert state._population_obj is not None
        assert state._neat_config is not None
        assert len(state._genomes_by_id) == 6

    def test_best_policy(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        policy = state.best_policy(env)
        assert policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))

    def test_current_pool(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        pool = state.current_pool()
        assert len(pool) == 6
        rng = np.random.default_rng(42)
        opp = pool.sample_opponent(rng)
        obs = np.random.randn(18).astype(np.float32)
        action = opp.get_action(obs)
        assert isinstance(action, (int, np.integer))

    def test_current_pool_blends_the_hall_of_fame(self, side, env):
        """``use_archive_in_opponent_pool`` must not be silently ignored."""
        from deap_pettingzoo.baselines.coevolution import RingHallOfFame
        from deap_pettingzoo.baselines.coevolution.pools import CompositePool

        side.hof = RingHallOfFame(capacity=4)
        side.use_archive_in_opponent_pool = True
        side.archive_mix = 0.5
        state = make_side_state(side)
        state.initialise(env)
        # Empty archive -> current generation only.
        assert not isinstance(state.current_pool(), CompositePool)

        state.snapshot_best()
        pool = state.current_pool()
        assert isinstance(pool, CompositePool)
        assert len(pool.sub_pools) == 2

    def test_real_snapshot_rebuild_step(self, side, env):
        """The real producer chain (train one
        generation -> current_pool() -> snapshot_opponents() -> rebuild ->
        step) must not crash with an input-size mismatch.
        """
        state = make_side_state(side)
        state.initialise(env)
        state.step_generation(lambda _side, _ev=_make_evaluator(state, env): _ev)

        pool = state.current_pool()
        rng = np.random.default_rng(0)
        [snapshot] = pool.snapshot_opponents(rng, 1)
        assert snapshot.kind == "hyperneat"
        # Workers must build the *same* substrate the driver did — without
        # the hidden layers they express a different phenotype.
        assert snapshot.payload["hidden_layers"] == SUBSTRATE_HIDDEN_LAYERS

        policy = rebuild_opponent_policy(
            snapshot, action_space=env.action_space(env.possible_agents[0])
        )
        assert (
            policy.substrate.hidden_coordinates
            == state._substrate.hidden_coordinates
        )

        obs = env.reset()
        agent0 = env.possible_agents[0]
        action = policy.get_action(obs[agent0])
        assert isinstance(action, (int, np.integer))

    def test_step_generation_uses_the_supplied_evaluator(self, side, env):
        """Fitness must come from the trainer's evaluator, not a private loop."""
        state = make_side_state(side)
        state.initialise(env)
        evaluator = _make_evaluator(state, env)

        stats = state.step_generation(lambda _side: evaluator)

        # 6 genomes x 1 opponent sample x 10 capped steps.
        assert evaluator.env_steps_consumed == 60
        assert stats.env_steps == 60


class TestESHyperNeatSideState:
    @pytest.fixture
    def side(self):
        return Side(
            name="es_hyperneat_agents",
            role_prefixes=("agent_0",),
            optimizer="es_hyperneat",
            optimizer_config={
                "pop_size": 6,
                "seed": 42,
                "n_generations_per_step": 1,
            },
        )

    @pytest.fixture
    def env(self):
        from deap_pettingzoo.environments.pettingzoo_wrapper import (
            PettingZooWrapper,
        )
        return PettingZooWrapper(
            "simple_spread_v3",
            env_config={"max_cycles": 25, "continuous_actions": False},
        )

    def test_factory_dispatches(self, side):
        state = make_side_state(side)
        assert isinstance(state, ESHyperNeatSideState)

    def test_initialise(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        assert state._population_obj is not None
        assert len(state._genomes_by_id) == 6

    def test_best_policy(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        policy = state.best_policy(env)
        assert policy is not None
        obs = np.random.randn(18).astype(np.float32)
        action = policy.get_action(obs)
        assert isinstance(action, (int, np.integer))

    def test_real_snapshot_rebuild_step(self, side, env):
        """Test the real snapshot rebuild step,
        mirroring `test_real_snapshot_rebuild_step` for ES-HyperNEAT."""
        state = make_side_state(side)
        state.initialise(env)
        state.step_generation(lambda _side, _ev=_make_evaluator(state, env): _ev)

        pool = state.current_pool()
        rng = np.random.default_rng(0)
        [snapshot] = pool.snapshot_opponents(rng, 1)
        assert snapshot.kind == "es_hyperneat"

        policy = rebuild_opponent_policy(
            snapshot, action_space=env.action_space(env.possible_agents[0])
        )
        obs = env.reset()
        agent0 = env.possible_agents[0]
        action = policy.get_action(obs[agent0])
        assert isinstance(action, (int, np.integer))

    def test_phenotype_is_built_once_per_genome(self, side, env):
        """ES-HyperNEAT quadtree expansion must not run per agent per episode."""
        state = make_side_state(side)
        state.initialise(env)
        genome = next(iter(state._population_obj.population.values()))
        factory = state.learner_policy_factory()
        agent = env.possible_agents[0]
        space = env.action_space(agent)

        first = factory(genome, agent, space)
        second = factory(genome, env.possible_agents[1], space)
        assert first is second

    def test_learner_snapshot_carries_es_params(self, side, env):
        state = make_side_state(side)
        state.initialise(env)
        genome = next(iter(state._population_obj.population.values()))

        snapshot = state.learner_snapshot(GenomeIndividual(genome=genome).genome)
        assert snapshot.kind == "es_hyperneat"
        assert snapshot.payload["es_params"] == state._es_params
