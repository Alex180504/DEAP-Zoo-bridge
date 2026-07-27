"""Regression tests asserting that off-policy HPO must store transitions and call learn()."""

import pytest
from unittest.mock import patch

pytest.importorskip(
    "agilerl",
    reason="AgileRL not installed; run `pip install 'deap-pettingzoo[rl-baselines]'`.",
)
pytest.importorskip("torch", reason="PyTorch not installed.")
pytest.importorskip("mpe2", reason="mpe2 not installed.")
pytestmark = pytest.mark.slow

from deap_pettingzoo.baselines.agilerl_hpo_trainer import MADDPGHPOTrainer
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


def _hpo_config(
    seed: int = 0,
    *,
    budget: int = 200,
    evo_every: int = 50,
    batch_size: int = 2,
    memory_size: int = 500,
    learning_delay: int = 0,
    learn_step: int = 1,
) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=budget,
        num_seeds=1,
        seed=seed,
        algorithm_params={
            "max_steps_per_episode": 25,
            "population_size": 2,
            "evo_every_n_steps": evo_every,
            "learning_delay": learning_delay,
            "learn_step": learn_step,
            "batch_size": batch_size,
            "memory_size": memory_size,
        },
        env_config={"max_cycles": 25},
    )


class TestHPOStoresAndLearns:
    """TOff-policy HPO must call learn()."""

    def test_hpo_stores_and_learns(self):
        """
        learn() is invoked: proves a real buffer was built and sampled.
        """
        from agilerl.algorithms.maddpg import MADDPG

        learn_calls: list = []
        _orig = MADDPG.learn

        def _spy(self, *args, **kwargs):
            learn_calls.append(True)
            return _orig(self, *args, **kwargs)

        config = _hpo_config(
            seed=0,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=1,
        )
        with patch.object(MADDPG, "learn", _spy):
            trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
            trainer.train(config)

        assert len(learn_calls) > 0, (
            "agent.learn() was never invoked — buffer may be None or never "
            "reached batch_size. This is the B3 regression."
        )

    def test_hpo_replay_buffer_persists_across_generations(self):
        """The off-policy replay buffer is built once per population slot.

        Regression for "rebuilt from empty every generation": with N slots and
        G>=2 generations, exactly N buffers should be constructed for the whole
        run (one per persistent slot), not N*G.
        """
        from agilerl.components.multi_agent_replay_buffer import (
            MultiAgentReplayBuffer,
        )

        init_calls: list = []
        _orig_init = MultiAgentReplayBuffer.__init__

        def _init_spy(self, *args, **kwargs):
            init_calls.append(True)
            return _orig_init(self, *args, **kwargs)

        config = _hpo_config(
            seed=0,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=1,
        )
        with patch.object(MultiAgentReplayBuffer, "__init__", _init_spy):
            trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
            result = trainer.train(config)

        n_generations = len(result.metrics["fitness_history"])
        assert n_generations >= 2, (
            f"Test needs >=2 generations to be meaningful (got {n_generations})"
        )
        assert len(init_calls) == 2, (
            f"Expected one replay buffer per population slot (2), got "
            f"{len(init_calls)} — the buffer is being rebuilt each generation."
        )

    def test_eval_episodes_surfaced_and_hall_of_fame_selection(self):
        """`eval_episodes`/`final_eval_episodes` drive the eval counts, and the
        final champion is a single best-ever hall-of-fame re-score.

        The per-generation selection eval runs `eval_episodes` episodes for
        every agent; the champion is then re-scored by exactly ONE final eval
        (the best-ever HOF member), not one eval per final-population agent.
        """
        from deap_pettingzoo.baselines.agilerl_hpo_trainer import EvolveHPOTrainer

        calls: list = []
        _orig = EvolveHPOTrainer._evaluate_agent

        def _spy(agent, env, agent_ids, max_steps, seed, rng, num_episodes=3):
            calls.append(num_episodes)
            return _orig(
                agent, env, agent_ids, max_steps, seed, rng, num_episodes=num_episodes
            )

        config = _hpo_config(
            seed=0,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=1,
        )
        config.algorithm_params["eval_episodes"] = 2
        config.algorithm_params["final_eval_episodes"] = 4

        with patch.object(EvolveHPOTrainer, "_evaluate_agent", staticmethod(_spy)):
            trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
            result = trainer.train(config)

        pop = 2
        n_gen = len(result.metrics["fitness_history"])
        assert n_gen >= 2, f"Test needs >=2 generations (got {n_gen})"

        per_gen = calls[: pop * n_gen]
        final = calls[pop * n_gen :]
        assert per_gen == [2] * (pop * n_gen), (
            f"Per-generation eval must use eval_episodes=2; got {per_gen}"
        )
        assert final == [4], (
            f"Expected exactly one final HOF re-score with final_eval_episodes=4; "
            f"got {final} (a per-final-population re-select would be length {pop})."
        )
        assert result.policy is not None

    def test_hpo_hall_of_fame_top_n_export(self, tmp_path):
        """hof_size>1 retains the top-N, exports checkpoints, and the summary is
        ordered by the clean final re-score (rank 1 == returned champion == max)."""
        config = _hpo_config(
            seed=0,
            budget=300,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=1,
        )
        config.algorithm_params["population_size"] = 3
        config.algorithm_params["hof_size"] = 3
        config.algorithm_params["final_eval_episodes"] = 3
        object.__setattr__(config, "output_dir", tmp_path)

        trainer = MADDPGHPOTrainer(population_size=3, device="cpu")
        result = trainer.train(config)

        # In-memory top-N, best first; [0] is the returned champion.
        assert result.hall_of_fame is not None
        assert len(result.hall_of_fame) == 3
        assert result.hall_of_fame[0] is result.policy

        # Summary ranked 1..3 with fitness descending under the final re-score,
        # and final_fitness is the top (this is the finding-1 regression guard).
        summary = result.metrics["hall_of_fame"]
        assert [row["rank"] for row in summary] == [1, 2, 3]
        fits = [row["fitness"] for row in summary]
        assert fits == sorted(fits, reverse=True)
        assert result.metrics["final_fitness"] == pytest.approx(fits[0])

        # Disk: champion agent.pt + hof_02.pt + hof_03.pt.
        ckpts = {p.name for p in tmp_path.glob("**/checkpoints/*.pt")}
        assert {"agent.pt", "hof_02.pt", "hof_03.pt"} <= ckpts

    def test_hpo_champion_improves(self):
        """Peak champion fitness across generations does not fall below gen-1.

        Guards against a catastrophically broken rollout-and-learn loop where
        the population never receives any learning signal.
        """
        config = _hpo_config(
            seed=2,
            budget=800,
            evo_every=100,
            batch_size=4,
            memory_size=2000,
            learning_delay=0,
            learn_step=10,
        )
        trainer = MADDPGHPOTrainer(population_size=2, device="cpu")
        result = trainer.train(config)

        history = result.metrics["fitness_history"]
        assert len(history) >= 2, "Expected at least 2 generations for comparison"
        gen1_max = history[0]["max"]
        # Peak across all generations (not just final) to avoid lucky-dip regression.
        peak_max = max(g["max"] for g in history[1:])
        assert peak_max >= gen1_max, (
            f"Peak post-gen-1 fitness {peak_max:.2f} is below gen-1 max "
            f"{gen1_max:.2f} — the population may be gaining no learning signal."
        )

    def test_hpo_is_reproducible(self):
        """Two same-seed HPO runs yield identical champion fitness histories."""
        config_a = _hpo_config(
            seed=7,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=5,
        )
        config_b = _hpo_config(
            seed=7,
            budget=200,
            evo_every=50,
            batch_size=2,
            memory_size=500,
            learning_delay=0,
            learn_step=5,
        )
        trainer_a = MADDPGHPOTrainer(population_size=2, device="cpu")
        trainer_b = MADDPGHPOTrainer(population_size=2, device="cpu")

        result_a = trainer_a.train(config_a)
        result_b = trainer_b.train(config_b)

        hist_a = result_a.metrics["fitness_history"]
        hist_b = result_b.metrics["fitness_history"]

        assert len(hist_a) == len(hist_b), (
            f"Different number of generations: {len(hist_a)} vs {len(hist_b)}"
        )
        for i, (ga, gb) in enumerate(zip(hist_a, hist_b)):
            assert abs(ga["max"] - gb["max"]) < 1e-6, (
                f"Gen {i + 1} max fitness mismatch: "
                f"{ga['max']:.6f} vs {gb['max']:.6f}"
            )


def _coev_hpo_side(seed: int = 0):
    """Build and initialise an off-policy rl_hpo co-evolution side on a raw env.

    Returns ``(state, env, learner_ids)``. The env is a *raw* mpe2 parallel
    env — the format ``RLEvolveHPOSideState._train_agent`` consumes (its
    ``reset()`` returns ``(obs, info)`` and ``step()`` a 5-tuple), matching the
    ``evaluator._env`` the trainer hands it.
    """
    import mpe2
    import numpy as np

    from deap_pettingzoo.baselines.coevolution import Side, make_side_state

    env = mpe2.simple_spread_v3.parallel_env(N=3, max_cycles=25)
    env.reset(seed=seed)
    side = Side(
        "learners", ("agent",), "rl_hpo",
        {"algorithm": "maddpg_hpo", "population_size": 2,
         "evo_every_n_steps": 60, "device": "cpu", "seed": seed},
    )
    state = make_side_state(side)
    state.initialise(env, rng=np.random.default_rng(seed))
    return state, env, state._learner_ids


class TestCoevolutionHPOSideLearns:
    """The rl_hpo co-evolution side must store transitions and call learn()..
    """

    def test_coev_hpo_side_stores_and_learns(self):
        """save_to_memory and learn() are both invoked during off-policy training."""
        from agilerl.algorithms.maddpg import MADDPG
        from agilerl.components.multi_agent_replay_buffer import (
            MultiAgentReplayBuffer,
        )
        from deap_pettingzoo.baselines.coevolution.rl_hpo_side import (
            RLEvolveHPOSideState,
        )

        state, env, learner_ids = _coev_hpo_side(seed=0)
        agent = state._population[0]
        # Force learning to fire quickly within a small step budget.
        agent.batch_size = 4
        agent.learn_step = 1
        agent.learning_delay = 0

        store_calls: list = []
        learn_calls: list = []
        _orig_store = MultiAgentReplayBuffer.save_to_memory
        _orig_learn = MADDPG.learn

        def _store_spy(self, *args, **kwargs):
            store_calls.append(True)
            return _orig_store(self, *args, **kwargs)

        def _learn_spy(self, *args, **kwargs):
            learn_calls.append(True)
            return _orig_learn(self, *args, **kwargs)

        try:
            with patch.object(MultiAgentReplayBuffer, "save_to_memory", _store_spy), \
                    patch.object(MADDPG, "learn", _learn_spy):
                RLEvolveHPOSideState._train_agent(
                    agent, env, learner_ids, target_steps=60,
                )
        finally:
            env.close()

        assert len(store_calls) > 0, (
            "save_to_memory was never called — the co-evolution HPO side built "
            "no replay buffer."
        )
        assert len(learn_calls) > 0, (
            "agent.learn() was never invoked — the co-evolution HPO side does "
            "no learning."
        )

    def test_coev_hpo_side_buffer_persists_across_generations(self):
        """The rl_hpo side builds one persistent replay buffer per slot and reuses it."""
        from agilerl.components.multi_agent_replay_buffer import (
            MultiAgentReplayBuffer,
        )
        from deap_pettingzoo.baselines.coevolution.rl_hpo_side import (
            RLEvolveHPOSideState,
        )

        init_calls: list = []
        _orig_init = MultiAgentReplayBuffer.__init__

        def _init_spy(self, *args, **kwargs):
            init_calls.append(True)
            return _orig_init(self, *args, **kwargs)

        with patch.object(MultiAgentReplayBuffer, "__init__", _init_spy):
            state, env, learner_ids = _coev_hpo_side(seed=0)  # population_size=2

        try:
            # One persistent buffer per population slot, built at initialise().
            assert len(state._replay_slots) == state.population_size == 2
            assert len(init_calls) == 2, (
                f"Expected one buffer per slot at init (2), got {len(init_calls)}."
            )

            # Training against a provided slot buffer reuses it (no new build).
            before = len(init_calls)
            agent = state._population[0]
            agent.batch_size = 4
            agent.learn_step = 1
            agent.learning_delay = 0
            with patch.object(MultiAgentReplayBuffer, "__init__", _init_spy):
                steps, total = RLEvolveHPOSideState._train_agent(
                    agent,
                    env,
                    learner_ids,
                    target_steps=40,
                    memory=state._replay_slots[0],
                    total_transitions=state._slot_transitions[0],
                )
            assert len(init_calls) == before, (
                "A provided per-slot buffer must be reused, not rebuilt."
            )
            assert steps >= 1 and total >= steps
        finally:
            env.close()

    def test_coev_hpo_side_learn_updates_params(self):
        """Actor parameters change after _train_agent — end-to-end proof of learning."""
        import torch

        from deap_pettingzoo.baselines.coevolution.rl_hpo_side import (
            RLEvolveHPOSideState,
        )

        state, env, learner_ids = _coev_hpo_side(seed=1)
        agent = state._population[0]
        agent.batch_size = 4
        agent.learn_step = 1
        agent.learning_delay = 0

        actor = list(agent.actors.values())[0]
        before = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()

        try:
            RLEvolveHPOSideState._train_agent(
                agent, env, learner_ids, target_steps=80,
            )
        finally:
            env.close()

        after = torch.cat(
            [p.detach().flatten() for p in actor.parameters()]
        ).clone()
        assert not torch.allclose(before, after), (
            "Actor parameters did not change — no gradient updates occurred."
        )


class TestCoevolutionHPOSideForwardsCtorKwargs:
    """Regression test: ``agent_ctor_kwargs`` must reach the agents.
    """

    def test_ctor_kwargs_reach_constructed_agents(self):
        """Non-default ctor kwargs land on every population member."""
        import mpe2
        import numpy as np

        from deap_pettingzoo.baselines.coevolution import Side, make_side_state

        # Non-default values, distinct from MADDPG's defaults
        # (batch_size=64, lr_actor=1e-3, learn_step=5, gamma=0.95, tau=0.01).
        ctor_kwargs = {
            "batch_size": 7,
            "lr_actor": 3e-4,
            "learn_step": 11,
            "gamma": 0.9,
            "tau": 0.02,
        }

        env = mpe2.simple_spread_v3.parallel_env(N=3, max_cycles=25)
        env.reset(seed=0)
        try:
            side = Side(
                "learners", ("agent",), "rl_hpo",
                {"algorithm": "maddpg_hpo", "population_size": 2,
                 "evo_every_n_steps": 60, "device": "cpu", "seed": 0,
                 "agent_ctor_kwargs": ctor_kwargs},
            )
            state = make_side_state(side)
            state.initialise(env, rng=np.random.default_rng(0))

            assert len(state._population) == 2
            for agent in state._population:
                assert agent.batch_size == 7
                assert agent.lr_actor == pytest.approx(3e-4)
                assert agent.learn_step == 11
                assert agent.gamma == pytest.approx(0.9)
                assert agent.tau == pytest.approx(0.02)
        finally:
            env.close()
