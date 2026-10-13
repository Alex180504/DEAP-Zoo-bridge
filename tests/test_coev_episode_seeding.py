"""Every evaluation path must seed episodes by the same rule.

Before :func:`episode_seed_for` the paths disagreed: the evaluator derived
``seed + i`` from a caller seed nothing actually passed, the QDax
single-policy evaluator hard-coded ``seed=i`` (so a whole run trained against
the same k starting states), and only the population-batch path honoured
``common_random_seeds`` — which made that flag a silent no-op on serial runs
and for every QDax/NEAT side. None of that is visible in an archive, so the
tests below pin the rule rather than any one path's behaviour.
"""

from __future__ import annotations

import pytest

from deap_pettingzoo.baselines.coevolution.evaluator import episode_seed_for


class TestSeedRule:
    def test_unseeded_stays_unseeded(self):
        """No caller seed and no CRN means the env picks its own start."""
        assert episode_seed_for(
            base_seed=None, episode_index=0, k_opponents=4
        ) is None

    def test_caller_seed_offsets_by_episode(self):
        seeds = [
            episode_seed_for(base_seed=100, episode_index=i, k_opponents=4)
            for i in range(4)
        ]
        assert seeds == [100, 101, 102, 103]

    def test_crn_ignores_the_caller_seed(self):
        """Under CRN the seed must not depend on per-individual state."""
        a = episode_seed_for(
            base_seed=None, episode_index=2, k_opponents=4,
            generation_index=3, common_random_seeds=True,
        )
        b = episode_seed_for(
            base_seed=99_999, episode_index=2, k_opponents=4,
            generation_index=3, common_random_seeds=True,
        )
        assert a == b == 3 * 4 + 2

    def test_crn_generations_do_not_reuse_seeds(self):
        """Otherwise the run trains against one fixed set of starts forever."""
        k = 5
        gen0 = {
            episode_seed_for(
                base_seed=None, episode_index=i, k_opponents=k,
                generation_index=0, common_random_seeds=True,
            )
            for i in range(k)
        }
        gen1 = {
            episode_seed_for(
                base_seed=None, episode_index=i, k_opponents=k,
                generation_index=1, common_random_seeds=True,
            )
            for i in range(k)
        }
        assert gen0.isdisjoint(gen1)

    def test_crn_is_stable_within_a_generation(self):
        """The property the whole technique rests on."""
        per_individual = [
            [
                episode_seed_for(
                    base_seed=None, episode_index=i, k_opponents=3,
                    generation_index=7, common_random_seeds=True,
                )
                for i in range(3)
            ]
            for _ in range(5)
        ]
        assert all(block == per_individual[0] for block in per_individual)

    @pytest.mark.parametrize("k", [0, -1])
    def test_degenerate_k_does_not_collapse_generations(self, k):
        """A bad k must not make every generation share one seed."""
        a = episode_seed_for(
            base_seed=None, episode_index=0, k_opponents=k,
            generation_index=1, common_random_seeds=True,
        )
        b = episode_seed_for(
            base_seed=None, episode_index=0, k_opponents=k,
            generation_index=2, common_random_seeds=True,
        )
        assert a != b


class TestPathsAgree:
    """The evaluator and the QDax single-policy path apply the same rule."""

    @staticmethod
    def _evaluator(common_random_seeds, generation_index, k=4):
        from deap_pettingzoo.baselines.coevolution.evaluator import (
            CoEvolutionEvaluator,
        )

        ev = CoEvolutionEvaluator.__new__(CoEvolutionEvaluator)
        ev.k_opponents = k
        ev.common_random_seeds = common_random_seeds
        ev.generation_index = generation_index
        return ev

    def test_evaluator_matches_the_rule_under_crn(self):
        ev = self._evaluator(True, 2)
        assert [ev._episode_seed(None, i) for i in range(4)] == [8, 9, 10, 11]

    def test_evaluator_matches_the_rule_without_crn(self):
        ev = self._evaluator(False, 2)
        assert [ev._episode_seed(None, i) for i in range(4)] == [None] * 4
        assert [ev._episode_seed(50, i) for i in range(4)] == [50, 51, 52, 53]

    def test_qdax_path_keeps_its_historical_seeds_without_crn(self):
        """``base_seed=0`` reproduces the old hard-coded ``seed=i``."""
        assert [
            episode_seed_for(base_seed=0, episode_index=i, k_opponents=4)
            for i in range(4)
        ] == [0, 1, 2, 3]

    def test_qdax_path_varies_across_generations_under_crn(self):
        """The old behaviour pinned the same starts for the whole run."""
        gen0 = [
            episode_seed_for(
                base_seed=0, episode_index=i, k_opponents=4,
                generation_index=0, common_random_seeds=True,
            )
            for i in range(4)
        ]
        gen5 = [
            episode_seed_for(
                base_seed=0, episode_index=i, k_opponents=4,
                generation_index=5, common_random_seeds=True,
            )
            for i in range(4)
        ]
        assert gen0 == [0, 1, 2, 3]
        assert gen5 == [20, 21, 22, 23]


class TestTrainerPropagatesGeneration:
    """CRN is useless if the evaluator never learns which generation it is."""

    def test_evaluator_receives_the_flag_and_generation(self):
        from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
        from deap_pettingzoo.baselines.coevolution.evaluator import (
            CoEvolutionEvaluator,
        )
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        pytest.importorskip("pettingzoo")
        tiny = {"population_size": 4, "hidden_size": 8, "seed": 0}
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pred", ("adversary",), "ea", dict(tiny)),
                Side("prey", ("agent",), "ea", dict(tiny)),
            ],
            k_opponents=2,
            common_random_seeds=True,
            max_steps_per_episode=5,
        )
        captured = {}
        original = CoEvolutionEvaluator.__init__

        def spy(self, *args, **kwargs):
            original(self, *args, **kwargs)
            captured.setdefault("crn", self.common_random_seeds)

        CoEvolutionEvaluator.__init__ = spy
        try:
            CoEvolutionTrainer().train(
                TrainingConfig(
                    env_name="simple_tag_v3",
                    budget_mode=BudgetMode.EVAL_EPISODES,
                    budget_value=20,
                    coevolution_config=cfg,
                )
            )
        finally:
            CoEvolutionEvaluator.__init__ = original

        assert captured.get("crn") is True
