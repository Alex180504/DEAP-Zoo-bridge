"""Surface tests for AgileRL HPO trainer.

:class:`TestEvolveHPOTrainerInit`, :class:`TestConcreteTrainers` and
:class:`TestBudgetMode` require only the core package (no agilerl).
:class:`TestBuildHPConfig` directly imports agilerl and is skipped when
the ``rl-baselines`` extra is not installed.
"""
import importlib

import pytest

_AGILERL_AVAILABLE = importlib.util.find_spec("agilerl") is not None

import inspect

from deap_pettingzoo.baselines.agilerl_hpo_trainer import (
    EvolveHPOTrainer,
    GroupedIPPOHPOTrainer,
    IPPOHPOTrainer,
    MADDPGHPOTrainer,
    MATD3HPOTrainer,
)
from deap_pettingzoo.baselines.base import BudgetMode


class TestEvolveHPOTrainerInit:
    def test_defaults(self):
        t = EvolveHPOTrainer()
        assert t._hpo_population_size >= 2
        assert t._evo_every_n_steps > 0

    def test_population_size_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="population_size"):
            EvolveHPOTrainer(population_size=1)

    def test_mutation_probs_defaults(self):
        t = EvolveHPOTrainer()
        assert "no_mutation" in t._mutation_probs
        assert "architecture" in t._mutation_probs
        assert "rl_hp" in t._mutation_probs

    def test_custom_mutation_probs(self):
        t = EvolveHPOTrainer(mutation_probs={"no_mutation": 0.5, "rl_hp": 0.5})
        assert t._mutation_probs["no_mutation"] == 0.5
        assert t._mutation_probs["rl_hp"] == 0.5
        assert "architecture" in t._mutation_probs  # filled by default


class TestConcreteTrainers:
    def test_maddpg_hpo_name(self):
        t = MADDPGHPOTrainer()
        assert t.ALGORITHM_NAME == "maddpg"
        assert not t.ON_POLICY

    def test_matd3_hpo_name(self):
        t = MATD3HPOTrainer()
        assert t.ALGORITHM_NAME == "matd3"
        assert not t.ON_POLICY

    def test_budget_support(self):
        for cls in (MADDPGHPOTrainer, MATD3HPOTrainer):
            t = cls()
            assert t.supports_budget_mode(BudgetMode.ENV_STEPS)
            assert t.supports_budget_mode(BudgetMode.EVAL_EPISODES)

    def test_capabilities(self):
        for cls in (MADDPGHPOTrainer, MATD3HPOTrainer):
            t = cls()
            caps = t.get_capabilities()
            assert caps["discrete_action"] is True
            assert caps["continuous_action"] is True

    def test_ippo_hpo_name(self):
        t = IPPOHPOTrainer()
        assert t.ALGORITHM_NAME == "ippo"
        assert t.ON_POLICY

    def test_grouped_ippo_hpo_name(self):
        t = GroupedIPPOHPOTrainer()
        assert t.ALGORITHM_NAME == "ippo-grouped"
        assert t.ON_POLICY

    def test_ippo_hpo_capabilities(self):
        for cls in (IPPOHPOTrainer, GroupedIPPOHPOTrainer):
            caps = cls().get_capabilities()
            assert caps["on_policy"] is True
            # On-policy bases support MultiDiscrete; HPO is never turn-based.
            assert caps["multidiscrete_action"] is True
            assert caps["supports_turn_based"] is False


class TestTrainSignature:
    def test_train_accepts_progress_callback(self):
        # The runner always calls train(config, progress_callback=...), so the
        # HPO override must accept it (regression for the missing-kwarg crash).
        sig = inspect.signature(EvolveHPOTrainer.train)
        assert "progress_callback" in sig.parameters
        assert sig.parameters["progress_callback"].kind is inspect.Parameter.KEYWORD_ONLY


class TestHyperparameterOverlay:
    def test_includes_hpo_knobs(self):
        names = {p.name for p in EvolveHPOTrainer.describe_hyperparameters()}
        for knob in ("population_size", "evo_every_n_steps", "tournament_size",
                     "mutation_sd", "prob_no_mutation", "prob_rl_hp"):
            assert knob in names, knob
        # The inherited AgileRL surface is still present.
        assert "seed" in names
        assert "batch_size" in names

    def test_surfaces_eval_episode_knobs(self):
        specs = {p.name: p for p in EvolveHPOTrainer.describe_hyperparameters()}
        assert "eval_episodes" in specs, "per-generation eval count not surfaced"
        assert "final_eval_episodes" in specs, "final eval count not surfaced"
        # Defaults match the historical hardcoded values (3 per-gen, 5 final).
        assert specs["eval_episodes"].default == 3
        assert specs["final_eval_episodes"].default == 5

    def test_surfaces_hof_size_knob(self):
        specs = {p.name: p for p in EvolveHPOTrainer.describe_hyperparameters()}
        assert "hof_size" in specs, "hall-of-fame size not surfaced"
        # Default 1 keeps the single-champion behaviour unchanged.
        assert specs["hof_size"].default == 1
        assert specs["hof_size"].minimum == 1


class TestAgentHallOfFame:
    """Unit tests for the best-N hall-of-fame (no agilerl needed)."""

    class _FakeAgent:
        """Minimal stand-in exposing the ``.clone()`` the HOF relies on."""

        def __init__(self, tag):
            self.tag = tag

        def clone(self):
            return TestAgentHallOfFame._FakeAgent(self.tag)

    def _pop(self, tags):
        return [self._FakeAgent(t) for t in tags]

    def test_size_one_keeps_single_best(self):
        from deap_pettingzoo.baselines.agilerl_hpo_trainer import _AgentHallOfFame

        hof = _AgentHallOfFame(1)
        hof.update(self._pop(["a", "b", "c"]), [1.0, 3.0, 2.0])
        assert len(hof) == 1
        assert hof.champion.tag == "b"
        assert hof.fitnesses == [3.0]

    def test_keeps_top_n_best_first(self):
        from deap_pettingzoo.baselines.agilerl_hpo_trainer import _AgentHallOfFame

        hof = _AgentHallOfFame(3)
        hof.update(self._pop(["a", "b", "c", "d", "e"]), [1.0, 5.0, 2.0, 4.0, 3.0])
        assert [m.tag for m in hof.items] == ["b", "d", "e"]
        assert hof.fitnesses == [5.0, 4.0, 3.0]

    def test_best_ever_survives_a_worse_generation(self):
        from deap_pettingzoo.baselines.agilerl_hpo_trainer import _AgentHallOfFame

        hof = _AgentHallOfFame(2)
        hof.update(self._pop(["hi", "mid"]), [10.0, 6.0])
        # A later, uniformly-worse generation must not evict the earlier best.
        hof.update(self._pop(["lo1", "lo2"]), [1.0, 2.0])
        assert [m.tag for m in hof.items] == ["hi", "mid"]
        assert hof.fitnesses == [10.0, 6.0]

    def test_clones_only_qualifying_candidates(self):
        from deap_pettingzoo.baselines.agilerl_hpo_trainer import _AgentHallOfFame

        clones = {"n": 0}

        class _CountingAgent:
            def __init__(self, tag):
                self.tag = tag

            def clone(self):
                clones["n"] += 1
                return _CountingAgent(self.tag)

        hof = _AgentHallOfFame(1)
        pop = [_CountingAgent(t) for t in ("a", "b", "c")]
        hof.update(pop, [1.0, 3.0, 2.0])
        # Size-1 HOF over a 3-agent generation clones exactly once (the best).
        assert clones["n"] == 1

    def test_mutation_probs_from_flat_keys(self):
        t = EvolveHPOTrainer()
        probs = t._assemble_mutation_probs(
            {"prob_no_mutation": 0.7, "prob_rl_hp": 0.1}
        )
        assert probs["no_mutation"] == 0.7
        assert probs["rl_hp"] == 0.1
        # Untouched keys keep the constructor defaults.
        assert "architecture" in probs

    def test_nested_mutation_probs_still_supported(self):
        t = EvolveHPOTrainer()
        probs = t._assemble_mutation_probs({"mutation_probs": {"architecture": 0.5}})
        assert probs["architecture"] == 0.5


@pytest.mark.skipif(not _AGILERL_AVAILABLE, reason="rl-baselines extra not installed")
class TestBuildHPConfig:
    def test_empty_spec_returns_empty_config(self):
        from agilerl.algorithms.core.registry import (
            HyperparameterConfig,
            RLParameter,
        )
        result = EvolveHPOTrainer._build_hp_config(
            {}, HyperparameterConfig, RLParameter,
        )
        assert not result  # bool(HyperparameterConfig()) is False

    def test_simple_range_spec(self):
        from agilerl.algorithms.core.registry import (
            HyperparameterConfig,
            RLParameter,
        )
        spec = {"lr": [1e-4, 1e-2]}
        result = EvolveHPOTrainer._build_hp_config(
            spec, HyperparameterConfig, RLParameter,
        )
        assert result  # non-empty
        assert "lr" in result
